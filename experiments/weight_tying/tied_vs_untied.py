from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import load_dataset, set_seed


torch.backends.cuda.matmul.allow_tf32 = True


DEFAULT_MODEL_DIM = 128
DEFAULT_MLP_DIM = 512
DEFAULT_N_HEADS = 4
CHUNK_SIZE = 128
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 3e-4
VAL_BATCH_SEED = 17_241
TRAIN_EVAL_BATCH_SEED = 9_137
TRAIN_BATCH_SEED_OFFSET = 100_000


class CausalSelfAttention(nn.Module):
    def __init__(self, *, model_dim: int, n_heads: int) -> None:
        super().__init__()
        if model_dim % n_heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by n_heads={n_heads}.")

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x: Float[Tensor, "batch seq dim"]) -> Float[Tensor, "batch seq dim"]:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        else:
            scale = self.head_dim**-0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
            weights = scores.softmax(dim=-1)
            attention = torch.matmul(weights, v)

        merged = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        return self.out_proj(merged)


class TransformerBlock(nn.Module):
    def __init__(self, *, model_dim: int, n_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(model_dim)
        self.attn = CausalSelfAttention(model_dim=model_dim, n_heads=n_heads)
        self.mlp_norm = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, model_dim),
        )

    def forward(self, x: Float[Tensor, "batch seq dim"]) -> Float[Tensor, "batch seq dim"]:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TiedTransformer(nn.Module):
    def __init__(
        self,
        *,
        token_embedding: nn.Embedding,
        position_embedding: nn.Embedding,
        block: TransformerBlock,
        final_norm: nn.LayerNorm,
        lm_head: nn.Linear,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.position_embedding = position_embedding
        self.block = block
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.n_layers = n_layers

    def forward(self, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch seq vocab"]:
        _, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        for _ in range(self.n_layers):
            hidden = self.block(hidden)
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)


class UntiedTransformer(nn.Module):
    def __init__(
        self,
        *,
        token_embedding: nn.Embedding,
        position_embedding: nn.Embedding,
        blocks: nn.ModuleList,
        final_norm: nn.LayerNorm,
        lm_head: nn.Linear,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.position_embedding = position_embedding
        self.blocks = blocks
        self.final_norm = final_norm
        self.lm_head = lm_head

    def forward(self, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch seq vocab"]:
        _, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tied vs untied transformer depth.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=DEFAULT_MODEL_DIM)
    parser.add_argument("--mlp-dim", type=int, default=None, help="MLP hidden dim (default: 4*d_model)")
    parser.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    parser.add_argument("--tied-only", action="store_true", help="Only run the tied model")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}.")
    if args.log_every <= 0:
        raise ValueError(f"--log-every must be positive, got {args.log_every}.")
    if args.n_layers <= 0:
        raise ValueError(f"--n-layers must be positive, got {args.n_layers}.")
    if args.d_model <= 0:
        raise ValueError(f"--d-model must be positive, got {args.d_model}.")
    if args.d_model % args.n_heads != 0:
        raise ValueError(f"--d-model ({args.d_model}) must be divisible by --n-heads ({args.n_heads}).")
    if args.mlp_dim is None:
        args.mlp_dim = args.d_model * 4

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed.")
    args.seeds = seeds
    return args


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_models(
    *,
    vocab_size: int,
    n_layers: int,
    model_dim: int,
    mlp_dim: int,
    n_heads: int,
    device: torch.device,
    tied_only: bool = False,
) -> tuple[TiedTransformer, UntiedTransformer | None]:
    base_token_embedding = nn.Embedding(vocab_size, model_dim)
    base_position_embedding = nn.Embedding(CHUNK_SIZE, model_dim)
    base_block = TransformerBlock(model_dim=model_dim, n_heads=n_heads, mlp_dim=mlp_dim)
    base_final_norm = nn.LayerNorm(model_dim)
    base_lm_head = nn.Linear(model_dim, vocab_size)

    tied_model = TiedTransformer(
        token_embedding=copy.deepcopy(base_token_embedding),
        position_embedding=copy.deepcopy(base_position_embedding),
        block=base_block,
        final_norm=copy.deepcopy(base_final_norm),
        lm_head=copy.deepcopy(base_lm_head),
        n_layers=n_layers,
    )
    untied_model = None
    if not tied_only:
        untied_model = UntiedTransformer(
            token_embedding=copy.deepcopy(base_token_embedding),
            position_embedding=copy.deepcopy(base_position_embedding),
            blocks=nn.ModuleList([copy.deepcopy(base_block) for _ in range(n_layers)]),
            final_norm=copy.deepcopy(base_final_norm),
            lm_head=copy.deepcopy(base_lm_head),
        )
    return tied_model.to(device), untied_model.to(device) if untied_model else None


def prepare_dataset() -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    (train_inputs, train_next_tokens), (val_inputs, val_next_tokens), vocab_size = load_dataset(
        context_size=CHUNK_SIZE
    )
    train_targets = torch.cat((train_inputs[:, 1:], train_next_tokens.unsqueeze(1)), dim=1)
    val_targets = torch.cat((val_inputs[:, 1:], val_next_tokens.unsqueeze(1)), dim=1)
    return train_inputs, train_targets, val_inputs, val_targets, vocab_size


def sample_indices(*, total: int, count: int, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, total, (count,), generator=generator)


def compute_ce(
    model: nn.Module,
    *,
    inputs: Tensor,
    targets: Tensor,
    indices: Tensor,
    device: torch.device,
    vocab_size: int,
) -> float:
    batch_inputs = inputs[indices].to(device)
    batch_targets = targets[indices].to(device)
    with torch.inference_mode():
        logits = model(batch_inputs)
        return F.cross_entropy(logits.reshape(-1, vocab_size), batch_targets.reshape(-1)).item()


def verify_identical_initialization(
    tied_model: TiedTransformer,
    untied_model: UntiedTransformer,
    *,
    inputs: Tensor,
    indices: Tensor,
    device: torch.device,
) -> None:
    batch_inputs = inputs[indices].to(device)
    with torch.inference_mode():
        tied_logits = tied_model(batch_inputs)
        untied_logits = untied_model(batch_inputs)
    if not torch.allclose(tied_logits, untied_logits, atol=0.0, rtol=0.0):
        max_diff = (tied_logits - untied_logits).abs().max().item()
        raise RuntimeError(
            "Tied and untied models are not identical at initialization. "
            f"Maximum logit difference: {max_diff:.8f}"
        )


def log_metrics(
    *,
    step: int,
    tied_model: TiedTransformer,
    untied_model: UntiedTransformer | None,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    train_eval_indices: Tensor,
    val_eval_indices: Tensor,
    device: torch.device,
    vocab_size: int,
) -> tuple[float, float | None]:
    tied_train_ce = compute_ce(
        tied_model,
        inputs=train_inputs,
        targets=train_targets,
        indices=train_eval_indices,
        device=device,
        vocab_size=vocab_size,
    )
    tied_val_ce = compute_ce(
        tied_model,
        inputs=val_inputs,
        targets=val_targets,
        indices=val_eval_indices,
        device=device,
        vocab_size=vocab_size,
    )

    if untied_model:
        untied_train_ce = compute_ce(
            untied_model,
            inputs=train_inputs,
            targets=train_targets,
            indices=train_eval_indices,
            device=device,
            vocab_size=vocab_size,
        )
        untied_val_ce = compute_ce(
            untied_model,
            inputs=val_inputs,
            targets=val_targets,
            indices=val_eval_indices,
            device=device,
            vocab_size=vocab_size,
        )
        print(
            f"step={step} "
            f"tied_train_ce={tied_train_ce:.4f} untied_train_ce={untied_train_ce:.4f} "
            f"tied_val_ce={tied_val_ce:.4f} untied_val_ce={untied_val_ce:.4f}",
            flush=True,
        )
        return tied_val_ce, untied_val_ce
    else:
        print(
            f"step={step} tied_train_ce={tied_train_ce:.4f} tied_val_ce={tied_val_ce:.4f}",
            flush=True,
        )
        return tied_val_ce, None


def train_one_seed(
    *,
    seed: int,
    steps: int,
    log_every: int,
    n_layers: int,
    model_dim: int,
    mlp_dim: int,
    n_heads: int,
    tied_only: bool,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    vocab_size: int,
    device: torch.device,
) -> tuple[float, float | None, int, int | None]:
    set_seed(seed)
    tied_model, untied_model = build_models(
        vocab_size=vocab_size,
        n_layers=n_layers,
        model_dim=model_dim,
        mlp_dim=mlp_dim,
        n_heads=n_heads,
        device=device,
        tied_only=tied_only,
    )
    tied_optimizer = torch.optim.Adam(tied_model.parameters(), lr=LEARNING_RATE)
    untied_optimizer = torch.optim.Adam(untied_model.parameters(), lr=LEARNING_RATE) if untied_model else None

    tied_params = count_parameters(tied_model)
    untied_params = count_parameters(untied_model) if untied_model else None

    train_eval_indices = sample_indices(
        total=train_inputs.shape[0], count=EVAL_BATCH_SIZE, seed=TRAIN_EVAL_BATCH_SEED
    )
    val_eval_indices = sample_indices(total=val_inputs.shape[0], count=EVAL_BATCH_SIZE, seed=VAL_BATCH_SEED)

    if untied_model:
        verify_identical_initialization(
            tied_model,
            untied_model,
            inputs=train_inputs,
            indices=train_eval_indices,
            device=device,
        )

    print(f"=== Seed {seed} ===", flush=True)
    if untied_params:
        print(f"tied_params={tied_params} untied_params={untied_params}", flush=True)
    else:
        print(f"tied_params={tied_params} d_model={model_dim} n_layers={n_layers}", flush=True)

    tied_final_val_ce, untied_final_val_ce = log_metrics(
        step=0,
        tied_model=tied_model,
        untied_model=untied_model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        train_eval_indices=train_eval_indices,
        val_eval_indices=val_eval_indices,
        device=device,
        vocab_size=vocab_size,
    )

    batch_generator = torch.Generator().manual_seed(TRAIN_BATCH_SEED_OFFSET + seed)
    sample_count = train_inputs.shape[0]

    for step in range(1, steps + 1):
        batch_indices = torch.randint(0, sample_count, (BATCH_SIZE,), generator=batch_generator)
        batch_inputs = train_inputs[batch_indices].to(device)
        batch_targets = train_targets[batch_indices].to(device)

        tied_optimizer.zero_grad(set_to_none=True)
        tied_logits = tied_model(batch_inputs)
        tied_loss = F.cross_entropy(tied_logits.reshape(-1, vocab_size), batch_targets.reshape(-1))
        tied_loss.backward()
        torch.nn.utils.clip_grad_norm_(tied_model.parameters(), max_norm=1.0)
        tied_optimizer.step()

        if untied_model and untied_optimizer:
            untied_optimizer.zero_grad(set_to_none=True)
            untied_logits = untied_model(batch_inputs)
            untied_loss = F.cross_entropy(untied_logits.reshape(-1, vocab_size), batch_targets.reshape(-1))
            untied_loss.backward()
            torch.nn.utils.clip_grad_norm_(untied_model.parameters(), max_norm=1.0)
            untied_optimizer.step()

        if step % log_every == 0 or step == steps:
            tied_final_val_ce, untied_final_val_ce = log_metrics(
                step=step,
                tied_model=tied_model,
                untied_model=untied_model,
                train_inputs=train_inputs,
                train_targets=train_targets,
                val_inputs=val_inputs,
                val_targets=val_targets,
                train_eval_indices=train_eval_indices,
                val_eval_indices=val_eval_indices,
                device=device,
                vocab_size=vocab_size,
            )

    return tied_final_val_ce, untied_final_val_ce, tied_params, untied_params


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_inputs, train_targets, val_inputs, val_targets, vocab_size = prepare_dataset()

    tied_final_val_ces: list[float] = []
    untied_final_val_ces: list[float] = []
    tied_params: int | None = None
    untied_params: int | None = None

    for seed in args.seeds:
        tied_final_val_ce, untied_final_val_ce, tied_params_for_seed, untied_params_for_seed = train_one_seed(
            seed=seed,
            steps=args.steps,
            log_every=args.log_every,
            n_layers=args.n_layers,
            model_dim=args.d_model,
            mlp_dim=args.mlp_dim,
            n_heads=args.n_heads,
            tied_only=args.tied_only,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            vocab_size=vocab_size,
            device=device,
        )
        tied_final_val_ces.append(tied_final_val_ce)
        untied_final_val_ces.append(untied_final_val_ce)
        tied_params = tied_params_for_seed
        untied_params = untied_params_for_seed

    print(
        json.dumps(
            {
                "seeds": args.seeds,
                "tied_final_val_ce": tied_final_val_ces,
                "untied_final_val_ce": untied_final_val_ces,
                "tied_params": tied_params,
                "untied_params": untied_params,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
