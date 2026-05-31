from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import load_dataset, set_seed
from experiments.weight_tying.tied_vs_untied import CausalSelfAttention, TransformerBlock


torch.backends.cuda.matmul.allow_tf32 = True


SEED = 0
CHUNK_SIZE = 128
TRAIN_BATCH_SIZE = 32
ANALYSIS_BATCH_SIZE = 64
TRAIN_STEPS = 1000
LEARNING_RATE = 3e-4
GRAD_CLIP_NORM = 1.0
MODEL_DIM = 256
MLP_DIM = 1024
N_HEADS = 4
N_LAYERS = 8
LOG_EVERY = 100
TRAIN_BATCH_SEED_OFFSET = 100_000


class TiedTransformerWithIntermediates(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        model_dim: int,
        mlp_dim: int,
        n_heads: int,
        n_layers: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(chunk_size, model_dim)
        self.block = TransformerBlock(model_dim=model_dim, n_heads=n_heads, mlp_dim=mlp_dim)
        self.final_norm = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size)
        self.n_layers = n_layers
        self.chunk_size = chunk_size

    def forward(self, tokens: Tensor) -> Tensor:
        _, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)

        logits_by_depth: list[Tensor] = []
        for _ in range(self.n_layers):
            hidden = self.block(hidden)
            normalized = self.final_norm(hidden)
            logits_by_depth.append(self.lm_head(normalized))

        return torch.stack(logits_by_depth, dim=0)


def prepare_dataset() -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    (train_inputs, train_next_tokens), (val_inputs, val_next_tokens), vocab_size = load_dataset(
        context_size=CHUNK_SIZE
    )
    train_targets = torch.cat((train_inputs[:, 1:], train_next_tokens.unsqueeze(1)), dim=1)
    val_targets = torch.cat((val_inputs[:, 1:], val_next_tokens.unsqueeze(1)), dim=1)
    return train_inputs, train_targets, val_inputs, val_targets, vocab_size


def compute_mean_ce(logits: Tensor, targets: Tensor) -> float:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1)).item()


def train_model(
    *,
    model: TiedTransformerWithIntermediates,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    vocab_size: int,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    batch_generator = torch.Generator().manual_seed(TRAIN_BATCH_SEED_OFFSET + SEED)
    sample_count = train_inputs.shape[0]

    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        batch_indices = torch.randint(0, sample_count, (TRAIN_BATCH_SIZE,), generator=batch_generator)
        batch_inputs = train_inputs[batch_indices].to(device)
        batch_targets = train_targets[batch_indices].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_inputs)[-1]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == TRAIN_STEPS:
            model.eval()
            with torch.inference_mode():
                train_eval_logits = model(train_inputs[:256].to(device))[-1]
                val_eval_logits = model(val_inputs[:256].to(device))[-1]
                train_ce = compute_mean_ce(train_eval_logits, train_targets[:256].to(device))
                val_ce = compute_mean_ce(val_eval_logits, val_targets[:256].to(device))
            print(f"step={step} train_ce={train_ce:.4f} val_ce={val_ce:.4f}", flush=True)
            model.train()


def analyze_validation_set(
    *,
    model: TiedTransformerWithIntermediates,
    val_inputs: Tensor,
    val_targets: Tensor,
    device: torch.device,
) -> Tensor:
    model.eval()
    per_batch_losses: list[Tensor] = []

    with torch.inference_mode():
        for start in range(0, val_inputs.shape[0], ANALYSIS_BATCH_SIZE):
            stop = start + ANALYSIS_BATCH_SIZE
            batch_inputs = val_inputs[start:stop].to(device)
            batch_targets = val_targets[start:stop].to(device)
            logits_by_depth = model(batch_inputs)
            flat_targets = batch_targets.reshape(-1)

            depth_losses: list[Tensor] = []
            for depth_logits in logits_by_depth:
                per_token_ce = F.cross_entropy(
                    depth_logits.reshape(-1, depth_logits.shape[-1]),
                    flat_targets,
                    reduction="none",
                ).reshape(batch_targets.shape)
                depth_losses.append(per_token_ce.cpu())

            per_batch_losses.append(torch.stack(depth_losses, dim=0))

    return torch.cat(per_batch_losses, dim=1)


def print_analysis_table(losses_by_depth: Tensor) -> None:
    flat_losses = losses_by_depth.reshape(losses_by_depth.shape[0], -1)
    best_loss_per_token = flat_losses.min(dim=0).values
    final_loss_per_token = flat_losses[-1]

    within_point_one = flat_losses <= (best_loss_per_token.unsqueeze(0) + 0.1)
    within_point_zero_five = flat_losses <= (best_loss_per_token.unsqueeze(0) + 0.05)
    optimal_n_mask = flat_losses <= (final_loss_per_token.unsqueeze(0) + 0.05)
    optimal_n = optimal_n_mask.float().argmax(dim=0) + 1

    print("Iteration | Mean Val CE | Frac within 0.1 of best | Frac within 0.05 of best", flush=True)
    for depth_index in range(losses_by_depth.shape[0]):
        mean_ce = flat_losses[depth_index].mean().item()
        frac_within_point_one = within_point_one[depth_index].float().mean().item()
        frac_within_point_zero_five = within_point_zero_five[depth_index].float().mean().item()
        print(
            f"{depth_index + 1:>9} | {mean_ce:>11.4f} | {frac_within_point_one:>24.4f} | {frac_within_point_zero_five:>25.4f}",
            flush=True,
        )

    n2_sufficient = within_point_one[1].float().mean().item() * 100.0
    mean_optimal_n = optimal_n.float().mean().item()
    percentile_95_optimal_n = torch.quantile(optimal_n.float(), 0.95).item()

    print(
        f"Tokens where N=2 is sufficient (within 0.1 of best): {n2_sufficient:.2f}%",
        flush=True,
    )
    print(f"Mean optimal N across all tokens: {mean_optimal_n:.2f}", flush=True)
    print(f"95th percentile optimal N: {int(torch.ceil(torch.tensor(percentile_95_optimal_n)).item())}", flush=True)


def main() -> None:
    _ = CausalSelfAttention
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    train_inputs, train_targets, val_inputs, val_targets, vocab_size = prepare_dataset()

    model = TiedTransformerWithIntermediates(
        vocab_size=vocab_size,
        model_dim=MODEL_DIM,
        mlp_dim=MLP_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        chunk_size=CHUNK_SIZE,
    ).to(device)

    print(
        f"device={device} seed={SEED} d_model={MODEL_DIM} mlp_dim={MLP_DIM} n_heads={N_HEADS} n_layers={N_LAYERS}",
        flush=True,
    )
    train_model(
        model=model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        vocab_size=vocab_size,
        device=device,
    )
    losses_by_depth = analyze_validation_set(
        model=model,
        val_inputs=val_inputs,
        val_targets=val_targets,
        device=device,
    )
    print_analysis_table(losses_by_depth)


if __name__ == "__main__":
    main()
