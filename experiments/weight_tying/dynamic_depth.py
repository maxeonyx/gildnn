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
THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


class TiedTransformerDynamicDepth(nn.Module):
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

    def forward(self, tokens: Tensor) -> Tensor:
        _, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)

        for _ in range(self.n_layers):
            hidden = self.block(hidden)
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)

    def forward_all_iterations(self, tokens: Tensor) -> Tensor:
        _, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        logits_by_depth: list[Tensor] = []

        for _ in range(self.n_layers):
            hidden = self.block(hidden)
            logits_by_depth.append(self.lm_head(self.final_norm(hidden)))

        return torch.stack(logits_by_depth, dim=0)


def prepare_dataset() -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    (train_inputs, train_next_tokens), (val_inputs, val_next_tokens), vocab_size = load_dataset(
        context_size=CHUNK_SIZE
    )
    train_targets = torch.cat((train_inputs[:, 1:], train_next_tokens.unsqueeze(1)), dim=1)
    val_targets = torch.cat((val_inputs[:, 1:], val_next_tokens.unsqueeze(1)), dim=1)
    return train_inputs, train_targets, val_inputs, val_targets, vocab_size


def compute_cross_entropy_sum(logits: Tensor, targets: Tensor) -> Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="sum",
    )


def compute_per_token_ce(logits: Tensor, targets: Tensor) -> Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)


def train_model(
    *,
    model: TiedTransformerDynamicDepth,
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
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == TRAIN_STEPS:
            model.eval()
            with torch.inference_mode():
                train_eval_logits = model(train_inputs[:256].to(device))
                val_eval_logits = model(val_inputs[:256].to(device))
                train_ce = F.cross_entropy(
                    train_eval_logits.reshape(-1, vocab_size),
                    train_targets[:256].to(device).reshape(-1),
                ).item()
                val_ce = F.cross_entropy(
                    val_eval_logits.reshape(-1, vocab_size),
                    val_targets[:256].to(device).reshape(-1),
                ).item()
            print(f"step={step} train_ce={train_ce:.4f} val_ce={val_ce:.4f}", flush=True)
            model.train()


def evaluate_validation_ce(
    *,
    model: TiedTransformerDynamicDepth,
    val_inputs: Tensor,
    val_targets: Tensor,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.inference_mode():
        for start in range(0, val_inputs.shape[0], ANALYSIS_BATCH_SIZE):
            stop = start + ANALYSIS_BATCH_SIZE
            batch_inputs = val_inputs[start:stop].to(device)
            batch_targets = val_targets[start:stop].to(device)
            logits = model(batch_inputs)
            total_loss += compute_cross_entropy_sum(logits, batch_targets).item()
            total_tokens += batch_targets.numel()

    return total_loss / total_tokens


def collect_validation_losses_by_iteration(
    *,
    model: TiedTransformerDynamicDepth,
    val_inputs: Tensor,
    val_targets: Tensor,
    device: torch.device,
) -> Tensor:
    per_batch_losses: list[Tensor] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, val_inputs.shape[0], ANALYSIS_BATCH_SIZE):
            stop = start + ANALYSIS_BATCH_SIZE
            batch_inputs = val_inputs[start:stop].to(device)
            batch_targets = val_targets[start:stop].to(device)
            logits_by_depth = model.forward_all_iterations(batch_inputs)
            depth_losses = [compute_per_token_ce(depth_logits, batch_targets).cpu() for depth_logits in logits_by_depth]
            per_batch_losses.append(torch.stack(depth_losses, dim=0))

    return torch.cat(per_batch_losses, dim=1)


def evaluate_ce_exit_threshold(losses_by_depth: Tensor, threshold: float) -> tuple[float, float]:
    flat_losses = losses_by_depth.reshape(losses_by_depth.shape[0], -1)
    token_count = flat_losses.shape[1]
    exited = torch.zeros(token_count, dtype=torch.bool)
    exit_iterations = torch.zeros(token_count, dtype=torch.long)
    exit_losses = torch.zeros(token_count)

    for depth_index in range(1, flat_losses.shape[0]):
        improvement = flat_losses[depth_index - 1] - flat_losses[depth_index]
        newly_exited = (~exited) & (improvement < threshold)
        exit_iterations[newly_exited] = depth_index + 1
        exit_losses[newly_exited] = flat_losses[depth_index][newly_exited]
        exited = exited | newly_exited

    exit_iterations[~exited] = flat_losses.shape[0]
    exit_losses[~exited] = flat_losses[-1][~exited]
    return exit_iterations.float().mean().item(), exit_losses.mean().item()


def print_oracle_distribution(losses_by_depth: Tensor) -> None:
    flat_losses = losses_by_depth.reshape(losses_by_depth.shape[0], -1)
    acceptable_mask = flat_losses <= (flat_losses[-1].unsqueeze(0) + 0.05)
    optimal_n = acceptable_mask.float().argmax(dim=0) + 1
    print("Oracle early-exit distribution (minimum N within 0.05 nats of N=8):", flush=True)
    for depth in range(1, losses_by_depth.shape[0] + 1):
        fraction = (optimal_n == depth).float().mean().item() * 100.0
        print(f"N={depth}: {fraction:.2f}%", flush=True)


def compute_pareto_operating_point(rows: list[dict[str, float]]) -> dict[str, float]:
    pareto_rows: list[dict[str, float]] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = other["mean_iters"] <= row["mean_iters"] and other["ce"] <= row["ce"]
            strictly_better = other["mean_iters"] < row["mean_iters"] or other["ce"] < row["ce"]
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_rows.append(row)

    small_loss_rows = [row for row in pareto_rows if row["ce_overhead"] <= 0.05]
    candidate_rows = small_loss_rows if small_loss_rows else pareto_rows
    return max(candidate_rows, key=lambda row: row["compute_savings"])


def main() -> None:
    _ = CausalSelfAttention
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    train_inputs, train_targets, val_inputs, val_targets, vocab_size = prepare_dataset()

    model = TiedTransformerDynamicDepth(
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

    losses_by_depth = collect_validation_losses_by_iteration(
        model=model,
        val_inputs=val_inputs,
        val_targets=val_targets,
        device=device,
    )
    baseline_ce = losses_by_depth[-1].mean().item()

    rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        mean_iters, ce = evaluate_ce_exit_threshold(losses_by_depth, threshold)
        compute_fraction = mean_iters / N_LAYERS
        rows.append(
            {
                "threshold": threshold,
                "mean_iters": mean_iters,
                "ce": ce,
                "compute_fraction": compute_fraction,
                "compute_savings": 1.0 - compute_fraction,
                "ce_overhead": ce - baseline_ce,
                "baseline_ce": baseline_ce,
            }
        )

    print(f"Baseline N={N_LAYERS} validation CE: {baseline_ce:.4f}", flush=True)
    print("Threshold | Mean iters | Val CE | Compute fraction | CE overhead vs N=8", flush=True)
    for row in rows:
        print(
            f"{row['threshold']:>8.2f} | {row['mean_iters']:>10.2f} | {row['ce']:>6.4f} | {row['compute_fraction']:>16.4f} | {row['ce_overhead']:+.4f}",
            flush=True,
        )

    pareto_row = compute_pareto_operating_point(rows)
    print(
        "Pareto-optimal operating point: "
        f"threshold={pareto_row['threshold']:.2f} gives {pareto_row['compute_savings'] * 100.0:.1f}% compute savings "
        f"with only {pareto_row['ce_overhead']:.4f} nats quality loss",
        flush=True,
    )
    print_oracle_distribution(losses_by_depth)


if __name__ == "__main__":
    main()
