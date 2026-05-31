"""Learned halting predictor for dynamic depth.

Trains a tiny regression head on a frozen weight-tied transformer to predict
remaining CE gain at each iteration depth. Compares against oracle, entropy
heuristic, and fixed-depth baselines.

Pathway 5 (Dynamic Depth & Early Exit) — bridges the oracle→deployable gap.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import Tensor, nn

from core.fixed_window_char import load_dataset, set_seed
from experiments.weight_tying.dynamic_depth import (
    ANALYSIS_BATCH_SIZE,
    CHUNK_SIZE,
    ENTROPY_THRESHOLDS,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MLP_DIM,
    MODEL_DIM,
    N_HEADS,
    N_LAYERS,
    SEED,
    THRESHOLDS,
    TiedTransformerDynamicDepth,
    compute_per_token_ce,
    compute_per_token_entropy,
    evaluate_ce_exit_threshold,
    evaluate_entropy_exit_threshold,
    prepare_dataset,
    train_model,
)
from experiments.weight_tying.tied_vs_untied import CausalSelfAttention, TransformerBlock


torch.backends.cuda.matmul.allow_tf32 = True

# --- Predictor hyperparameters ---
PREDICTOR_WINDOW_BATCH_SIZE = 64
PREDICTOR_TRAIN_STEPS = 500
PREDICTOR_LOG_EVERY = 50
PREDICTOR_LEARNING_RATE = 1e-3
PREDICTOR_HIDDEN_DIM = 64
PREDICTOR_HUBER_DELTA = 0.1
TRAIN_SPLIT_FRACTION = 0.8
PREDICTOR_DEPTHS = N_LAYERS - 1  # depths 1..7 (depth 8 has gain=0 by definition)
HIDDEN_FEATURE_DIM = MODEL_DIM + 5  # hidden_state + 5 scalar features
ENTROPY_ONLY_FEATURE_DIM = 4  # entropy, entropy_drop, margin, normalized_depth
LEARNED_THRESHOLDS = [0.0, *THRESHOLDS]
SPLIT_SEED_OFFSET = 20_000
PREDICTOR_TRAIN_SEED_OFFSET = 30_000

# Suppress unused import warnings (needed for import side effects)
_ = (CausalSelfAttention, TransformerBlock, load_dataset)


@dataclass
class FeatureBatch:
    hidden_features: Tensor  # [depths, batch, seq, hidden_feature_dim]
    entropy_features: Tensor  # [depths, batch, seq, entropy_feature_dim]
    gain_targets: Tensor  # [depths, batch, seq]
    losses_by_depth: Tensor  # [n_layers, batch, seq]
    entropies_by_depth: Tensor  # [n_layers, batch, seq]


@dataclass
class EvaluationArtifacts:
    losses_by_depth: Tensor
    entropies_by_depth: Tensor
    true_gains_by_depth: Tensor
    hidden_predictions_by_depth: Tensor
    entropy_predictions_by_depth: Tensor


class GainPredictor(nn.Module):
    """Tiny MLP that predicts remaining CE gain from current-depth features."""

    def __init__(self, *, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, PREDICTOR_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(PREDICTOR_HIDDEN_DIM, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1)


def split_validation_indices(sample_count: int) -> tuple[Tensor, Tensor]:
    """80/20 split of validation windows for predictor train/eval."""
    generator = torch.Generator().manual_seed(SEED + SPLIT_SEED_OFFSET)
    permutation = torch.randperm(sample_count, generator=generator)
    train_count = int(sample_count * TRAIN_SPLIT_FRACTION)
    return permutation[:train_count], permutation[train_count:]


def batch_index_chunks(indices: Tensor, batch_size: int) -> list[Tensor]:
    return [indices[start : start + batch_size] for start in range(0, indices.numel(), batch_size)]


def forward_all_iterations_with_hidden(
    model: TiedTransformerDynamicDepth,
    tokens: Tensor,
) -> tuple[Tensor, Tensor]:
    """Forward pass returning hidden states AND logits at each iteration depth."""
    _, seq_len = tokens.shape
    positions = torch.arange(seq_len, device=tokens.device)
    hidden = model.token_embedding(tokens) + model.position_embedding(positions).unsqueeze(0)
    hidden_by_depth: list[Tensor] = []
    logits_by_depth: list[Tensor] = []

    for _ in range(model.n_layers):
        hidden = model.block(hidden)
        hidden_by_depth.append(hidden)
        logits_by_depth.append(model.lm_head(model.final_norm(hidden)))

    return torch.stack(hidden_by_depth, dim=0), torch.stack(logits_by_depth, dim=0)


def compute_logit_margin(logits_by_depth: Tensor) -> Tensor:
    """Top-1 minus top-2 logit value — confidence signal."""
    top2 = torch.topk(logits_by_depth, k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def collect_feature_batch(
    model: TiedTransformerDynamicDepth,
    tokens: Tensor,
    targets: Tensor,
) -> FeatureBatch:
    """Collect all features needed for predictor training from one batch."""
    hidden_by_depth, logits_by_depth = forward_all_iterations_with_hidden(model, tokens)

    losses_by_depth = torch.stack(
        [compute_per_token_ce(depth_logits, targets) for depth_logits in logits_by_depth],
        dim=0,
    )
    entropies_by_depth = torch.stack(
        [compute_per_token_entropy(depth_logits) for depth_logits in logits_by_depth],
        dim=0,
    )
    margins_by_depth = compute_logit_margin(logits_by_depth)
    normalized_hidden_by_depth = torch.stack(
        [model.final_norm(hidden) for hidden in hidden_by_depth], dim=0
    )

    # Entropy drop: entropy_d - entropy_{d-1} (0 at depth 1)
    entropy_delta_by_depth = torch.zeros_like(entropies_by_depth)
    entropy_delta_by_depth[1:] = entropies_by_depth[1:] - entropies_by_depth[:-1]

    # Hidden state change magnitude: ||h_d - h_{d-1}|| / sqrt(d_model) (0 at depth 1)
    hidden_change_by_depth = torch.zeros_like(entropies_by_depth)
    hidden_change_by_depth[1:] = (
        (hidden_by_depth[1:] - hidden_by_depth[:-1]).pow(2).mean(dim=-1).sqrt()
    )

    # Normalized depth scalars
    depth_scalars = (
        torch.arange(1, model.n_layers + 1, device=tokens.device, dtype=normalized_hidden_by_depth.dtype)
        / model.n_layers
    )

    # Assemble hidden features: [depths-1, batch, seq, 261]
    # Only depths 1..7 (index 0..6) — depth 8 has gain=0 by definition
    hidden_features = torch.cat(
        (
            normalized_hidden_by_depth[:-1],  # [7, batch, seq, 256]
            depth_scalars[:-1].view(PREDICTOR_DEPTHS, 1, 1, 1).expand(
                -1, tokens.shape[0], tokens.shape[1], 1
            ),
            entropies_by_depth[:-1].unsqueeze(-1),
            entropy_delta_by_depth[:-1].unsqueeze(-1),
            margins_by_depth[:-1].unsqueeze(-1),
            hidden_change_by_depth[:-1].unsqueeze(-1),
        ),
        dim=-1,
    )

    # Entropy-only features: [depths-1, batch, seq, 4]
    entropy_features = torch.stack(
        (
            entropies_by_depth[:-1],
            entropy_delta_by_depth[:-1],
            margins_by_depth[:-1],
            depth_scalars[:-1]
            .view(PREDICTOR_DEPTHS, 1, 1)
            .expand(-1, tokens.shape[0], tokens.shape[1]),
        ),
        dim=-1,
    )

    # Target: remaining gain g_d = L_d - L_8
    gain_targets = losses_by_depth[:-1] - losses_by_depth[-1].unsqueeze(0)

    return FeatureBatch(
        hidden_features=hidden_features,
        entropy_features=entropy_features,
        gain_targets=gain_targets,
        losses_by_depth=losses_by_depth,
        entropies_by_depth=entropies_by_depth,
    )


def flatten_feature_tensor(features_by_depth: Tensor) -> Tensor:
    """Reshape [depths, batch, seq, feat] -> [depths*batch*seq, feat]."""
    return features_by_depth.permute(1, 2, 0, 3).reshape(-1, features_by_depth.shape[-1])


def flatten_target_tensor(targets_by_depth: Tensor) -> Tensor:
    """Reshape [depths, batch, seq] -> [depths*batch*seq]."""
    return targets_by_depth.permute(1, 2, 0).reshape(-1)


def reshape_predictions(flat_predictions: Tensor, batch_size: int, seq_len: int) -> Tensor:
    """Reshape flat predictions back to [depths, batch, seq]."""
    return flat_predictions.reshape(batch_size, seq_len, PREDICTOR_DEPTHS).permute(2, 0, 1)


def train_predictor(
    *,
    predictor: GainPredictor,
    feature_kind: str,
    model: TiedTransformerDynamicDepth,
    val_inputs: Tensor,
    val_targets: Tensor,
    train_indices: Tensor,
    device: torch.device,
) -> None:
    """Train the gain predictor on frozen model features."""
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=PREDICTOR_LEARNING_RATE)
    loss_fn = nn.HuberLoss(delta=PREDICTOR_HUBER_DELTA)
    batch_generator = torch.Generator().manual_seed(SEED + PREDICTOR_TRAIN_SEED_OFFSET)
    predictor.train()

    for step in range(1, PREDICTOR_TRAIN_STEPS + 1):
        sampled_positions = train_indices[
            torch.randint(0, train_indices.numel(), (PREDICTOR_WINDOW_BATCH_SIZE,), generator=batch_generator)
        ]
        batch_inputs = val_inputs[sampled_positions].to(device)
        batch_targets = val_targets[sampled_positions].to(device)

        with torch.inference_mode():
            feature_batch = collect_feature_batch(model, batch_inputs, batch_targets)
            if feature_kind == "hidden":
                features = flatten_feature_tensor(feature_batch.hidden_features)
            elif feature_kind == "entropy":
                features = flatten_feature_tensor(feature_batch.entropy_features)
            else:
                raise ValueError(f"Unsupported feature kind: {feature_kind}")
            targets = flatten_target_tensor(feature_batch.gain_targets)

        # Clone out of inference_mode context for gradient computation
        features = features.to(dtype=torch.float32).clone()
        targets = targets.to(dtype=torch.float32).clone()

        optimizer.zero_grad(set_to_none=True)
        predictions = predictor(features)
        loss = loss_fn(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        if step % PREDICTOR_LOG_EVERY == 0 or step == PREDICTOR_TRAIN_STEPS:
            mae = (predictions.detach() - targets).abs().mean().item()
            print(
                f"predictor={feature_kind} step={step} loss={loss.item():.6f} mae={mae:.6f}",
                flush=True,
            )


def collect_evaluation_artifacts(
    *,
    model: TiedTransformerDynamicDepth,
    hidden_predictor: GainPredictor,
    entropy_predictor: GainPredictor,
    val_inputs: Tensor,
    val_targets: Tensor,
    eval_indices: Tensor,
    device: torch.device,
) -> EvaluationArtifacts:
    """Collect predictions and ground truth over the eval split."""
    losses_by_depth_batches: list[Tensor] = []
    entropies_by_depth_batches: list[Tensor] = []
    true_gains_by_depth_batches: list[Tensor] = []
    hidden_prediction_batches: list[Tensor] = []
    entropy_prediction_batches: list[Tensor] = []

    hidden_predictor.eval()
    entropy_predictor.eval()

    with torch.inference_mode():
        for batch_indices in batch_index_chunks(eval_indices, ANALYSIS_BATCH_SIZE):
            batch_inputs = val_inputs[batch_indices].to(device)
            batch_targets = val_targets[batch_indices].to(device)
            feature_batch = collect_feature_batch(model, batch_inputs, batch_targets)

            hidden_features = flatten_feature_tensor(feature_batch.hidden_features).to(dtype=torch.float32)
            entropy_features = flatten_feature_tensor(feature_batch.entropy_features).to(dtype=torch.float32)

            hidden_predictions = reshape_predictions(
                hidden_predictor(hidden_features),
                batch_inputs.shape[0],
                batch_inputs.shape[1],
            )
            entropy_predictions = reshape_predictions(
                entropy_predictor(entropy_features),
                batch_inputs.shape[0],
                batch_inputs.shape[1],
            )

            losses_by_depth_batches.append(feature_batch.losses_by_depth.cpu())
            entropies_by_depth_batches.append(feature_batch.entropies_by_depth.cpu())
            true_gains_by_depth_batches.append(feature_batch.gain_targets.cpu())
            hidden_prediction_batches.append(hidden_predictions.cpu())
            entropy_prediction_batches.append(entropy_predictions.cpu())

    return EvaluationArtifacts(
        losses_by_depth=torch.cat(losses_by_depth_batches, dim=1),
        entropies_by_depth=torch.cat(entropies_by_depth_batches, dim=1),
        true_gains_by_depth=torch.cat(true_gains_by_depth_batches, dim=1),
        hidden_predictions_by_depth=torch.cat(hidden_prediction_batches, dim=1),
        entropy_predictions_by_depth=torch.cat(entropy_prediction_batches, dim=1),
    )


def evaluate_predicted_gain_exit_threshold(
    losses_by_depth: Tensor,
    predicted_gains_by_depth: Tensor,
    threshold: float,
) -> tuple[float, float]:
    """Halt at first depth d where predicted remaining gain < threshold."""
    flat_losses = losses_by_depth.reshape(losses_by_depth.shape[0], -1)
    flat_predictions = predicted_gains_by_depth.reshape(predicted_gains_by_depth.shape[0], -1)
    token_count = flat_losses.shape[1]
    exited = torch.zeros(token_count, dtype=torch.bool)
    exit_iterations = torch.zeros(token_count, dtype=torch.long)
    exit_losses = torch.zeros(token_count)

    for depth_index in range(flat_predictions.shape[0]):
        newly_exited = (~exited) & (flat_predictions[depth_index] < threshold)
        exit_iterations[newly_exited] = depth_index + 1
        exit_losses[newly_exited] = flat_losses[depth_index][newly_exited]
        exited = exited | newly_exited

    # Tokens that never exited early get full depth
    exit_iterations[~exited] = losses_by_depth.shape[0]
    exit_losses[~exited] = flat_losses[-1][~exited]
    return exit_iterations.float().mean().item(), exit_losses.mean().item()


def compute_pearson_correlation(x: Tensor, y: Tensor) -> float:
    x = x.reshape(-1).to(dtype=torch.float64)
    y = y.reshape(-1).to(dtype=torch.float64)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if denominator.item() == 0.0:
        return float("nan")
    return (torch.dot(x_centered, y_centered) / denominator).item()


def make_result_row(
    *,
    method: str,
    threshold: str,
    mean_iters: float,
    ce: float,
    baseline_ce: float,
) -> dict[str, float | str]:
    compute_savings = 1.0 - (mean_iters / N_LAYERS)
    return {
        "method": method,
        "threshold": threshold,
        "mean_iters": mean_iters,
        "ce": ce,
        "compute_savings": compute_savings,
        "ce_overhead": ce - baseline_ce,
    }


def print_comparison_table(rows: list[dict[str, float | str]]) -> None:
    print(
        "Method                         | Threshold | Mean Iters | Val CE | Compute Savings | CE Overhead vs N=8",
        flush=True,
    )
    print(
        "------------------------------ | --------- | ---------- | ------ | --------------- | ------------------",
        flush=True,
    )
    for row in rows:
        print(
            f"{str(row['method']):<30} | {str(row['threshold']):>9} | {float(row['mean_iters']):>10.2f} | "
            f"{float(row['ce']):>6.4f} | {float(row['compute_savings']) * 100.0:>14.2f}% | "
            f"{float(row['ce_overhead']):>+18.4f}",
            flush=True,
        )


def print_correlation_table(
    *,
    heading: str,
    true_gains_by_depth: Tensor,
    predicted_gains_by_depth: Tensor,
) -> None:
    print(heading, flush=True)
    print("Depth | Pearson r", flush=True)
    for depth in range(PREDICTOR_DEPTHS):
        correlation = compute_pearson_correlation(
            true_gains_by_depth[depth], predicted_gains_by_depth[depth]
        )
        print(f"{depth + 1:>5} | {correlation:>9.4f}", flush=True)
    overall = compute_pearson_correlation(true_gains_by_depth, predicted_gains_by_depth)
    print(f"{'all':>5} | {overall:>9.4f}", flush=True)


def build_oracle_efficiency_rows(
    *,
    learned_rows: list[dict[str, float | str]],
    oracle_rows: list[dict[str, float | str]],
) -> list[dict[str, str | float]]:
    """For each learned operating point, find oracle at matched CE overhead."""
    rows: list[dict[str, str | float]] = []
    for learned_row in learned_rows:
        matched_oracle_row = min(
            oracle_rows,
            key=lambda oracle_row: abs(float(oracle_row["ce_overhead"]) - float(learned_row["ce_overhead"])),
        )
        oracle_savings = float(matched_oracle_row["compute_savings"])
        learned_savings = float(learned_row["compute_savings"])
        efficiency = float("nan") if oracle_savings <= 0.0 else learned_savings / oracle_savings
        rows.append(
            {
                "method": str(learned_row["method"]),
                "threshold": str(learned_row["threshold"]),
                "matched_threshold": str(matched_oracle_row["threshold"]),
                "learned_savings": learned_savings,
                "oracle_savings": oracle_savings,
                "efficiency": efficiency,
            }
        )
    return rows


def print_oracle_efficiency_table(rows: list[dict[str, str | float]]) -> None:
    print("Oracle efficiency at matched CE overhead", flush=True)
    print(
        "Method                         | Threshold | Oracle Thr | Learned Savings | Oracle Savings | Efficiency",
        flush=True,
    )
    print(
        "------------------------------ | --------- | ---------- | --------------- | -------------- | ----------",
        flush=True,
    )
    for row in rows:
        print(
            f"{str(row['method']):<30} | {str(row['threshold']):>9} | {str(row['matched_threshold']):>10} | "
            f"{float(row['learned_savings']) * 100.0:>14.2f}% | {float(row['oracle_savings']) * 100.0:>13.2f}% | "
            f"{float(row['efficiency']):>10.4f}",
            flush=True,
        )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    train_inputs, train_targets, val_inputs, val_targets, vocab_size = prepare_dataset()

    # --- Phase 1: Train base model ---
    model = TiedTransformerDynamicDepth(
        vocab_size=vocab_size,
        model_dim=MODEL_DIM,
        mlp_dim=MLP_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        chunk_size=CHUNK_SIZE,
    ).to(device)

    print(
        f"device={device} seed={SEED} d_model={MODEL_DIM} n_layers={N_LAYERS} ctx={CHUNK_SIZE}",
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

    # --- Phase 2: Freeze model, prepare splits ---
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_indices, eval_indices = split_validation_indices(val_inputs.shape[0])
    print(
        f"validation_windows train={train_indices.numel()} eval={eval_indices.numel()} "
        f"tokens_per_window={val_inputs.shape[1]}",
        flush=True,
    )

    # --- Phase 3: Train halting predictors ---
    hidden_predictor = GainPredictor(input_dim=HIDDEN_FEATURE_DIM).to(device)
    entropy_predictor = GainPredictor(input_dim=ENTROPY_ONLY_FEATURE_DIM).to(device)

    print("\n--- Training hidden+scalars predictor ---", flush=True)
    train_predictor(
        predictor=hidden_predictor,
        feature_kind="hidden",
        model=model,
        val_inputs=val_inputs,
        val_targets=val_targets,
        train_indices=train_indices,
        device=device,
    )

    print("\n--- Training entropy-only predictor ---", flush=True)
    train_predictor(
        predictor=entropy_predictor,
        feature_kind="entropy",
        model=model,
        val_inputs=val_inputs,
        val_targets=val_targets,
        train_indices=train_indices,
        device=device,
    )

    # --- Phase 4: Evaluate on held-out split ---
    print("\n--- Collecting evaluation artifacts ---", flush=True)
    evaluation = collect_evaluation_artifacts(
        model=model,
        hidden_predictor=hidden_predictor,
        entropy_predictor=entropy_predictor,
        val_inputs=val_inputs,
        val_targets=val_targets,
        eval_indices=eval_indices,
        device=device,
    )

    baseline_ce = evaluation.losses_by_depth[-1].mean().item()
    print(f"eval_windows={eval_indices.numel()} eval_baseline_ce={baseline_ce:.4f}\n", flush=True)

    # --- Phase 5: Build comparison table ---
    rows: list[dict[str, float | str]] = []

    # Full depth baseline
    rows.append(
        make_result_row(
            method="Full depth (N=8)",
            threshold="-",
            mean_iters=float(N_LAYERS),
            ce=baseline_ce,
            baseline_ce=baseline_ce,
        )
    )

    # Fixed depth baselines
    for depth in (4, 5, 6, 7):
        rows.append(
            make_result_row(
                method=f"Fixed depth {depth}",
                threshold="-",
                mean_iters=float(depth),
                ce=evaluation.losses_by_depth[depth - 1].mean().item(),
                baseline_ce=baseline_ce,
            )
        )

    # Oracle CE-based
    oracle_rows: list[dict[str, float | str]] = []
    for threshold in THRESHOLDS:
        mean_iters, ce = evaluate_ce_exit_threshold(evaluation.losses_by_depth, threshold)
        oracle_rows.append(
            make_result_row(
                method="Oracle CE-based",
                threshold=f"{threshold:.3f}",
                mean_iters=mean_iters,
                ce=ce,
                baseline_ce=baseline_ce,
            )
        )
    rows.extend(oracle_rows)

    # Entropy heuristic
    heuristic_rows: list[dict[str, float | str]] = []
    for threshold in ENTROPY_THRESHOLDS:
        mean_iters, ce = evaluate_entropy_exit_threshold(
            evaluation.losses_by_depth,
            evaluation.entropies_by_depth,
            threshold,
        )
        heuristic_rows.append(
            make_result_row(
                method="Entropy heuristic",
                threshold=f"{threshold:.1f}",
                mean_iters=mean_iters,
                ce=ce,
                baseline_ce=baseline_ce,
            )
        )
    rows.extend(heuristic_rows)

    # Learned (hidden+scalars)
    learned_hidden_rows: list[dict[str, float | str]] = []
    for threshold in LEARNED_THRESHOLDS:
        mean_iters, ce = evaluate_predicted_gain_exit_threshold(
            evaluation.losses_by_depth,
            evaluation.hidden_predictions_by_depth,
            threshold,
        )
        learned_hidden_rows.append(
            make_result_row(
                method="Learned (hidden+scalars)",
                threshold=f"{threshold:.3f}",
                mean_iters=mean_iters,
                ce=ce,
                baseline_ce=baseline_ce,
            )
        )
    rows.extend(learned_hidden_rows)

    # Learned (entropy-only)
    learned_entropy_rows: list[dict[str, float | str]] = []
    for threshold in LEARNED_THRESHOLDS:
        mean_iters, ce = evaluate_predicted_gain_exit_threshold(
            evaluation.losses_by_depth,
            evaluation.entropy_predictions_by_depth,
            threshold,
        )
        learned_entropy_rows.append(
            make_result_row(
                method="Learned (entropy-only)",
                threshold=f"{threshold:.3f}",
                mean_iters=mean_iters,
                ce=ce,
                baseline_ce=baseline_ce,
            )
        )
    rows.extend(learned_entropy_rows)

    # --- Print results ---
    print("=" * 110, flush=True)
    print("COMPARISON TABLE", flush=True)
    print("=" * 110, flush=True)
    print_comparison_table(rows)
    print(flush=True)

    print_correlation_table(
        heading="Predicted remaining-gain correlation: learned (hidden+scalars)",
        true_gains_by_depth=evaluation.true_gains_by_depth,
        predicted_gains_by_depth=evaluation.hidden_predictions_by_depth,
    )
    print(flush=True)
    print_correlation_table(
        heading="Predicted remaining-gain correlation: learned (entropy-only)",
        true_gains_by_depth=evaluation.true_gains_by_depth,
        predicted_gains_by_depth=evaluation.entropy_predictions_by_depth,
    )
    print(flush=True)

    print_oracle_efficiency_table(
        build_oracle_efficiency_rows(learned_rows=learned_hidden_rows, oracle_rows=oracle_rows)
    )
    print(flush=True)
    print_oracle_efficiency_table(
        build_oracle_efficiency_rows(learned_rows=learned_entropy_rows, oracle_rows=oracle_rows)
    )


if __name__ == "__main__":
    main()
