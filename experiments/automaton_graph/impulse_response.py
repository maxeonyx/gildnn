from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import Tensor
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.automaton_graph import GraphCellularAutomaton, l2_normalize


SNAPSHOT_STEPS = (0, 4, 8, 16, 32, 64, 128)
PALETTE = " .:-=+*#%@"


def module_index(*, band: int, col: int, n_cols: int) -> int:
    return band * n_cols + col


def inject_impulse(
    model: GraphCellularAutomaton,
    states: Tensor,
    global_buffer: Tensor,
    *,
    band: int,
    col: int,
    magnitude: float,
) -> None:
    target_module = module_index(band=band, col=col, n_cols=model.n_cols)
    impulse = torch.randn(
        model.d_stream,
        device=states.device,
        dtype=states.dtype,
    )
    impulse = l2_normalize(impulse.unsqueeze(0)).squeeze(0) * magnitude
    states[target_module, 0] = impulse
    global_buffer[target_module, 0] = impulse


def step_automaton(
    model: GraphCellularAutomaton,
    states: Tensor,
    global_buffer: Tensor,
    predictions: Tensor,
    has_predicted: Tensor,
    refractory_levels: Tensor,
    *,
    timestep: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    fires = torch.remainder(
        torch.tensor(timestep, device=states.device, dtype=torch.long) + model.module_phases,
        model.module_rates,
    ) == 0
    fire_mask = fires[:, None, None]

    if model.refractory:
        refractory_levels = refractory_levels * model.refractory_decay

    neighbor_sum = model._neighbor_sum(global_buffer, refractory_levels)
    combined = l2_normalize(states + neighbor_sum)
    hidden = F.gelu(model._stacked_linear(combined, model.w1, model.b1))
    output = model._stacked_linear(hidden, model.w2, model.b2)
    new_predictions = model._stacked_linear(output, model.pred_w, model.pred_b)

    states = torch.where(fire_mask, output, states)
    predictions = torch.where(fire_mask, new_predictions, predictions)
    global_buffer = torch.where(fire_mask, output.detach(), global_buffer)
    has_predicted = has_predicted | fires

    if model.refractory:
        output_norms = output.float().norm(dim=-1).amax(dim=1)
        fired_strongly = fires & (output_norms > model.refractory_threshold)
        refractory_levels = torch.where(
            fired_strongly,
            torch.ones_like(refractory_levels),
            refractory_levels,
        )

    return states, global_buffer, predictions, has_predicted, refractory_levels


def record_norms(model: GraphCellularAutomaton, states: Tensor) -> Tensor:
    return states[:, 0].float().norm(dim=-1).reshape(model.n_bands, model.n_cols).cpu()


def render_heatmap(snapshot: Tensor) -> list[str]:
    peak = float(snapshot.max().item())
    if peak == 0.0:
        return [" " * snapshot.shape[1] for _ in range(snapshot.shape[0])]

    rows: list[str] = []
    for band in range(snapshot.shape[0]):
        chars: list[str] = []
        for col in range(snapshot.shape[1]):
            scaled = float(snapshot[band, col].item()) / peak
            palette_index = min(int(round(scaled * (len(PALETTE) - 1))), len(PALETTE) - 1)
            chars.append(PALETTE[palette_index])
        rows.append("".join(chars))
    return rows


def print_snapshot(timestep: int, snapshot: Tensor) -> None:
    print(
        f"t={timestep:>3} max_norm={snapshot.max().item():6.3f} "
        f"mean_norm={snapshot.mean().item():6.3f} total_norm={snapshot.sum().item():7.3f}"
    )
    print("     cols: 000000000011111111112222")
    print("           012345678901234567890123")
    for band, row in enumerate(render_heatmap(snapshot)):
        print(f"band {band}: {row}")
    print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to trained checkpoint")
    parser.add_argument("--band", type=int, default=4)
    parser.add_argument("--col", type=int, default=12)
    parser.add_argument("--magnitude", type=float, default=10.0)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")

    model = GraphCellularAutomaton(
        vocab_size=128,
        noise_std=0.0,
        refractory=True,
        refractory_threshold=0.5,
        refractory_decay=0.9,
    ).to(device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from step {ckpt['step']} (ce_loss={ckpt['ce_loss']:.4f})")
    else:
        print("Using random initialization")

    model.eval()

    with torch.no_grad():
        states, global_buffer, predictions, has_predicted, refractory_levels = model.initial_recurrent_state(
            1,
            device=device,
        )
        inject_impulse(model, states, global_buffer, band=args.band, col=args.col, magnitude=args.magnitude)

        snapshots: dict[int, Tensor] = {0: record_norms(model, states)}
        for timestep in range(1, max(SNAPSHOT_STEPS) + 1):
            states, global_buffer, predictions, has_predicted, refractory_levels = step_automaton(
                model,
                states,
                global_buffer,
                predictions,
                has_predicted,
                refractory_levels,
                timestep=timestep - 1,
            )
            if timestep in SNAPSHOT_STEPS:
                snapshots[timestep] = record_norms(model, states)

    print(f"Graph cellular automaton impulse response")
    print(f"impulse: band={args.band} col={args.col} magnitude={args.magnitude}")
    print()

    for timestep in SNAPSHOT_STEPS:
        print_snapshot(timestep, snapshots[timestep])


if __name__ == "__main__":
    main()
