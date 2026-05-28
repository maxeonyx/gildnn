import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


DTYPE = torch.float64
DEVICE = torch.device("cpu")
SEED = 42
DIM = 16
PAIRS = DIM // 2
HIDDEN = 64
TRAIN_STEPS = 140
LOG_EVERY = 35
BATCH = 48
SEQ_LEN = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
EVAL_BATCHES = 8
NOISE_LEVELS = (0.0, 0.1, 0.2, 0.5, 1.0)
PHASE_SPEED_MIN = 0.35
PHASE_SPEED_MAX = 0.80
PHASE_ACCEL_SCALE = 0.015
AMPLITUDE_MODULATION = 0.08
OBSERVATION_NOISE = 0.03


@dataclass(frozen=True)
class NoiseRunResult:
    noise_sigma: float
    final_train_mse: float
    eval_mse: float
    copy_baseline_mse: float
    absolute_improvement: float
    relative_gain: float
    beats_copy: bool


class RecurrentPredictor(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=dim, hidden_size=hidden, batch_first=True)
        self.readout = nn.Linear(hidden, dim)

    def forward(self, noisy_inputs: torch.Tensor) -> torch.Tensor:
        hidden_states, _final_state = self.gru(noisy_inputs)
        return self.readout(hidden_states)


def l2_normalize(vectors: torch.Tensor) -> torch.Tensor:
    return F.normalize(vectors, dim=-1)


def generate_clean_sequences(*, batch: int, seq_len: int, generator: torch.Generator) -> torch.Tensor:
    initial_phase = torch.rand(batch, PAIRS, generator=generator, dtype=DTYPE, device=DEVICE) * (2.0 * math.pi)
    direction_sign = torch.where(
        torch.rand(batch, PAIRS, generator=generator, dtype=DTYPE, device=DEVICE) > 0.5,
        1.0,
        -1.0,
    )
    angular_speed = (
        PHASE_SPEED_MIN
        + (PHASE_SPEED_MAX - PHASE_SPEED_MIN) * torch.rand(batch, PAIRS, generator=generator, dtype=DTYPE, device=DEVICE)
    ) * direction_sign
    angular_acceleration = PHASE_ACCEL_SCALE * torch.randn(batch, PAIRS, generator=generator, dtype=DTYPE, device=DEVICE)
    base_amplitude = 0.7 + 0.3 * torch.rand(batch, PAIRS, generator=generator, dtype=DTYPE, device=DEVICE)

    time_index = torch.arange(seq_len, dtype=DTYPE, device=DEVICE).view(1, seq_len, 1)
    phase = initial_phase[:, None, :] + angular_speed[:, None, :] * time_index + 0.5 * angular_acceleration[:, None, :] * time_index.square()
    amplitude = base_amplitude[:, None, :] * (1.0 + AMPLITUDE_MODULATION * torch.sin(0.5 * phase + 0.3))
    circular_pairs = torch.stack((amplitude * torch.cos(phase), amplitude * torch.sin(phase)), dim=-1)
    sequences = circular_pairs.reshape(batch, seq_len, DIM)
    sequences = sequences + OBSERVATION_NOISE * torch.randn(batch, seq_len, DIM, generator=generator, dtype=DTYPE, device=DEVICE)
    return l2_normalize(sequences)


def add_gaussian_noise(clean_sequences: torch.Tensor, *, sigma: float, generator: torch.Generator) -> torch.Tensor:
    if sigma == 0.0:
        return clean_sequences.clone()
    noise = torch.randn(clean_sequences.shape, generator=generator, dtype=DTYPE, device=DEVICE)
    return clean_sequences + sigma * noise


def predictor_loss(predictions: torch.Tensor, clean_sequences: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predictions[:, :-1], clean_sequences[:, 1:])


def copy_baseline_loss(noisy_sequences: torch.Tensor, clean_sequences: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(noisy_sequences[:, :-1], clean_sequences[:, 1:])


def train_noise_level(*, noise_sigma: float, seed: int) -> NoiseRunResult:
    train_generator = torch.Generator(device=DEVICE).manual_seed(seed)
    eval_generator = torch.Generator(device=DEVICE).manual_seed(seed + 10_000)

    model = RecurrentPredictor(dim=DIM, hidden=HIDDEN).to(device=DEVICE, dtype=DTYPE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    final_train_mse = math.nan
    for step in range(1, TRAIN_STEPS + 1):
        clean_sequences = generate_clean_sequences(batch=BATCH, seq_len=SEQ_LEN, generator=train_generator)
        noisy_sequences = add_gaussian_noise(clean_sequences, sigma=noise_sigma, generator=train_generator)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(noisy_sequences)
        loss = predictor_loss(predictions, clean_sequences)
        loss.backward()
        optimizer.step()

        final_train_mse = loss.item()
        if step == 1 or step % LOG_EVERY == 0:
            copy_mse = copy_baseline_loss(noisy_sequences, clean_sequences).item()
            print(
                f"noise_sigma={noise_sigma:>3.1f} "
                f"step={step:3d} "
                f"train_mse={final_train_mse:.6f} "
                f"copy_mse={copy_mse:.6f}"
            )

    eval_losses: list[float] = []
    copy_losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            clean_sequences = generate_clean_sequences(batch=BATCH, seq_len=SEQ_LEN, generator=eval_generator)
            noisy_sequences = add_gaussian_noise(clean_sequences, sigma=noise_sigma, generator=eval_generator)
            predictions = model(noisy_sequences)
            eval_losses.append(predictor_loss(predictions, clean_sequences).item())
            copy_losses.append(copy_baseline_loss(noisy_sequences, clean_sequences).item())

    eval_mse = sum(eval_losses) / len(eval_losses)
    copy_mse = sum(copy_losses) / len(copy_losses)
    absolute_improvement = copy_mse - eval_mse
    relative_gain = absolute_improvement / copy_mse
    return NoiseRunResult(
        noise_sigma=noise_sigma,
        final_train_mse=final_train_mse,
        eval_mse=eval_mse,
        copy_baseline_mse=copy_mse,
        absolute_improvement=absolute_improvement,
        relative_gain=relative_gain,
        beats_copy=eval_mse < copy_mse,
    )


def print_summary(results: list[NoiseRunResult], *, elapsed_seconds: float) -> None:
    print()
    print("=== Noise on lateral summary ===")
    print(f"seed={SEED}  dtype={DTYPE}  device={DEVICE}  dim={DIM}  hidden={HIDDEN}  steps={TRAIN_STEPS}  elapsed_s={elapsed_seconds:.2f}")
    print()
    print("sigma  final_train_mse  eval_mse   copy_mse   abs_improve  gain_vs_copy  beats_copy")
    print("-----  ---------------  ---------  ---------  -----------  ------------  ----------")
    for result in results:
        print(
            f"{result.noise_sigma:5.1f}  {result.final_train_mse:15.6f}  {result.eval_mse:9.6f}  "
            f"{result.copy_baseline_mse:9.6f}  {result.absolute_improvement:11.6f}  {result.relative_gain:12.2%}  {str(result.beats_copy):>10}"
        )


def main() -> int:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    start_time = time.perf_counter()
    results = [train_noise_level(noise_sigma=noise_sigma, seed=SEED + 1_000 * index) for index, noise_sigma in enumerate(NOISE_LEVELS)]
    elapsed_seconds = time.perf_counter() - start_time

    print_summary(results, elapsed_seconds=elapsed_seconds)

    failures: list[str] = []
    if elapsed_seconds >= 30.0:
        failures.append(f"runtime exceeded budget: {elapsed_seconds:.2f}s")
    if not results[0].beats_copy:
        failures.append("sigma=0.0 did not beat copy baseline")
    if not all(result.beats_copy for result in results[1:]):
        failures.append("at least one noisy condition failed to beat copy baseline")
    if any(current.eval_mse + 1e-12 < previous.eval_mse for previous, current in zip(results, results[1:], strict=False)):
        failures.append("eval_mse did not worsen monotonically with noise")

    if failures:
        print()
        print("FAILURES:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
