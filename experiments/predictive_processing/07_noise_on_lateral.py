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
    use_recurrence: bool
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


def matching_feedforward_hidden(dim: int, hidden: int) -> int:
    recurrent_parameter_count = sum(parameter.numel() for parameter in RecurrentPredictor(dim=dim, hidden=hidden).parameters())
    return max(1, (recurrent_parameter_count - dim) // (2 * dim + 1))


def feedforward_parameter_remainder(dim: int, hidden: int, *, feedforward_hidden: int) -> int:
    recurrent_parameter_count = sum(parameter.numel() for parameter in RecurrentPredictor(dim=dim, hidden=hidden).parameters())
    feedforward_parameter_count = (2 * dim + 1) * feedforward_hidden + dim
    return recurrent_parameter_count - feedforward_parameter_count


class ScalarPolynomialCorrection(nn.Module):
    def __init__(self, parameter_count: int) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(torch.zeros(parameter_count)) if parameter_count > 0 else None

    def forward(self, noisy_inputs: torch.Tensor) -> torch.Tensor:
        if self.coefficients is None:
            return torch.zeros((*noisy_inputs.shape[:-1], 1), dtype=noisy_inputs.dtype, device=noisy_inputs.device)

        summary = noisy_inputs.mean(dim=-1, keepdim=True)
        basis_terms = torch.cat([summary.pow(power) for power in range(1, self.coefficients.numel() + 1)], dim=-1)
        return (basis_terms * self.coefficients).sum(dim=-1, keepdim=True)


class FeedforwardPredictor(nn.Module):
    def __init__(self, dim: int, hidden: int, *, match_parameter_count_to_hidden: int) -> None:
        super().__init__()
        feedforward_hidden = matching_feedforward_hidden(dim=dim, hidden=match_parameter_count_to_hidden)
        parameter_remainder = feedforward_parameter_remainder(
            dim=dim,
            hidden=match_parameter_count_to_hidden,
            feedforward_hidden=feedforward_hidden,
        )
        self.input_layer = nn.Linear(dim, feedforward_hidden)
        self.output_layer = nn.Linear(feedforward_hidden, dim)
        self.activation = nn.GELU()
        self.correction = ScalarPolynomialCorrection(parameter_remainder)

    def forward(self, noisy_inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.input_layer(noisy_inputs))
        predictions = self.output_layer(hidden_states)
        predictions[..., :1] = predictions[..., :1] + self.correction(noisy_inputs)
        return predictions


def build_predictor(*, dim: int, hidden: int, use_recurrence: bool) -> nn.Module:
    if use_recurrence:
        return RecurrentPredictor(dim=dim, hidden=hidden)
    return FeedforwardPredictor(dim=dim, hidden=hidden, match_parameter_count_to_hidden=hidden)


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


def train_noise_level(*, noise_sigma: float, seed: int, use_recurrence: bool) -> NoiseRunResult:
    train_generator = torch.Generator(device=DEVICE).manual_seed(seed)
    eval_generator = torch.Generator(device=DEVICE).manual_seed(seed + 10_000)

    torch.manual_seed(seed)
    model = build_predictor(dim=DIM, hidden=HIDDEN, use_recurrence=use_recurrence).to(device=DEVICE, dtype=DTYPE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    variant_label = "gru" if use_recurrence else "ff"

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
                f"variant={variant_label:>3s} "
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
        use_recurrence=use_recurrence,
        noise_sigma=noise_sigma,
        final_train_mse=final_train_mse,
        eval_mse=eval_mse,
        copy_baseline_mse=copy_mse,
        absolute_improvement=absolute_improvement,
        relative_gain=relative_gain,
        beats_copy=eval_mse < copy_mse,
    )


def print_summary(results: list[NoiseRunResult], *, elapsed_seconds: float) -> None:
    grouped_results = {
        result.noise_sigma: {variant.use_recurrence: variant for variant in results if variant.noise_sigma == result.noise_sigma}
        for result in results
    }

    print()
    print("=== Noise on lateral summary ===")
    print(f"seed={SEED}  dtype={DTYPE}  device={DEVICE}  dim={DIM}  hidden={HIDDEN}  steps={TRAIN_STEPS}  elapsed_s={elapsed_seconds:.2f}")
    print()
    print("sigma  gru_eval   ff_eval    copy_mse   gru_gain  ff_gain   recurrence_advantage  gru_beats_copy  ff_beats_copy")
    print("-----  ---------  ---------  ---------  --------  --------  --------------------  --------------  -------------")
    for noise_sigma in NOISE_LEVELS:
        gru_result = grouped_results[noise_sigma][True]
        ff_result = grouped_results[noise_sigma][False]
        recurrence_advantage = gru_result.relative_gain - ff_result.relative_gain
        print(
            f"{noise_sigma:5.1f}  {gru_result.eval_mse:9.6f}  {ff_result.eval_mse:9.6f}  "
            f"{gru_result.copy_baseline_mse:9.6f}  {gru_result.relative_gain:8.2%}  {ff_result.relative_gain:8.2%}  "
            f"{recurrence_advantage:20.2%}  {str(gru_result.beats_copy):>14}  {str(ff_result.beats_copy):>13}"
        )


def main() -> int:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    start_time = time.perf_counter()
    results = [
        train_noise_level(noise_sigma=noise_sigma, seed=SEED + 1_000 * index, use_recurrence=use_recurrence)
        for index, noise_sigma in enumerate(NOISE_LEVELS)
        for use_recurrence in (True, False)
    ]
    elapsed_seconds = time.perf_counter() - start_time

    print_summary(results, elapsed_seconds=elapsed_seconds)

    grouped_results = {
        noise_sigma: {result.use_recurrence: result for result in results if result.noise_sigma == noise_sigma}
        for noise_sigma in NOISE_LEVELS
    }

    failures: list[str] = []
    if elapsed_seconds >= 30.0:
        failures.append(f"runtime exceeded budget: {elapsed_seconds:.2f}s")
    if not all(grouped_results[noise_sigma][True].beats_copy for noise_sigma in NOISE_LEVELS):
        failures.append("gru predictor failed to beat copy baseline for at least one noise level")
    if not all(grouped_results[noise_sigma][False].beats_copy for noise_sigma in NOISE_LEVELS):
        failures.append("feedforward predictor failed to beat copy baseline for at least one noise level")
    gru_results = [grouped_results[noise_sigma][True] for noise_sigma in NOISE_LEVELS]
    ff_results = [grouped_results[noise_sigma][False] for noise_sigma in NOISE_LEVELS]
    if any(current.eval_mse + 1e-12 < previous.eval_mse for previous, current in zip(gru_results, gru_results[1:], strict=False)):
        failures.append("gru eval_mse did not worsen monotonically with noise")
    if any(current.eval_mse + 1e-12 < previous.eval_mse for previous, current in zip(ff_results, ff_results[1:], strict=False)):
        failures.append("feedforward eval_mse did not worsen monotonically with noise")

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
