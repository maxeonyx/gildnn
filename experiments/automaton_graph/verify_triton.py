from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.automaton_graph import GraphCellularAutomaton
from core.triton_forward import triton_forward_chunk


def _clone_state(state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.clone() for tensor in state)


def _assert_close(name: str, actual: torch.Tensor | None, expected: torch.Tensor | None) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{name} mismatch")
        return
    if actual.dtype in (torch.bool, torch.int32, torch.int64, torch.long):
        if not torch.equal(actual, expected):
            raise AssertionError(f"{name} mismatch")
        return
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3, msg=name)


def _run_eager(
    model: GraphCellularAutomaton,
    tokens: torch.Tensor,
    state: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return model.forward_chunk(tokens, *state, global_step_offset=0)


def _run_triton(
    model: GraphCellularAutomaton,
    tokens: torch.Tensor,
    state: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return triton_forward_chunk(model, tokens, *state, global_step_offset=0)


def _benchmark(label: str, fn, *, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    print(f"{label}: {elapsed_ms:.3f} ms")
    return elapsed_ms


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton verification.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size = 4
    seq_len = 16
    vocab_size = 128

    eager_model = GraphCellularAutomaton(vocab_size=vocab_size).to(device)
    triton_model = copy.deepcopy(eager_model).to(device)
    eager_model.eval()
    triton_model.eval()

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    initial_state = eager_model.initial_recurrent_state(batch_size, device=device)
    eager_state = _clone_state(initial_state)
    triton_state = _clone_state(initial_state)

    with torch.no_grad():
        torch.manual_seed(1234)
        eager_outputs = _run_eager(eager_model, tokens, eager_state)
        torch.manual_seed(1234)
        triton_outputs = _run_triton(triton_model, tokens, triton_state)

    names = [
        "logits",
        "per_band_logits",
        "states",
        "global_buffer",
        "predictions",
        "has_predicted",
        "refractory_levels",
        "prediction_loss_sums",
        "prediction_counts",
    ]
    for name, eager_value, triton_value in zip(names, eager_outputs, triton_outputs, strict=True):
        _assert_close(name, triton_value, eager_value)

    print("numerical agreement: PASS")

    eager_benchmark_state = _clone_state(initial_state)
    triton_benchmark_state = _clone_state(initial_state)

    def eager_fn() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            torch.manual_seed(4321)
            return _run_eager(eager_model, tokens, _clone_state(eager_benchmark_state))

    def triton_fn() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            torch.manual_seed(4321)
            return _run_triton(triton_model, tokens, _clone_state(triton_benchmark_state))

    eager_ms = _benchmark("eager", eager_fn)
    triton_ms = _benchmark("triton", triton_fn)
    print(f"speedup: {eager_ms / triton_ms:.3f}x")


if __name__ == "__main__":
    main()
