from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an automaton training log JSONL file.")
    parser.add_argument("--log-path", type=Path, default=Path("experiments/automaton/artifacts/run_v1.jsonl"))
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes:d}m {seconds:02d}s"


def print_row(label: str, value: str) -> None:
    print(f"{label:<26} {value}")


def main() -> None:
    args = parse_args()
    entries = [json.loads(line) for line in args.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"Log file is empty: {args.log_path}")

    first = entries[0]
    last = entries[-1]
    baseline = math.log(65)
    start_ce = first["ce_loss"]
    end_ce = last["ce_loss"]
    ce_improvement = start_ce - end_ce
    avg_wall_time = sum(entry["wall_time_s"] for entry in entries) / len(entries)
    total_training_time = avg_wall_time * last["step"]
    start_losses = first["prediction_losses"]
    end_losses = last["prediction_losses"]
    start_gradient = start_losses[0] / start_losses[-1]
    end_gradient = end_losses[0] / end_losses[-1]
    learned = end_ce < baseline and end_ce < start_ce

    print(f"Log: {args.log_path}")
    print()
    print("Summary")
    print("-" * 40)
    print_row("First step", str(first["step"]))
    print_row("Last step", str(last["step"]))
    print_row("CE loss start", f"{start_ce:.4f}")
    print_row("CE loss end", f"{end_ce:.4f}")
    print_row("CE improvement", f"{ce_improvement:+.4f}")
    print_row("Random baseline", f"ln(65) = {baseline:.4f}")
    print_row("End vs baseline", f"{baseline - end_ce:+.4f}")
    print_row("Avg wall time / step", f"{avg_wall_time:.2f}s")
    print_row("Total training time", f"~{format_duration(total_training_time)}")
    print_row("Model learned", "yes" if learned else "no")
    print_row("Pred gradient start", f"{start_gradient:.3f}")
    print_row("Pred gradient end", f"{end_gradient:.3f}")
    print_row("Gradient differentiated", "yes" if end_gradient > 1.0 else "no")

    print()
    print("Per-level prediction losses")
    print("-" * 40)
    print(f"{'level':<8}{'start':>12}{'end':>12}{'delta':>12}")
    for level, (start_loss, end_loss) in enumerate(zip(start_losses, end_losses, strict=True)):
        print(f"{level:<8}{start_loss:>12.6f}{end_loss:>12.6f}{(start_loss - end_loss):>12.6f}")

    print()
    print("Learned check")
    print("-" * 40)
    print(f"CE below random baseline: {'yes' if end_ce < baseline else 'no'}")
    print(f"CE decreasing: {'yes' if end_ce < start_ce else 'no'}")


if __name__ == "__main__":
    main()
