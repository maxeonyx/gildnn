from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def run_comparison(*, repo_root: Path, context_size: int, output_dir: Path, device: str) -> None:
    command = [
        sys.executable,
        "-m",
        "experiments.pytorch_char_shakespeare_comparison",
        "--device",
        device,
        "--context-size",
        str(context_size),
        "--predictive-aux-weights",
        "1.0",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (
        repo_root
        / "research"
        / "questions"
        / "predictive-chain"
        / "artifacts"
        / "context_scaling"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    context_sizes = [5, 10, 20]
    summaries: list[dict[str, object]] = []

    for context_size in context_sizes:
        context_dir = output_root / f"context_{context_size}"
        run_comparison(
            repo_root=repo_root,
            context_size=context_size,
            output_dir=context_dir,
            device=device,
        )
        comparison_summary = load_json(context_dir / "comparison_summary.json")
        models = comparison_summary["models"]
        summaries.append(
            {
                "context_size": context_size,
                "output_dir": str(context_dir.relative_to(repo_root)).replace("\\", "/"),
                "models": models,
            }
        )

    architecture_names = [
        "feedforward_baseline",
        "rnn_baseline",
        "transformer_baseline",
        "predictive_chain_aux_1_detach",
    ]
    val_loss_table = {
        architecture: {
            str(summary["context_size"]): next(
                model["val_loss"]
                for model in summary["models"]
                if model["model_name"] == architecture
            )
            for summary in summaries
        }
        for architecture in architecture_names
    }

    best_by_context = {
        str(summary["context_size"]): min(
            summary["models"], key=lambda model: model["val_loss"]
        )
        for summary in summaries
    }

    write_json(
        output_root / "summary.json",
        {
            "device": device,
            "context_sizes": context_sizes,
            "runs": summaries,
            "val_loss_table": val_loss_table,
            "best_by_context": best_by_context,
        },
    )


if __name__ == "__main__":
    main()
