from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODEL_SPECS = {
    "pil": {
        "label": "PIL-APS",
        "color": "#2E86AB",
        "candidates": ["pil_aps_metrics.json", "pil_metrics.json"],
    },
    "dpmac": {
        "label": "DPMAC",
        "color": "#F18F01",
        "candidates": ["dpmac_metrics.json"],
    },
    "i2c": {
        "label": "I2C",
        "color": "#4F772D",
        "candidates": ["i2c_metrics.json"],
    },
    "maddpg": {
        "label": "MADDPG",
        "color": "#D1495B",
        "candidates": ["maddpg_metrics.json"],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot per-model convergence curves for Exp 1 single-trainer outputs.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(ROOT / "experiments" / "exp_runs" / "01_single_trainers"),
        help="Directory containing Exp 1 JSON metric files.",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(ROOT / "plots" / "exp_runs" / "01_single_trainers"),
        help="Directory used for generated convergence plots.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="pil,dpmac,i2c,maddpg",
        help="Comma-separated list drawn from: pil,dpmac,i2c,maddpg",
    )
    return parser


def load_runs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "runs" in payload:
        return list(payload["runs"])
    return [payload]


def extract_series(runs: list[dict], extractor: Callable[[dict], float]) -> np.ndarray:
    series = [[extractor(entry) for entry in run.get("history", [])] for run in runs]
    return np.asarray(series, dtype=np.float64)


def maybe_scalar_list(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size > 0 else 0.0


def pick_file(input_dir: Path, model_key: str) -> Path | None:
    for candidate in MODEL_SPECS[model_key]["candidates"]:
        path = input_dir / candidate
        if path.exists():
            return path
    return None


def plot_model(
    *,
    model_key: str,
    runs: list[dict],
    plots_dir: Path,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if not runs or not runs[0].get("history"):
        return None

    label = MODEL_SPECS[model_key]["label"]
    color = MODEL_SPECS[model_key]["color"]

    metric_specs = [
        ("team_reward", "Team Reward", lambda entry: float(entry["team_reward"])),
        ("welfare_regret", "Welfare Regret", lambda entry: float(entry["welfare_regret"])),
        ("mean_kl", "Mean KL Distortion", lambda entry: maybe_scalar_list(entry.get("kl_distortion", []))),
        ("mean_eps", "Mean Epsilon", lambda entry: maybe_scalar_list(entry.get("privacy", {}).get("epsilon", []))),
    ]

    block_axis = np.arange(len(runs[0]["history"]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    axes = axes.flatten()

    for ax, (_, ylabel, extractor) in zip(axes, metric_specs):
        values = extract_series(runs, extractor)
        mean = values.mean(axis=0)
        std = values.std(axis=0) if values.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(block_axis, mean, color=color, linewidth=2.0, label=label)
        if np.any(std > 0.0):
            ax.fill_between(block_axis, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)

    axes[2].set_xlabel("Block")
    axes[3].set_xlabel("Block")
    fig.suptitle(f"Exp 1 Convergence: {label}")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = plots_dir / f"{model_key}_convergence.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    models = [model.strip().lower() for model in args.models.split(",") if model.strip()]
    invalid = [model for model in models if model not in MODEL_SPECS]
    if invalid:
        raise ValueError(f"Unsupported models: {invalid}")

    generated = {}
    for model_key in models:
        source_path = pick_file(input_dir, model_key)
        if source_path is None:
            print(f"[skip] {model_key}: no metrics JSON found in {input_dir}")
            continue
        runs = load_runs(source_path)
        output_path = plot_model(model_key=model_key, runs=runs, plots_dir=plots_dir)
        if output_path is None:
            print(f"[skip] {model_key}: plotting unavailable or empty history in {source_path}")
            continue
        generated[model_key] = {
            "source": str(source_path),
            "plot": str(output_path),
            "num_runs": len(runs),
            "num_blocks": len(runs[0].get("history", [])),
        }

    print(json.dumps({"generated": generated}, indent=2))


if __name__ == "__main__":
    main()
