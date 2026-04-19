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


ALGO_STYLES = {
    "pil": ("PIL-APS", "#2E86AB"),
    "dpmac": ("DPMAC", "#F18F01"),
    "i2c": ("I2C", "#4F772D"),
    "tarmac": ("TarMAC", "#7B2CBF"),
    "maddpg": ("MADDPG", "#D1495B"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot convergence curves for matrix-game and MPE suite outputs.")
    parser.add_argument(
        "--matrix_dir",
        type=str,
        default=str(ROOT / "experiments" / "exp_runs" / "03_matrix_suite"),
        help="Directory containing saved matrix-suite JSON files.",
    )
    parser.add_argument(
        "--rl_dir",
        type=str,
        default=str(ROOT / "experiments" / "exp_runs" / "04_mpe_suite"),
        help="Directory containing saved MPE-suite JSON files.",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(ROOT / "plots" / "exp_runs"),
        help="Root directory for generated convergence plots.",
    )
    return parser


def load_runs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "runs" in payload:
        return list(payload["runs"])
    return [payload]


def extract_series(runs: list[dict], extractor: Callable[[dict], float]) -> np.ndarray:
    return np.asarray([[extractor(entry) for entry in run["history"]] for run in runs], dtype=np.float64)


def mean_list(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def plot_with_band(ax, x_axis: np.ndarray, series: np.ndarray, *, label: str, color: str) -> None:
    mean = series.mean(axis=0)
    std = series.std(axis=0) if series.shape[0] > 1 else np.zeros_like(mean)
    ax.plot(x_axis, mean, linewidth=2.0, color=color, label=label)
    if np.any(std > 0.0):
        ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)


def plot_matrix_suite(matrix_dir: Path, plots_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    generated: dict[str, str] = {}
    games = ["binary_sum", "multi_round_sum"]
    algorithms = [key for key in ALGO_STYLES if (matrix_dir / f"{games[0]}_{key}.json").exists() or (matrix_dir / f"{games[1]}_{key}.json").exists()]

    target_dir = plots_dir / "03_matrix_suite"
    target_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("average_episode_reward", "Reward", lambda entry: float(entry["average_episode_reward"])),
        ("prediction_error", "Prediction Error", lambda entry: float(entry["prediction_error"])),
        ("epsilon", "Mean Epsilon", lambda entry: mean_list(entry.get("privacy", {}).get("epsilon", []))),
        ("kl", "Mean KL Distortion", lambda entry: mean_list(entry.get("kl_distortion", []))),
    ]

    for game in games:
        available = [(algo, matrix_dir / f"{game}_{algo}.json") for algo in algorithms if (matrix_dir / f"{game}_{algo}.json").exists()]
        if not available:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True)
        axes = axes.flatten()

        for algo, path in available:
            label, color = ALGO_STYLES[algo]
            runs = load_runs(path)
            x_axis = np.arange(1, len(runs[0]["history"]) + 1)
            for ax, (_, ylabel, extractor) in zip(axes, metric_specs):
                series = extract_series(runs, extractor)
                plot_with_band(ax, x_axis, series, label=label, color=color)
                ax.set_ylabel(ylabel)

        axes[2].set_xlabel("Block")
        axes[3].set_xlabel("Block")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=min(5, len(handles)), frameon=False)
        fig.suptitle(f"Matrix Game Convergence: {game}")
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        output_path = target_dir / f"{game}_convergence.png"
        fig.savefig(output_path)
        plt.close(fig)
        generated[game] = str(output_path)

    return generated


def plot_rl_suite(rl_dir: Path, plots_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    generated: dict[str, str] = {}
    scenarios = ["cn", "ccn", "pp"]
    algorithms = [key for key in ALGO_STYLES if any((rl_dir / f"{scenario}_{key}.json").exists() for scenario in scenarios)]

    target_dir = plots_dir / "04_mpe_suite"
    target_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("average_episode_reward", "Eval Reward", lambda entry: float(entry["average_episode_reward"])),
        ("running_train_reward", "Train Reward", lambda entry: float(entry["running_train_reward"])),
        ("epsilon", "Mean Epsilon", lambda entry: mean_list(entry.get("privacy", {}).get("epsilon", []))),
        ("leakage", "Empirical Leakage", lambda entry: mean_list(entry.get("empirical_leakage", []))),
    ]

    for scenario in scenarios:
        available = [(algo, rl_dir / f"{scenario}_{algo}.json") for algo in algorithms if (rl_dir / f"{scenario}_{algo}.json").exists()]
        if not available:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True)
        axes = axes.flatten()

        for algo, path in available:
            label, color = ALGO_STYLES[algo]
            runs = load_runs(path)
            x_axis = np.asarray([entry["episode"] for entry in runs[0]["history"]], dtype=np.float64)
            for ax, (_, ylabel, extractor) in zip(axes, metric_specs):
                series = extract_series(runs, extractor)
                plot_with_band(ax, x_axis, series, label=label, color=color)
                ax.set_ylabel(ylabel)

        axes[2].set_xlabel("Episode")
        axes[3].set_xlabel("Episode")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=min(5, len(handles)), frameon=False)
        fig.suptitle(f"MPE Convergence: {scenario.upper()}")
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        output_path = target_dir / f"{scenario}_convergence.png"
        fig.savefig(output_path)
        plt.close(fig)
        generated[scenario] = str(output_path)

    return generated


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        generated = {
            "matrix": plot_matrix_suite(Path(args.matrix_dir), plots_dir),
            "rl": plot_rl_suite(Path(args.rl_dir), plots_dir),
        }
    except Exception as exc:
        raise SystemExit(f"plotting failed: {exc}") from exc

    print(json.dumps(generated, indent=2))


if __name__ == "__main__":
    main()
