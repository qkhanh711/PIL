from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot PIL ablation summary metrics.")
    parser.add_argument(
        "--summary",
        type=str,
        default=str(ROOT / "experiments" / "exp_runs" / "05_pil_ablation" / "ablation_summary.json"),
        help="Path to ablation_summary.json",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(ROOT / "plots" / "exp_runs" / "05_pil_ablation"),
        help="Directory to save generated plots.",
    )
    return parser


def metric_bar(ax, names: list[str], values: np.ndarray, title: str, ylabel: str, higher_is_better: bool = True) -> None:
    x = np.arange(len(names))
    colors = ["#2E86AB" if name == "default" else "#F18F01" for name in names]
    bars = ax.bar(x, values, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    # Highlight best variant for quick reading
    best_idx = int(np.argmax(values) if higher_is_better else np.argmin(values))
    bars[best_idx].set_edgecolor("#111111")
    bars[best_idx].set_linewidth(2.0)

    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)


def delta_bar(ax, names: list[str], values: np.ndarray, title: str, ylabel: str) -> None:
    x = np.arange(len(names))
    colors = ["#2E86AB" if name == "default" else ("#4F772D" if v >= 0.0 else "#D1495B") for name, v in zip(names, values)]
    bars = ax.bar(x, values, color=colors, alpha=0.9)
    ax.axhline(0.0, color="#111111", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    max_abs = float(np.max(np.abs(values))) if values.size > 0 else 1.0
    ypad = max(0.001, max_abs * 0.15)
    ax.set_ylim(-max_abs - ypad, max_abs + ypad)

    for i, v in enumerate(values):
        va = "bottom" if v >= 0.0 else "top"
        offset = ypad * 0.1 if v >= 0.0 else -ypad * 0.1
        ax.text(i, v + offset, f"{v:+.4f}", ha="center", va=va, fontsize=8)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    payload = json.loads(summary_path.read_text())
    variants = payload.get("variants", {})
    if not variants:
        raise ValueError("No variants found in summary payload")

    names = list(variants.keys())
    rewards = np.asarray([variants[name]["metrics"]["team_reward"] for name in names], dtype=np.float64)
    regrets = np.asarray([variants[name]["metrics"]["welfare_regret"] for name in names], dtype=np.float64)
    epsilons = np.asarray([variants[name]["metrics"]["mean_epsilon"] for name in names], dtype=np.float64)
    kls = np.asarray([variants[name]["metrics"]["mean_kl_distortion"] for name in names], dtype=np.float64)

    reward_deltas = np.asarray([variants[name]["delta_vs_default"]["team_reward"] for name in names], dtype=np.float64)
    regret_deltas = np.asarray([variants[name]["delta_vs_default"]["welfare_regret"] for name in names], dtype=np.float64)
    epsilon_deltas = np.asarray([variants[name]["delta_vs_default"]["mean_epsilon"] for name in names], dtype=np.float64)
    kl_deltas = np.asarray([variants[name]["delta_vs_default"]["mean_kl_distortion"] for name in names], dtype=np.float64)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to plot ablation charts") from exc

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    metric_bar(
        axes[0, 0],
        names,
        rewards,
        title="Ablation: Team Reward (higher is better)",
        ylabel="Team Reward",
        higher_is_better=True,
    )
    metric_bar(
        axes[0, 1],
        names,
        regrets,
        title="Ablation: Welfare Regret (lower is better)",
        ylabel="Welfare Regret",
        higher_is_better=False,
    )
    metric_bar(
        axes[1, 0],
        names,
        epsilons,
        title="Ablation: Mean Epsilon (lower is better)",
        ylabel="Mean Epsilon",
        higher_is_better=False,
    )
    metric_bar(
        axes[1, 1],
        names,
        kls,
        title="Ablation: Mean KL Distortion (lower is better)",
        ylabel="Mean KL Distortion",
        higher_is_better=False,
    )

    source = payload.get("checkpoint_source", "final")
    seeds = payload.get("seeds", [])
    fig.suptitle(f"PIL Ablation Summary | source={source} | seeds={seeds}", fontsize=12, y=1.02)
    fig.tight_layout()

    out_png = plots_dir / "pil_ablation_summary.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    delta_fig, delta_axes = plt.subplots(2, 2, figsize=(13, 8.5))
    delta_bar(
        delta_axes[0, 0],
        names,
        reward_deltas,
        title="Delta vs Default: Team Reward",
        ylabel="Δ Team Reward",
    )
    delta_bar(
        delta_axes[0, 1],
        names,
        regret_deltas,
        title="Delta vs Default: Welfare Regret",
        ylabel="Δ Welfare Regret",
    )
    delta_bar(
        delta_axes[1, 0],
        names,
        epsilon_deltas,
        title="Delta vs Default: Mean Epsilon",
        ylabel="Δ Mean Epsilon",
    )
    delta_bar(
        delta_axes[1, 1],
        names,
        kl_deltas,
        title="Delta vs Default: Mean KL Distortion",
        ylabel="Δ Mean KL Distortion",
    )
    delta_fig.suptitle(f"PIL Ablation Delta vs Default | source={source} | seeds={seeds}", fontsize=12, y=1.02)
    delta_fig.tight_layout()

    out_delta_png = plots_dir / "pil_ablation_delta_vs_default.png"
    delta_fig.savefig(out_delta_png, dpi=220, bbox_inches="tight")
    plt.close(delta_fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_delta_png}")


if __name__ == "__main__":
    main()
