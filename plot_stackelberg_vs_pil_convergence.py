"""
Plot convergence curves:
1. Stackelberg Privacy-Pricing Game (lambda evolution)
2. PIL-APS Team Reward Convergence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Stackelberg privacy-pricing game convergence vs PIL-APS team reward convergence."
    )
    parser.add_argument(
        "--pil_metrics",
        type=str,
        default=str(ROOT / "experiments" / "exp_runs" / "02_compare" / "pil_vs_dpmac" / "pil_aps_metrics.json"),
        help="Path to PIL-APS metrics JSON (single or multi-seed with 'runs' key).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(ROOT / "plots" / "exp_runs" / "00_stackelberg_vs_pil_convergence.png"),
        help="Output path for the convergence plot.",
    )
    return parser


def load_runs(path: Path) -> list[dict]:
    """Load runs from JSON, handling both single run and multi-run formats."""
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "runs" in payload:
        return list(payload["runs"])
    return [payload]


def extract_series(runs: list[dict], extractor: Callable[[dict], float]) -> np.ndarray:
    """Extract metric series from runs."""
    series = [[extractor(entry) for entry in run.get("history", [])] for run in runs]
    return np.asarray(series, dtype=np.float64)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pil_path = Path(args.pil_metrics)
    if not pil_path.exists():
        print(f"Error: PIL metrics file not found at {pil_path}")
        sys.exit(1)

    pil_runs = load_runs(pil_path)
    if not pil_runs:
        print("Error: No PIL runs found")
        sys.exit(1)

    # Extract convergence metrics
    # Team reward: should increase (we want high reward)
    team_reward_series = extract_series(pil_runs, lambda entry: float(entry.get("team_reward", 0.0)))
    
    # Price (Privacy price lambda) from Stackelberg Game
    # This is the mechanism's scalar parameter that agents respond to
    price_series = extract_series(
        pil_runs, lambda entry: float(entry.get("price", 0.0))
    )
    
    # Posterior uncertainty: used to adaptively update price in Stackelberg game
    posterior_uncertainty_series = extract_series(
        pil_runs, lambda entry: float(np.mean(entry.get("posterior_uncertainty", [0.0])))
    )
    
    # Privacy budget (epsilon): cumulative RDP accounting
    epsilon_series = extract_series(
        pil_runs, lambda entry: float(np.mean(entry.get("privacy", {}).get("epsilon", [0.0])))
    )

    # Average across seeds
    team_reward_mean = team_reward_series.mean(axis=0)
    team_reward_std = team_reward_series.std(axis=0)
    price_mean = price_series.mean(axis=0)
    price_std = price_series.std(axis=0)
    posterior_uncertainty_mean = posterior_uncertainty_series.mean(axis=0)
    posterior_uncertainty_std = posterior_uncertainty_series.std(axis=0)
    epsilon_mean = epsilon_series.mean(axis=0)
    epsilon_std = epsilon_series.std(axis=0)

    block_axis = np.arange(len(team_reward_mean))

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Error: matplotlib not available: {e}")
        sys.exit(1)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Team Reward Convergence
    ax = axes[0, 0]
    ax.plot(block_axis, team_reward_mean, label="PIL-APS Team Reward", color="#2E86AB", linewidth=2.5)
    ax.fill_between(block_axis, team_reward_mean - team_reward_std, team_reward_mean + team_reward_std, 
                      alpha=0.2, color="#2E86AB")
    ax.set_xlabel("Block", fontsize=11)
    ax.set_ylabel("Team Reward", fontsize=11)
    ax.set_title("PIL-APS Team Reward Convergence", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Privacy Price (Lambda) Evolution - Stackelberg Game
    ax = axes[0, 1]
    ax.plot(block_axis, price_mean, label="Privacy Price (Stackelberg)", color="#F18F01", linewidth=2.5)
    ax.fill_between(block_axis, price_mean - price_std, price_mean + price_std, 
                      alpha=0.2, color="#F18F01")
    ax.set_xlabel("Block", fontsize=11)
    ax.set_ylabel("Price (λ)", fontsize=11)
    ax.set_title("Stackelberg Privacy-Pricing Game: λ Evolution", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Privacy Budget (Epsilon) Convergence
    ax = axes[1, 0]
    ax.plot(block_axis, epsilon_mean, label="Mean Epsilon (RDP)", color="#4F772D", linewidth=2.5)
    ax.fill_between(block_axis, epsilon_mean - epsilon_std, epsilon_mean + epsilon_std, 
                      alpha=0.2, color="#4F772D")
    ax.set_xlabel("Block", fontsize=11)
    ax.set_ylabel("Mean Epsilon", fontsize=11)
    ax.set_title("Cumulative Privacy Budget (RDP Epsilon)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Dual Convergence - Normalized Metrics
    ax = axes[1, 1]
    # Normalize both to [0, 1] range for comparison
    team_reward_norm = (team_reward_mean - team_reward_mean.min()) / (team_reward_mean.max() - team_reward_mean.min() + 1e-8)
    epsilon_norm = (epsilon_mean - epsilon_mean.min()) / (epsilon_mean.max() - epsilon_mean.min() + 1e-8)
    price_norm = (price_mean - price_mean.min()) / (price_mean.max() - price_mean.min() + 1e-8)
    
    ax.plot(block_axis, team_reward_norm, label="Team Reward (normalized)", color="#2E86AB", linewidth=2.5, marker='o', markersize=3)
    ax.plot(block_axis, epsilon_norm, label="Epsilon Spent (normalized)", color="#4F772D", linewidth=2.5, marker='s', markersize=3)
    ax.plot(block_axis, price_norm, label="Stackelberg Price (normalized)", color="#F18F01", linewidth=2.5, marker='^', markersize=3)
    
    ax.set_xlabel("Block", fontsize=11)
    ax.set_ylabel("Normalized Value [0, 1]", fontsize=11)
    ax.set_title("Dual Convergence: Reward vs Privacy Adaptation", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Convergence plot saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("CONVERGENCE SUMMARY")
    print("="*70)
    print(f"Team Reward (initial → final):  {team_reward_mean[0]:.4f} → {team_reward_mean[-1]:.4f}")
    print(f"Team Reward improvement:        +{team_reward_mean[-1] - team_reward_mean[0]:.4f}")
    print(f"\nMean Epsilon (final):            {epsilon_mean[-1]:.4f}")
    print(f"Privacy Budget Growth:          {epsilon_mean[-1] - epsilon_mean[0]:.4f}")
    print(f"\nStackelberg Price (initial → final): {price_mean[0]:.4f} → {price_mean[-1]:.4f}")
    print(f"Price Change:                   {price_mean[-1] - price_mean[0]:+.4f}")
    print(f"\nPosterior Uncertainty (final):  {posterior_uncertainty_mean[-1]:.4f}")
    print(f"\n#Blocks: {len(block_axis)}, #Seeds: {len(pil_runs)}")
    print("="*70)


if __name__ == "__main__":
    main()
