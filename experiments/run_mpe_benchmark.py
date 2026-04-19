from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.mpe_suite import MPEBenchmarkConfig, MPEBenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a lightweight MPE benchmark experiment.")
    for field in fields(MPEBenchmarkConfig):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "experiments" / "mpe_benchmark_metrics.json"),
        help="Path to the output JSON file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = MPEBenchmarkConfig.from_namespace(args)
    runner = MPEBenchmarkRunner(config)
    results = runner.run()
    runner.save_results(results, args.output)
    final = results["final"]
    print(f"{results['scenario']} / {results['algorithm']} final eval reward: {final.get('average_episode_reward', 0.0):.4f}")
    privacy = final.get("privacy", {})
    epsilon = privacy.get("epsilon", [])
    leakage = final.get("empirical_leakage", [])
    if epsilon:
        print(f"{results['scenario']} / {results['algorithm']} final mean epsilon: {sum(epsilon) / len(epsilon):.4f}")
    if leakage:
        print(f"{results['scenario']} / {results['algorithm']} final mean leakage: {sum(leakage) / len(leakage):.4f}")


if __name__ == "__main__":
    main()
