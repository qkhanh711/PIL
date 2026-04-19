from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.matrix_games import MatrixGameConfig, MatrixGameRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a matrix-game communication benchmark.")
    for field in fields(MatrixGameConfig):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "experiments" / "matrix_game_metrics.json"),
        help="Path to the output JSON file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = MatrixGameConfig.from_namespace(args)
    runner = MatrixGameRunner(config)
    results = runner.run()
    runner.save_results(results, args.output)
    final = results["final"]
    print(
        f"{results['game']} / {results['algorithm']} final reward: "
        f"{final.get('average_episode_reward', 0.0):.4f}, "
        f"prediction error: {final.get('prediction_error', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()
