from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dpmac_trainer import DPMACTrainer
from core.trainer import PILConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed-privacy DPMAC-style baseline.")
    for field in fields(PILConfig):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "experiments" / "dpmac_metrics.json"),
        help="Path to the output JSON file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = PILConfig.from_namespace(args)
    trainer = DPMACTrainer(config)
    results = trainer.run()
    trainer.save_results(results, args.output)
    final = results["final"]
    print(f"DPMAC final team reward: {final['team_reward']:.4f}")
    print(f"DPMAC final welfare regret: {final['welfare_regret']:.4f}")
    print(f"DPMAC final mean epsilon: {sum(final['privacy']['epsilon']) / len(final['privacy']['epsilon']):.4f}")


if __name__ == "__main__":
    main()
