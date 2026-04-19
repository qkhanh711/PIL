from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dpmac_trainer import DPMACTrainer
from core.i2c_trainer import I2CTrainer
from core.maddpg_trainer import MADDPGTrainer
from core.trainer import PILConfig, PILTrainer


TRAINER_REGISTRY = {
    "pil": ("PIL", PILTrainer),
    "dpmac": ("DPMAC", DPMACTrainer),
    "i2c": ("I2C", I2CTrainer),
    "maddpg": ("MADDPG", MADDPGTrainer),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one synthetic communication benchmark baseline.")
    for field in fields(PILConfig):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="pil",
        choices=sorted(TRAINER_REGISTRY.keys()),
        help="Which synthetic baseline to run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "experiments" / "synthetic_metrics.json"),
        help="Path to the output JSON file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = PILConfig.from_namespace(args)
    label, trainer_cls = TRAINER_REGISTRY[args.algorithm.lower()]
    trainer = trainer_cls(config)
    results = trainer.run()
    trainer.save_results(results, args.output)

    final = results["final"]
    print(f"{label} final team reward: {final.get('team_reward', 0.0):.4f}")
    print(f"{label} final welfare regret: {final.get('welfare_regret', 0.0):.4f}")
    privacy = final.get("privacy", {})
    eps = privacy.get("epsilon")
    if eps:
        print(f"{label} final mean epsilon: {sum(eps) / len(eps):.4f}")


if __name__ == "__main__":
    main()
