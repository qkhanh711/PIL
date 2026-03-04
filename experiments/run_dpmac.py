from __future__ import annotations

import json
from pathlib import Path

from PIL.core.dpmac_trainer import DPMACConfig, DPMACTrainer


def main() -> None:
    cfg = DPMACConfig()
    trainer = DPMACTrainer(cfg)
    history = trainer.train(num_iterations=1000, log_every=10)

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "dpmac_metrics.json"
    out_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
