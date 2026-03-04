from __future__ import annotations

import json
from pathlib import Path

from PIL.core.dpmac_trainer import DPMACConfig, DPMACTrainer
from PIL.core.trainer import PILAPSConfig, PILAPSTrainer


def _last(history: list[dict], key: str) -> float:
    return float(history[-1][key])


def main() -> None:
    n_iters = 300

    pil_cfg = PILAPSConfig()
    dpmac_cfg = DPMACConfig(
        n_agents=pil_cfg.n_agents,
        horizon=pil_cfg.horizon,
        batch_size=pil_cfg.batch_size,
        state_dim=pil_cfg.state_dim,
        obs_dim=pil_cfg.obs_dim,
        type_dim=pil_cfg.type_dim,
        latent_dim=pil_cfg.latent_dim,
        gamma=pil_cfg.gamma,
        device=pil_cfg.device,
        seed=pil_cfg.seed,
    )

    pil = PILAPSTrainer(pil_cfg)
    dpmac = DPMACTrainer(dpmac_cfg)

    print("Running PIL-APS...")
    pil_hist = pil.train(num_iterations=n_iters, log_every=50)
    print("Running DPMAC...")
    dpmac_hist = dpmac.train(num_iterations=n_iters, log_every=50)

    summary = {
        "iterations": n_iters,
        "pil_final": {
            "welfare": _last(pil_hist, "welfare"),
            "epsilon_total": _last(pil_hist, "epsilon_total"),
            "kl": _last(pil_hist, "kl"),
            "ic_violation": _last(pil_hist, "ic_violation"),
            "ir_violation": _last(pil_hist, "ir_violation"),
        },
        "dpmac_final": {
            "welfare": _last(dpmac_hist, "welfare"),
            "epsilon_total": _last(dpmac_hist, "epsilon_total"),
            "kl": _last(dpmac_hist, "kl"),
            "sigma2": _last(dpmac_hist, "sigma2"),
            "dp_condition_satisfied": _last(dpmac_hist, "dp_condition_satisfied"),
        },
        "delta_welfare_pil_minus_dpmac": _last(pil_hist, "welfare") - _last(dpmac_hist, "welfare"),
    }

    out_dir = Path(__file__).resolve().parent
    (out_dir / "pil_aps_metrics_compare.json").write_text(json.dumps(pil_hist, indent=2), encoding="utf-8")
    (out_dir / "dpmac_metrics_compare.json").write_text(json.dumps(dpmac_hist, indent=2), encoding="utf-8")
    summary_path = out_dir / "pil_vs_dpmac_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved comparison summary to: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
