from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trainer import PILConfig, PILTrainer


ABLATIONS: dict[str, dict[str, Any]] = {
    "default": {},
    "soft_contract": {
        "contract_uncertainty_coef": 0.32,
        "contract_sigma_coef": 0.12,
        "contract_price_coef": 0.02,
    },
    "kl_regularized": {
        "kl_loss_coef": 0.03,
        "sender_std_coef": 0.04,
        "rho_max": 0.9,
    },
    "balanced": {
        "kl_loss_coef": 0.025,
        "contract_uncertainty_coef": 0.3,
        "contract_sigma_coef": 0.1,
        "contract_price_coef": 0.02,
        "sender_std_coef": 0.035,
        "rho_max": 0.85,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small PIL ablation study over a few config presets.")
    for field in fields(PILConfig):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    parser.add_argument("--seeds", type=str, default="7,11,23", help="Comma-separated seeds to evaluate.")
    parser.add_argument(
        "--variants",
        type=str,
        default="default,soft_contract,kl_regularized,balanced",
        help="Comma-separated variant names.",
    )
    parser.add_argument(
        "--use_ema_last",
        action="store_true",
        help="Summarize EMA-smoothed last checkpoints when available.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "experiments" / "exp_runs" / "05_pil_ablation"),
        help="Directory used for per-variant JSON outputs.",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default=str(ROOT / "experiments" / "exp_runs" / "05_pil_ablation" / "ablation_summary.json"),
        help="Path to the aggregate ablation summary JSON file.",
    )
    return parser


def result_entry(result: dict[str, Any], use_ema_last: bool) -> dict[str, Any]:
    if use_ema_last and result.get("ema_last"):
        return result["ema_last"]
    return result["final"]


def summarize_result(result: dict[str, Any], *, use_ema_last: bool) -> dict[str, float]:
    entry = result_entry(result, use_ema_last)
    epsilon = np.asarray(entry["privacy"]["epsilon"], dtype=np.float64)
    kl = np.asarray(entry["kl_distortion"], dtype=np.float64)
    return {
        "team_reward": float(entry["team_reward"]),
        "oracle_team_reward": float(entry["oracle_team_reward"]),
        "welfare_regret": float(entry["welfare_regret"]),
        "mean_epsilon": float(epsilon.mean()),
        "mean_kl_distortion": float(kl.mean()),
    }


def average_summaries(summaries: list[dict[str, float]]) -> dict[str, float]:
    keys = summaries[0].keys()
    return {key: float(np.mean([summary[key] for summary in summaries])) for key in keys}


def apply_variant(config: PILConfig, variant_name: str) -> PILConfig:
    overrides = ABLATIONS[variant_name]
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    invalid = [variant for variant in variants if variant not in ABLATIONS]
    if invalid:
        raise ValueError(f"Unsupported variants: {invalid}")

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    total_runs = len(variants) * len(seeds)
    with tqdm(
        total=total_runs,
        desc="PIL Ablation",
        unit="run",
        dynamic_ncols=True,
        mininterval=0.1,
        smoothing=0.08,
        colour="#2E86AB",
    ) as progress:
        for variant in variants:
            for seed in seeds:
                config = PILConfig.from_namespace(args)
                config.seed = seed
                config = apply_variant(config, variant)
                trainer = PILTrainer(config)
                result = trainer.run(
                    show_progress=True,
                    run_label=f"{variant} s{seed}",
                    position=1,
                    leave_progress=False,
                )
                results_by_variant[variant].append(result)
                progress.update(1)
                progress.set_postfix(
                    {
                        "variant": variant,
                        "seed": seed,
                        "reward": f"{result_entry(result, args.use_ema_last)['team_reward']:.3f}",
                    },
                    refresh=False,
                )

    summary = {
        "seeds": seeds,
        "checkpoint_source": "ema_last" if args.use_ema_last else "final",
        "variants": {},
    }
    default_summary = None
    for variant in variants:
        payload = results_by_variant[variant][0] if len(results_by_variant[variant]) == 1 else {"runs": results_by_variant[variant]}
        (output_dir / f"{variant}_metrics.json").write_text(json.dumps(payload, indent=2))
        variant_summary = average_summaries(
            [summarize_result(result, use_ema_last=args.use_ema_last) for result in results_by_variant[variant]]
        )
        summary["variants"][variant] = {
            "overrides": ABLATIONS[variant],
            "metrics": variant_summary,
        }
        if variant == "default":
            default_summary = variant_summary

    if default_summary is not None:
        for variant in variants:
            metrics = summary["variants"][variant]["metrics"]
            summary["variants"][variant]["delta_vs_default"] = {
                key: metrics[key] - default_summary[key] for key in default_summary.keys()
            }

    Path(args.summary_output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
