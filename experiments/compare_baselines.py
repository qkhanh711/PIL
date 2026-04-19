from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dpmac_trainer import DPMACTrainer
from core.i2c_trainer import I2CTrainer
from core.maddpg_trainer import MADDPGTrainer
from core.trainer import PILConfig, PILTrainer


TRAINER_REGISTRY = {
    "pil": ("PIL-APS", PILTrainer, "#2E86AB"),
    "dpmac": ("DPMAC", DPMACTrainer, "#F18F01"),
    "i2c": ("I2C", I2CTrainer, "#4F772D"),
    "maddpg": ("MADDPG", MADDPGTrainer, "#D1495B"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PIL-APS against multiple communication baselines.")
    for field in fields(PILConfig):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    parser.add_argument("--seeds", type=str, default="7", help="Comma-separated seeds to evaluate.")
    parser.add_argument(
        "--use_ema_last",
        action="store_true",
        help="Summarize EMA-smoothed last checkpoints when available.",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="pil,dpmac,i2c,maddpg",
        help="Comma-separated list drawn from: pil,dpmac,i2c,maddpg",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "experiments"),
        help="Directory used for per-baseline JSON outputs.",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default=str(ROOT / "experiments" / "baseline_summary.json"),
        help="Path to the aggregate summary JSON file.",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(ROOT / "plots" / "exp_runs" / "02_compare" / "baselines"),
        help="Directory used for generated comparison plots.",
    )
    return parser


def result_entry(result: dict, use_ema_last: bool) -> dict:
    if use_ema_last and result.get("ema_last"):
        return result["ema_last"]
    return result["final"]


def summarize_final(result: dict, *, use_ema_last: bool) -> dict[str, float]:
    final = result_entry(result, use_ema_last)
    epsilon = np.asarray(final["privacy"]["epsilon"], dtype=np.float64)
    epsilon_ic = np.asarray(final["epsilon_ic"], dtype=np.float64)
    epsilon_ir = np.asarray(final["epsilon_ir"], dtype=np.float64)
    kl = np.asarray(final["kl_distortion"], dtype=np.float64)
    return {
        "team_reward": float(final["team_reward"]),
        "oracle_team_reward": float(final["oracle_team_reward"]),
        "welfare_regret": float(final["welfare_regret"]),
        "mean_epsilon": float(epsilon.mean()),
        "mean_epsilon_ic": float(epsilon_ic.mean()),
        "mean_epsilon_ir": float(epsilon_ir.mean()),
        "mean_kl_distortion": float(kl.mean()),
    }


def average_summaries(summaries: list[dict[str, float]]) -> dict[str, float]:
    keys = summaries[0].keys()
    return {key: float(np.mean([summary[key] for summary in summaries])) for key in keys}


def extract_series(results: list[dict], extractor: Callable[[dict], float]) -> np.ndarray:
    return np.asarray([[extractor(entry) for entry in result["history"]] for result in results], dtype=np.float64)


def maybe_plot(results_by_baseline: dict[str, list[dict]], plots_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    first_key = next(iter(results_by_baseline.keys()))
    block_axis = np.arange(len(results_by_baseline[first_key][0]["history"]))

    metric_specs = [
        ("team_reward", "baseline_reward_compare.png", "Team Reward"),
        ("welfare_regret", "baseline_welfare_compare.png", "Welfare Regret"),
        ("kl_distortion", "baseline_kl_compare.png", "Mean KL Distortion"),
    ]

    for metric_key, filename, ylabel in metric_specs:
        plt.figure(figsize=(7.5, 4.2))
        for baseline in results_by_baseline.keys():
            label, _, color = TRAINER_REGISTRY[baseline]
            if metric_key == "kl_distortion":
                series = extract_series(
                    results_by_baseline[baseline],
                    lambda entry: float(np.mean(entry["kl_distortion"])),
                ).mean(axis=0)
            else:
                series = extract_series(
                    results_by_baseline[baseline],
                    lambda entry: float(entry[metric_key]),
                ).mean(axis=0)
            plt.plot(block_axis, series, label=label, color=color)
        plt.xlabel("Block")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / filename)
        plt.close()


def run_one(
    trainer_cls: type,
    config: PILConfig,
    *,
    show_progress: bool,
    run_label: str,
    position: int,
    leave_progress: bool,
) -> dict:
    trainer = trainer_cls(config)
    return trainer.run(
        show_progress=show_progress,
        run_label=run_label,
        position=position,
        leave_progress=leave_progress,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    baselines = [baseline.strip().lower() for baseline in args.baselines.split(",") if baseline.strip()]

    invalid = [baseline for baseline in baselines if baseline not in TRAINER_REGISTRY]
    if invalid:
        raise ValueError(f"Unsupported baselines: {invalid}")

    results_by_baseline: dict[str, list[dict]] = {baseline: [] for baseline in baselines}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(seeds) * len(baselines)
    with tqdm(
        total=total_runs,
        desc="Baseline Compare",
        unit="run",
        dynamic_ncols=True,
        mininterval=0.1,
        smoothing=0.08,
        colour="#8E7DBE",
        position=0,
    ) as progress:
        for seed in seeds:
            config = PILConfig.from_namespace(args)
            config.seed = seed
            for baseline in baselines:
                label, trainer_cls, _ = TRAINER_REGISTRY[baseline]
                result = run_one(
                    trainer_cls,
                    config,
                    show_progress=True,
                    run_label=f"{label} s{seed}",
                    position=1,
                    leave_progress=False,
                )
                results_by_baseline[baseline].append(result)
                progress.update(1)
                progress.set_postfix(
                    {
                        "seed": seed,
                        "baseline": label,
                        "reward": f"{result_entry(result, args.use_ema_last)['team_reward']:.3f}",
                    },
                    refresh=False,
                )

    summary = {"seeds": seeds, "checkpoint_source": "ema_last" if args.use_ema_last else "final", "baselines": {}}
    pil_summary = None
    for baseline in baselines:
        label, _, _ = TRAINER_REGISTRY[baseline]
        payload = results_by_baseline[baseline][0] if len(results_by_baseline[baseline]) == 1 else {"runs": results_by_baseline[baseline]}
        (output_dir / f"{baseline}_metrics.json").write_text(json.dumps(payload, indent=2))
        baseline_summary = average_summaries(
            [summarize_final(result, use_ema_last=args.use_ema_last) for result in results_by_baseline[baseline]]
        )
        summary["baselines"][baseline] = {"label": label, "metrics": baseline_summary}
        if baseline == "pil":
            pil_summary = baseline_summary

    if pil_summary is not None:
        summary["pil_deltas"] = {}
        for baseline in baselines:
            if baseline == "pil":
                continue
            baseline_metrics = summary["baselines"][baseline]["metrics"]
            summary["pil_deltas"][baseline] = {
                key: pil_summary[key] - baseline_metrics[key] for key in pil_summary.keys()
            }

    Path(args.summary_output).write_text(json.dumps(summary, indent=2))
    maybe_plot(results_by_baseline, Path(args.plots_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
