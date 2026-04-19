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
from core.trainer import PILConfig, PILTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PIL-APS against the fixed-privacy DPMAC baseline.")
    for field in fields(PILConfig):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    parser.add_argument("--seeds", type=str, default="7", help="Comma-separated seeds to evaluate.")
    parser.add_argument(
        "--use_ema_last",
        action="store_true",
        help="Summarize EMA-smoothed last checkpoints when available.",
    )
    parser.add_argument("--pil_output", type=str, default=str(ROOT / "experiments" / "pil_aps_metrics.json"))
    parser.add_argument("--dpmac_output", type=str, default=str(ROOT / "experiments" / "dpmac_metrics.json"))
    parser.add_argument("--summary_output", type=str, default=str(ROOT / "experiments" / "pil_vs_dpmac_summary.json"))
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(ROOT / "plots" / "exp_runs" / "02_compare" / "pil_vs_dpmac"),
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


def maybe_plot(pil_results: list[dict], dpmac_results: list[dict], plots_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    block_axis = np.arange(len(pil_results[0]["history"]))

    pil_reward = extract_series(pil_results, lambda entry: float(entry["team_reward"])).mean(axis=0)
    dpmac_reward = extract_series(dpmac_results, lambda entry: float(entry["team_reward"])).mean(axis=0)
    plt.figure(figsize=(7, 4))
    plt.plot(block_axis, pil_reward, label="PIL-APS")
    plt.plot(block_axis, dpmac_reward, label="DPMAC")
    plt.xlabel("Block")
    plt.ylabel("Team Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "final_metrics_compare.png")
    plt.close()

    pil_welfare = extract_series(pil_results, lambda entry: float(entry["welfare_regret"])).mean(axis=0)
    dpmac_welfare = extract_series(dpmac_results, lambda entry: float(entry["welfare_regret"])).mean(axis=0)
    plt.figure(figsize=(7, 4))
    plt.plot(block_axis, pil_welfare, label="PIL-APS")
    plt.plot(block_axis, dpmac_welfare, label="DPMAC")
    plt.xlabel("Block")
    plt.ylabel("Welfare Regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "welfare_compare.png")
    plt.close()

    pil_kl = extract_series(pil_results, lambda entry: float(np.mean(entry["kl_distortion"]))).mean(axis=0)
    dpmac_kl = extract_series(dpmac_results, lambda entry: float(np.mean(entry["kl_distortion"]))).mean(axis=0)
    plt.figure(figsize=(7, 4))
    plt.plot(block_axis, pil_kl, label="PIL-APS")
    plt.plot(block_axis, dpmac_kl, label="DPMAC")
    plt.xlabel("Block")
    plt.ylabel("Mean KL Distortion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "kl_compare.png")
    plt.close()


def run_one(
    trainer_cls: type,
    config: PILConfig,
    *,
    show_progress: bool = True,
    run_label: str | None = None,
    position: int = 0,
    leave_progress: bool = True,
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
    pil_results = []
    dpmac_results = []
    with tqdm(
        seeds,
        desc="Compare",
        unit="seed",
        dynamic_ncols=True,
        mininterval=0.1,
        smoothing=0.08,
        colour="#8E7DBE",
        position=0,
    ) as seed_progress:
        for seed in seed_progress:
            config = PILConfig.from_namespace(args)
            config.seed = seed
            pil_result = run_one(
                PILTrainer,
                config,
                show_progress=True,
                run_label=f"PIL s{seed}",
                position=1,
                leave_progress=False,
            )
            dpmac_result = run_one(
                DPMACTrainer,
                config,
                show_progress=True,
                run_label=f"DPMAC s{seed}",
                position=1,
                leave_progress=False,
            )
            pil_results.append(pil_result)
            dpmac_results.append(dpmac_result)
            seed_progress.set_postfix(
                {
                    "seed": seed,
                    "pil": f"{result_entry(pil_result, args.use_ema_last)['team_reward']:.3f}",
                    "dpmac": f"{result_entry(dpmac_result, args.use_ema_last)['team_reward']:.3f}",
                },
                refresh=False,
            )

    pil_payload = pil_results[0] if len(pil_results) == 1 else {"runs": pil_results}
    dpmac_payload = dpmac_results[0] if len(dpmac_results) == 1 else {"runs": dpmac_results}

    Path(args.pil_output).write_text(json.dumps(pil_payload, indent=2))
    Path(args.dpmac_output).write_text(json.dumps(dpmac_payload, indent=2))

    pil_summary = average_summaries([summarize_final(result, use_ema_last=args.use_ema_last) for result in pil_results])
    dpmac_summary = average_summaries(
        [summarize_final(result, use_ema_last=args.use_ema_last) for result in dpmac_results]
    )
    summary = {
        "seeds": seeds,
        "checkpoint_source": "ema_last" if args.use_ema_last else "final",
        "pil_aps": pil_summary,
        "dpmac": dpmac_summary,
        "deltas": {key: pil_summary[key] - dpmac_summary[key] for key in pil_summary.keys()},
    }
    Path(args.summary_output).write_text(json.dumps(summary, indent=2))

    maybe_plot(pil_results, dpmac_results, Path(args.plots_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
