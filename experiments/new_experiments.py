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

from benchmarks.matrix_games import MatrixGameConfig, MatrixGameRunner
from benchmarks.mpe_suite import MPEBenchmarkConfig, MPEBenchmarkRunner
from core.dpmac_trainer import DPMACTrainer
from core.i2c_trainer import I2CTrainer
from core.maddpg_trainer import MADDPGTrainer
from core.trainer import PILConfig, PILTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a paper-aligned experiment bundle for new_PIL.tex using the available repo benchmarks."
    )
    parser.add_argument(
        "--sections",
        type=str,
        default="synthetic,matrix,mpe",
        help="Comma-separated sections to run: synthetic,matrix,mpe",
    )
    parser.add_argument("--seeds", type=str, default="7,11,23", help="Comma-separated seeds shared by all sections.")
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT / "experiments" / "exp_runs" / "new_experiments"),
        help="Directory used for outputs from this bundle.",
    )
    parser.add_argument(
        "--plots_root",
        type=str,
        default=str(ROOT / "plots" / "exp_runs" / "new_experiments"),
        help="Directory used for plots from this bundle.",
    )

    added_fields: set[str] = set()
    for config_cls in (PILConfig, MatrixGameConfig, MPEBenchmarkConfig):
        for field in fields(config_cls):
            if field.name in added_fields:
                continue
            parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
            added_fields.add(field.name)

    parser.add_argument("--matrix_games", type=str, default="binary_sum,multi_round_sum")
    parser.add_argument("--matrix_algorithms", type=str, default="pil,dpmac,i2c,tarmac,maddpg")
    parser.add_argument("--mpe_scenarios", type=str, default="cn,ccn,pp")
    parser.add_argument("--mpe_algorithms", type=str, default="pil,dpmac,i2c,tarmac,maddpg")
    return parser


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def result_payload(results: list[dict[str, Any]]) -> dict[str, Any]:
    return results[0] if len(results) == 1 else {"runs": results}


def mean_list(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize_trainer_result(result: dict[str, Any]) -> dict[str, float]:
    final = result["final"]
    privacy = final.get("privacy", {})
    naive = privacy.get("naive_unclipped_counterexample", {})
    return {
        "team_reward": float(final.get("team_reward", 0.0)),
        "oracle_team_reward": float(final.get("oracle_team_reward", 0.0)),
        "welfare_regret": float(final.get("welfare_regret", 0.0)),
        "mean_epsilon": mean_list([float(x) for x in privacy.get("epsilon", [])]),
        "mean_kl_distortion": mean_list([float(x) for x in final.get("kl_distortion", [])]),
        "mean_posterior_error": mean_list([float(x) for x in final.get("posterior_error", [])]),
        "mean_clip_rate": mean_list([float(x) for x in final.get("clip_rate", [])]),
        "max_overspend_ratio": max([float(x) for x in naive.get("overspend_ratio", [1.0])] or [1.0]),
    }


def average_summaries(summaries: list[dict[str, float]]) -> dict[str, float]:
    keys = summaries[0].keys()
    return {key: float(np.mean([summary[key] for summary in summaries])) for key in keys}


def _maybe_import_plt():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _save_line_plot(
    out_path: Path,
    x_values: list[float] | np.ndarray,
    series_by_label: dict[str, np.ndarray],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    plt = _maybe_import_plt()
    if plt is None or not series_by_label:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.2))
    for label, series in series_by_label.items():
        plt.plot(x_values, series, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def maybe_plot_synthetic(results_by_variant: dict[str, list[dict[str, Any]]], plots_dir: Path) -> None:
    if not results_by_variant:
        return
    first_key = next(iter(results_by_variant))
    if not results_by_variant[first_key]:
        return
    history = results_by_variant[first_key][0].get("history", [])
    if not history:
        return
    x_axis = [entry["block"] + 1 for entry in history]

    reward_series = {
        variant: np.asarray([[entry["team_reward"] for entry in result["history"]] for result in results], dtype=np.float64).mean(axis=0)
        for variant, results in results_by_variant.items()
    }
    _save_line_plot(plots_dir / "synthetic_reward.png", x_axis, reward_series, xlabel="Block", ylabel="Team Reward", title="Synthetic Reward")

    regret_series = {
        variant: np.asarray([[entry["welfare_regret"] for entry in result["history"]] for result in results], dtype=np.float64).mean(axis=0)
        for variant, results in results_by_variant.items()
    }
    _save_line_plot(plots_dir / "synthetic_welfare_regret.png", x_axis, regret_series, xlabel="Block", ylabel="Welfare Regret", title="Synthetic Welfare Regret")

    epsilon_series = {
        variant: np.asarray(
            [[float(np.mean(entry.get("privacy", {}).get("epsilon", [0.0]))) for entry in result["history"]] for result in results],
            dtype=np.float64,
        ).mean(axis=0)
        for variant, results in results_by_variant.items()
    }
    _save_line_plot(plots_dir / "synthetic_epsilon.png", x_axis, epsilon_series, xlabel="Block", ylabel="Mean Epsilon", title="Synthetic Privacy")

    overspend_series: dict[str, np.ndarray] = {}
    for variant, results in results_by_variant.items():
        values = []
        for result in results:
            seq = []
            for entry in result["history"]:
                naive = entry.get("privacy", {}).get("naive_unclipped_counterexample", {})
                seq.append(float(max(naive.get("overspend_ratio", [1.0]) or [1.0])))
            values.append(seq)
        if values and any(any(v != 1.0 for v in seq) for seq in values):
            overspend_series[variant] = np.asarray(values, dtype=np.float64).mean(axis=0)
    _save_line_plot(plots_dir / "synthetic_overspend.png", x_axis, overspend_series, xlabel="Block", ylabel="Overspend Ratio", title="Synthetic DP Audit")


def maybe_plot_matrix(results_by_key: dict[tuple[str, str], list[dict[str, Any]]], plots_dir: Path) -> None:
    if not results_by_key:
        return
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for (game, algorithm), results in results_by_key.items():
        grouped.setdefault(game, {})[algorithm] = results
    for game, algo_results in grouped.items():
        first_algo = next(iter(algo_results))
        history = algo_results[first_algo][0].get("history", [])
        if not history:
            continue
        x_axis = [entry["block"] + 1 for entry in history]
        reward_series = {
            algo: np.asarray([[entry["average_episode_reward"] for entry in result["history"]] for result in results], dtype=np.float64).mean(axis=0)
            for algo, results in algo_results.items()
        }
        _save_line_plot(plots_dir / f"{game}_reward.png", x_axis, reward_series, xlabel="Block", ylabel="Average Reward", title=f"Matrix {game} Reward")

        error_series = {
            algo: np.asarray([[entry["prediction_error"] for entry in result["history"]] for result in results], dtype=np.float64).mean(axis=0)
            for algo, results in algo_results.items()
        }
        _save_line_plot(plots_dir / f"{game}_prediction_error.png", x_axis, error_series, xlabel="Block", ylabel="Prediction Error", title=f"Matrix {game} Prediction Error")

        kl_series = {
            algo: np.asarray([[float(np.mean(entry.get("kl_distortion", [0.0]))) for entry in result["history"]] for result in results], dtype=np.float64).mean(axis=0)
            for algo, results in algo_results.items()
        }
        _save_line_plot(plots_dir / f"{game}_kl.png", x_axis, kl_series, xlabel="Block", ylabel="Mean KL Distortion", title=f"Matrix {game} KL Distortion")


def maybe_plot_mpe(results_by_key: dict[tuple[str, str], list[dict[str, Any]]], plots_dir: Path) -> None:
    if not results_by_key:
        return
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for (scenario, algorithm), results in results_by_key.items():
        grouped.setdefault(scenario, {})[algorithm] = results
    for scenario, algo_results in grouped.items():
        first_algo = next(iter(algo_results))
        history = algo_results[first_algo][0].get("history", [])
        if not history:
            continue
        x_axis = [entry["episode"] for entry in history]
        reward_series = {
            algo: np.asarray([[entry["average_episode_reward"] for entry in result["history"]] for result in results], dtype=np.float64).mean(axis=0)
            for algo, results in algo_results.items()
        }
        _save_line_plot(plots_dir / f"{scenario}_reward.png", x_axis, reward_series, xlabel="Episode", ylabel="Average Eval Reward", title=f"MPE {scenario.upper()} Reward")

        leakage_series = {
            algo: np.asarray([[float(np.mean(entry.get("empirical_leakage", [0.0]))) for entry in result["history"]] for result in results], dtype=np.float64).mean(axis=0)
            for algo, results in algo_results.items()
        }
        _save_line_plot(plots_dir / f"{scenario}_leakage.png", x_axis, leakage_series, xlabel="Episode", ylabel="Empirical Leakage", title=f"MPE {scenario.upper()} Leakage")

        epsilon_series = {
            algo: np.asarray([[float(np.mean(entry.get("privacy", {}).get("epsilon", [0.0]))) for entry in result["history"]] for result in results], dtype=np.float64).mean(axis=0)
            for algo, results in algo_results.items()
        }
        _save_line_plot(plots_dir / f"{scenario}_epsilon.png", x_axis, epsilon_series, xlabel="Episode", ylabel="Mean Epsilon", title=f"MPE {scenario.upper()} Privacy")


def run_synthetic_section(args: argparse.Namespace, seeds: list[int], output_root: Path, plots_root: Path) -> dict[str, Any]:
    section_dir = output_root / "synthetic"
    section_dir.mkdir(parents=True, exist_ok=True)

    variants: dict[str, tuple[type, dict[str, Any]]] = {
        "clip_la": (PILTrainer, {"scheduler_mode": "clip_la", "scheduler_variant": "heuristic"}),
        "naive_la": (PILTrainer, {"scheduler_mode": "naive_la", "scheduler_variant": "heuristic"}),
        "exact_wf": (PILTrainer, {"scheduler_mode": "clip_la", "scheduler_variant": "exact_wf"}),
        "dpmac": (DPMACTrainer, {}),
        "i2c": (I2CTrainer, {}),
        "maddpg": (MADDPGTrainer, {}),
    }

    results_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
    total_runs = len(variants) * len(seeds)
    with tqdm(
        total=total_runs,
        desc="New Synthetic",
        unit="run",
        dynamic_ncols=True,
        mininterval=0.1,
        smoothing=0.08,
        colour="#2E86AB",
    ) as progress:
        for variant_name, (trainer_cls, overrides) in variants.items():
            for seed in seeds:
                config = PILConfig.from_namespace(args)
                config.seed = seed
                for key, value in overrides.items():
                    setattr(config, key, value)
                trainer = trainer_cls(config)
                result = trainer.run(
                    show_progress=True,
                    run_label=f"{variant_name} s{seed}",
                    position=1,
                    leave_progress=False,
                )
                results_by_variant[variant_name].append(result)
                progress.update(1)
                progress.set_postfix(
                    {
                        "variant": variant_name,
                        "seed": seed,
                        "reward": f"{result['final'].get('team_reward', 0.0):.3f}",
                    },
                    refresh=False,
                )

    summary = {"seeds": seeds, "variants": {}}
    for variant_name, results in results_by_variant.items():
        (section_dir / f"{variant_name}.json").write_text(json.dumps(result_payload(results), indent=2))
        summary["variants"][variant_name] = average_summaries([summarize_trainer_result(result) for result in results])

    clip_la = summary["variants"].get("clip_la")
    if clip_la is not None:
        summary["clip_la_deltas"] = {}
        for variant_name, metrics in summary["variants"].items():
            if variant_name == "clip_la":
                continue
            summary["clip_la_deltas"][variant_name] = {
                key: clip_la[key] - metrics[key] for key in clip_la.keys()
            }

    summary_path = section_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    maybe_plot_synthetic(results_by_variant, plots_root / "synthetic")
    return summary


def summarize_matrix_result(result: dict[str, Any]) -> dict[str, float]:
    final = result["final"]
    return {
        "average_episode_reward": float(final.get("average_episode_reward", 0.0)),
        "prediction_error": float(final.get("prediction_error", 0.0)),
        "mean_epsilon": mean_list([float(x) for x in final.get("privacy", {}).get("epsilon", [])]),
        "mean_kl_distortion": mean_list([float(x) for x in final.get("kl_distortion", [])]),
    }


def run_matrix_section(args: argparse.Namespace, seeds: list[int], output_root: Path, plots_root: Path) -> dict[str, Any]:
    section_dir = output_root / "matrix"
    section_dir.mkdir(parents=True, exist_ok=True)

    games = parse_csv(args.matrix_games)
    algorithms = parse_csv(args.matrix_algorithms)
    results_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}

    total_runs = len(games) * len(algorithms) * len(seeds)
    with tqdm(
        total=total_runs,
        desc="New Matrix",
        unit="run",
        dynamic_ncols=True,
        mininterval=0.1,
        smoothing=0.08,
        colour="#5C7AEA",
    ) as progress:
        for game in games:
            for algorithm in algorithms:
                key = (game, algorithm)
                results_by_key[key] = []
                for seed in seeds:
                    config = MatrixGameConfig.from_namespace(args)
                    config.seed = seed
                    config.game = game
                    config.algorithm = algorithm
                    runner = MatrixGameRunner(config)
                    result = runner.run(show_progress=True, position=1, leave_progress=False)
                    results_by_key[key].append(result)
                    progress.update(1)
                    progress.set_postfix(
                        {
                            "game": game,
                            "algo": algorithm,
                            "seed": seed,
                            "reward": f"{result['final'].get('average_episode_reward', 0.0):.3f}",
                        },
                        refresh=False,
                    )

    summary = {"seeds": seeds, "games": {}}
    for (game, algorithm), results in results_by_key.items():
        (section_dir / f"{game}_{algorithm}.json").write_text(json.dumps(result_payload(results), indent=2))
        summary["games"].setdefault(game, {})[algorithm] = average_summaries(
            [summarize_matrix_result(result) for result in results]
        )

    summary_path = section_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    maybe_plot_matrix(results_by_key, plots_root / "matrix")
    return summary


def summarize_mpe_result(result: dict[str, Any]) -> dict[str, float]:
    final = result["final"]
    return {
        "average_episode_reward": float(final.get("average_episode_reward", 0.0)),
        "mean_epsilon": mean_list([float(x) for x in final.get("privacy", {}).get("epsilon", [])]),
        "mean_empirical_leakage": mean_list([float(x) for x in final.get("empirical_leakage", [])]),
        "mean_kl_distortion": mean_list([float(x) for x in final.get("kl_distortion", [])]),
    }


def run_mpe_section(args: argparse.Namespace, seeds: list[int], output_root: Path, plots_root: Path) -> dict[str, Any]:
    section_dir = output_root / "mpe"
    section_dir.mkdir(parents=True, exist_ok=True)

    scenarios = parse_csv(args.mpe_scenarios)
    algorithms = parse_csv(args.mpe_algorithms)
    results_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}

    total_runs = len(scenarios) * len(algorithms) * len(seeds)
    with tqdm(
        total=total_runs,
        desc="New MPE",
        unit="run",
        dynamic_ncols=True,
        mininterval=0.1,
        smoothing=0.08,
        colour="#7B2CBF",
    ) as progress:
        for scenario in scenarios:
            for algorithm in algorithms:
                key = (scenario, algorithm)
                results_by_key[key] = []
                for seed in seeds:
                    config = MPEBenchmarkConfig.from_namespace(args)
                    config.seed = seed
                    config.scenario = scenario
                    config.algorithm = algorithm
                    runner = MPEBenchmarkRunner(config)
                    result = runner.run(show_progress=True, position=1, leave_progress=False)
                    results_by_key[key].append(result)
                    progress.update(1)
                    progress.set_postfix(
                        {
                            "scenario": scenario,
                            "algo": algorithm,
                            "seed": seed,
                            "reward": f"{result['final'].get('average_episode_reward', 0.0):.2f}",
                        },
                        refresh=False,
                    )

    summary = {"seeds": seeds, "scenarios": {}}
    for (scenario, algorithm), results in results_by_key.items():
        (section_dir / f"{scenario}_{algorithm}.json").write_text(json.dumps(result_payload(results), indent=2))
        summary["scenarios"].setdefault(scenario, {})[algorithm] = average_summaries(
            [summarize_mpe_result(result) for result in results]
        )

    summary_path = section_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    maybe_plot_mpe(results_by_key, plots_root / "mpe")
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sections = parse_csv(args.sections)
    seeds = parse_seeds(args.seeds)
    output_root = Path(args.output_root)
    plots_root = Path(args.plots_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"sections": sections, "seeds": seeds}
    if "synthetic" in sections:
        summary["synthetic"] = run_synthetic_section(args, seeds, output_root, plots_root)
    if "matrix" in sections:
        summary["matrix"] = run_matrix_section(args, seeds, output_root, plots_root)
    if "mpe" in sections:
        summary["mpe"] = run_mpe_section(args, seeds, output_root, plots_root)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
