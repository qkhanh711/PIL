from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.mpe_suite import MPEBenchmarkConfig, MPEBenchmarkRunner


ALGO_COLORS = {
    "pil": "#2E86AB",
    "dpmac": "#F18F01",
    "i2c": "#4F772D",
    "tarmac": "#7B2CBF",
    "maddpg": "#D1495B",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multiple baselines on the PettingZoo MPE suite.")
    for field in fields(MPEBenchmarkConfig):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    parser.add_argument("--scenarios", type=str, default="cn,ccn,pp")
    parser.add_argument("--algorithms", type=str, default="pil,dpmac,i2c,tarmac,maddpg")
    parser.add_argument("--seeds", type=str, default="7")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "experiments" / "mpe_suite"))
    parser.add_argument("--summary_output", type=str, default=str(ROOT / "experiments" / "mpe_suite_summary.json"))
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(ROOT / "plots" / "exp_runs" / "04_mpe_suite"),
        help="Directory used for generated suite plots.",
    )
    return parser


def maybe_plot(results_by_key: dict[tuple[str, str], list[dict]], plots_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, list[dict]]] = {}
    for (scenario, algorithm), results in results_by_key.items():
        grouped.setdefault(scenario, {})[algorithm] = results

    for scenario, algo_results in grouped.items():
        plt.figure(figsize=(7.5, 4.2))
        for algorithm, results in algo_results.items():
            series = np.asarray(
                [[entry["average_episode_reward"] for entry in result["history"]] for result in results],
                dtype=np.float64,
            ).mean(axis=0)
            x_axis = [entry["episode"] for entry in results[0]["history"]]
            plt.plot(x_axis, series, label=algorithm.upper(), color=ALGO_COLORS.get(algorithm, None))
        plt.xlabel("Episode")
        plt.ylabel("Average Eval Reward")
        plt.title(f"MPE {scenario.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"mpe_{scenario}_reward_compare.png")
        plt.close()

        plt.figure(figsize=(7.5, 4.2))
        for algorithm, results in algo_results.items():
            series = np.asarray(
                [
                    [float(np.mean(entry.get("privacy", {}).get("epsilon", [0.0]))) for entry in result["history"]]
                    for result in results
                ],
                dtype=np.float64,
            ).mean(axis=0)
            x_axis = [entry["episode"] for entry in results[0]["history"]]
            plt.plot(x_axis, series, label=algorithm.upper(), color=ALGO_COLORS.get(algorithm, None))
        plt.xlabel("Episode")
        plt.ylabel("Mean Epsilon")
        plt.title(f"MPE {scenario.upper()} Privacy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"mpe_{scenario}_epsilon_compare.png")
        plt.close()

        plt.figure(figsize=(7.5, 4.2))
        for algorithm, results in algo_results.items():
            series = np.asarray(
                [
                    [float(np.mean(entry.get("empirical_leakage", [0.0]))) for entry in result["history"]]
                    for result in results
                ],
                dtype=np.float64,
            ).mean(axis=0)
            x_axis = [entry["episode"] for entry in results[0]["history"]]
            plt.plot(x_axis, series, label=algorithm.upper(), color=ALGO_COLORS.get(algorithm, None))
        plt.xlabel("Episode")
        plt.ylabel("Empirical Leakage")
        plt.title(f"MPE {scenario.upper()} Leakage")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"mpe_{scenario}_leakage_compare.png")
        plt.close()

        plt.figure(figsize=(7.5, 4.2))
        for algorithm, results in algo_results.items():
            series = np.asarray(
                [
                    [float(np.mean(entry.get("kl_distortion", [0.0]))) for entry in result["history"]]
                    for result in results
                ],
                dtype=np.float64,
            ).mean(axis=0)
            x_axis = [entry["episode"] for entry in results[0]["history"]]
            plt.plot(x_axis, series, label=algorithm.upper(), color=ALGO_COLORS.get(algorithm, None))
        plt.xlabel("Episode")
        plt.ylabel("Mean KL Distortion")
        plt.title(f"MPE {scenario.upper()} Distortion")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"mpe_{scenario}_kl_compare.png")
        plt.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    scenarios = [item.strip().lower() for item in args.scenarios.split(",") if item.strip()]
    algorithms = [item.strip().lower() for item in args.algorithms.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_key: dict[tuple[str, str], list[dict]] = {}
    total_runs = len(scenarios) * len(algorithms) * len(seeds)
    with tqdm(
        total=total_runs,
        desc="MPE Suite",
        unit="run",
        dynamic_ncols=True,
        mininterval=0.1,
        smoothing=0.08,
        colour="#5C7AEA",
    ) as progress:
        for scenario in scenarios:
            for algorithm in algorithms:
                key = (scenario, algorithm)
                results_by_key[key] = []
                for seed in seeds:
                    config = MPEBenchmarkConfig.from_namespace(args)
                    config.scenario = scenario
                    config.algorithm = algorithm
                    config.seed = seed
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

    summary = {"scenarios": {}, "seeds": seeds}
    for (scenario, algorithm), results in results_by_key.items():
        payload = results[0] if len(results) == 1 else {"runs": results}
        (output_dir / f"{scenario}_{algorithm}.json").write_text(json.dumps(payload, indent=2))
        final_rewards = [result["final"].get("average_episode_reward", 0.0) for result in results]
        final_epsilons = [
            float(np.mean(result["final"].get("privacy", {}).get("epsilon", [0.0])))
            for result in results
        ]
        final_leakage = [float(np.mean(result["final"].get("empirical_leakage", [0.0]))) for result in results]
        final_kl = [float(np.mean(result["final"].get("kl_distortion", [0.0]))) for result in results]
        summary["scenarios"].setdefault(scenario, {})[algorithm] = {
            "mean_final_reward": float(np.mean(final_rewards)),
            "std_final_reward": float(np.std(final_rewards)),
            "mean_final_epsilon": float(np.mean(final_epsilons)),
            "std_final_epsilon": float(np.std(final_epsilons)),
            "mean_final_leakage": float(np.mean(final_leakage)),
            "std_final_leakage": float(np.std(final_leakage)),
            "mean_final_kl": float(np.mean(final_kl)),
            "std_final_kl": float(np.std(final_kl)),
        }

    Path(args.summary_output).write_text(json.dumps(summary, indent=2))
    maybe_plot(results_by_key, Path(args.plots_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
