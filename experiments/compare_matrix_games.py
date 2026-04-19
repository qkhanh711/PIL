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

from benchmarks.matrix_games import MatrixGameConfig, MatrixGameRunner


ALGO_COLORS = {
    "pil": "#2E86AB",
    "dpmac": "#F18F01",
    "i2c": "#4F772D",
    "tarmac": "#7B2CBF",
    "maddpg": "#D1495B",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multiple baselines on the matrix-game suite.")
    for field in fields(MatrixGameConfig):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    parser.add_argument("--games", type=str, default="binary_sum,multi_round_sum")
    parser.add_argument("--algorithms", type=str, default="pil,dpmac,i2c,tarmac,maddpg")
    parser.add_argument("--seeds", type=str, default="7")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "experiments" / "matrix_games"))
    parser.add_argument("--summary_output", type=str, default=str(ROOT / "experiments" / "matrix_games_summary.json"))
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(ROOT / "plots" / "exp_runs" / "03_matrix_suite"),
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
    for (game, algorithm), results in results_by_key.items():
        grouped.setdefault(game, {})[algorithm] = results

    for game, algo_results in grouped.items():
        reward_path = plots_dir / f"matrix_{game}_reward_compare.png"
        error_path = plots_dir / f"matrix_{game}_error_compare.png"
        kl_path = plots_dir / f"matrix_{game}_kl_compare.png"

        plt.figure(figsize=(7.5, 4.2))
        for algorithm, results in algo_results.items():
            series = np.asarray(
                [[entry["average_episode_reward"] for entry in result["history"]] for result in results],
                dtype=np.float64,
            ).mean(axis=0)
            x_axis = [entry["block"] + 1 for entry in results[0]["history"]]
            plt.plot(x_axis, series, label=algorithm.upper(), color=ALGO_COLORS.get(algorithm))
        plt.xlabel("Block")
        plt.ylabel("Average Reward")
        plt.title(f"Matrix Game: {game}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(reward_path)
        plt.close()

        plt.figure(figsize=(7.5, 4.2))
        for algorithm, results in algo_results.items():
            series = np.asarray(
                [[entry["prediction_error"] for entry in result["history"]] for result in results],
                dtype=np.float64,
            ).mean(axis=0)
            x_axis = [entry["block"] + 1 for entry in results[0]["history"]]
            plt.plot(x_axis, series, label=algorithm.upper(), color=ALGO_COLORS.get(algorithm))
        plt.xlabel("Block")
        plt.ylabel("Prediction Error")
        plt.title(f"Matrix Game: {game}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(error_path)
        plt.close()

        plt.figure(figsize=(7.5, 4.2))
        for algorithm, results in algo_results.items():
            series = np.asarray(
                [[float(np.mean(entry["kl_distortion"])) for entry in result["history"]] for result in results],
                dtype=np.float64,
            ).mean(axis=0)
            x_axis = [entry["block"] + 1 for entry in results[0]["history"]]
            plt.plot(x_axis, series, label=algorithm.upper(), color=ALGO_COLORS.get(algorithm))
        plt.xlabel("Block")
        plt.ylabel("Mean KL Distortion")
        plt.title(f"Matrix Game: {game}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(kl_path)
        plt.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    games = [item.strip().lower() for item in args.games.split(",") if item.strip()]
    algorithms = [item.strip().lower() for item in args.algorithms.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_key: dict[tuple[str, str], list[dict]] = {}
    total_runs = len(games) * len(algorithms) * len(seeds)
    with tqdm(
        total=total_runs,
        desc="Matrix Suite",
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
                    config.game = game
                    config.algorithm = algorithm
                    config.seed = seed
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

    summary = {"games": {}, "seeds": seeds}
    for (game, algorithm), results in results_by_key.items():
        payload = results[0] if len(results) == 1 else {"runs": results}
        (output_dir / f"{game}_{algorithm}.json").write_text(json.dumps(payload, indent=2))
        final_rewards = [result["final"].get("average_episode_reward", 0.0) for result in results]
        final_errors = [result["final"].get("prediction_error", 0.0) for result in results]
        summary["games"].setdefault(game, {})[algorithm] = {
            "mean_final_reward": float(np.mean(final_rewards)),
            "std_final_reward": float(np.std(final_rewards)),
            "mean_final_error": float(np.mean(final_errors)),
            "std_final_error": float(np.std(final_errors)),
        }

    Path(args.summary_output).write_text(json.dumps(summary, indent=2))
    maybe_plot(results_by_key, Path(args.plots_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
