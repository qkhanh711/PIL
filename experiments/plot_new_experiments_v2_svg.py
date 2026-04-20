from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "experiments" / "exp_runs" / "new_experiments_v2"
OUTPUT_ROOT = ROOT / "plots" / "exp_runs" / "new_experiments_v2"


def _safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(mean(vals)) if vals else 0.0


def _load_runs(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "runs" in data:
        return data["runs"]
    if isinstance(data, dict):
        return [data]
    return data


def _svg_line_chart(title: str, x_label: str, y_label: str, series: Dict[str, List[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1000, 600
    left, right, top, bottom = 90, 30, 60, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_values = [v for arr in series.values() for v in arr]
    if not all_values:
        return
    y_min = min(all_values)
    y_max = max(all_values)
    if y_max == y_min:
        y_max = y_min + 1.0

    max_len = max(len(arr) for arr in series.values())
    x_max = max(1, max_len - 1)

    def sx(i: int) -> float:
        return left + (i / x_max) * plot_w if x_max > 0 else left

    def sy(v: float) -> float:
        return top + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    colors = {
        "pil": "#1b9e77",
        "dpmac": "#e41a1c",
        "i2c": "#984ea3",
        "tarmac": "#377eb8",
        "maddpg": "#4daf4a",
        "clip_la": "#ff7f00",
        "naive_la": "#a65628",
        "exact_wf": "#f781bf",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
    ]

    for j in range(6):
        yv = y_min + (y_max - y_min) * j / 5
        yy = sy(yv)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{yv:.3g}</text>')

    for i in range(max_len):
        xx = sx(i)
        if i % max(1, max_len // 10) == 0 or i == max_len - 1:
            parts.append(f'<text x="{xx:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-size="12" font-family="Arial">{i+1}</text>')

    parts.append(f'<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle" font-size="14" font-family="Arial">{x_label}</text>')
    parts.append(f'<text x="25" y="{top+plot_h/2}" transform="rotate(-90 25 {top+plot_h/2})" text-anchor="middle" font-size="14" font-family="Arial">{y_label}</text>')

    legend_x, legend_y = left + plot_w - 170, top + 10
    k = 0
    for name, arr in series.items():
        if not arr:
            continue
        color = colors.get(name, "#333")
        pts = " ".join(f"{sx(i):.2f},{sy(v):.2f}" for i, v in enumerate(arr))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
        parts.append(f'<rect x="{legend_x}" y="{legend_y + 20*k}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{legend_y + 10 + 20*k}" font-size="12" font-family="Arial">{name}</text>')
        k += 1

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def _svg_grouped_bar_chart(title: str, categories: list[str], series_names: list[str], values: Dict[str, Dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1200, 650
    left, right, top, bottom = 100, 30, 60, 140
    plot_w = width - left - right
    plot_h = height - top - bottom

    flat_vals = [values.get(cat, {}).get(s, 0.0) for cat in categories for s in series_names]
    y_min = min(0.0, min(flat_vals) if flat_vals else 0.0)
    y_max = max(0.0, max(flat_vals) if flat_vals else 1.0)
    if y_max == y_min:
        y_max = y_min + 1.0

    def sy(v: float) -> float:
        return top + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    group_w = plot_w / max(1, len(categories))
    bar_w = group_w / max(1, len(series_names)) * 0.75

    colors = {
        "pil": "#1b9e77",
        "dpmac": "#e41a1c",
        "i2c": "#984ea3",
        "tarmac": "#377eb8",
        "maddpg": "#4daf4a",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>',
        f'<line x1="{left}" y1="{sy(0):.1f}" x2="{left+plot_w}" y2="{sy(0):.1f}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
    ]

    for j in range(6):
        yv = y_min + (y_max - y_min) * j / 5
        yy = sy(yv)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{yv:.3g}</text>')

    for ci, cat in enumerate(categories):
        gx = left + ci * group_w
        for si, s in enumerate(series_names):
            v = float(values.get(cat, {}).get(s, 0.0))
            x0 = gx + si * (group_w / max(1, len(series_names))) + (group_w / max(1, len(series_names)) - bar_w) / 2
            y0 = sy(max(v, 0.0))
            y1 = sy(min(v, 0.0))
            h = max(1.0, abs(y1 - y0))
            color = colors.get(s, "#666")
            parts.append(f'<rect x="{x0:.1f}" y="{min(y0,y1):.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{gx + group_w/2:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-size="12" font-family="Arial">{cat}</text>')

    lx, ly = left, height - 70
    for i, s in enumerate(series_names):
        color = colors.get(s, "#666")
        parts.append(f'<rect x="{lx + 130*i}" y="{ly}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx + 130*i + 18}" y="{ly+10}" font-size="12" font-family="Arial">{s}</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def _collect_matrix_runs(matrix_dir: Path) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for path in sorted(matrix_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        stem = path.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        algorithm = parts[-1]
        game = "_".join(parts[:-1])
        grouped.setdefault(game, {})[algorithm] = _load_runs(path)
    return grouped


def _history_metric(runs: list[dict], getter) -> list[float]:
    if not runs:
        return []
    lengths = [len(r.get("history", [])) for r in runs]
    t = min(lengths) if lengths else 0
    out = []
    for i in range(t):
        out.append(_safe_mean(getter(r["history"][i]) for r in runs))
    return out


def plot_matrix() -> list[Path]:
    produced: list[Path] = []
    matrix_dir = INPUT_ROOT / "matrix"
    out_dir = OUTPUT_ROOT / "matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not matrix_dir.exists():
        return produced

    grouped = _collect_matrix_runs(matrix_dir)
    for game, algo_runs in grouped.items():
        reward_series = {algo: _history_metric(runs, lambda h: float(h.get("average_episode_reward", 0.0))) for algo, runs in algo_runs.items()}
        err_series = {algo: _history_metric(runs, lambda h: float(h.get("prediction_error", 0.0))) for algo, runs in algo_runs.items()}
        kl_series = {
            algo: _history_metric(runs, lambda h: _safe_mean(float(x) for x in h.get("kl_distortion", [0.0])))
            for algo, runs in algo_runs.items()
        }
        eps_series = {
            algo: _history_metric(runs, lambda h: _safe_mean(float(x) for x in h.get("privacy", {}).get("epsilon", [0.0])))
            for algo, runs in algo_runs.items()
        }

        p1 = out_dir / f"{game}_reward.svg"
        p2 = out_dir / f"{game}_prediction_error.svg"
        p3 = out_dir / f"{game}_kl.svg"
        p4 = out_dir / f"{game}_epsilon.svg"
        _svg_line_chart(f"Matrix {game} - Reward", "Block", "Average Episode Reward", reward_series, p1)
        _svg_line_chart(f"Matrix {game} - Prediction Error", "Block", "Prediction Error", err_series, p2)
        _svg_line_chart(f"Matrix {game} - KL Distortion", "Block", "Mean KL Distortion", kl_series, p3)
        _svg_line_chart(f"Matrix {game} - Mean Epsilon", "Block", "Mean Epsilon", eps_series, p4)
        produced.extend([p1, p2, p3, p4])

    summary = matrix_dir / "summary.json"
    if summary.exists():
        data = json.loads(summary.read_text()).get("games", {})
        game_names = sorted(data.keys())
        if game_names:
            algos = sorted({a for g in game_names for a in data[g].keys()})
            reward_vals = {g: {a: float(data[g].get(a, {}).get("average_episode_reward", 0.0)) for a in algos} for g in game_names}
            err_vals = {g: {a: float(data[g].get(a, {}).get("prediction_error", 0.0)) for a in algos} for g in game_names}
            p5 = out_dir / "summary_reward_grouped.svg"
            p6 = out_dir / "summary_prediction_error_grouped.svg"
            _svg_grouped_bar_chart("Matrix Summary - Average Episode Reward", game_names, algos, reward_vals, p5)
            _svg_grouped_bar_chart("Matrix Summary - Prediction Error", game_names, algos, err_vals, p6)
            produced.extend([p5, p6])

    return [p for p in produced if p.exists()]


def plot_mpe() -> list[Path]:
    produced: list[Path] = []
    mpe_dir = INPUT_ROOT / "mpe"
    out_dir = OUTPUT_ROOT / "mpe"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not mpe_dir.exists():
        return produced

    files = [p for p in mpe_dir.glob("*.json") if p.name != "summary.json"]
    if not files:
        return produced

    grouped: dict[str, dict[str, list[dict]]] = {}
    for path in files:
        stem = path.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        algo = parts[-1]
        scenario = "_".join(parts[:-1])
        grouped.setdefault(scenario, {})[algo] = _load_runs(path)

    for scenario, algo_runs in grouped.items():
        reward_series = {algo: _history_metric(runs, lambda h: float(h.get("average_episode_reward", 0.0))) for algo, runs in algo_runs.items()}
        leakage_series = {
            algo: _history_metric(runs, lambda h: _safe_mean(float(x) for x in h.get("empirical_leakage", [0.0])))
            for algo, runs in algo_runs.items()
        }
        eps_series = {
            algo: _history_metric(runs, lambda h: _safe_mean(float(x) for x in h.get("privacy", {}).get("epsilon", [0.0])))
            for algo, runs in algo_runs.items()
        }

        p1 = out_dir / f"{scenario}_reward.svg"
        p2 = out_dir / f"{scenario}_leakage.svg"
        p3 = out_dir / f"{scenario}_epsilon.svg"
        _svg_line_chart(f"MPE {scenario} - Reward", "Episode", "Average Eval Reward", reward_series, p1)
        _svg_line_chart(f"MPE {scenario} - Leakage", "Episode", "Empirical Leakage", leakage_series, p2)
        _svg_line_chart(f"MPE {scenario} - Mean Epsilon", "Episode", "Mean Epsilon", eps_series, p3)
        produced.extend([p1, p2, p3])

    return [p for p in produced if p.exists()]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    matrix_plots = plot_matrix()
    mpe_plots = plot_mpe()

    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Matrix plots: {len(matrix_plots)}")
    for p in matrix_plots:
        print(f"- {p.relative_to(ROOT)}")
    print(f"MPE plots: {len(mpe_plots)}")
    for p in mpe_plots:
        print(f"- {p.relative_to(ROOT)}")
    if not mpe_plots:
        print("(No MPE json found yet; skipped MPE plotting.)")


if __name__ == "__main__":
    main()
