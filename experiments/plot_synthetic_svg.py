from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]


def _load_runs(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "runs" in data:
        return data["runs"]
    if isinstance(data, dict):
        return [data]
    return data


def _safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(mean(vals)) if vals else 0.0


def _variant_history_metric(runs: list[dict], getter) -> list[float]:
    if not runs:
        return []
    lengths = [len(run.get("history", [])) for run in runs]
    t = min(lengths) if lengths else 0
    out: list[float] = []
    for i in range(t):
        out.append(_safe_mean(getter(run["history"][i]) for run in runs))
    return out


def _summary_metrics(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text()).get("variants", {})


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
        "clip_la": "#1b9e77",
        "exact_wf": "#377eb8",
        "dpmac": "#e41a1c",
        "naive_la": "#ff7f00",
        "i2c": "#984ea3",
        "maddpg": "#4daf4a",
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

    legend_x, legend_y = left + plot_w - 160, top + 10
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


def _svg_bar_chart(title: str, metrics: dict, field: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(metrics.keys())
    values = [float(metrics[n].get(field, 0.0)) for n in names]

    width, height = 1000, 600
    left, right, top, bottom = 90, 30, 60, 120
    plot_w = width - left - right
    plot_h = height - top - bottom

    v_min = min(values) if values else 0.0
    v_max = max(values) if values else 1.0
    y_min = min(0.0, v_min)
    y_max = max(0.0, v_max)
    if y_max == y_min:
        y_max = y_min + 1.0

    def sy(v: float) -> float:
        return top + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    bar_w = plot_w / max(1, len(names)) * 0.6
    gap = plot_w / max(1, len(names))

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

    for i, (name, v) in enumerate(zip(names, values)):
        cx = left + i * gap + gap / 2
        x0 = cx - bar_w / 2
        y0 = sy(max(0.0, v))
        y1 = sy(min(0.0, v))
        h = max(1.0, abs(y1 - y0))
        color = "#4C78A8" if v >= 0 else "#E45756"
        parts.append(f'<rect x="{x0:.1f}" y="{min(y0,y1):.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{sy(0)+20:.1f}" transform="rotate(35 {cx:.1f} {sy(0)+20:.1f})" font-size="11" font-family="Arial">{name}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{(min(y0,y1)-6):.1f}" text-anchor="middle" font-size="11" font-family="Arial">{v:.3g}</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def main() -> None:
    input_dir = ROOT / "experiments" / "exp_runs" / "new_experiments" / "synthetic"
    output_dir = ROOT / "plots" / "exp_runs" / "new_experiments" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = ["clip_la", "naive_la", "exact_wf", "dpmac", "i2c", "maddpg"]
    runs_by_variant: dict[str, list[dict]] = {}
    for v in variants:
        p = input_dir / f"{v}.json"
        if p.exists():
            runs_by_variant[v] = _load_runs(p)

    if not runs_by_variant:
        raise FileNotFoundError(f"No variant json found in {input_dir}")

    reward_series = {
        v: _variant_history_metric(runs, lambda h: float(h.get("team_reward", 0.0)))
        for v, runs in runs_by_variant.items()
    }
    regret_series = {
        v: _variant_history_metric(runs, lambda h: float(h.get("welfare_regret", 0.0)))
        for v, runs in runs_by_variant.items()
    }
    eps_series = {
        v: _variant_history_metric(runs, lambda h: _safe_mean(float(x) for x in h.get("privacy", {}).get("epsilon", [0.0])))
        for v, runs in runs_by_variant.items()
    }
    overspend_series = {
        v: _variant_history_metric(
            runs,
            lambda h: max(float(x) for x in h.get("privacy", {}).get("overspend_ratio", [1.0]))
            if h.get("privacy", {}).get("overspend_ratio")
            else 1.0,
        )
        for v, runs in runs_by_variant.items()
    }

    _svg_line_chart("Synthetic Team Reward", "Block", "Team Reward", reward_series, output_dir / "synthetic_reward.svg")
    _svg_line_chart("Synthetic Welfare Regret", "Block", "Welfare Regret", regret_series, output_dir / "synthetic_welfare_regret.svg")
    _svg_line_chart("Synthetic Mean Epsilon", "Block", "Mean Epsilon", eps_series, output_dir / "synthetic_epsilon.svg")
    _svg_line_chart("Synthetic Overspend Ratio", "Block", "Overspend Ratio", overspend_series, output_dir / "synthetic_overspend.svg")

    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        metrics = _summary_metrics(summary_path)
        if metrics:
            _svg_bar_chart("Final Team Reward (mean over seeds)", metrics, "team_reward", output_dir / "bar_team_reward.svg")
            _svg_bar_chart("Final Welfare Regret (mean over seeds)", metrics, "welfare_regret", output_dir / "bar_welfare_regret.svg")
            _svg_bar_chart("Final Overspend Ratio (mean over seeds)", metrics, "max_overspend_ratio", output_dir / "bar_overspend_ratio.svg")

    print(f"Saved plots to: {output_dir}")
    for p in sorted(output_dir.glob("*.svg")):
        print("-", p.name)


if __name__ == "__main__":
    main()
