"""Render primary contextual-bandit plots from the canonical SSOT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABELS = {
    "all_data": "All data",
    "diagnostic_filtered": "Disagreement-filtered",
    "global_random_2": "Size-matched random",
    "direction_matched_earlier_year": "Direction-matched earlier year",
}
COLORS = {
    "all_data": "#1f77b4",
    "diagnostic_filtered": "#d62728",
    "global_random_2": "#9467bd",
    "direction_matched_earlier_year": "#2ca02c",
}


def seed_summary(seed_rows: dict[str, list[dict]], split: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = [seed_rows[key] for key in sorted(seed_rows, key=int)]
    values = np.asarray([[row[split][metric] for row in rows] for rows in ordered])
    return values.mean(0), values.std(0, ddof=1), values


def render(stats_path: Path, output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    root = json.loads(stats_path.read_text(encoding="utf-8"))
    result = root["contextual_bandit"]
    conditions = result["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    step_axis, reward_axis = axes
    for condition, data in conditions.items():
        exact = data["trajectories"]["exact"]
        steps = np.asarray([row["step"] for row in exact])
        exact_human = np.asarray([row["test"]["human_cesa"] for row in exact])
        exact_proxy = np.asarray([row["test"]["standardized_proxy"] for row in exact])
        mean_human, std_human, seed_human = seed_summary(
            data["trajectories"]["reinforce"], "test", "human_cesa"
        )
        mean_proxy, _, seed_proxy = seed_summary(
            data["trajectories"]["reinforce"], "test", "standardized_proxy"
        )
        color, label = COLORS[condition], LABELS[condition]
        step_axis.plot(steps, exact_human, color=color, linewidth=2.2, label=f"{label} — exact")
        step_axis.plot(steps, mean_human, color=color, linewidth=1.5, linestyle="--", label=f"{label} — REINFORCE")
        step_axis.fill_between(steps, mean_human - std_human, mean_human + std_human, color=color, alpha=0.13)
        reward_axis.plot(exact_proxy, exact_human, color=color, linewidth=2.2, label=f"{label} — exact")
        for proxy, human in zip(seed_proxy, seed_human, strict=True):
            reward_axis.plot(proxy, human, color=color, linewidth=0.8, alpha=0.18)
        reward_axis.plot(mean_proxy, mean_human, color=color, linewidth=1.5, linestyle="--", label=f"{label} — REINFORCE")
        proxy_best = data["baselines"]["test"]["proxy_best"]
        proxy_best_standardized = (
            proxy_best["raw_proxy"] - data["reward_train_mean"]
        ) / data["reward_train_std"]
        reward_axis.scatter(
            [proxy_best_standardized],
            [proxy_best["human_cesa"]],
            marker="X",
            s=65,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
            label=f"{label} — direct proxy-best",
        )

    first = next(iter(conditions.values()))["baselines"]["test"]
    uniform = first["uniform"]["human_cesa"]
    ceiling = first["human_best_ceiling"]["human_cesa"]
    for axis in axes:
        axis.axhline(uniform, color="black", linestyle=":", linewidth=1.4, label="Uniform human quality")
        axis.axhline(ceiling, color="gray", linestyle="-.", linewidth=1.2, label="Human-best ceiling")
        axis.grid(alpha=0.25)
        axis.set_ylabel("Expected held-out human cESA")
    step_axis.set_xlabel("Policy update")
    step_axis.set_title("Human quality during optimization")
    reward_axis.set_xlabel("Achieved standardized proxy reward")
    reward_axis.set_title("Human quality versus achieved proxy reward")
    handles, labels = step_axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=root / "results/paper_stats.json")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=root / "results" / "figures" / "contextual_bandit_primary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.stats, args.output_stem)


if __name__ == "__main__":
    main()
