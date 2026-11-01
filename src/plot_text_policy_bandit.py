"""Plot held-out-system source--candidate text-policy trajectories from the SSOT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot_contextual_bandit import COLORS, LABELS


def render(stats_path: Path, output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    root = json.loads(stats_path.read_text(encoding="utf-8"))
    conditions = root["text_policy_bandit"]["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    step_axis, reward_axis = axes
    for condition, data in conditions.items():
        exact = data["trajectories"]["exact"]
        steps = np.asarray([row["step"] for row in exact])
        human = np.asarray([row["test_heldout_systems"]["human_cesa"] for row in exact])
        proxy = np.asarray(
            [row["test_heldout_systems"]["standardized_proxy"] for row in exact]
        )
        seed_rows = [
            data["trajectories"]["reinforce"][seed]
            for seed in sorted(data["trajectories"]["reinforce"], key=int)
        ]
        seed_human = np.asarray(
            [[row["test_heldout_systems"]["human_cesa"] for row in rows] for rows in seed_rows]
        )
        seed_proxy = np.asarray(
            [[row["test_heldout_systems"]["standardized_proxy"] for row in rows] for rows in seed_rows]
        )
        mean_human = seed_human.mean(0)
        color, label = COLORS[condition], LABELS[condition]
        step_axis.plot(steps, human, color=color, linewidth=2.2, label=f"{label} — exact")
        step_axis.plot(steps, mean_human, color=color, linestyle="--", linewidth=1.4)
        step_axis.fill_between(
            steps,
            seed_human.min(0),
            seed_human.max(0),
            color=color,
            alpha=0.13,
        )
        reward_axis.plot(proxy, human, color=color, linewidth=2.2, label=label)
        for seed_x, seed_y in zip(seed_proxy, seed_human, strict=True):
            reward_axis.plot(seed_x, seed_y, color=color, linewidth=0.8, alpha=0.18)

    uniform = next(iter(conditions.values()))["baselines"]["uniform_human_cesa"]
    ceiling = next(iter(conditions.values()))["baselines"]["human_best_ceiling_cesa"]
    for axis in axes:
        axis.axhline(uniform, color="black", linestyle=":", linewidth=1.4, label="Uniform")
        axis.axhline(ceiling, color="gray", linestyle="-.", linewidth=1.2, label="Human-best ceiling")
        axis.set_ylabel("Expected cESA on unseen systems")
        axis.grid(alpha=0.25)
    step_axis.set_xlabel("Policy update")
    step_axis.set_title("Generalization during optimization")
    reward_axis.set_xlabel("Achieved standardized proxy reward")
    reward_axis.set_title("Human quality versus proxy reward")
    handles, labels = step_axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
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
        default=root / "results/figures/text_policy_bandit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.stats, args.output_stem)


if __name__ == "__main__":
    main()
