"""Plot frozen best-of-N human outcomes for the reward-overoptimization paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


CONDITIONS = (
    ("all_data", "All data", "#1f77b4"),
    ("diagnostic_filtered", "Disagreement-filtered", "#d62728"),
    ("global_random_2", "Size-matched random", "#9467bd"),
    ("direction_matched_earlier_year", "Direction-matched earlier year", "#2ca02c"),
)


def plot(args: argparse.Namespace) -> None:
    root = json.loads(args.stats.read_text(encoding="utf-8"))
    wmt25 = root["wmt25_best_of_n"]
    wmt26 = root["wmt26_best_of_n"]
    panels = (
        (
            wmt25["pool_sizes"],
            wmt25["selection_results"]["combined"]["conditions"],
            "WMT25 MQM",
            "Selected human MQM score",
        ),
        (wmt26["pool_sizes"], wmt26["conditions"], "WMT26 cESA", "Selected human cESA"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.35))
    for axis, (pool_sizes, results, title, ylabel) in zip(axes, panels, strict=True):
        for condition, label, color in CONDITIONS:
            rows = [results[condition][str(size)] for size in pool_sizes]
            means = [row["mean"] for row in rows]
            lows = [row["ci_low"] for row in rows]
            highs = [row["ci_high"] for row in rows]
            axis.plot(pool_sizes, means, marker="o", linewidth=2, label=label, color=color)
            axis.fill_between(pool_sizes, lows, highs, color=color, alpha=0.10)
        axis.set_title(title)
        axis.set_xlabel("Candidate-pool size N")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.set_xticks(pool_sizes)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=True)
    figure.tight_layout(rect=(0, 0.19, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=root / "results/paper_stats.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "paper/reward_overoptimization_acl/figures/best_of_n.pdf",
    )
    return parser.parse_args()


if __name__ == "__main__":
    plot(parse_args())
