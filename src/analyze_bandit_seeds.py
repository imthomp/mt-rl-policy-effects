#!/usr/bin/env python3
"""Summarize exact-bandit outcomes across reward-training seeds and merge them into the SSOT."""
import json
from pathlib import Path

import numpy as np

from analyze_wmt25_best_of_n import CONDITIONS

ROOT = Path(__file__).resolve().parents[1]
FROZEN = 200
FINAL = 3200


def average_ranks(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    return float(np.corrcoef(average_ranks(left), average_ranks(right))[0, 1])


def main() -> None:
    conditions: dict = {}
    entropy, human, uniform = [], [], None
    below, cells = 0, 0
    files = sorted((ROOT / "results/bandit_seeds").glob("seed_*.json"))
    if len(files) < 2:
        raise ValueError("Reward-seed analysis requires multiple independent seed files")
    for file in files:
        run = json.loads(file.read_text(encoding="utf-8"))["contextual_bandit"]
        if set(run["conditions"]) != set(CONDITIONS):
            raise ValueError(f"Incomplete condition coverage in {file}")
        for condition, value in run["conditions"].items():
            test = {row["step"]: row["test"] for row in value["trajectories"]["exact"]}
            uniform = value["baselines"]["test"]["uniform"]["human_cesa"]
            row = conditions.setdefault(condition, {"frozen": [], "final": []})
            row["frozen"].append(test[FROZEN]["human_cesa"])
            row["final"].append(test[FINAL]["human_cesa"])
            entropy.append(test[FROZEN]["entropy"])
            human.append(test[FROZEN]["human_cesa"])
            cells += 1
            below += test[FINAL]["human_cesa"] < uniform
    summary = {
        "reward_seeds": len(files), "cells": cells, "final_step": FINAL, "uniform": uniform,
        "cells_below_uniform_final": int(below),
        "cells_above_uniform_frozen": int(sum(h > uniform for h in human)),
        "spearman_entropy_human_frozen": spearman(entropy, human),
        "conditions": {
            c: {"frozen_min": min(v["frozen"]), "frozen_max": max(v["frozen"]),
                "final_mean": float(np.mean(v["final"])), "final_sd": float(np.std(v["final"], ddof=1))}
            for c, v in conditions.items()},
    }
    expected_cells = len(files) * len(CONDITIONS)
    if cells != expected_cells or any(len(row["final"]) != len(files) for row in conditions.values()):
        raise AssertionError("Not every paper row traces to every reward-seed result")
    path = ROOT / "results/paper_stats.json"
    root = json.loads(path.read_text(encoding="utf-8"))
    root["bandit_seed_replication"] = summary
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
