"""Analyze the frozen WMT26 cESA best-of-N external check."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from analyze_wmt25_best_of_n import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    COMPARATORS,
    CONDITIONS,
    PERMUTATIONS,
    PERMUTATION_SEED,
    _mean_ci,
    candidate_order,
)
from wmt26_candidate_pool import EXCLUDED_NON_MACHINE_SYSTEMS, machine_only_rows


POOL_SIZES = (1, 2, 4, 8, 16, "full")


def read_conditions(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    metadata: dict[str, Any] = {}
    conditions: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = root / f"{condition}.jsonl"
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not lines or lines[0].get("condition") != condition:
            raise ValueError(f"Missing or invalid metadata: {path}")
        rows = {row["row_key"]: row for row in lines[1:]}
        if len(rows) != len(lines) - 1:
            raise ValueError(f"Duplicate rows: {path}")
        metadata[condition], conditions[condition] = lines[0], rows
    baseline_keys = set(conditions[CONDITIONS[0]])
    for condition in CONDITIONS[1:]:
        if set(conditions[condition]) != baseline_keys:
            raise ValueError(f"Candidate identity mismatch: {condition}")
    for row_key in baseline_keys:
        baseline = conditions[CONDITIONS[0]][row_key]
        identity = (
            baseline["lp"], baseline["segment_key"], baseline["system_id"], baseline["human_cesa"]
        )
        for condition in CONDITIONS[1:]:
            row = conditions[condition][row_key]
            if (row["lp"], row["segment_key"], row["system_id"], row["human_cesa"]) != identity:
                raise ValueError(f"Candidate content mismatch: {row_key} / {condition}")
    conditions = {condition: machine_only_rows(rows) for condition, rows in conditions.items()}
    return metadata, conditions


def analyze(conditions: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    baseline = conditions[CONDITIONS[0]]
    grouped: dict[str, list[str]] = defaultdict(list)
    for row_key, row in baseline.items():
        grouped[row["segment_key"]].append(row_key)
    candidate_counts = {len(keys) for keys in grouped.values()}
    if len(candidate_counts) != 1:
        raise ValueError(f"Nonrectangular pool: {sorted(candidate_counts)}")
    full_size = next(iter(candidate_counts))
    pool_sizes = [full_size if size == "full" else size for size in POOL_SIZES]
    selected: dict[str, dict[int, list[float]]] = {
        condition: defaultdict(list) for condition in CONDITIONS
    }
    for segment, keys in sorted(grouped.items()):
        base_candidates = [baseline[key] for key in keys]
        for replicate in range(PERMUTATIONS):
            order = [row["system_id"] for row in candidate_order(base_candidates, segment, replicate)]
            for pool_size in pool_sizes:
                available = set(order[:pool_size])
                for condition in CONDITIONS:
                    candidates = [
                        conditions[condition][key]
                        for key in keys
                        if conditions[condition][key]["system_id"] in available
                    ]
                    winner = max(candidates, key=lambda row: (row["predicted_reward"], row["system_id"]))
                    selected[condition][pool_size].append(float(winner["human_cesa"]))
    segment_count = len(grouped)
    per_segment: dict[str, dict[int, np.ndarray]] = {condition: {} for condition in CONDITIONS}
    condition_results: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_results[condition] = {}
        for pool_size in pool_sizes:
            values = np.asarray(selected[condition][pool_size])
            if len(values) != segment_count * PERMUTATIONS:
                raise ValueError("Unexpected WMT26 selection count")
            means = values.reshape(segment_count, PERMUTATIONS).mean(axis=1)
            per_segment[condition][pool_size] = means
            condition_results[condition][str(pool_size)] = _mean_ci(means, pool_size + 2000)
    for condition in CONDITIONS[1:]:
        if not np.allclose(per_segment[condition][pool_sizes[0]], per_segment[CONDITIONS[0]][pool_sizes[0]]):
            raise AssertionError("N=1 outcomes must be identical across reward models")
    contrasts: dict[str, Any] = {}
    for comparator in COMPARATORS:
        name = f"diagnostic_filtered_minus_{comparator}"
        contrasts[name] = {}
        for pool_size in pool_sizes:
            delta = per_segment["diagnostic_filtered"][pool_size] - per_segment[comparator][pool_size]
            contrasts[name][str(pool_size)] = _mean_ci(delta, pool_size + 3000)
        pressure_delta = (
            per_segment["diagnostic_filtered"][full_size]
            - per_segment["diagnostic_filtered"][pool_sizes[0]]
            - per_segment[comparator][full_size]
            + per_segment[comparator][pool_sizes[0]]
        )
        contrasts[name]["full_minus_first_difference_in_change"] = _mean_ci(pressure_delta, 3999)
    correlations = {}
    for condition in CONDITIONS:
        rows = list(conditions[condition].values())
        correlations[condition] = {
            "spearman": float(spearmanr(
                [row["predicted_reward"] for row in rows],
                [row["human_cesa"] for row in rows],
            ).statistic),
            "candidates": len(rows),
        }
    return {
        "status": "complete",
        "endpoint": "zho_Hans---jpn_Jpan cESA",
        "candidate_pool_size": full_size,
        "excluded_non_machine_systems": sorted(EXCLUDED_NON_MACHINE_SYSTEMS),
        "pool_sizes": pool_sizes,
        "permutations": PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "correlations": correlations,
        "conditions": condition_results,
        "contrasts": contrasts,
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing["wmt26_best_of_n"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=root / "artifacts/reward_models/wmt26_predictions")
    parser.add_argument("--output", type=Path, default=root / "results/paper_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata, conditions = read_conditions(args.prediction_root)
    result = analyze(conditions)
    result["prediction_metadata"] = metadata
    merge_ssot(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
