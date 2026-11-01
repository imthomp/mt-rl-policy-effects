"""Analyze frozen WMT25 best-of-N reward selection and merge results into the SSOT."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


CONDITIONS = (
    "all_data",
    "diagnostic_filtered",
    "global_random_2",
    "direction_matched_earlier_year",
)
COMPARATORS = tuple(condition for condition in CONDITIONS if condition != "diagnostic_filtered")
POOL_SIZES = (1, 2, 4, 8, 16, "full")
PERMUTATIONS = 100
PERMUTATION_SEED = 20261103
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261104


def _read_predictions(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    if not lines or lines[0].get("record_type") != "metadata":
        raise ValueError(f"Missing metadata row: {path}")
    rows = {row["row_key"]: row for row in lines[1:]}
    if len(rows) != len(lines) - 1 or any(
        row.get("record_type") != "candidate" for row in rows.values()
    ):
        raise ValueError(f"Duplicate or invalid candidate rows: {path}")
    return lines[0], rows


def load_conditions(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    metadata: dict[str, dict[str, Any]] = {}
    by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        meta, rows = _read_predictions(root / f"{condition}.jsonl")
        if meta.get("condition") != condition:
            raise ValueError(f"Condition mismatch for {condition}")
        metadata[condition], by_condition[condition] = meta, rows
    baseline_keys = set(by_condition[CONDITIONS[0]])
    for condition in CONDITIONS[1:]:
        if set(by_condition[condition]) != baseline_keys:
            raise ValueError(f"Candidate identity mismatch: {condition}")
    for key in baseline_keys:
        baseline = by_condition[CONDITIONS[0]][key]
        identity = (baseline["lp"], baseline["segment_key"], baseline["system_id"], baseline["human_mqm"])
        for condition in CONDITIONS[1:]:
            row = by_condition[condition][key]
            if (row["lp"], row["segment_key"], row["system_id"], row["human_mqm"]) != identity:
                raise ValueError(f"Candidate content mismatch for {key}: {condition}")
    return metadata, by_condition


def candidate_order(rows: list[dict[str, Any]], segment: str, replicate: int) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> bytes:
        payload = f"{PERMUTATION_SEED}|{replicate}|{segment}|{row['system_id']}".encode()
        return hashlib.sha256(payload).digest()
    return sorted(rows, key=key)


def _mean_ci(values: np.ndarray, seed_offset: int = 0) -> dict[str, float]:
    if values.ndim != 1 or not np.isfinite(values).all() or not len(values):
        raise ValueError("Bootstrap values must be a nonempty finite vector")
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    means = np.empty(BOOTSTRAP_RESAMPLES)
    chunk = 1000
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        stop = min(start + chunk, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "segments": int(len(values)),
    }


def analyze(by_condition: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    baseline_rows = by_condition[CONDITIONS[0]]
    grouped_keys: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, row in baseline_rows.items():
        grouped_keys[(row["lp"], row["segment_key"])].append(key)
    candidate_counts = {len(keys) for keys in grouped_keys.values()}
    if len(candidate_counts) != 1:
        raise ValueError(f"Nonrectangular candidate pools: {sorted(candidate_counts)}")
    full_size = next(iter(candidate_counts))
    numeric_pool_sizes = [full_size if value == "full" else value for value in POOL_SIZES]
    if any(value > full_size for value in numeric_pool_sizes):
        raise ValueError(f"Frozen pool size exceeds candidate count {full_size}")

    selected: dict[str, dict[str, dict[int, list[float]]]] = {
        condition: defaultdict(lambda: defaultdict(list)) for condition in CONDITIONS
    }
    for (lp, segment), keys in sorted(grouped_keys.items()):
        base_candidates = [baseline_rows[key] for key in keys]
        for replicate in range(PERMUTATIONS):
            ordered_systems = [
                row["system_id"] for row in candidate_order(base_candidates, segment, replicate)
            ]
            for pool_size in numeric_pool_sizes:
                systems = set(ordered_systems[:pool_size])
                for condition in CONDITIONS:
                    candidates = [
                        by_condition[condition][key]
                        for key in keys
                        if by_condition[condition][key]["system_id"] in systems
                    ]
                    winner = max(candidates, key=lambda row: (row["predicted_reward"], row["system_id"]))
                    selected[condition][lp][pool_size].append(float(winner["human_mqm"]))

    pair_results: dict[str, Any] = {}
    language_pairs = sorted({lp for lp, _ in grouped_keys})
    for lp_group in language_pairs + ["combined"]:
        segment_count = sum(1 for lp, _ in grouped_keys if lp_group == "combined" or lp == lp_group)
        expected = segment_count * PERMUTATIONS
        group_result: dict[str, Any] = {"conditions": {}, "contrasts": {}}
        per_segment: dict[str, dict[int, np.ndarray]] = {condition: {} for condition in CONDITIONS}
        for condition in CONDITIONS:
            group_result["conditions"][condition] = {}
            for pool_size in numeric_pool_sizes:
                if lp_group == "combined":
                    values = np.concatenate([
                        np.asarray(selected[condition][lp][pool_size]) for lp in language_pairs
                    ])
                else:
                    values = np.asarray(selected[condition][lp_group][pool_size])
                if len(values) != expected:
                    raise ValueError("Unexpected selection count")
                segment_means = values.reshape(segment_count, PERMUTATIONS).mean(axis=1)
                per_segment[condition][pool_size] = segment_means
                group_result["conditions"][condition][str(pool_size)] = _mean_ci(segment_means, pool_size)
        first_pool = numeric_pool_sizes[0]
        for condition in CONDITIONS[1:]:
            if not np.allclose(per_segment[condition][first_pool], per_segment[CONDITIONS[0]][first_pool]):
                raise AssertionError("N=1 human outcomes must be identical across reward models")
        for comparator in COMPARATORS:
            name = f"diagnostic_filtered_minus_{comparator}"
            group_result["contrasts"][name] = {}
            for pool_size in numeric_pool_sizes:
                delta = per_segment["diagnostic_filtered"][pool_size] - per_segment[comparator][pool_size]
                group_result["contrasts"][name][str(pool_size)] = _mean_ci(delta, pool_size + 100)
            pressure_delta = (
                per_segment["diagnostic_filtered"][full_size]
                - per_segment["diagnostic_filtered"][first_pool]
                - per_segment[comparator][full_size]
                + per_segment[comparator][first_pool]
            )
            group_result["contrasts"][name]["full_minus_first_difference_in_change"] = _mean_ci(
                pressure_delta, 999
            )
        pair_results[lp_group] = group_result

    correlations: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = list(by_condition[condition].values())
        correlations[condition] = {}
        for lp_group in language_pairs + ["combined"]:
            subset = [row for row in rows if lp_group == "combined" or row["lp"] == lp_group]
            correlations[condition][lp_group] = {
                "spearman": float(spearmanr(
                    [row["predicted_reward"] for row in subset],
                    [row["human_mqm"] for row in subset],
                ).statistic),
                "candidates": len(subset),
            }
    return {
        "status": "complete",
        "candidate_pool_size": full_size,
        "pool_sizes": numeric_pool_sizes,
        "permutations": PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "correlations": correlations,
        "selection_results": pair_results,
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing["wmt25_best_of_n"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=root / "artifacts/reward_models/wmt25_predictions")
    parser.add_argument("--output", type=Path, default=root / "results/paper_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata, rows = load_conditions(args.prediction_root)
    result = analyze(rows)
    result["prediction_metadata"] = metadata
    merge_ssot(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
