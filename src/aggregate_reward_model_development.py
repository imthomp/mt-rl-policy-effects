"""Validate reward-model development runs and merge their metrics into the SSOT."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_CONDITIONS = (
    "all_data",
    "diagnostic_filtered",
    "global_random_2",
    "direction_matched_earlier_year",
)
NUMERIC_FIELDS = (
    "dev_mse",
    "dev_pearson",
    "dev_spearman",
    "elapsed_seconds",
    "peak_gpu_allocated_gib",
)
SHARED_FIELDS = (
    "model_name",
    "model_snapshot_hash",
    "indices_manifest_sha256",
    "dev_indices_sha256",
    "train_rows",
    "dev_rows",
    "steps",
    "batch_size",
    "gradient_accumulation",
    "learning_rate",
    "warmup_steps",
    "max_length",
    "seed",
    "executing_commit",
)


def load_and_validate(run_root: Path) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {}
    for condition in REQUIRED_CONDITIONS:
        path = run_root / condition / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics for {condition}: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("condition") != condition:
            raise ValueError(f"Condition mismatch in {path}")
        if metrics.get("git_worktree_dirty") is not False:
            raise ValueError(f"Run was not executed from a clean worktree: {path}")
        for field in NUMERIC_FIELDS:
            value = metrics.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {field} in {path}: {value!r}")
        model_dir = path.parent / "model"
        if not (model_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"Missing saved model for {condition}: {model_dir}")
        runs[condition] = metrics

    baseline = runs[REQUIRED_CONDITIONS[0]]
    for condition, metrics in runs.items():
        for field in SHARED_FIELDS:
            if metrics.get(field) != baseline.get(field):
                raise ValueError(
                    f"Non-comparable {field}: {condition}={metrics.get(field)!r}, "
                    f"baseline={baseline.get(field)!r}"
                )

    return {
        "status": "complete_models_frozen_before_human_outcome_access",
        "shared_configuration": {field: baseline[field] for field in SHARED_FIELDS},
        "conditions": {
            condition: {
                key: value
                for key, value in metrics.items()
                if key not in SHARED_FIELDS and key != "condition"
            }
            for condition, metrics in runs.items()
        },
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(existing, dict):
        raise ValueError(f"SSOT root must be an object: {path}")
    existing["reward_model_development"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "artifacts" / "reward_models" / "development",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "results" / "paper_stats.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = load_and_validate(args.run_root)
    merge_ssot(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
