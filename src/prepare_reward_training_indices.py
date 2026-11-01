"""Prepare equal-budget row-index manifests for reward-model development runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from inventory_da_slices import sha256_file


DEFAULT_CONDITIONS = (
    "all_data",
    "diagnostic_filtered",
    "global_random_2",
    "direction_matched_earlier_year",
)


def hash_rank(seed: int, namespace: str, index: int) -> bytes:
    return hashlib.sha256(f"{seed}|{namespace}|{index}".encode()).digest()


def choose(indices: list[int], count: int, seed: int, namespace: str) -> list[int]:
    if len(indices) < count:
        raise ValueError(f"Requested {count} rows from only {len(indices)} eligible rows")
    return sorted(sorted(indices, key=lambda index: hash_rank(seed, namespace, index))[:count])


def slice_key(language_pair: str, year: int) -> str:
    return f"{language_pair}|{year}"


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    existing = json.loads(path.read_text(encoding="utf-8"))
    existing["reward_development_indices"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-dir", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=50_000)
    parser.add_argument("--dev-rows", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--condition-names", nargs="+", default=list(DEFAULT_CONDITIONS))
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output", type=Path, default=root / "data" / "reward_da" / "development_indices.json"
    )
    parser.add_argument("--ssot", type=Path, default=root / "results" / "paper_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_rows <= 0 or args.dev_rows <= 0:
        raise ValueError("train-rows and dev-rows must be positive")
    try:
        from datasets import Dataset, concatenate_datasets
    except ImportError as error:
        raise RuntimeError("Use the project data environment with datasets installed") from error
    shards = sorted(args.arrow_dir.glob("*.arrow"))
    dataset = concatenate_datasets([Dataset.from_file(str(path)) for path in shards])
    metadata = dataset.select_columns(["lp", "year"])
    conditions_payload = json.loads(args.conditions.read_text(encoding="utf-8"))
    conditions = conditions_payload["conditions"]
    missing = set(args.condition_names) - set(conditions)
    if missing:
        raise ValueError(f"Unknown conditions: {sorted(missing)}")

    ssot = json.loads(args.ssot.read_text(encoding="utf-8"))
    decisions = {
        slice_key(row["language_pair"], int(row["year"])): row["decision"]
        for row in ssot["reward_da_slice_diagnostics"]["rows"]
    }
    row_slices = [
        slice_key(str(language_pair), int(year))
        for language_pair, year in zip(metadata["lp"], metadata["year"], strict=True)
    ]
    clear_indices = [
        index for index, key in enumerate(row_slices) if decisions[key] == "retained_clear"
    ]
    dev_indices = choose(clear_indices, args.dev_rows, args.seed, "common_dev")
    dev_set = set(dev_indices)

    condition_rows: dict[str, Any] = {}
    for name in args.condition_names:
        excluded = set(conditions[name]["excluded_slices"])
        eligible = [
            index
            for index, key in enumerate(row_slices)
            if index not in dev_set and key not in excluded
        ]
        selected = choose(eligible, args.train_rows, args.seed, f"train|{name}")
        condition_rows[name] = {
            "train_indices": selected,
            "excluded_slices": sorted(excluded),
            "eligible_rows": len(eligible),
            "train_rows": len(selected),
            "train_indices_sha256": hashlib.sha256(
                json.dumps(selected, separators=(",", ":")).encode()
            ).hexdigest(),
        }

    payload = {
        "seed": args.seed,
        "train_rows_per_condition": args.train_rows,
        "common_dev_rows": args.dev_rows,
        "common_dev_source": "retained_clear slices only",
        "dev_indices": dev_indices,
        "dev_indices_sha256": hashlib.sha256(
            json.dumps(dev_indices, separators=(",", ":")).encode()
        ).hexdigest(),
        "conditions": condition_rows,
        "input_shards": [
            {"name": path.name, "sha256": sha256_file(path)} for path in shards
        ],
        "conditions_manifest_sha256": sha256_file(args.conditions),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    summary = {
        key: value for key, value in payload.items() if key not in {"dev_indices", "conditions"}
    }
    summary["conditions"] = {
        name: {key: value for key, value in row.items() if key != "train_indices"}
        for name, row in condition_rows.items()
    }
    summary["manifest_path"] = str(
        args.output.resolve().relative_to(Path(__file__).resolve().parents[1])
    )
    summary["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    merge_ssot(args.ssot, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
