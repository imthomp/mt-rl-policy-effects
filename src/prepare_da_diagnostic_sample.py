"""Create the frozen per-slice WMT DA diagnostic sample.

The JSONL contains public dataset text and labels and is intentionally ignored
by Git. Provenance, row counts, and the output hash are merged into the SSOT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from inventory_da_slices import sha256_file


def select_indices(metadata: pd.DataFrame, cap: int, seed: int) -> dict[tuple[str, int], list[int]]:
    selected: dict[tuple[str, int], list[int]] = {}
    for (language_pair, year), group in metadata.groupby(["lp", "year"], sort=True):
        candidates = list(map(int, group.index))
        local_seed = int.from_bytes(
            hashlib.sha256(f"{seed}|{language_pair}|{int(year)}".encode()).digest()[:8],
            byteorder="big",
        )
        generator = random.Random(local_seed)
        count = min(cap, len(candidates))
        selected[(str(language_pair), int(year))] = sorted(generator.sample(candidates, count))
    return selected


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError(f"SSOT root must be a JSON object: {path}")
    else:
        existing = {}
    existing["reward_da_diagnostic_sample"] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-dir", type=Path, required=True)
    parser.add_argument("--sample-per-slice", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261031)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output", type=Path, default=root / "data" / "reward_da" / "diagnostic_sample.jsonl"
    )
    parser.add_argument("--ssot", type=Path, default=root / "results" / "paper_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_per_slice <= 0:
        raise ValueError("sample-per-slice must be positive")
    try:
        from datasets import Dataset, concatenate_datasets
    except ImportError as error:
        raise RuntimeError("Use the project data environment with datasets installed") from error

    shards = sorted(args.arrow_dir.glob("*.arrow"))
    if not shards:
        raise ValueError(f"No Arrow shards found under {args.arrow_dir}")
    dataset = concatenate_datasets([Dataset.from_file(str(path)) for path in shards])
    metadata = dataset.select_columns(["lp", "year"]).to_pandas()
    chosen = select_indices(metadata, args.sample_per_slice, args.seed)
    flat_indices = sorted(index for indices in chosen.values() for index in indices)
    sampled = dataset.select(flat_indices)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for global_index, row in zip(flat_indices, sampled, strict=True):
            record = {
                "global_row_index": global_index,
                "language_pair": row["lp"],
                "year": int(row["year"]),
                "domain": row["domain"],
                "source": row["src"],
                "hypothesis": row["mt"],
                "reference": row["ref"],
                "da_score": row["score"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = {
        "sample_unit": "language_pair_by_year",
        "sample_per_slice_cap": args.sample_per_slice,
        "seed": args.seed,
        "input_shards": [
            {"name": path.name, "sha256": sha256_file(path)} for path in shards
        ],
        "output_path": str(
            args.output.resolve().relative_to(Path(__file__).resolve().parents[1])
        ),
        "output_sha256": sha256_file(args.output),
        "sampled_row_count": len(flat_indices),
        "slice_count": len(chosen),
        "slice_rows": [
            {"language_pair": pair, "year": year, "sampled_rows": len(indices)}
            for (pair, year), indices in sorted(chosen.items())
        ],
        "threshold_rule": {
            "diagnostic": "spearman(COMET, BLEURT) - spearman(COMET, DA)",
            "bootstrap_interval": "paired percentile 95%",
            "flag_when": "lower endpoint > 0.50",
            "crossing_interval": "inconclusive; retain slice",
        },
    }
    merge_ssot(args.ssot, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
