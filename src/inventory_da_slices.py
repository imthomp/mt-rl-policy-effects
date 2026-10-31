"""Inventory WMT DA pair-year slices and merge every row into the SSOT.

This script reads only dataset metadata needed for intervention design. It does
not inspect score values or translation text. The generated CSV is a convenient
view; ``results/paper_stats.json`` remains the source of truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_ID = "RicardoRei/wmt-da-human-evaluation"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def direction_class(language_pair: str) -> str:
    source, target = language_pair.split("-", maxsplit=1)
    if source == "en":
        return "en_to_x"
    if target == "en":
        return "x_to_en"
    return "x_to_y"


def build_inventory(frame: pd.DataFrame, discovery: set[tuple[str, int]]) -> list[dict[str, Any]]:
    required = {"lp", "year", "domain", "annotators"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DA metadata missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for (language_pair, year), group in frame.groupby(["lp", "year"], sort=True):
        domain_counts = {
            str(key): int(value)
            for key, value in group["domain"].fillna("<missing>").value_counts().sort_index().items()
        }
        annotators = group["annotators"]
        rows.append(
            {
                "language_pair": str(language_pair),
                "year": int(year),
                "label_rows": int(len(group)),
                "direction": direction_class(str(language_pair)),
                "domain_counts": domain_counts,
                "annotator_count_min": int(annotators.min()),
                "annotator_count_max": int(annotators.max()),
                "annotator_count_mean": float(annotators.mean()),
                "used_in_diagnostic_discovery": (str(language_pair), int(year)) in discovery,
            }
        )
    return rows


def load_arrow_metadata(paths: list[Path]) -> pd.DataFrame:
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError(
            "Reading Hugging Face Arrow shards requires the datasets package; "
            "use the project's data environment."
        ) from error
    columns = ["lp", "year", "domain", "annotators"]
    frames = [Dataset.from_file(str(path)).select_columns(columns).to_pandas() for path in paths]
    if not frames:
        raise ValueError("No Arrow shards supplied")
    return pd.concat(frames, ignore_index=True)


def load_discovery_slices(path: Path) -> set[tuple[str, int]]:
    frame = pd.read_csv(path, usecols=["testset", "lp"])
    years = frame["testset"].str.extract(r"wmt(\d{2})", expand=False).astype(int) + 2000
    return set(zip(frame["lp"].astype(str), years.astype(int), strict=True))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "language_pair",
        "year",
        "label_rows",
        "direction",
        "domain_counts_json",
        "annotator_count_min",
        "annotator_count_max",
        "annotator_count_mean",
        "used_in_diagnostic_discovery",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["domain_counts_json"] = json.dumps(output.pop("domain_counts"), sort_keys=True)
            writer.writerow(output)


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError(f"SSOT root must be a JSON object: {path}")
    else:
        existing = {}
    existing["reward_da_slice_inventory"] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-discovery-csv", type=Path, required=True)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output-csv", type=Path, default=root / "results" / "da_slice_inventory.csv")
    parser.add_argument("--ssot", type=Path, default=root / "results" / "paper_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards = sorted(args.arrow_dir.glob("*.arrow"))
    frame = load_arrow_metadata(shards)
    discovery = load_discovery_slices(args.diagnostic_discovery_csv)
    rows = build_inventory(frame, discovery)
    write_csv(args.output_csv, rows)
    result = {
        "dataset_id": DATASET_ID,
        "input_shards": [
            {"name": path.name, "sha256": sha256_file(path)} for path in shards
        ],
        "slice_unit": "language_pair_by_year",
        "rows": rows,
        "slice_count": len(rows),
        "label_row_count": int(sum(row["label_rows"] for row in rows)),
        "diagnostic_discovery_slice_count": int(
            sum(row["used_in_diagnostic_discovery"] for row in rows)
        ),
        "system_identity_available": False,
        "available_matching_fields": [
            "label volume",
            "year",
            "direction",
            "domain distribution",
            "annotator-count distribution",
            "language pair when another year is available",
        ],
        "unmatchable_field": "system composition",
        "score_values_accessed": False,
        "translation_text_accessed": False,
    }
    merge_ssot(args.ssot, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
