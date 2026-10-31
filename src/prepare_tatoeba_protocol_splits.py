"""Create sentence-disjoint Tatoeba train/calibration/dev/test manifests.

Reads the reviewed, WMT-overlap-filtered pool produced by
prepare_tatoeba_manifests.py. It greedily assigns rows using both sentence IDs
and normalized text as grouping keys, so no sentence or exact text crosses a
split. Unassigned eligible rows remain unused.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path("data/tatoeba_en_de")
OUTPUT = ROOT / "protocol_v1"
POOL = ROOT / "eligible_pairs.jsonl"
AUDIT = ROOT / "candidate_audit.json"
SPLIT_SEED = 314159
TARGETS = {
    "train": 2048,
    "calibration": 1024,
    "development": 1024,
    "test_in_domain": 2048,
}
ATTRIBUTION_FIELDS = (
    "source_sentence_id",
    "source_contributor",
    "source_license",
    "reference_sentence_id",
    "reference_contributor",
    "reference_license",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_keys(row: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return (
        ("id", row["source_sentence_id"]),
        ("id", row["reference_sentence_id"]),
        ("text", row["source"]),
        ("text", row["reference"]),
    )


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    pool_meta = audit.get("eligible_pair_pool")
    if not pool_meta or pool_meta.get("path") != str(POOL):
        raise ValueError("candidate audit does not identify the eligible pair pool")
    if sha256(POOL) != pool_meta["sha256"]:
        raise ValueError("eligible pair pool SHA-256 does not match candidate audit")
    rows = [json.loads(line) for line in POOL.read_text(encoding="utf-8").splitlines()]
    if len(rows) != pool_meta["pair_count"]:
        raise ValueError("eligible pair pool count does not match candidate audit")
    if any(not all(row.get(field) for field in ATTRIBUTION_FIELDS) for row in rows):
        raise ValueError("eligible pair pool has missing sentence attribution fields")

    shuffled = list(rows)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    split_names = list(TARGETS)
    selected: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}
    key_owner: dict[tuple[str, Any], str] = {}

    for row in shuffled:
        keys = row_keys(row)
        assigned = {key_owner[key] for key in keys if key in key_owner}
        if len(assigned) > 1:
            continue
        if assigned:
            destination = next(iter(assigned))
        else:
            available = [
                name for name in split_names
                if len(selected[name]) < TARGETS[name]
            ]
            if not available:
                continue
            destination = max(
                available,
                key=lambda name: (
                    (TARGETS[name] - len(selected[name])) / TARGETS[name],
                    -split_names.index(name),
                ),
            )
        if len(selected[destination]) >= TARGETS[destination]:
            continue
        selected[destination].append(row)
        for key in keys:
            key_owner[key] = destination

    short = {
        name: {"required": TARGETS[name], "actual": len(selected[name])}
        for name in split_names if len(selected[name]) != TARGETS[name]
    }
    if short:
        raise ValueError(f"sentence-disjoint allocation did not fill target sizes: {short}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    split_meta: dict[str, dict[str, Any]] = {}
    for name, split_rows in selected.items():
        path = OUTPUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in split_rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        split_meta[name] = {
            "path": str(path),
            "pair_count": len(split_rows),
            "sha256": sha256(path),
            "unique_source_contributors": len({row["source_contributor"] for row in split_rows}),
            "unique_reference_contributors": len({row["reference_contributor"] for row in split_rows}),
        }

    sentence_ids = {
        name: {
            row["source_sentence_id"] for row in split_rows
        } | {
            row["reference_sentence_id"] for row in split_rows
        }
        for name, split_rows in selected.items()
    }
    texts = {
        name: {
            row["source"] for row in split_rows
        } | {
            row["reference"] for row in split_rows
        }
        for name, split_rows in selected.items()
    }
    split_overlaps: dict[str, dict[str, int]] = {}
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            split_overlaps[f"{left}-{right}"] = {
                "sentence_id_overlap": len(sentence_ids[left] & sentence_ids[right]),
                "exact_text_overlap": len(texts[left] & texts[right]),
                "pair_overlap": len(
                    {(row["source"], row["reference"]) for row in selected[left]}
                    & {(row["source"], row["reference"]) for row in selected[right]}
                ),
            }
    if any(any(count for count in overlap.values()) for overlap in split_overlaps.values()):
        raise AssertionError(f"split leakage detected: {split_overlaps}")

    output_audit = {
        "status": "sentence-disjoint-protocol-splits-created",
        "source_dataset": audit["dataset"],
        "source_snapshot_date": audit["snapshot_date"],
        "source_license": audit["license"],
        "source_pair_pool_path": str(POOL),
        "source_pair_pool_count": len(rows),
        "source_pair_pool_sha256": pool_meta["sha256"],
        "split_seed": SPLIT_SEED,
        "selection_rule": (
            "shuffle eligible pool; greedily assign rows to the split with largest normalized "
            "size deficit; sentence IDs and normalized source/reference text may belong to only "
            "one split; discard conflicting rows and leave unused rows unassigned"
        ),
        "target_split_sizes": TARGETS,
        "actual_split_sizes": {name: len(value) for name, value in selected.items()},
        "unused_eligible_pairs": len(rows) - sum(map(len, selected.values())),
        "split_manifests": split_meta,
        "pairwise_overlaps": split_overlaps,
        "attribution_fields_preserved": list(ATTRIBUTION_FIELDS),
        "wmt23_wmt24_exact_text_overlap": 0,
        "interpretation": (
            "All three training seeds must reuse the same 2,048-row train split; seed variation "
            "then reflects optimization and action-sampling randomness, not different data subsets."
        ),
    }
    out_path = OUTPUT / "split_audit.json"
    out_path.write_text(json.dumps(output_audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output_audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
