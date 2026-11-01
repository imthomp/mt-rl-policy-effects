"""Outcome-blind coverage audit for prospective reward-failure confirmation data.

Only identifiers, source text, candidate text, and rater identifiers are
accessed. MQM category, severity, metadata, and derived scores are deliberately
ignored. The output contains hashes and coverage summaries, never text or
human-outcome values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from audit_reward_holdouts import sha256_file


DEFAULT_ROOT = Path(
    "/home/it238/projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation"
)
DEFAULT_FILES = (
    "generalMT2023/ende/mqm_generalMT2023_ende.tsv",
    "generalMT2023/zhen/mqm_generalMT2023_zhen.tsv",
    "generalMT2024/mqm_generalMT2024_ende.tsv",
    "generalMT2024/mqm_generalMT2024_enes.tsv",
    "generalMT2024/mqm_generalMT2024_jazh.tsv",
)
REPOSITORY_URL = "https://github.com/google/wmt-mqm-human-evaluation"
REPOSITORY_REVISION = "7fadea281aff8db61cc72fc3d469a4dca78b82a6"
LICENSE = "Apache-2.0"
NON_MACHINE = re.compile(r"^(?:ref|human)", re.IGNORECASE)


def _distribution(values: list[int], key: str) -> list[dict[str, int]]:
    counts: dict[int, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return [{key: value, "segments": counts[value]} for value in sorted(counts)]


def _text_hash(text: str) -> str:
    # MQM repeats a candidate for each marked error span, moving only the <v>
    # highlight tags. Removing those tags reconstructs the underlying output
    # without reading category or severity.
    unmarked = re.sub(r"</?v>", "", text)
    return hashlib.sha256(unmarked.encode("utf-8")).hexdigest()


def audit_tsv(path: Path) -> dict[str, Any]:
    candidates: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    candidate_texts: dict[tuple[str, str], set[str]] = defaultdict(set)
    raters: dict[tuple[str, str], set[str]] = defaultdict(set)
    documents: set[str] = set()
    accessed = {"system", "doc", "docSegId", "globalSegId", "rater", "source", "target"}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        index = {name: header.index(name) for name in accessed}
        for row_number, row in enumerate(reader, 2):
            if len(row) <= max(index.values()):
                raise ValueError(f"Missing required identifier field in {path}:{row_number}")
            system = row[index["system"]]
            document = row[index["doc"]]
            doc_segment = row[index["docSegId"]]
            global_segment = row[index["globalSegId"]]
            rater = row[index["rater"]]
            source_hash = _text_hash(row[index["source"]])
            target_hash = _text_hash(row[index["target"]])
            segment = f"{document}|{doc_segment}|{global_segment}"
            documents.add(document)
            candidates[segment].add(system)
            raters[(segment, system)].add(rater)
            sources[segment].add(source_hash)
            key = (segment, system)
            candidate_texts[key].add(target_hash)

    systems = sorted(set().union(*candidates.values()))
    excluded = sorted(system for system in systems if NON_MACHINE.search(system))
    machine_systems = sorted(set(systems) - set(excluded))
    machine_sets = {
        segment: observed & set(machine_systems) for segment, observed in candidates.items()
    }
    candidate_counts = [len(observed) for observed in machine_sets.values()]
    canonical = set(machine_systems)
    rater_counts = [
        len(raters[(segment, system)])
        for segment, observed in machine_sets.items()
        for system in observed
    ]
    ambiguous_candidates = [
        key
        for key, variants in candidate_texts.items()
        if key[1] in canonical and len(variants) > 1
    ]
    ambiguous_segments = {segment for segment, _ in ambiguous_candidates}
    source_variant_segments = {
        segment for segment, variants in sources.items() if len(variants) > 1
    }
    clean_segments = {
        segment
        for segment, observed in machine_sets.items()
        if observed == canonical
        and segment not in ambiguous_segments
        and segment not in source_variant_segments
    }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "documents": len(documents),
        "segments": len(candidates),
        "all_systems": systems,
        "excluded_non_machine_systems": excluded,
        "machine_systems": machine_systems,
        "machine_system_count": len(machine_systems),
        "machine_candidate_rows": sum(candidate_counts),
        "machine_candidate_count_min": min(candidate_counts),
        "machine_candidate_count_max": max(candidate_counts),
        "machine_candidate_count_mean": statistics.fmean(candidate_counts),
        "machine_candidate_count_distribution": _distribution(
            candidate_counts, "machine_candidates"
        ),
        "rectangular_machine_pool": all(observed == canonical for observed in machine_sets.values()),
        "fully_covered_machine_segments": sum(
            observed == canonical for observed in machine_sets.values()
        ),
        "source_variant_segments": len(source_variant_segments),
        "ambiguous_candidate_texts": len(ambiguous_candidates),
        "segments_with_ambiguous_candidate_text": len(ambiguous_segments),
        "cleaning_rule": (
            "exclude reference/human systems by name prefix; retain segments with every "
            "machine system, one source-text variant, and one target-text variant per candidate"
        ),
        "clean_rectangular_segments": len(clean_segments),
        "clean_machine_candidate_rows": len(clean_segments) * len(machine_systems),
        "rater_count_min_per_machine_candidate": min(rater_counts),
        "rater_count_max_per_machine_candidate": max(rater_counts),
        "rater_count_distribution": _distribution(rater_counts, "raters"),
        "accessed_fields": sorted(accessed),
        "prohibited_outcome_fields": ["category", "severity", "metadata"],
        "outcome_fields_accessed": [],
    }


def build_audit(dataset_root: Path, relative_files: tuple[str, ...]) -> dict[str, Any]:
    datasets = {
        relative: audit_tsv(dataset_root / relative) for relative in relative_files
    }
    eligible = [
        relative
        for relative, row in datasets.items()
        if row["clean_rectangular_segments"] > 0 and row["machine_system_count"] > 1
    ]
    return {
        "status": "outcome_blind_feasibility_complete",
        "repository_url": REPOSITORY_URL,
        "repository_revision": REPOSITORY_REVISION,
        "license": LICENSE,
        "dataset_root": str(dataset_root),
        "datasets": datasets,
        "eligible_rectangular_datasets": eligible,
        "outcomes_unopened_for_analysis": True,
        "prior_exposure": {
            "amanamt_followup": (
                "WMT23/24 were named as prospective additions, but the project handoff says "
                "these MQM files and protocols were not used."
            ),
            "reinforcement_learning": (
                "WMT23/24 source/reference text was used only for exact-overlap filtering "
                "in the earlier NLLB/Tatoeba work; MQM outcomes were not used."
            ),
            "interpretation": (
                "Eligible for a prospectively frozen reward-selection confirmation, but not "
                "an unanticipated dataset because its existence was already documented."
            ),
        },
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["predictor_confirmation_data_audit"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output", type=Path, default=root / "results/predictor_confirmation_data_audit.json"
    )
    parser.add_argument("--ssot", type=Path, default=root / "results/paper_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_audit(args.dataset_root, DEFAULT_FILES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    merge_ssot(args.ssot, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
