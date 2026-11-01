"""Freeze outcome-blind WMT candidate manifests for predictor confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from audit_predictor_confirmation_data import (
    DEFAULT_ROOT,
    LICENSE,
    NON_MACHINE,
    REPOSITORY_REVISION,
    REPOSITORY_URL,
    _text_hash,
)
from audit_reward_holdouts import sha256_file
from run_text_policy_bandit import SYSTEM_SPLIT_SEED, system_split


SELECTIONS = (
    (
        "primary",
        "wmt24_ja_zh",
        "generalMT2024/mqm_generalMT2024_jazh.tsv",
        "clean rectangular pool with no source or candidate-text ambiguity",
    ),
    (
        "secondary",
        "wmt23_zh_en",
        "generalMT2023/zhen/mqm_generalMT2023_zhen.tsv",
        "largest clean pool and a different year and translation direction",
    ),
)
CALIBRATION_SPLIT_SEED = 20261201
CALIBRATION_DOCUMENTS = 64


def freeze_dataset(
    path: Path,
    *,
    split_name: str | None = None,
    calibration_document_count: int | None = None,
) -> dict[str, Any]:
    sources: dict[str, set[str]] = defaultdict(set)
    targets: dict[tuple[str, str], set[str]] = defaultdict(set)
    systems_by_segment: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        names = ("system", "doc", "docSegId", "globalSegId", "source", "target")
        index = {name: header.index(name) for name in names}
        for row_number, row in enumerate(reader, 2):
            if len(row) <= max(index.values()):
                raise ValueError(f"Missing required blind field in {path}:{row_number}")
            system = row[index["system"]]
            segment = "|".join(
                (row[index["doc"]], row[index["docSegId"]], row[index["globalSegId"]])
            )
            systems_by_segment[segment].add(system)
            sources[segment].add(_text_hash(row[index["source"]]))
            targets[(segment, system)].add(_text_hash(row[index["target"]]))

    all_systems = sorted(set().union(*systems_by_segment.values()))
    excluded_systems = sorted(system for system in all_systems if NON_MACHINE.search(system))
    machine_systems = sorted(set(all_systems) - set(excluded_systems))
    machine_set = set(machine_systems)
    included = []
    exclusions = defaultdict(int)
    content_digest = hashlib.sha256()
    for segment in sorted(systems_by_segment):
        observed = systems_by_segment[segment] & machine_set
        reasons = []
        if observed != machine_set:
            reasons.append("incomplete_machine_pool")
        if len(sources[segment]) != 1:
            reasons.append("source_text_variants")
        if any(len(targets[(segment, system)]) != 1 for system in observed):
            reasons.append("candidate_text_variants")
        if "refA" not in excluded_systems or len(targets[(segment, "refA")]) != 1:
            reasons.append("reference_text_variants")
        if reasons:
            for reason in reasons:
                exclusions[reason] += 1
            continue
        source_hash = next(iter(sources[segment]))
        included.append(segment)
        content_digest.update(f"{segment}\t{source_hash}\n".encode("utf-8"))
        for system in machine_systems:
            target_hash = next(iter(targets[(segment, system)]))
            content_digest.update(f"{system}\t{target_hash}\n".encode("utf-8"))

    if not included:
        raise ValueError(f"No clean confirmation segments in {path}")
    result = {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "excluded_non_machine_systems": excluded_systems,
        "machine_systems": machine_systems,
        "included_segment_ids": included,
        "candidate_content_sha256": content_digest.hexdigest(),
        "excluded_segment_counts_by_reason": dict(sorted(exclusions.items())),
        "cleaning_rule": (
            "exclude reference/human systems by name prefix; retain segments with every "
            "machine system, one source-text variant, one target-text variant per machine "
            "candidate, and one refA text variant for reference-based reward scoring"
        ),
        "outcome_fields_accessed": [],
        "prohibited_outcome_fields": ["category", "severity", "metadata"],
    }
    if calibration_document_count is not None:
        if split_name is None:
            raise ValueError("split_name is required for a calibration split")
        by_document: dict[str, list[str]] = defaultdict(list)
        for segment in included:
            by_document[segment.split("|", 1)[0]].append(segment)
        if calibration_document_count >= len(by_document):
            raise ValueError(
                f"Calibration needs fewer than {len(by_document)} documents: {path}"
            )
        ordered_documents = sorted(
            by_document,
            key=lambda document: hashlib.sha256(
                f"{CALIBRATION_SPLIT_SEED}|{split_name}|document|{document}".encode()
            ).digest(),
        )
        calibration_documents = ordered_documents[:calibration_document_count]
        calibration_document_set = set(calibration_documents)
        calibration_segments = [
            min(
                by_document[document],
                key=lambda segment: hashlib.sha256(
                    f"{CALIBRATION_SPLIT_SEED}|{split_name}|segment|{segment}".encode()
                ).digest(),
            )
            for document in calibration_documents
        ]
        optimization_segments = [
            segment
            for segment in included
            if segment.split("|", 1)[0] in calibration_document_set
        ]
        evaluation_segments = [
            segment
            for segment in included
            if segment.split("|", 1)[0] not in calibration_document_set
        ]
        unlabeled_optimization_segments = sorted(
            set(optimization_segments) - set(calibration_segments)
        )
        if not evaluation_segments:
            raise AssertionError("Document-disjoint evaluation set is empty")
        result["calibration_evaluation_split"] = {
            "seed": CALIBRATION_SPLIT_SEED,
            "rule": (
                "hash-order documents; choose one hash-ordered segment from each calibration "
                "document; evaluate only on all segments from remaining documents"
            ),
            "calibration_document_ids": calibration_documents,
            "calibration_segment_ids": calibration_segments,
            "optimization_segment_ids": optimization_segments,
            "unlabeled_optimization_segment_ids": unlabeled_optimization_segments,
            "evaluation_document_ids": ordered_documents[calibration_document_count:],
            "evaluation_segment_ids": evaluation_segments,
            "outcome_fields_accessed": [],
        }
    return result


def build_manifest(dataset_root: Path) -> dict[str, Any]:
    datasets = {}
    for role, name, relative, rationale in SELECTIONS:
        row = freeze_dataset(
            dataset_root / relative,
            split_name=name,
            calibration_document_count=CALIBRATION_DOCUMENTS,
        )
        train_systems, heldout_systems = system_split(row["machine_systems"])
        row["policy_system_split"] = {
            "seed": SYSTEM_SPLIT_SEED,
            "rule": "same deterministic hash split as the development text policy",
            "train_systems": [row["machine_systems"][index] for index in train_systems],
            "heldout_systems": [
                row["machine_systems"][index] for index in heldout_systems
            ],
            "outcome_fields_accessed": [],
        }
        row.update({"role": role, "name": name, "relative_path": relative, "rationale": rationale})
        datasets[name] = row
    return {
        "status": "datasets_frozen_outcomes_sealed",
        "repository_url": REPOSITORY_URL,
        "repository_revision": REPOSITORY_REVISION,
        "license": LICENSE,
        "datasets": datasets,
        "confirmation_constraints": {
            "current_wmt25_wmt26_outcomes": "development only",
            "selected_mqm_outcomes": "sealed until predictor and decision rule are frozen",
            "reward_checkpoints": "confirmation requires reward-training seeds not used to fit the predictor",
            "policy_pressure": "compare at a prespecified matched entropy or KL grid, not fixed steps",
            "decision_options": ["optimize", "keep_initial"],
            "calibration_documents": CALIBRATION_DOCUMENTS,
            "calibration_and_evaluation": "document-disjoint",
        },
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["predictor_confirmation_manifest"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = build_manifest(DEFAULT_ROOT)
    output = root / "results/predictor_confirmation_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    merge_ssot(root / "results/paper_stats.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
