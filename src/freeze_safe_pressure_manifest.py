"""Freeze WMT24 English--Spanish safe-pressure confirmation splits.

Only identifiers, source text, candidate text, and the outcome-free WMT XML
domain metadata are accessed. MQM category, severity, and metadata fields are
never indexed or stored.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


CALIBRATION_DOCUMENTS = 64
SPLIT_SALT = "safe-pressure-v1"
NON_MACHINE = re.compile(r"^(?:ref|human)", re.IGNORECASE)
MARKUP = re.compile(r"</?v>")
ACCESSED_MQM_FIELDS = {
    "system", "doc", "docSegId", "globalSegId", "rater", "source", "target"
}
PROHIBITED_OUTCOME_FIELDS = {"category", "severity", "metadata"}
ATTENTION_CHECK_DOCUMENTS = {"canary"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(label: str, value: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}|{label}|{value}".encode()).hexdigest()


def unmark(value: str) -> str:
    return MARKUP.sub("", value)


def xml_domains(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    result = {}
    for document in root.iter("doc"):
        document_id = document.get("id")
        if not document_id:
            raise ValueError("XML document lacks id")
        result[document_id] = document.get("domain") or "unspecified"
    return result


def structural_rows(path: Path) -> dict[str, Any]:
    candidates: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    targets: dict[tuple[str, str], set[str]] = defaultdict(set)
    documents: dict[str, set[str]] = defaultdict(set)
    raters: dict[tuple[str, str], set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        missing = ACCESSED_MQM_FIELDS - set(header)
        if missing:
            raise ValueError(f"MQM file lacks required structural fields: {sorted(missing)}")
        index = {name: header.index(name) for name in ACCESSED_MQM_FIELDS}
        for row_number, row in enumerate(reader, 2):
            if len(row) <= max(index.values()):
                raise ValueError(f"Short MQM row {row_number}")
            system = row[index["system"]]
            document = row[index["doc"]]
            if document.lower() in ATTENTION_CHECK_DOCUMENTS:
                continue
            segment = (
                f"{document}|{row[index['docSegId']]}|{row[index['globalSegId']]}"
            )
            candidates[segment].add(system)
            documents[document].add(segment)
            sources[segment].add(unmark(row[index["source"]]))
            targets[(segment, system)].add(unmark(row[index["target"]]))
            raters[(segment, system)].add(row[index["rater"]])

    all_systems = sorted(set().union(*candidates.values()))
    excluded = sorted(system for system in all_systems if NON_MACHINE.search(system))
    systems = sorted(set(all_systems) - set(excluded))
    system_set = set(systems)
    clean_segments = []
    for segment, observed in candidates.items():
        if observed & system_set != system_set:
            continue
        if len(sources[segment]) != 1:
            continue
        if any(len(targets[(segment, system)]) != 1 for system in systems):
            continue
        clean_segments.append(segment)
    clean = set(clean_segments)
    clean_documents = {
        document: sorted(segments & clean)
        for document, segments in documents.items()
        if segments & clean
    }
    return {
        "systems": systems,
        "excluded_systems": excluded,
        "excluded_attention_check_documents": sorted(ATTENTION_CHECK_DOCUMENTS),
        "documents": clean_documents,
        "clean_segments": sorted(clean),
        "candidate_text_hashes": {
            f"{segment}|{system}": hashlib.sha256(
                next(iter(targets[(segment, system)])).encode()
            ).hexdigest()
            for segment in clean_segments
            for system in systems
        },
        "source_text_hashes": {
            segment: hashlib.sha256(next(iter(sources[segment])).encode()).hexdigest()
            for segment in clean_segments
        },
        "rater_count_range": [
            min(len(raters[(segment, system)]) for segment in clean for system in systems),
            max(len(raters[(segment, system)]) for segment in clean for system in systems),
        ],
    }


def stratified_documents(
    documents: list[str], domains: dict[str, str], count: int
) -> tuple[list[str], dict[str, int]]:
    by_domain: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        if document not in domains:
            raise ValueError(f"Document absent from outcome-free XML: {document}")
        by_domain[domains[document]].append(document)
    total = len(documents)
    raw = {domain: count * len(rows) / total for domain, rows in by_domain.items()}
    quota = {domain: math.floor(value) for domain, value in raw.items()}
    remainder = count - sum(quota.values())
    order = sorted(by_domain, key=lambda domain: (-(raw[domain] - quota[domain]), domain))
    for domain in order[:remainder]:
        quota[domain] += 1
    selected = []
    for domain, rows in sorted(by_domain.items()):
        ranked = sorted(rows, key=lambda document: rank(f"document|{domain}", document))
        selected.extend(ranked[: quota[domain]])
    if len(selected) != count:
        raise ValueError("Stratified allocation has wrong size")
    return sorted(selected), quota


def build(root: Path) -> dict[str, Any]:
    protocol_path = root / "results/safe_pressure_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "design_frozen_fresh_confirmation_manifest_pending":
        raise ValueError("Safe-pressure design is not awaiting a manifest")
    if protocol["confirmation_data"]["split"]["calibration_documents"] != CALIBRATION_DOCUMENTS:
        raise ValueError("Calibration-document count differs from frozen protocol")

    mqm_root = Path(
        "/home/it238/projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation"
    )
    mqm_path = mqm_root / "generalMT2024/mqm_generalMT2024_enes.tsv"
    xml_root = root / "data/wmt24_news_systems"
    xml_path = xml_root / "xml/wmttest2024.en-es.all.xml"
    structural = structural_rows(mqm_path)
    domains = xml_domains(xml_path)
    document_ids = sorted(structural["documents"])
    calibration_documents, domain_quota = stratified_documents(
        document_ids, domains, CALIBRATION_DOCUMENTS
    )
    calibration_set = set(calibration_documents)
    evaluation_documents = sorted(set(document_ids) - calibration_set)
    calibration_segments = sorted(
        segment
        for document in calibration_documents
        for segment in structural["documents"][document]
    )
    calibration_label_segments = sorted(
        min(
            structural["documents"][document],
            key=lambda segment: rank(f"segment|{document}", segment),
        )
        for document in calibration_documents
    )
    evaluation_segments = sorted(
        segment
        for document in evaluation_documents
        for segment in structural["documents"][document]
    )
    ranked_systems = sorted(
        structural["systems"], key=lambda system: rank("system", system)
    )
    midpoint = len(ranked_systems) // 2
    train_systems = sorted(ranked_systems[:midpoint])
    heldout_systems = sorted(ranked_systems[midpoint:])
    if not train_systems or not heldout_systems:
        raise ValueError("System split is empty")

    return {
        "status": "fresh_confirmation_manifest_frozen_outcomes_sealed",
        "protocol": "results/safe_pressure_protocol.json",
        "protocol_sha256": sha256(protocol_path),
        "dataset": "wmt24_en_es_mqm",
        "language_direction": "en-es",
        "human_outcomes_accessed": False,
        "outcome_access_statement": (
            "Only identifier, source, target, and rater fields were indexed. MQM category, "
            "severity, and metadata were neither indexed nor stored."
        ),
        "sources": {
            "candidate_xml": {
                "repository": "https://github.com/wmt-conference/wmt24-news-systems",
                "revision": "fabe1f81d65449b3e3f9f8d37da280fa6cf3650a",
                "relative_path": "xml/wmttest2024.en-es.all.xml",
                "sha256": sha256(xml_path),
            },
            "human_mqm": {
                "repository": "https://github.com/google/wmt-mqm-human-evaluation",
                "revision": "7fadea281aff8db61cc72fc3d469a4dca78b82a6",
                "relative_path": "generalMT2024/mqm_generalMT2024_enes.tsv",
                "sha256": sha256(mqm_path),
                "accessed_fields": sorted(ACCESSED_MQM_FIELDS),
                "prohibited_outcome_fields": sorted(PROHIBITED_OUTCOME_FIELDS),
                "outcome_fields_accessed": [],
            },
        },
        "prior_exposure": {
            "candidate_existence_known": True,
            "candidate_text_previously_structurally_audited": True,
            "human_outcomes_previously_analyzed": False,
            "reinforcement_learning_use": "none beyond outcome-blind structural audit",
            "related_amanamt_use": "no WMT24 English-Spanish use found in code or documentation search",
        },
        "cleaning": {
            "rule": (
                "exclude ref/human systems by name; require every machine system and exactly "
                "one unmarked source and target text for each retained segment-system pair"
            ),
            "excluded_systems": structural["excluded_systems"],
            "excluded_attention_check_documents": structural[
                "excluded_attention_check_documents"
            ],
            "source_text_hashes": structural["source_text_hashes"],
            "candidate_text_hashes": structural["candidate_text_hashes"],
            "rater_count_range_without_scores": structural["rater_count_range"],
        },
        "split": {
            "salt": SPLIT_SALT,
            "domain_quota": domain_quota,
            "calibration_documents": calibration_documents,
            "calibration_optimization_segments": calibration_segments,
            "calibration_human_label_segments": calibration_label_segments,
            "evaluation_documents": evaluation_documents,
            "evaluation_segments": evaluation_segments,
            "train_systems": train_systems,
            "heldout_systems": heldout_systems,
        },
        "integrity": {
            "document_disjoint": not bool(calibration_set & set(evaluation_documents)),
            "segment_disjoint": not bool(set(calibration_segments) & set(evaluation_segments)),
            "system_disjoint": not bool(set(train_systems) & set(heldout_systems)),
            "all_clean_segments_assigned": (
                set(calibration_segments) | set(evaluation_segments)
                == set(structural["clean_segments"])
            ),
            "all_machine_systems_assigned": (
                set(train_systems) | set(heldout_systems)
                == set(structural["systems"])
            ),
        },
        "next_allowed_stage": (
            "score frozen candidates with frozen reward checkpoints and compute text embeddings; "
            "do not open MQM outcome fields"
        ),
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["safe_pressure_manifest"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results/safe_pressure_manifest.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    result = build(root)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    merge_ssot(root / "results/paper_stats.json", result)
    print(json.dumps({
        "status": result["status"],
        "dataset": result["dataset"],
        "calibration_documents": len(result["split"]["calibration_documents"]),
        "calibration_segments": len(result["split"]["calibration_optimization_segments"]),
        "evaluation_documents": len(result["split"]["evaluation_documents"]),
        "evaluation_segments": len(result["split"]["evaluation_segments"]),
        "train_systems": result["split"]["train_systems"],
        "heldout_systems": result["split"]["heldout_systems"],
        "integrity": result["integrity"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
