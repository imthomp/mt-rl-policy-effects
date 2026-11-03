"""Audit a WMT candidate XML without opening human evaluation outcomes.

The input must be the outcome-free ``*.all.xml`` release. Human-score TSV or
JSON files are intentionally unsupported by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


FORBIDDEN_NAME_PARTS = ("mqm", "human", "score", "annotation", "humeval")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text(element: ET.Element) -> str:
    return "".join(element.itertext()).strip()


def read_set(element: ET.Element) -> dict[str, dict[str, str]]:
    documents: dict[str, dict[str, str]] = {}
    for document_index, document in enumerate(element.findall(".//doc")):
        document_id = document.get("docid") or f"document_{document_index}"
        segments: dict[str, str] = {}
        for segment_index, segment in enumerate(document.findall(".//seg")):
            segment_id = segment.get("id") or str(segment_index)
            if segment_id in segments:
                raise ValueError(f"Duplicate segment {document_id}/{segment_id}")
            segments[segment_id] = text(segment)
        if not segments:
            raise ValueError(f"Document has no segments: {document_id}")
        if document_id in documents:
            raise ValueError(f"Duplicate document: {document_id}")
        documents[document_id] = segments
    return documents


def find_sets(root: ET.Element, tag: str) -> list[ET.Element]:
    return [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == tag]


def read_wmt24_dataset(
    root: ET.Element,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, str]]]]:
    """Read the outcome-free WMT24 dataset/collection/doc XML layout."""
    source: dict[str, dict[str, str]] = {}
    systems: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    documents = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "doc"]
    for document_index, document in enumerate(documents):
        document_id = document.get("id") or f"document_{document_index}"
        source_nodes = [
            child for child in document
            if child.tag.rsplit("}", 1)[-1] == "src"
        ]
        if len(source_nodes) != 1:
            raise ValueError(f"Expected one source in {document_id}")
        source[document_id] = {
            segment.get("id") or str(index): text(segment)
            for index, segment in enumerate(source_nodes[0].iter())
            if segment.tag.rsplit("}", 1)[-1] == "seg"
        }
        hypothesis_nodes = [
            child for child in document
            if child.tag.rsplit("}", 1)[-1] == "hyp"
        ]
        for hypothesis in hypothesis_nodes:
            system = hypothesis.get("system")
            if not system:
                raise ValueError(f"Hypothesis without system in {document_id}")
            systems[system][document_id] = {
                segment.get("id") or str(index): text(segment)
                for index, segment in enumerate(hypothesis.iter())
                if segment.tag.rsplit("}", 1)[-1] == "seg"
            }
    if not source or not systems:
        raise ValueError("WMT24 XML has no source documents or hypotheses")
    return source, dict(systems)


def audit(path: Path, candidate_id: str, source_revision: str) -> dict[str, Any]:
    lowered = path.name.lower()
    if any(part in lowered for part in FORBIDDEN_NAME_PARTS):
        raise ValueError("Refusing a path that appears to contain human outcomes")
    root = ET.parse(path).getroot()
    if root.tag.rsplit("}", 1)[-1] == "dataset":
        source, system_documents = read_wmt24_dataset(root)
    else:
        source_sets = find_sets(root, "srcset")
        system_sets = find_sets(root, "tstset")
        if len(source_sets) != 1:
            raise ValueError(f"Expected one source set, found {len(source_sets)}")
        if not system_sets:
            raise ValueError("No machine-system test sets found")
        source = read_set(source_sets[0])
        system_documents = {}
        for index, system_set in enumerate(system_sets):
            system = system_set.get("sysid") or system_set.get("system") or f"system_{index}"
            if system in system_documents:
                raise ValueError(f"Duplicate system: {system}")
            system_documents[system] = read_set(system_set)

    expected = {
        (document, segment)
        for document, segments in source.items()
        for segment in segments
    }
    coverage: dict[str, set[tuple[str, str]]] = defaultdict(set)
    empty_outputs = []
    for system, documents in system_documents.items():
        for document, segments in documents.items():
            for segment, candidate in segments.items():
                coverage[system].add((document, segment))
                if not candidate:
                    empty_outputs.append(f"{system}|{document}|{segment}")

    per_system_missing = {
        system: len(expected - observed)
        for system, observed in sorted(coverage.items())
    }
    per_system_extra = {
        system: len(observed - expected)
        for system, observed in sorted(coverage.items())
    }
    result = {
        "status": "outcome_blind_candidate_audit_complete",
        "candidate_id": candidate_id,
        "input": {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "source_revision": source_revision,
            "kind": "WMT source/reference/hypothesis XML; no human outcomes",
        },
        "coverage": {
            "documents": len(source),
            "source_segments": len(expected),
            "machine_systems": sorted(system_documents),
            "machine_system_count": len(system_documents),
            "per_system_missing_segments": per_system_missing,
            "per_system_extra_segments": per_system_extra,
            "empty_output_count": len(empty_outputs),
            "rectangular": (
                not empty_outputs
                and not any(per_system_missing.values())
                and not any(per_system_extra.values())
            ),
        },
        "human_outcomes_accessed": False,
        "eligibility_pending": [
            "local and related-project prior-use search",
            "human-outcome file revision and hash recorded without opening contents",
            "license and attribution record",
            "deterministic document and system split manifest",
        ],
        "row_trace": {
            "documents": sorted(source),
            "systems": sorted(system_documents),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"Refusing to overwrite {arguments.output}")
    result = audit(arguments.xml, arguments.candidate_id, arguments.source_revision)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
