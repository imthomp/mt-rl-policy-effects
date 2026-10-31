"""Audit WMT holdout candidate coverage without reading judgment outcomes.

The audit deliberately accesses only identifiers and annotation mapping keys.
It never reads MQM errors, cESA scores, preferences, or error spans.  Its
output is merged into the repository's paper-statistics SSOT so later writing
can cite generated values instead of hand-typed coverage claims.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


WMT25_FILES = (
    "Task2/mqm_generalMT2025_ja-zh_CN_with_errors.json",
    "Task2/mqm_generalMT2025_en-ko_KR_with_errors.json",
)
WMT25_SOURCE_URL = (
    "https://data.statmt.org/wmt25/metrics-qetask2/"
    "Task2-20260625T123413Z-3-001.zip"
)
WMT26_CAMPAIGN_SOURCE_URL = (
    "https://github.com/wmt-conference/wmt26-general-mt/releases/"
    "download/humeval/campaign_sources.zip"
)
WMT26_ANNOTATION_SOURCE_URL = (
    "https://github.com/wmt-conference/wmt26-general-mt/releases/"
    "download/humeval/annotations.json"
)
WMT26_FILTER_MANIFEST_URL = (
    "https://raw.githubusercontent.com/wmt-conference/wmt26-general-mt/"
    "main/synthetic_references/manifest.json"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: Iterable[int], label: str) -> list[dict[str, int]]:
    counts: dict[int, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return [{label: value, "segments": counts[value]} for value in sorted(counts)]


def _coverage_summary(candidate_sets: dict[str, set[str]]) -> dict[str, Any]:
    sizes = [len(value) for value in candidate_sets.values()]
    if not sizes:
        raise ValueError("No production candidate sets were found")
    canonical = next(iter(candidate_sets.values()))
    return {
        "segments": len(sizes),
        "unique_systems": len(set().union(*candidate_sets.values())),
        "candidate_count_min": min(sizes),
        "candidate_count_max": max(sizes),
        "candidate_count_mean": statistics.fmean(sizes),
        "candidate_count_distribution": _distribution(sizes, "candidates"),
        "rectangular_candidate_pool": all(value == canonical for value in candidate_sets.values()),
    }


def audit_wmt25_mqm(archive_path: Path) -> dict[str, Any]:
    language_pairs: dict[str, Any] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in WMT25_FILES:
            with archive.open(member) as raw:
                columns = json.load(io.TextIOWrapper(raw, encoding="utf-8"))

            required = ("lp", "set_id", "doc_id", "segment_id", "system_id", "method")
            missing = set(required) - columns.keys()
            if missing:
                raise ValueError(f"{member} is missing required columns: {sorted(missing)}")
            row_keys = set(columns["system_id"])
            if any(set(columns[name]) != row_keys for name in required):
                raise ValueError(f"{member} has misaligned identifier columns")

            candidates: dict[str, set[str]] = defaultdict(set)
            methods: set[str] = set()
            pairs: set[str] = set()
            for key in row_keys:
                segment = "|".join(
                    str(columns[name][key]) for name in ("set_id", "doc_id", "segment_id")
                )
                candidates[segment].add(str(columns["system_id"][key]))
                methods.add(str(columns["method"][key]))
                pairs.add(str(columns["lp"][key]))
            if len(pairs) != 1:
                raise ValueError(f"{member} contains multiple language-pair identifiers: {pairs}")

            pair = next(iter(pairs))
            summary = _coverage_summary(candidates)
            summary.update(
                {
                    "annotation_rows": len(row_keys),
                    "methods": sorted(methods),
                    "systems": sorted(set().union(*candidates.values())),
                    "archive_member": member,
                }
            )
            language_pairs[pair] = summary

    return {
        "role": "primary_rectangular_mqm_holdout",
        "source_url": WMT25_SOURCE_URL,
        "archive_sha256": sha256_file(archive_path),
        "language_pairs": language_pairs,
        "outcome_fields_accessed": [],
    }


def _production_candidates(campaign: dict[str, Any]) -> dict[str, set[str]]:
    candidates: dict[str, set[str]] = {}
    for document in campaign["data"]:
        for item in document:
            item_id = str(item["item_id"])
            if item_id in candidates:
                raise ValueError(f"Duplicate production item_id: {item_id}")
            candidates[item_id] = set(map(str, item["tgt"].keys()))
    return candidates


def _retained_content_lines(
    records: list[dict[str, Any]], production_ids: set[str]
) -> list[dict[str, Any]]:
    """Apply WMT26's dump-order RESET and non-content rules without outcomes."""
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        user = record.get("user_id")
        if not isinstance(user, str):
            continue
        if record.get("annotation") == "__RESET__":
            by_user[user] = []
            continue
        items = record.get("item")
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            continue
        if str(items[0].get("item_id")) in production_ids:
            by_user[user].append(record)
    return [record for records in by_user.values() for record in records]


def _has_valid_score(payload: object) -> bool:
    """Check evidence validity without retaining or reporting the score value."""
    if not isinstance(payload, dict):
        return False
    value = payload.get("score")
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0.0 <= value <= 100.0
    )


def audit_wmt26_cesa(
    campaign_archive_path: Path,
    annotation_path: Path,
    filter_manifest_path: Path,
) -> dict[str, Any]:
    with annotation_path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    with filter_manifest_path.open(encoding="utf-8") as handle:
        filter_manifest = json.load(handle)
    removed_user_counts = filter_manifest["low_reliability_users_removed"]

    language_pairs: dict[str, Any] = {}
    with zipfile.ZipFile(campaign_archive_path) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".json"))
        for member in members:
            with archive.open(member) as raw:
                campaign = json.load(io.TextIOWrapper(raw, encoding="utf-8"))
            campaign_id = str(campaign["campaign_id"])
            production = _production_candidates(campaign)
            judged: dict[str, set[str]] = defaultdict(set)

            retained = _retained_content_lines(
                annotations.get(campaign_id, []), set(production)
            )
            for record in retained:
                items = record.get("item", [])
                per_item_annotations = record.get("annotation", [])
                for item, annotation in zip(items, per_item_annotations, strict=False):
                    item_id = str(item.get("item_id"))
                    if item_id not in production or not isinstance(annotation, dict):
                        continue
                    displayed = item.get("tgt", {})
                    if not isinstance(displayed, dict):
                        continue
                    for system, payload in annotation.items():
                        if (
                            system in production[item_id]
                            and system in displayed
                            and _has_valid_score(payload)
                        ):
                            judged[item_id].add(str(system))

            candidate_summary = _coverage_summary(production)
            judged_sizes = [len(judged[item_id]) for item_id in production]
            complete = sum(
                judged[item_id] == candidates for item_id, candidates in production.items()
            )
            removed_users = int(removed_user_counts.get(campaign_id, 0))
            candidate_summary.update(
                {
                    "campaign_id": campaign_id,
                    "judged_candidate_count_min": min(judged_sizes),
                    "judged_candidate_count_max": max(judged_sizes),
                    "judged_candidate_count_mean": statistics.fmean(judged_sizes),
                    "judged_candidate_count_distribution": _distribution(
                        judged_sizes, "judged_candidates"
                    ),
                    "pre_reliability_fully_judged_segments": complete,
                    "pre_reliability_fully_judged_segment_fraction": complete / len(production),
                    "low_reliability_users_removed_by_official_pipeline": removed_users,
                    "post_reliability_coverage_confirmed": removed_users == 0,
                }
            )
            language_pairs[member.removesuffix(".json")] = candidate_summary

    return {
        "role": "secondary_sparse_cesa_holdout",
        "campaign_source_url": WMT26_CAMPAIGN_SOURCE_URL,
        "annotation_source_url": WMT26_ANNOTATION_SOURCE_URL,
        "filter_manifest_url": WMT26_FILTER_MANIFEST_URL,
        "campaign_archive_sha256": sha256_file(campaign_archive_path),
        "annotation_file_sha256": sha256_file(annotation_path),
        "filter_manifest_sha256": sha256_file(filter_manifest_path),
        "language_pairs": language_pairs,
        "outcome_fields_accessed": [
            "score presence/type/range for coverage validation; values not retained or summarized"
        ],
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if not isinstance(existing, dict):
            raise ValueError(f"SSOT root must be a JSON object: {path}")
    else:
        existing = {}
    existing["reward_holdout_feasibility"] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wmt25-archive", type=Path, required=True)
    parser.add_argument("--wmt26-campaign-archive", type=Path, required=True)
    parser.add_argument("--wmt26-annotations", type=Path, required=True)
    parser.add_argument("--wmt26-filter-manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "paper_stats.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "design_status": "coverage_only_outcomes_locked",
        "wmt25": audit_wmt25_mqm(args.wmt25_archive),
        "wmt26": audit_wmt26_cesa(
            args.wmt26_campaign_archive,
            args.wmt26_annotations,
            args.wmt26_filter_manifest,
        ),
    }
    merge_ssot(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
