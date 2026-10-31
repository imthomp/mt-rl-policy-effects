"""Freeze OPUS-100 training subsets and audit exact overlap with WMT24.

This script uses cached OPUS-100 train Parquet plus WMT24 source/refA
segments for exact overlap matching. WMT24 hypothesis fields are ignored and
no translation scores are computed. Pair text is NFC-normalized and outer whitespace stripped before
empty-row removal and exact pair deduplication.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import unicodedata
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from lxml import etree


OPUS_ROOT = Path(os.environ.get(
    "OPUS_ROOT",
    str(Path.home() / ".cache/huggingface/hub/datasets--Helsinki-NLP--opus-100/"
        "snapshots/805090dc28bf78897da9641cdf08b61287580df9/de-en"),
))
OPUS_INFO = Path(os.environ.get(
    "OPUS_INFO",
    str(Path.home() / ".cache/huggingface/datasets/Helsinki-NLP___opus-100/de-en/"
        "0.0.0/805090dc28bf78897da9641cdf08b61287580df9/dataset_info.json"),
))
WMT24_XML = Path(os.environ.get(
    "WMT24_XML", str(Path.home() / ".sacrebleu/wmt24/raw/xml/wmttest2024.en-de.all.xml"),
))
WMT23_SOURCE = Path(os.environ.get(
    "WMT23_SOURCE", str(Path.home() / ".sacrebleu/wmt23/wmt23.en-de.src"),
))
WMT23_REFERENCE = Path(os.environ.get(
    "WMT23_REFERENCE", str(Path.home() / ".sacrebleu/wmt23/wmt23.en-de.ref-refA"),
))
OUTPUT = Path("data/opus100_en_de")
RUN_SEEDS = (17, 29, 41)
PAIRS_PER_SEED = 2048


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_unique_train(path: Path) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    table = pq.read_table(path)
    if table.num_rows != 1_000_000:
        raise ValueError(f"expected cached 1,000,000-row OPUS train split; got {table.num_rows}")
    if table.schema.names != ["translation"]:
        raise ValueError(f"unexpected OPUS columns: {table.schema.names}")
    translation_type = table.schema.field("translation").type
    if translation_type.names != ["de", "en"]:
        raise ValueError(f"unexpected translation fields/order: {translation_type.names}")

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    empty_rows = 0
    duplicate_rows = 0
    for row in table.column("translation").to_pylist():
        # The cached config is named de-en; model inputs are English and targets German.
        source, reference = normalize(row["en"]), normalize(row["de"])
        if not source or not reference:
            empty_rows += 1
            continue
        pair = (source, reference)
        if pair in seen:
            duplicate_rows += 1
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs, {
        "input_rows": table.num_rows,
        "empty_rows_removed": empty_rows,
        "exact_duplicate_pairs_removed": duplicate_rows,
        "unique_nonempty_pairs": len(pairs),
        "fields": {"source": "translation.en", "reference": "translation.de"},
    }


def extract_wmt24_source_ref(path: Path) -> list[tuple[str, str]]:
    tree = etree.parse(str(path))
    pairs: list[tuple[str, str]] = []
    for doc in tree.findall(".//doc"):
        source_node = doc.find("src")
        ref_node = next(
            (ref for ref in doc.findall("ref") if ref.get("translator") == "refA"),
            None,
        )
        if source_node is None or ref_node is None:
            continue
        source_segments = {
            seg.get("id"): normalize("".join(seg.itertext()))
            for seg in source_node.findall(".//seg")
        }
        ref_segments = {
            seg.get("id"): normalize("".join(seg.itertext()))
            for seg in ref_node.findall(".//seg")
        }
        for segment_id in sorted(source_segments.keys() & ref_segments.keys(),
                                 key=lambda item: int(item) if item and item.isdigit() else item):
            source, reference = source_segments[segment_id], ref_segments[segment_id]
            if source and reference:
                pairs.append((source, reference))
    if not pairs:
        raise ValueError(f"no WMT24 en-de source/refA pairs parsed from {path}")
    return pairs


def extract_wmt23_source_ref(source_path: Path, reference_path: Path) -> list[tuple[str, str]]:
    sources = [normalize(line) for line in source_path.read_text(encoding="utf-8").splitlines()]
    references = [normalize(line) for line in reference_path.read_text(encoding="utf-8").splitlines()]
    if len(sources) != len(references):
        raise ValueError(f"WMT23 source/reference line counts differ: {len(sources)} vs {len(references)}")
    return [(source, reference) for source, reference in zip(sources, references, strict=True)
            if source and reference]


def main() -> None:
    train_path = OPUS_ROOT / "train-00000-of-00001.parquet"
    if not all(path.is_file() for path in (train_path, OPUS_INFO, WMT24_XML,
                                           WMT23_SOURCE, WMT23_REFERENCE)):
        raise FileNotFoundError("required cached OPUS train or WMT source/reference files are missing")

    eval_pairs = extract_wmt24_source_ref(WMT24_XML)
    eval_sources = {source for source, _ in eval_pairs}
    eval_references = {reference for _, reference in eval_pairs}
    eval_pairs_set = set(eval_pairs)
    calibration_pairs = extract_wmt23_source_ref(WMT23_SOURCE, WMT23_REFERENCE)
    calibration_sources = {source for source, _ in calibration_pairs}
    calibration_references = {reference for _, reference in calibration_pairs}
    calibration_pairs_set = set(calibration_pairs)
    unique_pairs, cleaning = load_unique_train(train_path)
    eval_excluded = [pair for pair in unique_pairs if pair[0] in eval_sources]
    calibration_excluded = [pair for pair in unique_pairs
                            if pair[0] in calibration_sources and pair[0] not in eval_sources]
    excluded_sources = eval_sources | calibration_sources
    eligible_pairs = [pair for pair in unique_pairs if pair[0] not in excluded_sources]
    cleaning["unique_pairs_excluded_for_exact_wmt24_source_overlap"] = len(eval_excluded)
    cleaning["unique_pairs_excluded_for_exact_wmt23_source_overlap"] = len(calibration_excluded)
    cleaning["eligible_pairs_after_wmt23_and_wmt24_source_exclusion"] = len(eligible_pairs)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[tuple[str, str]]] = {}
    for run_seed in RUN_SEEDS:
        subset_seed = 314159 + run_seed
        selected = random.Random(subset_seed).sample(eligible_pairs, PAIRS_PER_SEED)
        if len(set(selected)) != PAIRS_PER_SEED:
            raise AssertionError("manifest sampling produced duplicate pairs")
        path = OUTPUT / f"train_seed{run_seed}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for source, reference in selected:
                stream.write(json.dumps(
                    {"source": source, "reference": reference},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ) + "\n")
        manifests[str(run_seed)] = selected

    audit: dict[str, Any] = {
        "purpose": "data-integrity overlap audit only; no evaluation scores computed",
        "dataset": "Helsinki-NLP/opus-100 config de-en, cached parquet snapshot 805090dc28bf78897da9641cdf08b61287580df9",
        "normalization": "Unicode NFC then strip outer whitespace; preserve case and punctuation",
        "cleaning": cleaning,
        "sources": {
            "opus_train_path": str(train_path),
            "opus_train_sha256": sha256(train_path),
            "opus_dataset_info_sha256": sha256(OPUS_INFO),
            "wmt24_source_reference_xml": str(WMT24_XML),
            "wmt24_source_reference_xml_sha256": sha256(WMT24_XML),
            "wmt24_text_used": "src and refA segments used for exact matching; hypothesis values ignored; no scores computed",
            "wmt23_source_path": str(WMT23_SOURCE),
            "wmt23_source_sha256": sha256(WMT23_SOURCE),
            "wmt23_reference_path": str(WMT23_REFERENCE),
            "wmt23_reference_sha256": sha256(WMT23_REFERENCE),
            "wmt23_text_used": "source and refA only for exact matching; no scores computed",
        },
        "wmt23_source_refA_pair_count": len(calibration_pairs),
        "wmt23_unique_sources": len(calibration_sources),
        "wmt23_unique_references": len(calibration_references),
        "wmt24_source_refA_pair_count": len(eval_pairs),
        "wmt24_unique_sources": len(eval_sources),
        "wmt24_unique_references": len(eval_references),
        "seed_manifests": {},
        "cross_seed_pair_overlaps": {},
    }
    for seed, pairs in manifests.items():
        selected_pairs = set(pairs)
        selected_sources = {source for source, _ in pairs}
        selected_refs = {reference for _, reference in pairs}
        if selected_sources & excluded_sources:
            raise AssertionError(f"calibration/evaluation source overlap remains for seed {seed}")
        path = OUTPUT / f"train_seed{seed}.jsonl"
        audit["seed_manifests"][seed] = {
            "run_seed": int(seed),
            "subset_seed": 314159 + int(seed),
            "pair_count": len(pairs),
            "path": str(path),
            "sha256": sha256(path),
            "wmt23_exact_pair_overlap_count": len(selected_pairs & calibration_pairs_set),
            "wmt23_source_overlap_count": len(selected_sources & calibration_sources),
            "wmt23_reference_overlap_count": len(selected_refs & calibration_references),
            "wmt24_exact_pair_overlap_count": len(selected_pairs & eval_pairs_set),
            "wmt24_source_overlap_count": len(selected_sources & eval_sources),
            "wmt24_reference_overlap_count": len(selected_refs & eval_references),
        }
    for seed_a, pairs_a in manifests.items():
        for seed_b, pairs_b in manifests.items():
            if int(seed_a) < int(seed_b):
                audit["cross_seed_pair_overlaps"][f"{seed_a}-{seed_b}"] = len(set(pairs_a) & set(pairs_b))

    audit_path = OUTPUT / "manifest_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
