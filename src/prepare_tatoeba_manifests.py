"""Prepare an attributable English-German Tatoeba candidate for the pilot.

Uses Tatoeba's official detailed-sentence, translation-link, and review exports.
It selects only sentence pairs for which both sides have an explicit positive
community review and no negative review, then excludes exact WMT23/WMT24 text
overlap. It writes provenance IDs/contributors into each manifest row.
"""

from __future__ import annotations

import bz2
import csv
import hashlib
import json
import os
import random
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator

from lxml import etree


RAW = Path("data/tatoeba_raw")
OUTPUT = Path("data/tatoeba_en_de")
RUN_SEEDS = (17, 29, 41)
PAIRS_PER_SEED = 2048
WMT24_XML = Path(os.environ.get(
    "WMT24_XML", str(Path.home() / ".sacrebleu/wmt24/raw/xml/wmttest2024.en-de.all.xml"),
))
WMT23_SOURCE = Path(os.environ.get(
    "WMT23_SOURCE", str(Path.home() / ".sacrebleu/wmt23/wmt23.en-de.src"),
))
WMT23_REFERENCE = Path(os.environ.get(
    "WMT23_REFERENCE", str(Path.home() / ".sacrebleu/wmt23/wmt23.en-de.ref-refA"),
))
DOWNLOAD_URLS = {
    "sentences_detailed.tar.bz2": "https://downloads.tatoeba.org/exports/sentences_detailed.tar.bz2",
    "links.tar.bz2": "https://downloads.tatoeba.org/exports/links.tar.bz2",
    "users_sentences.csv": "https://downloads.tatoeba.org/exports/users_sentences.csv",
}
SNAPSHOT_DATE = "2026-10-31"


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_tsv(path: Path) -> Iterator[list[str]]:
    """Yield tab-separated lines from the first regular member in an archive."""
    with tarfile.open(path, mode="r|bz2") as archive:
        member = next((entry for entry in archive if entry.isfile()), None)
        if member is None:
            raise ValueError(f"no regular file in {path}")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"could not open {member.name} from {path}")
        for line_no, raw in enumerate(stream, 1):
            fields = raw.decode("utf-8").rstrip("\r\n").split("\t")
            yield fields


def load_wmt_overlap() -> tuple[set[str], set[str], dict[str, int]]:
    tree = etree.parse(str(WMT24_XML))
    wmt24_sources: set[str] = set()
    wmt24_refs: set[str] = set()
    for doc in tree.findall(".//doc"):
        src_node = doc.find("src")
        ref_node = next((ref for ref in doc.findall("ref") if ref.get("translator") == "refA"), None)
        if src_node is None or ref_node is None:
            continue
        srcs = {seg.get("id"): normalize("".join(seg.itertext())) for seg in src_node.findall(".//seg")}
        refs = {seg.get("id"): normalize("".join(seg.itertext())) for seg in ref_node.findall(".//seg")}
        for key in srcs.keys() & refs.keys():
            if srcs[key] and refs[key]:
                wmt24_sources.add(srcs[key])
                wmt24_refs.add(refs[key])

    wmt23_sources = [normalize(line) for line in WMT23_SOURCE.read_text(encoding="utf-8").splitlines()]
    wmt23_refs = [normalize(line) for line in WMT23_REFERENCE.read_text(encoding="utf-8").splitlines()]
    if len(wmt23_sources) != len(wmt23_refs):
        raise ValueError("WMT23 source/reference lengths differ")
    wmt23_source_set = {text for text in wmt23_sources if text}
    wmt23_ref_set = {text for text in wmt23_refs if text}
    return wmt24_sources | wmt23_source_set, wmt24_refs | wmt23_ref_set, {
        "wmt23_pair_count": len(wmt23_sources),
        "wmt24_pair_count": len(wmt24_sources),
        "wmt23_unique_source_count": len(wmt23_source_set),
        "wmt24_unique_source_count": len(wmt24_sources),
    }


def main() -> None:
    expected = tuple(RAW / name for name in DOWNLOAD_URLS)
    if any(not path.is_file() for path in expected):
        raise FileNotFoundError(f"missing official Tatoeba exports in {RAW}")
    for path in (WMT24_XML, WMT23_SOURCE, WMT23_REFERENCE):
        if not path.is_file():
            raise FileNotFoundError(path)

    sentences: dict[int, tuple[str, str, str]] = {}
    sentence_counts: Counter[str] = Counter()
    for row in archive_tsv(RAW / "sentences_detailed.tar.bz2"):
        if len(row) != 6:
            raise ValueError(f"unexpected detailed-sentence row with {len(row)} fields")
        sentence_id, language, text, contributor, _created, _modified = row
        if language not in {"eng", "deu"}:
            continue
        text = normalize(text)
        if not text:
            continue
        sid = int(sentence_id)
        sentences[sid] = (language, text, contributor)
        sentence_counts[language] += 1

    review_positive: Counter[int] = Counter()
    review_negative: set[int] = set()
    with (RAW / "users_sentences.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) != 5:
                raise ValueError(f"unexpected review row with {len(row)} fields")
            _username, sentence_id, rating, _added, _modified = row
            if rating == "1":
                review_positive[int(sentence_id)] += 1
            elif rating == "-1":
                review_negative.add(int(sentence_id))

    positive_eligible_ids = {
        sid for sid in sentences
        if review_positive[sid] > 0 and sid not in review_negative
    }

    wmt_sources, wmt_refs, wmt_counts = load_wmt_overlap()
    pairs: dict[tuple[str, str], dict[str, object]] = {}
    linked_en_de_rows = 0
    for row in archive_tsv(RAW / "links.tar.bz2"):
        if len(row) != 2:
            raise ValueError(f"unexpected link row with {len(row)} fields")
        source_id, target_id = map(int, row)
        source = sentences.get(source_id)
        target = sentences.get(target_id)
        if source is None or target is None:
            continue
        if source[0] != "eng" or target[0] != "deu":
            continue
        linked_en_de_rows += 1
        if source_id not in positive_eligible_ids or target_id not in positive_eligible_ids:
            continue
        key = (source[1], target[1])
        if key in pairs:
            continue
        pairs[key] = {
            "source": source[1],
            "reference": target[1],
            "source_sentence_id": source_id,
            "source_contributor": source[2],
            "source_license": "CC-BY-2.0-FR (weekly export-level default; row-specific CC0 status is not present in export)",
            "reference_sentence_id": target_id,
            "reference_contributor": target[2],
            "reference_license": "CC-BY-2.0-FR (weekly export-level default; row-specific CC0 status is not present in export)",
            "license": "CC-BY-2.0-FR (Tatoeba text; preserve attribution)",
        }

    all_reviewed = list(pairs.values())
    excluded_overlap = [p for p in all_reviewed if p["source"] in wmt_sources or p["reference"] in wmt_refs]
    eligible = [p for p in all_reviewed if p["source"] not in wmt_sources and p["reference"] not in wmt_refs]
    if len(eligible) < PAIRS_PER_SEED:
        summary = {
            "status": "candidate-insufficient-reviewed-data",
            "review_rule": "both sentences have >=1 positive review and no negative review",
            "english_sentences": sentence_counts["eng"],
            "german_sentences": sentence_counts["deu"],
            "positive_review_eligible_sentences": len(positive_eligible_ids),
            "linked_english_german_rows": linked_en_de_rows,
            "unique_reviewed_pairs": len(all_reviewed),
            "pairs_excluded_for_exact_wmt_text_overlap": len(excluded_overlap),
            "eligible_pairs_after_overlap_exclusion": len(eligible),
            "minimum_required_pairs": PAIRS_PER_SEED,
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "candidate_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return

    # Draw all rows without replacement so data subsets are disjoint across seeds.
    shuffled = list(eligible)
    random.Random(314159).shuffle(shuffled)
    manifests: dict[int, list[dict[str, object]]] = {}
    for index, seed in enumerate(RUN_SEEDS):
        start = index * PAIRS_PER_SEED
        manifests[seed] = shuffled[start:start + PAIRS_PER_SEED]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pool_path = OUTPUT / "eligible_pairs.jsonl"
    with pool_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in eligible:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest_metadata = {}
    for seed, rows in manifests.items():
        path = OUTPUT / f"train_seed{seed}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest_metadata[str(seed)] = {"path": str(path), "pair_count": len(rows), "sha256": sha256(path)}

    cross_seed = {}
    for i, left_seed in enumerate(RUN_SEEDS):
        for right_seed in RUN_SEEDS[i + 1:]:
            left = {(r["source"], r["reference"]) for r in manifests[left_seed]}
            right = {(r["source"], r["reference"]) for r in manifests[right_seed]}
            cross_seed[f"{left_seed}-{right_seed}"] = len(left & right)
    summary = {
        "status": "candidate-manifests-created",
        "dataset": "Tatoeba official English-German detailed sentences + translation links",
        "snapshot_date": SNAPSHOT_DATE,
        "license": "CC-BY-2.0-FR (text); attribution preserved per sentence",
        "license_scope": {
            "source_statement": "Tatoeba's weekly text export is released under CC-BY 2.0 FR; its downloads page also notes that some sentences are available under CC0",
            "row_level_license_metadata_in_exports": False,
            "project_policy": "Treat each selected row under the export-level CC-BY 2.0 FR terms and retain both sentence contributors; do not claim sentence-specific CC0 status from these files",
            "institutional_clearance": "not determined by this dataset audit",
        },
        "attribution_fields": ["source_sentence_id", "source_contributor", "reference_sentence_id", "reference_contributor"],
        "rights_fields": ["source_license", "reference_license"],
        "quality_filter": "each sentence has >=1 positive community review and no negative community review; review data is experimental per Tatoeba",
        "english_sentences": sentence_counts["eng"],
        "german_sentences": sentence_counts["deu"],
        "positive_review_eligible_sentences": len(positive_eligible_ids),
        "linked_english_german_rows": linked_en_de_rows,
        "unique_reviewed_pairs": len(all_reviewed),
        "pairs_excluded_for_exact_wmt_text_overlap": len(excluded_overlap),
        "eligible_pairs_after_overlap_exclusion": len(eligible),
        "wmt_overlap_source_counts": wmt_counts,
        "manifest_rows": manifest_metadata,
        "eligible_pair_pool": {
            "path": str(pool_path),
            "pair_count": len(eligible),
            "sha256": sha256(pool_path),
        },
        "cross_seed_pair_overlaps": cross_seed,
        "source_file_sha256": {path.name: sha256(path) for path in expected},
        "source_file_urls": DOWNLOAD_URLS,
        "selection_seed_rule": "shuffle eligible pairs with seed 314159 and allocate disjoint 2048-pair chunks in run-seed order",
    }
    (OUTPUT / "candidate_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
