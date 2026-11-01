"""Score the frozen WMT26 Chinese-to-Japanese cESA candidate pool."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from audit_reward_holdouts import _retained_content_lines, sha256_file
from train_da_reward_model import format_input


CAMPAIGN_MEMBER = "zho_Hans---jpn_Jpan.json"


def split_html_segments(text: str) -> list[str]:
    positions = [match.end() for match in re.finditer(re.escape("</p>"), text)]
    blocks: list[str] = []
    start = 0
    for end in positions:
        blocks.append(text[start:end])
        start = end
    if start < len(text):
        blocks.append(text[start:])
    return blocks


def load_references(document_path: Path) -> dict[str, list[str]]:
    references: dict[str, list[str]] = {}
    with document_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            doc_id = row.get("doc_id", "")
            marker = "_###_zho_Hans_###_jpn_Jpan_###_"
            if marker not in doc_id:
                continue
            domain, bare_doc = doc_id.removeprefix("WMT26_###_").split(marker, 1)
            reference = row.get("refs", {}).get("refPE")
            if not isinstance(reference, str):
                raise ValueError(f"Missing refPE for {doc_id}")
            references[f"{domain}_###_{bare_doc}"] = split_html_segments(reference.strip())
    return references


def load_human_scores(
    annotation_path: Path,
    campaign_id: str,
    production: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], float]:
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    retained = _retained_content_lines(annotations[campaign_id], set(production))
    observations: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for line in retained:
        items, item_annotations = line.get("item"), line.get("annotation")
        if not isinstance(items, list) or not isinstance(item_annotations, list):
            raise ValueError("Malformed WMT26 annotation row")
        for displayed, annotation in zip(items, item_annotations, strict=True):
            item_id = str(displayed.get("item_id"))
            if item_id not in production or not isinstance(annotation, dict):
                continue
            displayed_targets = displayed.get("tgt", {})
            for system, payload in annotation.items():
                if system not in production[item_id] or system not in displayed_targets:
                    continue
                if displayed_targets[system] != production[item_id][system]:
                    raise ValueError(f"Displayed candidate mismatch: {item_id} / {system}")
                score = payload.get("score") if isinstance(payload, dict) else None
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or not 0.0 <= score <= 100.0
                ):
                    continue
                observations[(item_id, system)].append(float(score))
    expected = {(item_id, system) for item_id, systems in production.items() for system in systems}
    if set(observations) != expected:
        missing = expected - set(observations)
        raise ValueError(f"Incomplete WMT26 cESA coverage: {len(missing)} candidates missing")
    return {key: statistics.mean(values) for key, values in observations.items()}


def load_candidates(
    campaign_archive: Path,
    annotation_path: Path,
    documents_path: Path,
    filter_manifest_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    with zipfile.ZipFile(campaign_archive) as archive:
        campaign = json.load(archive.open(CAMPAIGN_MEMBER))
    campaign_id = str(campaign["campaign_id"])
    manifest = json.loads(filter_manifest_path.read_text(encoding="utf-8"))
    if int(manifest["low_reliability_users_removed"].get(campaign_id, -1)) != 0:
        raise ValueError("Frozen endpoint requires zero low-reliability user removals")
    references = load_references(documents_path)
    production = {
        str(item["item_id"]): {str(system): str(text) for system, text in item["tgt"].items()}
        for document in campaign["data"]
        for item in document
    }
    human_scores = load_human_scores(annotation_path, campaign_id, production)
    rows: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    for document in campaign["data"]:
        document_key = str(document[0]["item_id"]).rsplit("_###_", 1)[0]
        if document_key in seen_documents:
            raise ValueError(f"Duplicate campaign document: {document_key}")
        seen_documents.add(document_key)
        reference_segments = references.get(document_key)
        if reference_segments is None or len(reference_segments) != len(document):
            raise ValueError(f"Reference alignment failure: {document_key}")
        for segment_index, (item, reference) in enumerate(zip(document, reference_segments, strict=True)):
            item_id = str(item["item_id"])
            if item_id.rsplit("_###_", 1) != [document_key, str(segment_index)]:
                raise ValueError(f"Unexpected segment order: {item_id}")
            source = str(item.get("src_text", item["src"]))
            for system, hypothesis in sorted(production[item_id].items()):
                rows.append(
                    {
                        "row_key": f"{item_id}|{system}",
                        "lp": "zho_Hans---jpn_Jpan",
                        "segment_key": item_id,
                        "system_id": system,
                        "source": source,
                        "hypothesis": hypothesis,
                        "reference": reference,
                        "human_cesa": human_scores[(item_id, system)],
                    }
                )
    return campaign_id, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--campaign-archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--filter-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    campaign_id, rows = load_candidates(
        args.campaign_archive, args.annotations, args.documents, args.filter_manifest
    )
    model_dir = args.model_root / args.condition / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to("cuda")
    model.eval()
    predictions: list[float] = []
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            texts = [
                format_input(row["source"], row["hypothesis"], row["reference"], tokenizer.sep_token)
                for row in batch
            ]
            inputs = tokenizer(
                texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt"
            )
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                output = model(**inputs).logits.squeeze(-1)
            predictions.extend(map(float, output.float().cpu().tolist()))
    if len(predictions) != len(rows) or not np.isfinite(predictions).all():
        raise ValueError("Incomplete or nonfinite predictions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "record_type": "metadata",
        "condition": args.condition,
        "campaign_id": campaign_id,
        "campaign_archive_sha256": sha256_file(args.campaign_archive),
        "annotation_sha256": sha256_file(args.annotations),
        "documents_sha256": sha256_file(args.documents),
        "filter_manifest_sha256": sha256_file(args.filter_manifest),
        "rows": len(rows),
        "segments": len({row["segment_key"] for row in rows}),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        for row, prediction in zip(rows, predictions, strict=True):
            output_row = {
                key: value for key, value in row.items()
                if key not in {"source", "hypothesis", "reference"}
            }
            output_row.update({"record_type": "candidate", "predicted_reward": prediction})
            handle.write(json.dumps(output_row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
