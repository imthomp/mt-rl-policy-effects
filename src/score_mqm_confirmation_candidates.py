"""Score a frozen MQM confirmation pool without reading human outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from audit_reward_holdouts import sha256_file
from train_da_reward_model import format_input


def clean_text(text: str) -> str:
    return re.sub(r"</?v>", "", text)


def load_candidates(
    source_path: Path, manifest_path: Path, dataset_name: str
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest["datasets"][dataset_name]
    if sha256_file(source_path) != dataset["source_sha256"]:
        raise ValueError("MQM source hash differs from frozen manifest")
    included = set(dataset["included_segment_ids"])
    machine_systems = dataset["machine_systems"]
    excluded = dataset["excluded_non_machine_systems"]
    if "refA" not in excluded:
        raise ValueError("Frozen confirmation scorer requires refA")
    sources: dict[str, set[str]] = defaultdict(set)
    targets: dict[tuple[str, str], set[str]] = defaultdict(set)
    accessed = ("system", "doc", "docSegId", "globalSegId", "source", "target")
    with source_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        index = {name: header.index(name) for name in accessed}
        for row_number, row in enumerate(reader, 2):
            if len(row) <= max(index.values()):
                raise ValueError(f"Missing required blind field at row {row_number}")
            segment = "|".join(
                (row[index["doc"]], row[index["docSegId"]], row[index["globalSegId"]])
            )
            if segment not in included:
                continue
            system = row[index["system"]]
            sources[segment].add(clean_text(row[index["source"]]))
            targets[(segment, system)].add(clean_text(row[index["target"]]))
    rows = []
    for segment in sorted(included):
        if len(sources[segment]) != 1 or len(targets[(segment, "refA")]) != 1:
            raise ValueError(f"Ambiguous source/reference in frozen segment: {segment}")
        source = next(iter(sources[segment]))
        reference = next(iter(targets[(segment, "refA")]))
        for system in machine_systems:
            values = targets[(segment, system)]
            if len(values) != 1:
                raise ValueError(f"Ambiguous candidate in frozen segment: {segment}/{system}")
            rows.append(
                {
                    "row_key": f"{segment}|{system}",
                    "segment_key": segment,
                    "system_id": system,
                    "source": source,
                    "hypothesis": next(iter(values)),
                    "reference": reference,
                }
            )
    expected = len(included) * len(machine_systems)
    if len(rows) != expected:
        raise AssertionError(f"Expected {expected} candidates, found {len(rows)}")
    metadata = {
        "dataset_name": dataset_name,
        "source_path": str(source_path),
        "source_sha256": dataset["source_sha256"],
        "candidate_content_sha256": dataset["candidate_content_sha256"],
        "segments": len(included),
        "systems": len(machine_systems),
        "rows": len(rows),
        "accessed_fields": list(accessed),
        "outcome_fields_accessed": [],
    }
    return metadata, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition", required=True)
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
    metadata, rows = load_candidates(args.source, args.manifest, args.dataset_name)
    model_dir = args.model_root / args.condition / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    ).to("cuda")
    model.eval()
    predictions = []
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            texts = [
                format_input(
                    row["source"], row["hypothesis"], row["reference"], tokenizer.sep_token
                )
                for row in batch
            ]
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                output = model(**inputs).logits.squeeze(-1)
            predictions.extend(map(float, output.float().cpu().tolist()))
    if len(predictions) != len(rows) or not np.isfinite(predictions).all():
        raise ValueError("Incomplete or nonfinite predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata.update(
        {
            "record_type": "metadata",
            "condition": args.condition,
            "model_metrics_sha256": sha256_file(args.model_root / args.condition / "metrics.json"),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        }
    )
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        for row, prediction in zip(rows, predictions, strict=True):
            output_row = {
                key: value
                for key, value in row.items()
                if key not in {"source", "hypothesis", "reference"}
            }
            output_row.update(
                {"record_type": "candidate", "predicted_reward": prediction}
            )
            handle.write(json.dumps(output_row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
