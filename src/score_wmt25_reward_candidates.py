"""Score every rectangular WMT25 MQM candidate with one frozen reward model."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from audit_reward_holdouts import WMT25_FILES, sha256_file
from train_da_reward_model import format_input


MQM_WEIGHTING_SOURCE = "https://www2.statmt.org/wmt25/mteval-subtask1.html"


def mqm_score(errors: list[dict[str, Any]]) -> float:
    if any(str(error.get("error_type", "")).lower() == "non-translation!" for error in errors):
        return -25.0
    penalty = 0.0
    for error in errors:
        severity = str(error.get("severity", "")).lower()
        error_type = str(error.get("error_type", "")).lower()
        if severity in {"major", "critical"}:
            penalty += 5.0
        elif severity == "minor":
            penalty += 0.1 if error_type == "fluency/punctuation" else 1.0
        else:
            raise ValueError(f"Unknown MQM severity: {severity!r}")
    return -penalty


def _ordered_keys(column: dict[str, Any]) -> list[str]:
    keys = list(column)
    try:
        return sorted(keys, key=int)
    except ValueError:
        return sorted(keys)


def iter_candidates(archive_path: Path) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(archive_path) as archive:
        for member in WMT25_FILES:
            columns = json.load(archive.open(member))
            required = (
                "lp", "set_id", "doc_id", "segment_id", "system_id",
                "source_segment", "hypothesis_segment", "reference_segment", "error_spans",
            )
            if missing := set(required) - columns.keys():
                raise ValueError(f"{member} missing columns: {sorted(missing)}")
            keys = _ordered_keys(columns["system_id"])
            for name in required:
                if set(columns[name]) != set(keys):
                    raise ValueError(f"Misaligned WMT25 column {name} in {member}")
            for key in keys:
                segment_key = "|".join(
                    str(columns[name][key]) for name in ("set_id", "doc_id", "segment_id")
                )
                yield {
                    "row_key": f"{member}|{key}",
                    "lp": str(columns["lp"][key]),
                    "segment_key": segment_key,
                    "system_id": str(columns["system_id"][key]),
                    "source": str(columns["source_segment"][key]),
                    "hypothesis": str(columns["hypothesis_segment"][key]),
                    "reference": str(columns["reference_segment"][key]),
                    "human_mqm": mqm_score(columns["error_spans"][key]),
                }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--archive", type=Path, required=True)
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
    model_dir = args.model_root / args.condition / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to("cuda")
    model.eval()
    rows = list(iter_candidates(args.archive))
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
        raise ValueError("Incomplete or nonfinite reward predictions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        metadata = {
            "record_type": "metadata", "condition": args.condition,
            "archive_sha256": sha256_file(args.archive),
            "mqm_weighting_source": MQM_WEIGHTING_SOURCE,
            "rows": len(rows), "batch_size": args.batch_size, "max_length": args.max_length,
        }
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
