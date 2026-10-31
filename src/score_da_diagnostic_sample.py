"""Score the frozen DA diagnostic sample with COMET or BLEURT.

Outputs are append-only JSONL files with resumable prefix validation. Raw
metric scores remain local artifacts; paper-facing slice diagnostics are
computed separately and merged into the SSOT.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MODEL_NAMES = {
    "comet": "Unbabel/wmt22-comet-da",
    "bleurt": "Elron/bleurt-large-512",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_prefix(output: Path, sample: list[dict[str, Any]]) -> int:
    if not output.exists():
        return 0
    completed = 0
    with output.open(encoding="utf-8") as handle:
        for completed, line in enumerate(handle, start=1):
            record = json.loads(line)
            expected = sample[completed - 1]["global_row_index"]
            if record.get("global_row_index") != expected:
                raise ValueError(
                    f"Resume prefix mismatch at row {completed - 1}: "
                    f"{record.get('global_row_index')} != {expected}"
                )
    if completed > len(sample):
        raise ValueError("Output contains more records than the frozen sample")
    return completed


def cached_revision(model_name: str) -> str | None:
    owner, name = model_name.split("/", maxsplit=1)
    ref = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{owner}--{name}"
        / "refs"
        / "main"
    )
    return ref.read_text(encoding="utf-8").strip() if ref.exists() else None


def score_comet(rows: list[dict[str, Any]], batch_size: int) -> list[float]:
    from comet import download_model, load_from_checkpoint

    checkpoint = download_model(MODEL_NAMES["comet"])
    model = load_from_checkpoint(checkpoint)
    data = [
        {"src": row["source"], "mt": row["hypothesis"], "ref": row["reference"]}
        for row in rows
    ]
    output = model.predict(data, batch_size=batch_size, gpus=1)
    return list(map(float, output.scores))


def score_bleurt(rows: list[dict[str, Any]], batch_size: int) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = MODEL_NAMES["bleurt"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, local_files_only=True
    ).eval().to("cuda")
    scores: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs = tokenizer(
            [row["reference"] for row in batch],
            [row["hypothesis"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits.squeeze(-1)
        scores.extend(map(float, logits.detach().cpu().tolist()))
    return scores


def write_metadata(
    path: Path,
    *,
    metric: str,
    input_path: Path,
    output_path: Path,
    completed: int,
    total: int,
) -> None:
    metadata = {
        "metric": metric,
        "model_name": MODEL_NAMES[metric],
        "model_snapshot_hash": cached_revision(MODEL_NAMES[metric]),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "completed_rows": completed,
        "total_rows": total,
        "complete": completed == total,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    if output_path.exists():
        metadata["output_sha256"] = sha256_file(output_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError("batch-size and chunk-size must be positive")
    sample = read_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    start = completed_prefix(args.output, sample)
    scorer = score_comet if args.metric == "comet" else score_bleurt

    write_metadata(
        metadata_path,
        metric=args.metric,
        input_path=args.input,
        output_path=args.output,
        completed=start,
        total=len(sample),
    )
    with args.output.open("a", encoding="utf-8") as handle:
        for chunk_start in range(start, len(sample), args.chunk_size):
            rows = sample[chunk_start : chunk_start + args.chunk_size]
            scores = scorer(rows, args.batch_size)
            if len(scores) != len(rows):
                raise ValueError(f"Scorer returned {len(scores)} values for {len(rows)} rows")
            for row, score in zip(rows, scores, strict=True):
                handle.write(
                    json.dumps(
                        {"global_row_index": row["global_row_index"], "score": score}
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
            completed = chunk_start + len(rows)
            write_metadata(
                metadata_path,
                metric=args.metric,
                input_path=args.input,
                output_path=args.output,
                completed=completed,
                total=len(sample),
            )
            gc.collect()


if __name__ == "__main__":
    main()
