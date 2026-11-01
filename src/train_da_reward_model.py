"""Train one fixed-budget XLM-R reward-model development arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


MODEL_NAME = "FacebookAI/xlm-roberta-base"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def repository_revision(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"executing_commit": commit, "git_worktree_dirty": dirty}


def cached_revision(model_name: str) -> str | None:
    owner, name = model_name.split("/", maxsplit=1)
    path = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{owner}--{name}"
        / "refs"
        / "main"
    )
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def format_input(source: str, hypothesis: str, reference: str, separator: str) -> str:
    return f"{source} {separator} {hypothesis} {separator} {reference}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--arrow-dir", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261102)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from datasets import Dataset, concatenate_datasets
    from scipy.stats import pearsonr, spearmanr
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    manifest = json.loads(args.indices.read_text(encoding="utf-8"))
    if args.condition not in manifest["conditions"]:
        raise ValueError(f"Condition not present in index manifest: {args.condition}")
    train_indices = manifest["conditions"][args.condition]["train_indices"]
    dev_indices = manifest["dev_indices"]
    shards = sorted(args.arrow_dir.glob("*.arrow"))
    dataset = concatenate_datasets([Dataset.from_file(str(path)) for path in shards])
    train_data = dataset.select(train_indices)
    dev_data = dataset.select(dev_indices)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression",
        local_files_only=True,
    ).to("cuda")

    def collate(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [
            format_input(row["src"], row["mt"], row["ref"], tokenizer.sep_token)
            for row in rows
        ]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([float(row["score"]) for row in rows])
        return encoded

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=0,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.steps
    )
    scaler_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    train_iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    started = time.monotonic()
    model.train()
    for step in range(args.steps):
        accumulated = 0.0
        for _ in range(args.gradient_accumulation):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            labels = batch.pop("labels").to("cuda")
            inputs = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=scaler_enabled):
                predictions = model(**inputs).logits.squeeze(-1)
                loss = F.mse_loss(predictions.float(), labels.float())
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(accumulated / args.gradient_accumulation)
        if (step + 1) % 25 == 0:
            print(json.dumps({"step": step + 1, "train_loss": losses[-1]}), flush=True)

    model.eval()
    predictions: list[float] = []
    labels_all: list[float] = []
    with torch.inference_mode():
        for batch in dev_loader:
            labels = batch.pop("labels")
            inputs = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=scaler_enabled):
                output = model(**inputs).logits.squeeze(-1)
            predictions.extend(map(float, output.float().cpu().tolist()))
            labels_all.extend(map(float, labels.tolist()))

    predictions_array = np.asarray(predictions)
    labels_array = np.asarray(labels_all)
    if not np.isfinite(predictions_array).all():
        raise ValueError("Nonfinite development predictions")
    elapsed = time.monotonic() - started
    metrics = {
        "condition": args.condition,
        "model_name": MODEL_NAME,
        "model_snapshot_hash": cached_revision(MODEL_NAME),
        "indices_manifest_sha256": sha256_file(args.indices),
        "train_indices_sha256": manifest["conditions"][args.condition][
            "train_indices_sha256"
        ],
        "dev_indices_sha256": manifest["dev_indices_sha256"],
        "train_rows": len(train_indices),
        "dev_rows": len(dev_indices),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_length": args.max_length,
        "seed": args.seed,
        "final_train_loss": losses[-1],
        "mean_last_25_train_loss": float(np.mean(losses[-25:])),
        "dev_mse": float(np.mean((predictions_array - labels_array) ** 2)),
        "dev_pearson": float(pearsonr(predictions_array, labels_array).statistic),
        "dev_spearman": float(spearmanr(predictions_array, labels_array).statistic),
        "elapsed_seconds": elapsed,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "gpu_name": torch.cuda.get_device_name(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        **repository_revision(Path(__file__).resolve().parents[1]),
    }
    if not all(
        math.isfinite(metrics[key])
        for key in ("final_train_loss", "dev_mse", "dev_pearson", "dev_spearman")
    ):
        raise ValueError("Nonfinite training or development metric")
    model.save_pretrained(args.output_dir / "model", safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir / "model")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
