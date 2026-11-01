"""Score frozen MQM confirmation pools with reference-free COMET controls."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from audit_reward_holdouts import sha256_file
from score_mqm_confirmation_candidates import load_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from comet import load_from_checkpoint

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    matches = glob.glob(
        str(
            Path.home()
            / ".cache/huggingface/hub"
            / f"models--Unbabel--{args.model_name}"
            / "snapshots/*/checkpoints/model.ckpt"
        )
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one cached COMET checkpoint, found {len(matches)}")
    checkpoint = Path(matches[0])
    metadata, rows = load_candidates(args.source, args.manifest, args.dataset_name)
    model = load_from_checkpoint(str(checkpoint))
    scores = model.predict(
        [{"src": row["source"], "mt": row["hypothesis"]} for row in rows],
        batch_size=args.batch_size,
        gpus=1,
    ).scores
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(rows) or not np.isfinite(scores).all():
        raise ValueError("Incomplete or nonfinite COMET predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata.update(
        {
            "record_type": "metadata",
            "condition": args.model_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "batch_size": args.batch_size,
        }
    )
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        for row, score in zip(rows, scores, strict=True):
            output_row = {
                key: value
                for key, value in row.items()
                if key not in {"source", "hypothesis", "reference"}
            }
            output_row.update(
                {"record_type": "candidate", "predicted_reward": float(score)}
            )
            handle.write(json.dumps(output_row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
