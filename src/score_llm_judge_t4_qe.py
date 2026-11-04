"""Score frozen T4 base and perturbed candidates with an external COMET-family QE model."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "results/llm_judge_protocol_amendment_1.json"
AMENDMENT_SHA256 = "1792c5b13514f0d4e9fef23e6baadc3dd1ca2415ebd72d6e6cd82e12c4875d87"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() != AMENDMENT_SHA256:
        raise ValueError("T4 amendment differs from frozen hash")
    if args.output.exists():
        raise FileExistsError(args.output)
    amendment = json.loads(AMENDMENT.read_text())
    items_path = ROOT / amendment["items_path"]
    if hashlib.sha256(items_path.read_bytes()).hexdigest() != amendment["items_sha256"]:
        raise ValueError("T4 items differ from frozen hash")
    items = [json.loads(line) for line in items_path.open(encoding="utf-8")]

    from comet import load_from_checkpoint

    matches = glob.glob(
        str(Path.home() / ".cache/huggingface/hub" / f"models--Unbabel--{args.model_name}"
            / "snapshots/*/checkpoints/model.ckpt")
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one cached checkpoint, found {len(matches)}")
    model = load_from_checkpoint(matches[0])
    scores = np.asarray(
        model.predict(
            [{"src": i["source"], "mt": i["hypothesis"]} for i in items],
            batch_size=args.batch_size,
            gpus=1,
        ).scores,
        dtype=float,
    )
    if len(scores) != len(items) or not np.isfinite(scores).all():
        raise ValueError("Incomplete or nonfinite predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "metadata", "model": args.model_name,
                                 "items_sha256": amendment["items_sha256"]}) + "\n")
        for item, score in zip(items, scores, strict=True):
            handle.write(json.dumps({
                "record_type": "candidate",
                "pool": item["pool"],
                "segment_key": item["segment_key"],
                "base_system": item["base_system"],
                "perturbation": item["perturbation"],
                "predicted_reward": float(score),
            }) + "\n")
    print(f"{args.model_name}: scored {len(items)}")


if __name__ == "__main__":
    main()
