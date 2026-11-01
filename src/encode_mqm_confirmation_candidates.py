"""Encode frozen MQM source--candidate pairs without reading human outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_text_policy_bandit import (
    ENCODER_BATCH_SIZE,
    ENCODER_NAME,
    ENCODER_REVISION,
    MAX_LENGTH,
    encode_pairs,
)
from score_mqm_confirmation_candidates import load_candidates


def arrange(rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[str], list[list[str]]]:
    contexts = sorted({row["segment_key"] for row in rows})
    actions = sorted({row["system_id"] for row in rows})
    keyed = {(row["segment_key"], row["system_id"]): row for row in rows}
    expected = {(context, action) for context in contexts for action in actions}
    if set(keyed) != expected:
        raise ValueError("Confirmation candidate environment is not rectangular")
    sources = []
    candidates = []
    for context in contexts:
        source_values = {keyed[(context, action)]["source"] for action in actions}
        if len(source_values) != 1:
            raise ValueError(f"Source text differs across candidates: {context}")
        sources.append(next(iter(source_values)))
        candidates.append([keyed[(context, action)]["hypothesis"] for action in actions])
    return contexts, actions, sources, candidates


def text_digest(sources: list[str], candidates: list[list[str]]) -> str:
    digest = hashlib.sha256()
    for source, rows in zip(sources, candidates, strict=True):
        digest.update(source.encode("utf-8"))
        for candidate in rows:
            digest.update(b"\0")
            digest.update(candidate.encode("utf-8"))
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import run_text_policy_bandit as text_bandit

    text_bandit.torch = torch

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    metadata, rows = load_candidates(args.source, args.manifest, args.dataset_name)
    contexts, actions, sources, candidates = arrange(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = encode_pairs(
        sources,
        candidates,
        model_name=ENCODER_NAME,
        revision=ENCODER_REVISION,
        max_length=MAX_LENGTH,
        batch_size=ENCODER_BATCH_SIZE,
        device=device,
    )
    digest = text_digest(sources, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        text_sha256=np.asarray(digest),
        contexts=np.asarray(contexts),
        actions=np.asarray(actions),
    )
    result = {
        **metadata,
        "output": str(args.output),
        "embedding_shape": list(embeddings.shape),
        "text_sha256": digest,
        "encoder_name": ENCODER_NAME,
        "encoder_revision": ENCODER_REVISION,
        "max_length": MAX_LENGTH,
        "outcome_fields_accessed": [],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
