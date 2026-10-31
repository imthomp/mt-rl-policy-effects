"""Check a saved NLLB checkpoint's deterministic output and optimizer reload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nllb_policy_rl import (
    DEFAULT_MODEL,
    checkpoint_probe,
    file_sha256,
    piece_bias,
)
from reproducibility import repository_metadata
from rights_clearance import clearance_metadata, require_rights_clearance


def recursively_equal(expected: Any, actual: Any, location: str = "state") -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(
            expected.detach().cpu(), actual.detach().cpu(),
        ):
            raise AssertionError(f"restored optimizer tensor differs at {location}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f"restored optimizer keys differ at {location}")
        for key in expected:
            recursively_equal(expected[key], actual[key], f"{location}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise AssertionError(f"restored optimizer sequence differs at {location}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            recursively_equal(left, right, f"{location}[{index}]")
    elif expected != actual:
        raise AssertionError(f"restored optimizer value differs at {location}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    rights_record = require_rights_clearance()
    repo_metadata = repository_metadata()
    if repo_metadata["git_worktree_dirty"]:
        raise RuntimeError("checkpoint verification requires a clean Git worktree")
    if not repo_metadata["git_commit_sha"]:
        raise RuntimeError("checkpoint verification requires a Git commit")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU allocation required")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)

    checkpoint_path = args.run_dir / "final.pt"
    integrity_path = args.run_dir / "final_checkpoint_integrity.json"
    config_path = args.run_dir / "run_config.json"
    output_path = args.run_dir / "checkpoint_roundtrip_verification.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    if not checkpoint_path.is_file() or not integrity_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("run directory must contain final.pt, integrity metadata, and run_config.json")

    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if file_sha256(checkpoint_path) != integrity["sha256"]:
        raise ValueError("checkpoint SHA-256 does not match its integrity record")
    if checkpoint_path.stat().st_size != integrity["size_bytes"]:
        raise ValueError("checkpoint size does not match its integrity record")
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    if run_config.get("mode") != "one-update-checkpoint-roundtrip":
        raise ValueError("checkpoint verifier requires a --checkpoint-check run")
    if run_config.get("model_snapshot_revision") != args.model_dir.name:
        raise ValueError("checkpoint model snapshot differs from the requested local model revision")

    device = torch.device("cuda")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state.get("probe"), dict):
        raise ValueError("checkpoint lacks its deterministic model-output probe")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.src_lang, tokenizer.tgt_lang = "eng_Latn", "deu_Latn"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.float32,
    ).to(device).eval()
    model.config.use_cache = False
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint parameter mismatch: missing={missing}, unexpected={unexpected}")
    bias = piece_bias(args.model_dir, tokenizer, model.config.vocab_size, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
    optimizer.load_state_dict(state["optimizer"])
    restored_optimizer = optimizer.state_dict()
    recursively_equal(state["optimizer"], restored_optimizer)
    optimizer_state_parameter_count = len(restored_optimizer["state"])
    if optimizer_state_parameter_count == 0:
        raise ValueError("checkpoint contains no restored AdamW state")

    observed_probe = checkpoint_probe(model, tokenizer, bias, device)
    expected_probe = state["probe"]
    if observed_probe["generated_sequence_ids"] != expected_probe["generated_sequence_ids"]:
        raise AssertionError("reloaded model's deterministic probe token IDs differ")
    logprob_error = abs(observed_probe["sequence_logprob"] - expected_probe["sequence_logprob"])
    if logprob_error > 1e-5:
        raise AssertionError(f"reloaded model probe log-probability error {logprob_error} exceeds 1e-5")

    report = {
        **repo_metadata,
        "rights_clearance": clearance_metadata(rights_record),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": integrity["sha256"],
        "training_git_commit": run_config["git_commit_sha"],
        "model_id": "facebook/nllb-200-distilled-600M",
        "model_revision": args.model_dir.name,
        "model_snapshot_hash": args.model_dir.name,
        "model_output_token_ids_match": True,
        "probe_logprob_absolute_error": logprob_error,
        "probe_logprob_tolerance": 1e-5,
        "optimizer_state_exactly_restored": True,
        "optimizer_state_parameter_count": optimizer_state_parameter_count,
        "resumed_update_tested": False,
        "interpretation": "Serialization, model-output reload, and optimizer-state restoration passed. This check does not test resumed optimizer updates; this trainer has no resume mode.",
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
