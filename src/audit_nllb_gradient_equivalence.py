"""Compare cached autoregressive scoring with uncached teacher-forced scoring.

Uses one exact rollout saved by the final-manifest preflight. It is a numerical
and gradient diagnostic only: no reward is recomputed and no optimizer step is
performed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rights_clearance import require_rights_clearance

from nllb_policy_rl import (
    DEFAULT_MODEL, TEMPERATURE, piece_bias, policy_log_probs, score_path,
)


MANIFEST = Path("data/opus100_en_de/train_seed17.jsonl")
PREFLIGHT = Path("artifacts/runs/nllb_rl_preflight_14936831")
OUTPUT = Path("artifacts/diagnostics/nllb_gradient_equivalence_v2.json")
BETA = 0.20
MANIFEST_INDEX = 1308


def full_sequence_score(
    model: Any,
    tokenizer: Any,
    source: str,
    path: list[int],
    bias: torch.Tensor,
    device: torch.device,
    autocast: bool,
    with_grad: bool,
) -> torch.Tensor:
    eos = tokenizer.eos_token_id
    lang = tokenizer.convert_tokens_to_ids("deu_Latn")
    encoded = tokenizer(source, return_tensors="pt", truncation=True, max_length=256)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    decoder_ids = torch.tensor([[eos, lang] + path[:-1]], dtype=torch.long, device=device)
    labels = torch.tensor(path, dtype=torch.long, device=device)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            logits = model(**encoded, decoder_input_ids=decoder_ids, use_cache=False).logits
        selected_logits = logits[:, 1 : 1 + len(path), :]
        logp = policy_log_probs(selected_logits, bias, BETA)
        return logp[0].gather(-1, labels.unsqueeze(-1)).sum()


def legacy_cached_score(
    model: Any,
    tokenizer: Any,
    source: str,
    path: list[int],
    bias: torch.Tensor,
    device: torch.device,
    autocast: bool,
    with_grad: bool,
) -> torch.Tensor:
    """Pre-optimization path, which re-encoded the source at every action."""
    eos, lang = tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("deu_Latn")
    encoded = tokenizer(source, return_tensors="pt", truncation=True, max_length=256)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            first = model(**encoded, decoder_input_ids=torch.tensor([[eos]], device=device),
                          use_cache=True)
        past = first.past_key_values
        step_input = torch.tensor([[lang]], dtype=torch.long, device=device)
        terms = []
        for token in path:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                output = model(**encoded, decoder_input_ids=step_input,
                               past_key_values=past, use_cache=True)
            past = output.past_key_values
            logp = policy_log_probs(output.logits[:, -1, :], bias, BETA)
            terms.append(logp[0, token])
            step_input = torch.tensor([[token]], dtype=torch.long, device=device)
        return torch.stack(terms).sum()


def capture_gradients(
    model: Any,
    score_fn: Any,
    autocast: bool,
) -> tuple[float, dict[str, torch.Tensor], float, float]:
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    score = score_fn(autocast=autocast, with_grad=True)
    score.backward()
    grads: dict[str, torch.Tensor] = {}
    norm_sq = torch.zeros((), dtype=torch.float64)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float().cpu().clone()
        if not torch.isfinite(grad).all():
            raise FloatingPointError(f"nonfinite gradient in {name}")
        norm_sq += grad.square().sum(dtype=torch.float64)
        grads[name] = grad
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return float(score.detach().item()), grads, float(norm_sq.sqrt().item()), peak_gib


def compare_gradients(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, float]:
    if a.keys() != b.keys():
        raise ValueError("cached and uncached paths produced different gradient parameter sets")
    dot = torch.zeros((), dtype=torch.float64)
    a_sq = torch.zeros((), dtype=torch.float64)
    b_sq = torch.zeros((), dtype=torch.float64)
    diff_sq = torch.zeros((), dtype=torch.float64)
    for name in a:
        x, y = a[name], b[name]
        dot += (x * y).sum(dtype=torch.float64)
        a_sq += x.square().sum(dtype=torch.float64)
        b_sq += y.square().sum(dtype=torch.float64)
        diff_sq += (x - y).square().sum(dtype=torch.float64)
    norm_a, norm_b, norm_diff = a_sq.sqrt(), b_sq.sqrt(), diff_sq.sqrt()
    return {
        "cached_gradient_norm": float(norm_a),
        "uncached_gradient_norm": float(norm_b),
        "gradient_cosine": float(dot / (norm_a * norm_b)),
        "relative_l2_gradient_error_vs_cached": float(norm_diff / norm_a),
    }


def main() -> None:
    require_rights_clearance()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this audit on a scheduled GPU node")
    manifest_rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    source = manifest_rows[MANIFEST_INDEX]["source"]
    step = json.loads((PREFLIGHT / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    saved_path = step["sampled_token_ids"][0]
    sampled_logprob = step["sampled_logprobs"][0]
    model_dir = Path(os.environ.get("NLLB_MODEL_DIR", str(DEFAULT_MODEL)))
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.src_lang, tokenizer.tgt_lang = "eng_Latn", "deu_Latn"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32,
    ).to(device).eval()
    bias = piece_bias(model_dir, tokenizer, model.config.vocab_size, device)

    result: dict[str, Any] = {
        "purpose": "encoder-reuse equivalence diagnostic; no optimizer update",
        "preflight_job_reference": "14936831",
        "audit_job_id": os.environ.get("SLURM_JOB_ID", "not-submitted"),
        "manifest_path": str(MANIFEST),
        "manifest_index": MANIFEST_INDEX,
        "sampled_actions": len(saved_path),
        "sampled_behavior_logprob_from_preflight": sampled_logprob,
        "beta": BETA,
        "temperature": TEMPERATURE,
        "model": str(model_dir),
        "device": torch.cuda.get_device_name(device),
        "modes": {},
    }

    for mode, autocast in (("fp32", False), ("bf16_autocast", True)):
        cached_score, cached_grads, cached_gradient_norm, cached_peak = capture_gradients(
            model,
            lambda autocast, with_grad: score_path(
                model, tokenizer, source, saved_path, bias, BETA, device, autocast, with_grad,
            ),
            autocast,
        )
        uncached_score, uncached_grads, uncached_gradient_norm, uncached_peak = capture_gradients(
            model,
            lambda autocast, with_grad: full_sequence_score(
                model, tokenizer, source, saved_path, bias, device, autocast, with_grad,
            ),
            autocast,
        )
        result["modes"][mode] = {
            "optimized_cached_logprob": cached_score,
            "optimized_cached_peak_gib": cached_peak,
            "optimized_gradient_norm": cached_gradient_norm,
            "legacy_cached_logprob": None,
            "legacy_cached_peak_gib": None,
            "optimized_vs_uncached_logprob_difference": abs(cached_score - uncached_score),
            "optimized_vs_uncached_peak_gib": uncached_peak,
            "uncached_gradient_norm": uncached_gradient_norm,
            "optimized_vs_uncached_gradient": compare_gradients(cached_grads, uncached_grads),
        }
        legacy_score, legacy_grads, legacy_gradient_norm, legacy_peak = capture_gradients(
            model,
            lambda autocast, with_grad: legacy_cached_score(
                model, tokenizer, source, saved_path, bias, device, autocast, with_grad,
            ),
            autocast,
        )
        result["modes"][mode].update({
            "legacy_cached_logprob": legacy_score,
            "legacy_cached_peak_gib": legacy_peak,
            "legacy_gradient_norm": legacy_gradient_norm,
            "optimized_vs_legacy_logprob_difference": abs(cached_score - legacy_score),
            "optimized_vs_legacy_gradient": compare_gradients(cached_grads, legacy_grads),
        })
        del cached_grads, uncached_grads, legacy_grads
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    result["bf16_cached_vs_saved_sample_absolute_logprob_difference"] = abs(
        result["modes"]["bf16_autocast"]["optimized_cached_logprob"] - sampled_logprob
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
