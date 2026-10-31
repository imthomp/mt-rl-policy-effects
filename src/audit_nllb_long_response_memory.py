"""Memory-only worst-case scorer/update stress test; writes no checkpoint."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rights_clearance import require_rights_clearance

from nllb_policy_rl import DEFAULT_MODEL, MAX_SOURCE, MAX_TARGET, piece_bias, score_path


MANIFEST = Path("data/opus100_en_de/train_seed17.jsonl")
PREFLIGHT = Path("artifacts/runs/nllb_rl_preflight_14936831")
BETA = 0.20


def main() -> None:
    require_rights_clearance()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this stress test on an allocated GPU")
    physical_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocator_limit_gib = float(os.environ.get("NLLB_ALLOCATOR_LIMIT_GIB", "0"))
    if allocator_limit_gib > 0:
        if allocator_limit_gib >= physical_memory_gib:
            raise ValueError("allocator cap must be lower than physical GPU memory")
        torch.cuda.set_per_process_memory_fraction(allocator_limit_gib / physical_memory_gib, 0)
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, local_files_only=True)
    tokenizer.src_lang, tokenizer.tgt_lang = "eng_Latn", "deu_Latn"
    measured = []
    for index, row in enumerate(rows):
        tokenized = tokenizer(row["source"], truncation=True, max_length=MAX_SOURCE)
        measured.append((len(tokenized["input_ids"]), index, row["source"]))
    selected = sorted(measured, reverse=True)[:8]

    model = AutoModelForSeq2SeqLM.from_pretrained(
        DEFAULT_MODEL, local_files_only=True, dtype=torch.float32,
    ).cuda().eval()
    model.config.use_cache = False
    bias = piece_bias(DEFAULT_MODEL, tokenizer, model.config.vocab_size, torch.device("cuda"))
    step = json.loads((PREFLIGHT / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ordinary_path = step["sampled_token_ids"][0]
    specials = set(tokenizer.all_special_ids)
    ordinary_token = next(token for token in ordinary_path if token not in specials)
    path = [ordinary_token] * MAX_TARGET  # cap-length; deliberately no EOS (memory stress only)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    scores = []
    for _, _, source in selected:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            score = score_path(
                model, tokenizer, source, path, bias, BETA, torch.device("cuda"),
                autocast=True, with_grad=True,
            )
        scores.append(float(score.detach().item()))
        (-score / len(selected)).backward()
    torch.cuda.synchronize()
    peak_before_step = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_before_step = torch.cuda.max_memory_reserved() / (1024**3)
    gradient_norm_sq = torch.zeros((), device="cuda", dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("non-finite gradient")
            gradient_norm_sq += parameter.grad.double().square().sum()
    optimizer.step()  # one throwaway update, solely to allocate AdamW states
    torch.cuda.synchronize()
    peak_with_step = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_with_step = torch.cuda.max_memory_reserved() / (1024**3)
    # Match the production trainer's update-norm diagnostic, which runs after
    # every optimizer step and can itself create large temporary tensors.
    update_norm_sq = torch.zeros((), device="cuda")
    for parameter in model.parameters():
        state = optimizer.state.get(parameter)
        if not state:
            continue
        step_number = int(state["step"].item())
        bias_correction1 = 1.0 - 0.9 ** step_number
        bias_correction2_sqrt = math.sqrt(1.0 - 0.999 ** step_number)
        delta = (state["exp_avg"] / bias_correction1) / (
            state["exp_avg_sq"].sqrt() / bias_correction2_sqrt + 1e-8
        )
        update_norm_sq += delta.float().square().sum()
    update_norm = float((1e-6 * update_norm_sq.sqrt()).item())
    torch.cuda.synchronize()
    peak_with_update_norm_gib = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_with_update_norm_gib = torch.cuda.max_memory_reserved() / (1024**3)
    result = {
        "purpose": "worst-case memory stress only; artificial cap-length token paths; no checkpoint",
        "gpu": torch.cuda.get_device_name(),
        "physical_gpu_memory_gib": physical_memory_gib,
        "pytorch_allocator_limit_gib": allocator_limit_gib or None,
        "manifest": str(MANIFEST),
        "selected_longest_source_rows": [
            {"index": index, "source_tokens": token_count} for token_count, index, _ in selected
        ],
        "target_actions_per_response": MAX_TARGET,
        "path_construction": "repeat one non-special token from saved real rollout; no EOS",
        "batch_size": len(selected),
        "autocast": "bfloat16; FP32 model parameters, gradients, and AdamW state",
        "finite_gradients": True,
        "gradient_norm": float(gradient_norm_sq.sqrt().item()),
        "peak_allocated_before_optimizer_step_gib": peak_before_step,
        "peak_reserved_before_optimizer_step_gib": reserved_before_step,
        "peak_allocated_with_optimizer_step_gib": peak_with_step,
        "peak_reserved_with_optimizer_step_gib": reserved_with_step,
        "production_update_norm": update_norm,
        "peak_allocated_with_update_norm_gib": peak_with_update_norm_gib,
        "peak_reserved_with_update_norm_gib": reserved_with_update_norm_gib,
        "fits_80_gib_with_10_percent_headroom": peak_with_update_norm_gib <= 72.0,
    }
    output = Path(f"artifacts/diagnostics/nllb_long_response_memory_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
