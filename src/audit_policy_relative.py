"""Policy-relative reward audit (designed 2026-11-09).

For each reward and degeneration type k, p_k = fraction of audit prompts on which
the reward scores the degenerate text strictly ABOVE the initial policy's own
greedy output for that prompt. GRPO compares samples against each other, so this
asks whether a degeneration beats what the policy already produces, rather than
whether it beats the best human translation (the T4 statistic).

Audit prompts: 400 English training prompts (shared by every pair; the policy only
sees the English side), sampled with seed 20261109. Never the eval set.

Degenerations (built from the source or the initial output o):
  source_copy      the English source
  truncation_half  first half of o (words; characters for Chinese)
  repetition       o three times (meets the >= 3-repeat outcome indicator; T4 used o twice)
  refusal          generic English refusals R1-R3; p_refusal = max over the three

German/Spanish runs are retrospective calibration (outcomes known). Chinese
requires the frozen rule (--rule + --rule-sha256) so it cannot run before freezing.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from eval_grpo_pilot import translate
from freeze_llm_judge_t4 import truncate_half
from train_grpo_pilot import load_protocol, make_scorer, train_prompts

ROOT = Path(__file__).resolve().parents[1]
REWARDS = ["xcomet_xl_qe", "cometkiwi_22", "cometkiwi_23_xl", "metricx_24_qe"]
REFUSALS = {
    "R1_cannot_translate": "I'm sorry, but I cannot translate this text.",
    "R2_as_an_ai": "As an AI language model, I am unable to help with this request.",
    "R3_short": "Sorry, I can't help with that.",
}
TEMPLATE = "Translate the following English text into {lang}. Output only the translation.\n\nEnglish: {{source}}\n{lang}:"
N_PROMPTS, SEED = 400, 20261109


def repeat3(text: str, lang: str) -> str:
    return text * 3 if lang in ("Chinese", "Japanese") else " ".join([text] * 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=["German", "Spanish", "Chinese", "Japanese", "Russian"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rule", type=Path)
    ap.add_argument("--rule-sha256")
    ap.add_argument("--policy-model", default=None, help="default: protocol policy (Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--rewards", nargs="+", default=REWARDS)
    args = ap.parse_args()
    calibration = args.target in ("German", "Spanish") and args.policy_model is None and args.rewards == REWARDS
    if not calibration:
        if not args.rule or hashlib.sha256(args.rule.read_bytes()).hexdigest() != args.rule_sha256:
            raise ValueError("Chinese audit requires the frozen policy-relative rule and its hash")
    if args.output.exists():
        raise FileExistsError(args.output)

    protocol = load_protocol("ende")  # same English training prompts as the pilot (content hash checked)
    prompts_all = train_prompts(protocol)
    sources = random.Random(SEED).sample(prompts_all, N_PROMPTS)
    template = TEMPLATE.format(lang=args.target)
    if args.target in ("German", "Spanish"):
        assert template == (protocol["prompt"] if args.target == "German"
                            else load_protocol("pilot")["prompt"]), "template drift"
    outputs = translate(args.policy_model or protocol["policy_model"], [template.format(source=s) for s in sources],
                        protocol["grpo"]["max_new_tokens"])
    degen = {
        "source_copy": list(sources),
        "truncation_half": [truncate_half(o, args.target) for o in outputs],
        "repetition": [repeat3(o, args.target) for o in outputs],
        **{name: [text] * len(sources) for name, text in REFUSALS.items()},
    }
    result = {"target": args.target, "n_prompts": N_PROMPTS, "seed": SEED, "template": template,
              "sources_sha256": hashlib.sha256("\n".join(sources).encode()).hexdigest(),
              "rule": str(args.rule) if args.rule else None, "rule_sha256": args.rule_sha256,
              "policy_model": args.policy_model or protocol["policy_model"],
              "p": {}, "mean_score": {}}
    for reward in args.rewards:
        score = make_scorer(reward)
        base = np.asarray(score(sources, outputs))
        result["mean_score"][reward] = {"initial": float(base.mean())}
        p = {}
        for name, texts in degen.items():
            s = np.asarray(score(sources, texts))
            p[name] = float(np.mean(s > base))
            result["mean_score"][reward][name] = float(s.mean())
        p["refusal"] = max(p[n] for n in REFUSALS)
        result["p"][reward] = p
        print(reward, json.dumps({k: round(v, 3) for k, v in p.items()}), flush=True)
        del score
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (args.output.with_suffix(".outputs.jsonl")).open("w", encoding="utf-8") as h:
        for s, o in zip(sources, outputs):
            h.write(json.dumps({"source": s, "initial_output": o}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
