"""Evaluate GRPO pilot checkpoints on the frozen eval set (greedy decoding)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hack_guards import flags, off_target
from score_mqm_confirmation_candidates import load_candidates
from train_grpo_pilot import NEW_REWARDS, PROTOCOLS, REWARD_CHECKPOINTS, load_protocol, make_scorer

ROOT = Path(__file__).resolve().parents[1]
MQM_ROOT = Path.home() / "projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation"


def eval_sources(protocol: dict) -> tuple[list[str], list[str]]:
    if "sources" in protocol["eval_set"]:  # frozen inline (en-de)
        segments, sources = protocol["eval_set"]["segments"], protocol["eval_set"]["sources"]
        if hashlib.sha256("\n".join(sources).encode()).hexdigest() != protocol["eval_set"]["sources_sha256"]:
            raise ValueError("Eval sources differ from frozen hash")
        return segments, sources
    judge = json.loads((ROOT / "results/llm_judge_protocol.json").read_text())
    pool = judge["pools"]["wmt24_en_es_mqm"]
    _, rows = load_candidates(MQM_ROOT / pool["source"], ROOT / pool["manifest"], "wmt24_en_es_mqm")
    by_segment = {r["segment_key"]: r["source"] for r in rows}
    segments = protocol["eval_set"]["segments"]
    sources = [by_segment[s] for s in segments]
    if hashlib.sha256("\n".join(sources).encode()).hexdigest() != protocol["eval_set"]["sources_sha256"]:
        raise ValueError("Eval sources differ from frozen hash")
    return segments, sources


def translate(model_path: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().eval()
    outputs = []
    for i in range(0, len(prompts), 32):
        batch = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts[i:i + 32]]
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outputs += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]
    del model
    torch.cuda.empty_cache()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-name", default="pilot", choices=sorted(PROTOCOLS))
    parser.add_argument("--steps", type=int, default=None,
                        help="en-de amendment 2 follow-up: evaluate 0..steps every 100")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol_name)
    target_lang = protocol.get("target_lang", "Spanish")
    g = protocol["grpo"]
    segments, sources = eval_sources(protocol)
    prompts = [protocol["prompt"].format(source=s) for s in sources]

    run_meta = json.loads((args.run_dir / "pilot_run.json").read_text()) if (args.run_dir / "pilot_run.json").exists() else {}
    policy_model = run_meta.get("policy_model") or protocol["policy_model"]
    reward_names = list(REWARD_CHECKPOINTS) + ([run_meta["reward"]] if run_meta.get("reward") in NEW_REWARDS else [])
    checkpoints = {}
    steps = g["checkpoints_evaluated"] if args.steps is None else list(range(0, args.steps + 1, 100))
    for step in steps:
        path = policy_model if step == 0 else str(args.run_dir / f"checkpoint-{step}")
        if step and not Path(path).exists():
            raise FileNotFoundError(path)
        checkpoints[step] = translate(path, prompts, g["max_new_tokens"])

    scores = {}
    for name in reward_names:
        scorer = make_scorer(name)
        for step, outs in checkpoints.items():
            scores[(name, step)] = scorer(sources, outs)
        del scorer

    summary, rows = {}, []
    for step, outs in checkpoints.items():
        f = [flags(o, s, target_lang) | {"off_target": off_target(o, target_lang)}
             for o, s in zip(outs, sources)]
        summary[step] = {
            "untranslated_rate": float(np.mean([x["untranslated"] for x in f])),
            "truncation_rate": float(np.mean([x["truncated"] for x in f])),
            "repetition_rate": float(np.mean([x["repetitive"] for x in f])),
            "off_target_rate": float(np.mean([x["off_target"] for x in f])),
            "mean_reward": {n: float(np.mean(scores[(n, step)])) for n in reward_names},
            "mean_length_ratio": float(np.mean([len(o) / max(1, len(s)) for o, s in zip(outs, sources)])),
        }
        for i, seg in enumerate(segments):
            rows.append({"step": step, "segment_key": seg, "output": outs[i], **f[i],
                         **{f"reward_{n}": scores[(n, step)][i] for n in reward_names}})
    with (args.run_dir / "eval_outputs.jsonl").open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.run_dir / "eval_summary.json").write_text(json.dumps(
        {"protocol_name": args.protocol_name, "protocol_sha256": PROTOCOLS[args.protocol_name][1],
         "summary": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
