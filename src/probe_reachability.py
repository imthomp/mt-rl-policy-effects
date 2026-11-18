"""Reachability probe (results/reachability_probe_protocol.json).

Samples G=8 rollouts per audit prompt from the INITIAL policy with TRL's GRPO rollout
sampling (temperature 1.0, top_p 1.0, no top-k, no repetition penalty, 256 new tokens,
left padding, chat template) and reports, per failure type, the fraction of rollouts
that already exhibit it. Reward-independent: one value per (policy, direction).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/reachability_probe_protocol.json"
PROTOCOL_SHA256 = "857d4705faae2142052244d01f73c4315331713eb948f20c9580ca19d025a848"
G = 8
# Greedy initial outputs on the same 400 audit prompts, saved by each audit (same prompt sample, seed 20261109).
GREEDY = {
    ("Qwen/Qwen2.5-1.5B-Instruct", "German"): "results/policy_relative_audit_en_de.outputs.jsonl",
    ("Qwen/Qwen2.5-1.5B-Instruct", "Spanish"): "results/policy_relative_audit_en_es.outputs.jsonl",
    ("Qwen/Qwen2.5-1.5B-Instruct", "Chinese"): "results/policy_relative_audit_en_zh.outputs.jsonl",
    ("Qwen/Qwen2.5-7B-Instruct", "German"): "results/policy_relative_audit_scale_A_en_de.outputs.jsonl",
    ("Qwen/Qwen2.5-7B-Instruct", "Chinese"): "results/policy_relative_audit_scale_A_en_zh.outputs.jsonl",
    ("Qwen/Qwen2.5-7B-Instruct", "Japanese"): "results/policy_relative_audit_gen_D_en_ja.outputs.jsonl",
    ("Qwen/Qwen2.5-7B-Instruct", "Russian"): "results/policy_relative_audit_gen_D_en_ru.outputs.jsonl",
}
CODE = {"German": "en_de", "Spanish": "en_es", "Chinese": "en_zh", "Japanese": "en_ja", "Russian": "en_ru"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--target", required=True)
    args = ap.parse_args()
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("reachability protocol differs from frozen hash")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from audit_policy_relative import TEMPLATE
    from hack_guards import refusal, repetitive, untranslated

    greedy_rows = [json.loads(l) for l in (ROOT / GREEDY[(args.policy, args.target)]).open(encoding="utf-8")]
    sources = [r["source"] for r in greedy_rows]
    greedy = [r["initial_output"] for r in greedy_rows]
    if len(sources) != 400:
        raise AssertionError(f"expected 400 audit prompts, found {len(sources)}")
    template = TEMPLATE.format(lang=args.target)

    tok = AutoTokenizer.from_pretrained(args.policy, padding_side="left")
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.policy, torch_dtype=torch.bfloat16).cuda().eval()
    gen = GenerationConfig(max_new_tokens=256, do_sample=True, pad_token_id=tok.pad_token_id,
                           bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, temperature=1.0,
                           top_p=1.0, top_k=None, min_p=None, repetition_penalty=1.0)
    torch.manual_seed(20261118)
    samples = []
    for i in range(0, len(sources), 8):
        batch = [tok.apply_chat_template([{"role": "user", "content": template.format(source=s)}],
                                         tokenize=False, add_generation_prompt=True) for s in sources[i:i + 8]]
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, generation_config=gen, num_return_sequences=G)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, t in enumerate(texts):
            samples.append({"prompt_index": i + j // G, "output": t.strip()})
    if len(samples) != G * len(sources):
        raise AssertionError("sample count mismatch")
    flags = {"source_copy": [], "refusal": [], "repetition": [], "truncation_half": []}
    for s in samples:
        src, out, g = sources[s["prompt_index"]], s["output"], greedy[s["prompt_index"]]
        copy = untranslated(out, src)
        flags["source_copy"].append(copy)
        flags["refusal"].append(refusal(out))
        flags["repetition"].append(repetitive(out, args.target) and not copy)
        flags["truncation_half"].append(len(out) / max(1, len(g.strip())) < 0.5)
    result = {"policy": args.policy, "target": args.target, "lp": CODE[args.target], "G": G,
              "n_prompts": len(sources), "n_samples": len(samples), "protocol_sha256": PROTOCOL_SHA256,
              "reach": {k: sum(v) / len(v) for k, v in flags.items()},
              "prompts_with_any": {k: len({s["prompt_index"] for s, f in zip(samples, v) if f}) for k, v in flags.items()}}
    tag = f"{'7b' if '7B' in args.policy else '1p5b'}_{CODE[args.target]}"
    (ROOT / f"results/reachability_probe_{tag}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    out_dir = ROOT / "artifacts/reachability_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{tag}.samples.jsonl").open("w", encoding="utf-8") as h:
        for s in samples:
            h.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(tag, json.dumps({k: round(v, 4) for k, v in result["reach"].items()}))


if __name__ == "__main__":
    main()
