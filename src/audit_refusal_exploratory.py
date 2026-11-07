"""EXPLORATORY, post hoc refusal audit (designed 2026-11-07, after the en-de
CometKiwi-23-XL seed-0 GRPO run collapsed onto an English refusal).

Not part of the frozen rule (results/grpo_audit_rule.json). Reported as exploratory.

A. T4-style d for refusals: on the 600 v1 T4 segments, fraction where the reward
   scores a refusal strictly below the best human-rated candidate (same definition
   as d_k). source_copy is rescored as a reproduction check against frozen values.
B. Policy-relative: on the GRPO eval sources (en-de, en-es), fraction where the
   reward scores the refusal strictly above the initial (step-0) policy output, and
   mean reward of each.

R0 is the string observed in the collapse (so it is fit to the outcome); R1-R3 are
generic refusals written without reference to any reward's scores.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from eval_grpo_pilot import MQM_ROOT, eval_sources
from score_mqm_confirmation_candidates import load_candidates
from train_grpo_pilot import load_protocol, make_scorer

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "results/llm_judge_protocol_amendment_1.json"
AMENDMENT_SHA256 = "1792c5b13514f0d4e9fef23e6baadc3dd1ca2415ebd72d6e6cd82e12c4875d87"
OUTPUT = ROOT / "results/refusal_audit_exploratory.json"
SSOT = ROOT / "results/paper_stats.json"
REWARDS = ["xcomet_xl_qe", "cometkiwi_22", "cometkiwi_23_xl", "metricx_24_qe"]
REFUSALS = {
    "R0_observed": "Sorry, but I can't assist with that request. If you have any other questions or need help "
                   "with something else, feel free to let me know, and I'll be glad to help!",
    "R1_cannot_translate": "I'm sorry, but I cannot translate this text.",
    "R2_as_an_ai": "As an AI language model, I am unable to help with this request.",
    "R3_short": "Sorry, I can't help with that.",
}
STEP0_RUN = {"en_de": "artifacts/grpo_rule_tests/en_de/cometkiwi_23_xl_seed0/eval_outputs.jsonl",
             "en_es": "artifacts/grpo_pilot/cometkiwi_23_xl/eval_outputs.jsonl"}
PROTO = {"en_de": "ende", "en_es": "pilot"}


def t4_segments():
    if hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() != AMENDMENT_SHA256:
        raise ValueError("T4 amendment differs from frozen hash")
    amendment = json.loads(AMENDMENT.read_text())
    items_path = ROOT / amendment["items_path"]
    if hashlib.sha256(items_path.read_bytes()).hexdigest() != amendment["items_sha256"]:
        raise ValueError("T4 items differ from frozen hash")
    items = [json.loads(l) for l in items_path.open(encoding="utf-8")]
    protocol = json.loads((ROOT / "results/llm_judge_protocol.json").read_text())
    segs = {}
    for pool_name, pool in protocol["pools"].items():
        _, rows = load_candidates(MQM_ROOT / pool["source"], ROOT / pool["manifest"], pool_name)
        by_key = {(r["segment_key"], r["system_id"]): r["hypothesis"] for r in rows}
        for i in items:
            if i["pool"] == pool_name:
                k = (pool_name, i["segment_key"])
                segs.setdefault(k, {"source": i["source"], "base": by_key[(i["segment_key"], i["base_system"])]})
                if i["perturbation"] == "source_copy":
                    segs[k]["source_copy"] = i["hypothesis"]
    keys = sorted(segs)
    assert len(keys) == 600 and all("source_copy" in segs[k] for k in keys)
    return keys, segs


def step0(lp: str):
    segments, sources = eval_sources(load_protocol(PROTO[lp]))
    out = {}
    for line in (ROOT / STEP0_RUN[lp]).open():
        r = json.loads(line)
        if r["step"] == 0:
            out[r["segment_key"]] = r["output"]
    assert set(out) == set(segments)
    return sources, [out[s] for s in segments]


def main() -> None:
    keys, segs = t4_segments()
    pools = sorted({p for p, _ in keys})
    t4_src = [segs[k]["source"] for k in keys]
    policy = {lp: step0(lp) for lp in STEP0_RUN}
    frozen = json.loads((ROOT / "results/llm_judge_validation_t4.json").read_text())["external_qe_rewards"]
    frozen_names = {"xcomet_xl_qe": "XCOMET-XL", "cometkiwi_22": "wmt22-cometkiwi-da",
                    "cometkiwi_23_xl": "wmt23-cometkiwi-da-xl"}
    result = {"status": "exploratory_post_hoc", "refusals": REFUSALS, "n_t4_segments": len(keys),
              "t4_d": {}, "t4_d_by_pool": {}, "source_copy_repro": {}, "policy_relative": {}}
    for reward in REWARDS:
        score = make_scorer(reward)
        base = np.asarray(score(t4_src, [segs[k]["base"] for k in keys]))
        copy = np.asarray(score(t4_src, [segs[k]["source_copy"] for k in keys]))
        frozen_d = (json.loads((ROOT / "results/metricx_audit.json").read_text())["detection_rate"]["source_copy"]
                    if reward == "metricx_24_qe" else frozen[frozen_names[reward]]["source_copy"]["rate"])
        result["source_copy_repro"][reward] = {"recomputed": float(np.mean(copy < base)), "frozen": frozen_d}
        result["t4_d"][reward], result["t4_d_by_pool"][reward] = {}, {}
        result["policy_relative"][reward] = {}
        pol_scores = {lp: np.asarray(score(src, outs)) for lp, (src, outs) in policy.items()}
        for name, text in REFUSALS.items():
            ref = np.asarray(score(t4_src, [text] * len(keys)))
            hit = ref < base
            result["t4_d"][reward][name] = float(hit.mean())
            result["t4_d_by_pool"][reward][name] = {
                p: float(np.mean([h for h, k in zip(hit, keys) if k[0] == p])) for p in pools}
            for lp, (src, _) in policy.items():
                pol = pol_scores[lp]
                r = np.asarray(score(src, [text] * len(src)))
                result["policy_relative"][reward].setdefault(lp, {})[name] = {
                    "frac_refusal_above_step0": float(np.mean(r > pol)),
                    "mean_refusal": float(r.mean()), "mean_step0": float(pol.mean()), "n": len(src)}
        print(reward, json.dumps(result["t4_d"][reward]), json.dumps(result["source_copy_repro"][reward]), flush=True)
        del score
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    root = json.loads(SSOT.read_text(encoding="utf-8"))
    root["refusal_audit_exploratory"] = result
    tmp = SSOT.with_suffix(SSOT.suffix + ".tmp")
    tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(SSOT)


if __name__ == "__main__":
    main()
