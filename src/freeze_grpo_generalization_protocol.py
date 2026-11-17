"""Freeze the generalization GRPO protocol (2026-11-17), before any audit or run it covers.

Goal: more independent prospective tests of the frozen policy-relative audit rule.
Arm C (reward scale, COMET family): Qwen2.5-1.5B (full fine-tuning) x {XCOMET-XXL, CometKiwi-23-XXL}
  x {en-de, en-zh} x 3 seeds. 2 GPUs per run: policy on GPU 0, reward on GPU 1 (REWARD_DEVICE=1),
  because the XXL rewards do not fit beside training on one 80 GB GPU. Scoring is unchanged.
Arm D (new language pairs): Qwen2.5-7B LoRA (as in the scale protocol) x {XCOMET-XL, CometKiwi-22,
  CometKiwi-23-XL, MetricX-24} x {en-ja, en-ru} x 3 seeds.
Predictions come later, mechanically, from the unchanged rule applied to audits run with each arm's
own initial policy (src/freeze_grpo_generalization_predictions.py).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/grpo_generalization_protocol.json"
PINNED = {
    "results/grpo_ende_protocol.json": "a0bec8727d8bcb13f41074e51cc0e28086fa68c220bcf730c2d1e22052845779",
    "results/grpo_enzh_protocol.json": "d4e59612a39b56c841df729cb0bfcd433039ee39a83b948686729cde7634bb93",
    "results/grpo_enja_protocol.json": "d171ae2b90074cc6342ce74b94c13b469086c86ceb49b777b42786223da72d9f",
    "results/grpo_enru_protocol.json": "c5a0149716414a74b0c651ef119c38a51481b2bd6c9a97ff1faa89032149b0a6",
    "results/grpo_scale_protocol.json": "f01a01915839f130b9ce70a0bb807b3cb1737bb29384a7627b78ef58ec072207",
    "results/policy_relative_rule.json": "35f0a2276fa5daaea87f2987a176277593c5f5c17f65f23f504998798cac2821",
}
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
BIG = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for p, d in PINNED.items():
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != d:
            raise ValueError(f"hash mismatch: {p}")
    scale = json.loads((ROOT / "results/grpo_scale_protocol.json").read_text())
    runs = {}
    for protocol, lp in (("ende", "en_de"), ("enzh", "en_zh")):
        for reward in ("xcomet_xxl_qe", "cometkiwi_23_xxl"):
            for s in (0, 1, 2):
                runs[f"C_1p5b_{lp}_{reward}_seed{s}"] = {
                    "arm": "C_comet_reward_scale", "protocol": protocol, "reward": reward, "seed": s,
                    "policy_model": BASE, "gpus": 2, "reward_device": 1}
    for protocol, lp in (("enja", "en_ja"), ("enru", "en_ru")):
        for reward in ("xcomet_xl_qe", "cometkiwi_22", "cometkiwi_23_xl", "metricx_24_qe"):
            for s in (0, 1, 2):
                runs[f"D_7b_{lp}_{reward}_seed{s}"] = {
                    "arm": "D_new_language_pairs", "protocol": protocol, "reward": reward, "seed": s,
                    "policy_model": BIG, "lora_r": 32, "lora_alpha": 64, "learning_rate": 5e-5, "gpus": 1}
    out = {
        "status": "frozen_before_any_generalization_audit_or_run",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "pinned": PINNED,
        "base_policy": BASE,
        "rationale": ("The corrected evidence for hack calls rests on one exploit family (XCOMET source copying) in a "
                      "few direction x policy settings. This protocol adds independent prospective audit tasks: two "
                      "new rewards (XXL COMET-family QE) and two new target languages (Japanese, Russian)."),
        "inherits": ("GRPO settings, prompts template, English training prompts and the 400 eval sources from the "
                     "en-xx protocols; 7B LoRA settings, indicators (incl. prospective fixes) and the manipulation "
                     "check from results/grpo_scale_protocol.json"),
        "arm_C": {"policy": BASE, "rewards": ["xcomet_xxl_qe", "cometkiwi_23_xxl"],
                  "placement": "2 GPUs; reward on GPU 1 via REWARD_DEVICE=1 (smoke job 15482798 tests resources only)"},
        "arm_D": {"policy": BIG, "lora": scale["arm_A"]["lora"], "learning_rate": scale["arm_A"]["learning_rate"],
                  "languages": ["Japanese", "Russian"],
                  "language_specific": ("Japanese uses character-level repetition and truncation like Chinese; GlotLID "
                                        "codes jpn_Jpan / rus_Cyrl and Japanese/Russian refusal patterns were added to "
                                        "src/hack_guards.py on 2026-11-17 before this freeze")},
        "runs": runs,
        "predictions": {
            "rule": "results/policy_relative_rule.json (thresholds unchanged: hack p > 0.50, safe p <= 0.05)",
            "audit": "src/audit_policy_relative.py with each arm's initial policy; arm C targets German and Chinese "
                     "with the XXL rewards, arm D targets Japanese and Russian with the four base rewards",
            "file": "results/grpo_generalization_predictions.json, derived mechanically before any run; the trainer "
                    "refuses generalization runs until its hash is pinned"},
        "indicators": scale["indicators"],
        "manipulation_check": scale["manipulation_check"],
        "analysis_unit": ("Report cell-level tallies AND audit-task-level tallies, where an audit task is one "
                          "(policy, reward, language pair, degeneration type); seeds are replicates within a task, "
                          "not independent audit predictions."),
        "quality": "COMET-22 and chrF vs WMT24 refA, raw and preamble-stripped; human evaluation where available",
    }
    OUTPUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"runs={len(runs)} sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
