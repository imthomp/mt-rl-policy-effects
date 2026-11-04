"""Freeze LLM-judge validation protocol v2 on segments disjoint from v1.

v2 was designed after v1 failed (results/llm_judge_validation*.json).
Changes from v1, all informed by v1 results:
  * gpt-oss-20b gets max_tokens 4096 (v1: 5,480/8,600 outputs hit 1024 while
    reasoning);
  * each judge is also evaluated as a hybrid with the deterministic guards in
    hack_guards.py (quality - 25 if any guard fires), because v1 judges had
    blind spots for degenerate outputs.
To avoid tuning on seen data, v2 is evaluated only on segments not sampled in
v1.  Acceptance thresholds are unchanged from v1.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from freeze_llm_judge_t4 import PERTURBATIONS, join, truncate_half
from mqm_human_scores import load_mqm_quality
from score_mqm_confirmation_candidates import load_candidates

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "results/llm_judge_protocol.json"
V1_SHA256 = "172360f175fe106d44d4c666582e9a56a5eceb3ba6ff4d785b9618a0a9786013"
OUTPUT = ROOT / "results/llm_judge_protocol_v2.json"
ITEMS = ROOT / "artifacts/llm_judge/v2/t4_items.jsonl"
MQM_ROOT = Path.home() / "projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation"
SEED = 20261105
SEGMENTS_PER_POOL = 150


def main() -> None:
    if hashlib.sha256(V1.read_bytes()).hexdigest() != V1_SHA256:
        raise ValueError("v1 protocol hash mismatch")
    for path in (OUTPUT, ITEMS):
        if path.exists():
            raise FileExistsError(path)
    v1 = json.loads(V1.read_text())
    rng = np.random.default_rng(SEED)
    pools, items = {}, []
    for name, v1_pool in v1["pools"].items():
        _, rows = load_candidates(MQM_ROOT / v1_pool["source"], ROOT / v1_pool["manifest"], name)
        used = set(v1_pool["sampled_segments"])
        available = sorted({r["segment_key"] for r in rows} - used)
        chosen = sorted(rng.choice(len(available), size=SEGMENTS_PER_POOL, replace=False).tolist())
        sampled = [available[i] for i in chosen]
        systems = v1_pool["systems"]
        pools[name] = {
            **{k: v1_pool[k] for k in ("source", "manifest", "source_lang", "target_lang",
                                       "source_sha256", "candidate_content_sha256", "systems")},
            "segments_available_after_v1": len(available),
            "sampled_segments": sampled,
            "sampled_candidates": len(sampled) * len(systems),
        }
        # T4 items on the v2 sample (same construction as amendment 1).
        text = {(r["segment_key"], r["system_id"]): r for r in rows}
        human, _ = load_mqm_quality(MQM_ROOT / v1_pool["source"], sampled, systems)
        bases = [text[(s, systems[b])] for s, b in zip(sampled, np.argmax(human, axis=1))]
        shift = int(rng.integers(1, len(bases)))
        lang = v1_pool["target_lang"]
        for i, base in enumerate(bases):
            donor = bases[(i + shift) % len(bases)]["hypothesis"]
            hyp = base["hypothesis"]
            variants = {
                "source_copy": base["source"],
                "truncation_half": truncate_half(hyp, lang),
                "appended_unrelated": join(hyp, donor, lang),
                "unrelated_fluent": donor,
                "repetition": join(hyp, hyp, lang),
            }
            for kind in PERTURBATIONS:
                items.append({"pool": name, "segment_key": base["segment_key"],
                              "base_system": base["system_id"], "perturbation": kind,
                              "source": base["source"], "hypothesis": variants[kind]})
    ITEMS.parent.mkdir(parents=True, exist_ok=True)
    with ITEMS.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    items_sha = hashlib.sha256(ITEMS.read_bytes()).hexdigest()

    protocol = {
        "version": 2,
        "status": "frozen_before_any_v2_judge_output",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes": {"path": "results/llm_judge_protocol.json", "sha256": V1_SHA256,
                       "v1_outcome": "no judge accepted"},
        "changes_from_v1_and_why": __doc__,
        "pools": pools,
        "sample_seed": SEED,
        "segments_per_pool": SEGMENTS_PER_POOL,
        "judges": v1["judges"],
        "prompt_template": v1["prompt_template"],
        "prompt_template_sha256": v1["prompt_template_sha256"],
        "decoding_by_judge": {
            "qwen2.5-32b": v1["decoding"],
            "gemma-2-27b": v1["decoding"],
            "gpt-oss-20b": {"temperature": 0.0, "max_tokens": 4096, "max_model_len": 8192},
        },
        "scoring": v1["scoring"],
        "guards": {
            "module": "src/hack_guards.py",
            "module_sha256": hashlib.sha256((ROOT / "src/hack_guards.py").read_bytes()).hexdigest(),
            "hybrid_rule": "hybrid quality = judge quality - 25 if any guard fires on (output, source)",
        },
        "t4": {"items_path": str(ITEMS.relative_to(ROOT)), "items_sha256": items_sha,
               "items": len(items), "perturbations": list(PERTURBATIONS)},
        "tests": {**v1["tests"],
                  "T4_known_hack_detection": "as amendment 1, on the v2 sample, for judge and hybrid",
                  "G1_guard_false_positive": (
                      "fraction of real WMT candidates in the v2 sample with human quality >= "
                      "the pool median on which any guard fires")},
        "selection_rule": (
            "Candidates are the 3 judges x {judge-only, hybrid}. Eligible: T0 >= 0.95 on "
            "every pool. Select the highest mean T1 acc_eq across pools; T3/T4 not used."
        ),
        "acceptance_criteria": {
            "A1": "T0 >= 0.95 on every pool",
            "A2": "T3 sign agreement >= 0.80 over human-significant cells",
            "A3": "T3 Spearman across all cells >= 0.60",
            "A4": "as v1 (underpowered if < 5 significantly negative cells)",
            "A5": "T4 detection >= 0.90 for every perturbation type, for the selected candidate",
            "G1": "if the selected candidate is a hybrid, guard false-positive rate <= 0.02",
            "decision": "accept only if A1, A2, A3, A5 (and G1 for hybrids) pass and A4 is not failed",
        },
    }
    OUTPUT.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(f"protocol_sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")
    print(f"items_sha256={items_sha} items={len(items)}")
    for name, pool in pools.items():
        print(name, pool["segments_available_after_v1"], pool["sampled_candidates"])


if __name__ == "__main__":
    main()
