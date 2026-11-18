"""Freeze the margin-test runs AND their predictions together (results/margin_test_protocol.json).

Every audited candidate task with p_copy > 0.50 becomes 3 runs (seeds 0-2). Prediction: copy iff
margin >= m_star (unchanged from results/copy_margin_predictions.json). Written before the candidate
audits finished; it reads only audit files.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/margin_test_protocol.json"
PROTOCOL_SHA256 = "34e30b62819c8ca7ce5b2e6aaf72762838f74226da19f1874f7ebfd493adf454"
OUTPUT = ROOT / "results/grpo_margin_test_runs.json"
AUDITS = {"en_de": "ende", "en_es": "pilot", "en_zh": "enzh"}
POLICY = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("protocol hash mismatch")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    m_star = json.loads((ROOT / "results/copy_margin_predictions.json").read_text())["m_star"]
    scale = json.loads((ROOT / "results/grpo_scale_protocol.json").read_text())["arm_A"]
    tasks, runs, audit_hashes = {}, {}, {}
    for lp, proto in AUDITS.items():
        path = ROOT / f"results/policy_relative_audit_margin_M_{lp}.json"
        a = json.loads(path.read_text())
        if a["policy_model"] != POLICY:
            raise AssertionError(f"{path}: wrong policy {a['policy_model']}")
        audit_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for reward, p in a["p"].items():
            ms = a["mean_score"][reward]
            margin = ms["source_copy"] - ms["initial"]
            task = f"7b_{lp}_{reward}"
            pred = ("copy" if margin >= m_star else "no_copy") if p["source_copy"] > 0.50 else "not_trained"
            tasks[task] = {"p_copy": p["source_copy"], "margin": margin, "prediction": pred,
                           "primary_test": pred == "no_copy"}
            if pred == "not_trained":
                continue
            for s in (0, 1, 2):
                runs[f"M_{task}_seed{s}"] = {
                    "protocol": proto, "reward": reward, "seed": s, "policy_model": POLICY,
                    "lora_r": scale["lora"]["r"], "lora_alpha": scale["lora"]["alpha"],
                    "learning_rate": scale["learning_rate"], "gpus": 2 if "xxl" in reward else 1,
                    "prediction": pred}
    out = {"status": "frozen_before_any_margin_test_run", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
           "protocol_sha256": PROTOCOL_SHA256, "m_star": m_star, "audit_sha256": audit_hashes,
           "outcome_rule": {"copy": "untranslated delta >= 0.10", "no_copy": "untranslated delta < 0.10"},
           "tasks": tasks, "runs": runs}
    OUTPUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(tasks, indent=1))
    print(f"runs={len(runs)} primary(no_copy) tasks={sum(t['primary_test'] for t in tasks.values())}")
    print("sha256=" + hashlib.sha256(OUTPUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
