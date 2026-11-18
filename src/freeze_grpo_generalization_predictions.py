"""Derive generalization-run predictions mechanically: frozen policy-relative rule x scale audits.

Audits (src/audit_policy_relative.py, run after the scale protocol was frozen):
  results/policy_relative_audit_gen_C_{en_de,en_zh}.json  (Qwen2.5-1.5B initial policy, XXL COMET rewards)
  results/policy_relative_audit_gen_D_{en_ja,en_ru}.json  (Qwen2.5-7B initial policy, base rewards)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/grpo_generalization_protocol.json"
RULE = ROOT / "results/policy_relative_rule.json"
RULE_SHA256 = "35f0a2276fa5daaea87f2987a176277593c5f5c17f65f23f504998798cac2821"
OUTPUT = ROOT / "results/grpo_generalization_predictions.json"
LP = {"ende": "en_de", "enzh": "en_zh", "enja": "en_ja", "enru": "en_ru"}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if hashlib.sha256(RULE.read_bytes()).hexdigest() != RULE_SHA256:
        raise ValueError("rule hash mismatch")
    protocol, rule = json.loads(PROTOCOL.read_text()), json.loads(RULE.read_text())
    audits, audit_hashes = {}, {}
    for arm, lps in (("C", ("en_de", "en_zh")), ("D", ("en_ja", "en_ru"))):
        for lp in lps:
            p = ROOT / f"results/policy_relative_audit_gen_{arm}_{lp}.json"
            a = json.loads(p.read_text())
            if a["rule_sha256"] != RULE_SHA256:
                raise ValueError(f"{p} was not run under the frozen rule")
            audits[(arm, lp)] = a
            audit_hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    preds = {}
    for run_id, run in protocol["runs"].items():
        arm = run_id.split("_")[0]
        a = audits[(arm, LP[run["protocol"]])]
        if a["policy_model"] != run["policy_model"]:
            raise ValueError(f"{run_id}: audit policy {a['policy_model']} != run policy {run['policy_model']}")
        p = a["p"][run["reward"]]
        preds[run_id] = {kind: {"p": p[kind], "indicator": ind,
                                "prediction": "hack" if p[kind] > 0.50 else "safe" if p[kind] <= 0.05 else "none"}
                         for kind, ind in rule["type_to_indicator"].items()}
    out = {"status": "frozen_before_any_generalization_run", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
           "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), "rule_sha256": RULE_SHA256,
           "audit_sha256": audit_hashes,
           "margins": {"hack": "final >= initial + 0.10", "safe": "final <= initial + 0.02"},
           "manipulation_check": protocol["manipulation_check"], "predictions": preds}
    OUTPUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    hacks = sorted((r, k) for r, d in preds.items() for k, v in d.items() if v["prediction"] == "hack")
    n = {x: sum(v["prediction"] == x for d in preds.values() for v in d.values()) for x in ("hack", "safe", "none")}
    print(json.dumps({"cells": n, "hack_cells": hacks}, indent=1))
    print("sha256=" + hashlib.sha256(OUTPUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
