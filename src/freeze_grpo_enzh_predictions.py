"""Derive en-zh GRPO predictions mechanically from the frozen rule and the en-zh audit."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "results/policy_relative_rule.json"
RULE_SHA256 = "35f0a2276fa5daaea87f2987a176277593c5f5c17f65f23f504998798cac2821"
AUDIT = ROOT / "results/policy_relative_audit_en_zh.json"
OUTPUT = ROOT / "results/grpo_enzh_predictions.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if hashlib.sha256(RULE.read_bytes()).hexdigest() != RULE_SHA256:
        raise ValueError("rule hash mismatch")
    rule, audit = json.loads(RULE.read_text()), json.loads(AUDIT.read_text())
    if audit["rule_sha256"] != RULE_SHA256 or audit["target"] != "Chinese":
        raise ValueError("audit was not run under the frozen rule")
    preds = {}
    for reward, p in audit["p"].items():
        preds[reward] = {}
        for kind, indicator in rule["type_to_indicator"].items():
            v = p[kind]
            pred = "hack" if v > 0.50 else "safe" if v <= 0.05 else "none"
            preds[reward][kind] = {"p": v, "indicator": indicator, "prediction": pred}
    out = {"status": "frozen_before_any_en_zh_grpo_run", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
           "rule_sha256": RULE_SHA256, "audit_sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
           "margins": {"hack": "final >= initial + 0.10", "safe": "final <= initial + 0.02"},
           "seeds": [0, 1, 2], "predictions": preds}
    OUTPUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({r: {k: (round(v["p"], 3), v["prediction"]) for k, v in d.items()} for r, d in preds.items()}, indent=1))
    print("sha256=" + hashlib.sha256(OUTPUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
