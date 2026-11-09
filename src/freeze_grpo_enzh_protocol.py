"""Freeze the en-zh GRPO protocol (prospective test of results/policy_relative_rule.json).

Everything is inherited from the en-de protocol except the target language and
prompt. Eval sources are the same 400 English segments as en-de (WMT24 en-xx tasks
share sources); no MQM file is read. WMT24 en-zh refA (MTME v2) is used only by the
later reference-based evaluation.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENDE = ROOT / "results/grpo_ende_protocol.json"
ENDE_SHA256 = "a0bec8727d8bcb13f41074e51cc0e28086fa68c220bcf730c2d1e22052845779"
RULE = ROOT / "results/policy_relative_rule.json"
RULE_SHA256 = "35f0a2276fa5daaea87f2987a176277593c5f5c17f65f23f504998798cac2821"
OUTPUT = ROOT / "results/grpo_enzh_protocol.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for p, d in ((ENDE, ENDE_SHA256), (RULE, RULE_SHA256)):
        if hashlib.sha256(p.read_bytes()).hexdigest() != d:
            raise ValueError(f"hash mismatch: {p}")
    ende = json.loads(ENDE.read_text())
    protocol = {k: ende[k] for k in ("policy_model", "train_prompts", "grpo", "rewards")}
    protocol.update({
        "status": "frozen_before_en_zh_audit_and_before_any_en_zh_grpo_run",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "inherits": {"path": "results/grpo_ende_protocol.json", "sha256": ENDE_SHA256,
                     "unchanged": "policy model, English training prompts, GRPO hyperparameters, checkpoints, rewards, eval sources"},
        "direction": "en-zh",
        "target_lang": "Chinese",
        "prompt": "Translate the following English text into Chinese. Output only the translation.\n\nEnglish: {source}\nChinese:",
        "eval_set": {k: ende["eval_set"][k] for k in ("segments", "sources", "count", "sources_sha256")}
                    | {"pool": "WMT24 English sources (same 400 as en-de; no MQM file read)",
                       "references": "MTME v2 wmt24 references/en-zh.refA.txt (reference-based eval only)"},
        "hack_indicators": json.loads(RULE.read_text())["indicator_definitions"]
                           | {"off_target_rate": "hack_guards.off_target (GlotLID top-1 != cmn_Hani, >= 4 whitespace words)"},
        "predictions": {"rule": "results/policy_relative_rule.json", "rule_sha256": RULE_SHA256,
                        "audit": "results/policy_relative_audit_en_zh.json (run after this freeze)",
                        "predictions_file": "results/grpo_enzh_predictions.json (derived mechanically from rule + audit before any GRPO run)"},
        "seeds": [0, 1, 2],
    })
    OUTPUT.write_text(json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
