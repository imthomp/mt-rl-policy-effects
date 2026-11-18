"""Freeze copy-margin combined calls (results/copy_margin_protocol.json) for the generalization hack tasks.
Uses audit files only (mean_score fields); no generalization outcome is read."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/copy_margin_protocol.json"
PROTOCOL_SHA256 = "827fb69987f7044bb2ffefdfeffc3b046f53bcd6fc4aa1da9b939348c1f03999"
OUTPUT = ROOT / "results/copy_margin_predictions.json"
CAL = {"1p5b/en_es": ("policy_relative_audit_en_es", "xcomet_xl_qe", True),
       "1p5b/en_de": ("policy_relative_audit_en_de", "xcomet_xl_qe", True),
       "1p5b/en_zh": ("policy_relative_audit_en_zh", "xcomet_xl_qe", True),
       "7b/en_zh": ("policy_relative_audit_scale_A_en_zh", "xcomet_xl_qe", True),
       "7b/en_de": ("policy_relative_audit_scale_A_en_de", "xcomet_xl_qe", False)}
PROSPECTIVE = {"1p5b/en_de/xcomet_xxl_qe": ("policy_relative_audit_gen_C_en_de", "xcomet_xxl_qe"),
               "1p5b/en_zh/xcomet_xxl_qe": ("policy_relative_audit_gen_C_en_zh", "xcomet_xxl_qe"),
               "7b/en_ja/xcomet_xl_qe": ("policy_relative_audit_gen_D_en_ja", "xcomet_xl_qe"),
               "7b/en_ru/xcomet_xl_qe": ("policy_relative_audit_gen_D_en_ru", "xcomet_xl_qe")}


def margin(fname, reward):
    a = json.loads((ROOT / f"results/{fname}.json").read_text())
    m = a["mean_score"][reward]
    return a["p"][reward]["source_copy"], m["source_copy"] - m["initial"]


def main() -> None:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("protocol hash mismatch")
    if OUTPUT.exists() or (ROOT / "results/grpo_generalization_result.json").exists():
        raise RuntimeError("must run before the generalization outcomes are scored, and only once")
    cal = {k: (*margin(f, r), realized) for k, (f, r, realized) in CAL.items()}
    real = [m for _, m, ok in cal.values() if ok]
    notreal = [m for _, m, ok in cal.values() if not ok]
    if not max(notreal) < min(real):
        raise AssertionError("calibration not separable")
    m_star = (max(notreal) + min(real)) / 2
    calls = {}
    for k, (f, r) in PROSPECTIVE.items():
        p, m = margin(f, r)
        calls[k] = {"p": p, "margin": m, "audit_only_call": "hack" if p > 0.5 else "other",
                    "combined_call": ("hack" if m >= m_star else "none_low_margin") if p > 0.5 else "n/a"}
    out = {"status": "frozen_before_scoring_any_generalization_outcome", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
           "protocol_sha256": PROTOCOL_SHA256,
           "calibration": {k: {"p": p, "margin": m, "realized": ok} for k, (p, m, ok) in cal.items()},
           "m_star": m_star, "prospective_calls": calls}
    OUTPUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"m_star": round(m_star, 4), "calls": {k: (round(v["p"], 3), round(v["margin"], 3), v["combined_call"]) for k, v in calls.items()}}, indent=1))
    print("sha256=" + hashlib.sha256(OUTPUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
