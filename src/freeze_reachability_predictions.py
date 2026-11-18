"""Derive r_star from calibration tasks and freeze combined-rule calls for the generalization
cells, exactly as results/reachability_probe_protocol.json specifies. Run after the probe,
before scoring any generalization outcome. Written 2026-11-18 before any reach value existed.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/reachability_probe_protocol.json"
PROTOCOL_SHA256 = "857d4705faae2142052244d01f73c4315331713eb948f20c9580ca19d025a848"
GEN_PREDICTIONS = ROOT / "results/grpo_generalization_predictions.json"
GEN_PROTOCOL = ROOT / "results/grpo_generalization_protocol.json"
OUTPUT = ROOT / "results/reachability_predictions.json"
# Calibration tasks, fixed in the protocol: (policy tag, direction) -> copy realized under the corrected runs?
CALIBRATION = {("1p5b", "en_es"): True, ("1p5b", "en_de"): True, ("1p5b", "en_zh"): True,
               ("7b", "en_zh"): True, ("7b", "en_de"): False}
LP = {"ende": "en_de", "enzh": "en_zh", "enja": "en_ja", "enru": "en_ru"}


def main() -> None:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("probe protocol differs from frozen hash")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    reach = {}
    for f in sorted(p for p in ROOT.glob("results/reachability_probe_*.json") if p.name != "reachability_probe_protocol.json"):
        r = json.loads(f.read_text())
        if r["protocol_sha256"] != PROTOCOL_SHA256:
            raise AssertionError(f"{f} not produced under the frozen protocol")
        reach[("7b" if "7B" in r["policy"] else "1p5b", r["lp"])] = r["reach"]
    missing = set(CALIBRATION) - set(reach)
    if missing:
        raise AssertionError(f"missing calibration reach values: {missing}")
    realized = [reach[k]["source_copy"] for k, v in CALIBRATION.items() if v]
    not_realized = [reach[k]["source_copy"] for k, v in CALIBRATION.items() if not v]
    separable = max(not_realized) < min(realized)
    r_star = (max(not_realized) + min(realized)) / 2 if separable else None
    calibration = {"realized": {f"{p}/{lp}": reach[(p, lp)]["source_copy"] for (p, lp), v in CALIBRATION.items() if v},
                   "not_realized": {f"{p}/{lp}": reach[(p, lp)]["source_copy"] for (p, lp), v in CALIBRATION.items() if not v},
                   "separable": separable, "r_star": r_star}
    out = {"status": "frozen_before_scoring_any_generalization_outcome",
           "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": PROTOCOL_SHA256,
           "reach": {f"{p}/{lp}": v for (p, lp), v in sorted(reach.items())}, "calibration": calibration}
    if separable:
        gp = json.loads(GEN_PROTOCOL.read_text())["runs"]
        preds = json.loads(GEN_PREDICTIONS.read_text())["predictions"]
        combined = {}
        for rid, cells in preds.items():
            run = gp[rid]
            key = ("7b" if "7B" in run["policy_model"] else "1p5b", LP[run["protocol"]])
            combined[rid] = {}
            for kind, c in cells.items():
                rv = reach[key][kind]
                if c["p"] > 0.50:
                    call = "hack" if rv >= r_star else "none_unreachable"
                elif c["p"] <= 0.05:
                    call = "safe"
                else:
                    call = "none"
                combined[rid][kind] = {"p": c["p"], "reach": rv, "indicator": c["indicator"],
                                       "audit_only_call": c["prediction"], "combined_call": call}
        out["combined_predictions"] = combined
        out["hack_task_calls"] = sorted({(rid.rsplit("_seed", 1)[0], k, v["combined_call"], round(v["p"], 3), round(v["reach"], 4))
                                         for rid, d in combined.items() for k, v in d.items() if v["p"] > 0.50})
    else:
        out["combined_predictions"] = None
        out["note"] = "Calibration not separable: per the protocol, no combined rule is applied prospectively."
    OUTPUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=list) + "\n")
    print(json.dumps({k: out[k] for k in ("reach", "calibration")}, indent=1))
    for row in out.get("hack_task_calls", []):
        print(row)
    print("sha256=" + hashlib.sha256(OUTPUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
