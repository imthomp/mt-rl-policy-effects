"""Score every GRPO run against the frozen audit rule (results/grpo_audit_rule.json).

Rates are recomputed from per-output records in eval_outputs.jsonl and checked
against each run's eval_summary.json. Refusal counts are an exploratory
post-hoc indicator (added 2026-11-07 after the en-de CometKiwi-23-XL seed-0
collapse); they are not part of the frozen rule.

Merges keys "grpo_runs" and "grpo_rule_scorecard" into results/paper_stats.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import argparse
from collections import defaultdict
from pathlib import Path

from hack_guards import refusal

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
T4_NAMES = {
    "xcomet_xl_qe": "XCOMET-XL",
    "cometkiwi_22": "wmt22-cometkiwi-da",
    "cometkiwi_23_xl": "wmt23-cometkiwi-da-xl",
}
REWARDS = ["xcomet_xl_qe", "cometkiwi_22", "cometkiwi_23_xl", "metricx_24_qe"]
FLAG = {"untranslated_rate": "untranslated", "truncation_rate": "truncated", "repetition_rate": "repetitive"}


def audit_d() -> dict[str, dict[str, float]]:
    t4 = json.loads((ROOT / "results/llm_judge_validation_t4.json").read_text())["external_qe_rewards"]
    d = {r: {k: v["rate"] for k, v in t4[name].items()} for r, name in T4_NAMES.items()}
    d["metricx_24_qe"] = json.loads((ROOT / "results/metricx_audit.json").read_text())["detection_rate"]
    return d


def discover(art: Path, corrected: bool = False) -> list[dict]:
    runs = []
    names = ({"pilot": "pilot", "replication": "replication",
              "rule_en_es": "rule_test_en_es", "rule_en_de": "rule_test_en_de",
              "rule_en_zh": "rule_test_en_zh"} if corrected else
             {"pilot": "grpo_pilot", "replication": "grpo_replication",
              "rule_en_es": "grpo_rule_tests/en_es", "rule_en_de": "grpo_rule_tests/en_de",
              "rule_en_zh": "grpo_rule_tests/en_zh"})
    for p in sorted(art.glob(f"{names['pilot']}/*/eval_outputs.jsonl")):
        runs.append({"group": "pilot", "lp": "en_es", "reward": p.parent.name, "guarded": False, "seed": 0, "path": p})
    for p in sorted(art.glob(f"{names['replication']}/*/eval_outputs.jsonl")):
        m = re.fullmatch(r"(.+?)(_guarded)?_seed(\d+)", p.parent.name)
        runs.append({"group": "replication", "lp": "en_es", "reward": m[1], "guarded": bool(m[2]), "seed": int(m[3]), "path": p})
    for lp in ("en_es", "en_de", "en_zh"):
        for p in sorted(art.glob(f"{names['rule_' + lp]}/*/eval_outputs.jsonl")):
            m = re.fullmatch(r"(.+)_seed(\d+)", p.parent.name)
            runs.append({"group": "rule_test", "lp": lp, "reward": m[1], "guarded": False, "seed": int(m[2]), "path": p})
    return runs


def summarize(run: dict) -> dict:
    by_step: dict[int, list[dict]] = defaultdict(list)
    for line in run["path"].open():
        r = json.loads(line)
        by_step[r["step"]].append(r)
    first, last = min(by_step), max(by_step)
    n = {len(v) for v in by_step.values()}
    if len(n) != 1:
        raise AssertionError(f"{run['path']}: uneven eval sizes {n}")
    summary = json.loads((run["path"].parent / "eval_summary.json").read_text())
    summary = summary.get("summary", summary)
    out = {"n_eval": n.pop(), "first_step": first, "final_step": last, "initial": {}, "final": {}}
    for step, key in [(first, "initial"), (last, "final")]:
        rows = by_step[step]
        for rate, flag in FLAG.items():
            v = sum(r[flag] for r in rows) / len(rows)
            if abs(v - summary[str(step)][rate]) > 1e-9:
                raise AssertionError(f"{run['path']} step {step} {rate}: {v} != summary")
            out[key][rate] = v
        out[key]["refusal_count"] = sum(refusal(r["output"]) for r in rows)
        out[key]["refusal_rate"] = out[key]["refusal_count"] / len(rows)
        out[key][f"mean_{run['reward']}"] = summary[str(step)]["mean_reward"][run["reward"]]
    out["refusal_count_by_step"] = {
        str(s): sum(refusal(r["output"]) for r in v) for s, v in sorted(by_step.items())
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrected", action="store_true", help="score artifacts/grpo_v2 instead of the legacy right-padded runs")
    args = ap.parse_args()
    rule = json.loads((ROOT / "results/grpo_audit_rule.json").read_text())
    hack_t = float(re.search(r"d_k < ([\d.]+)", rule["rule"]["predict_hack"])[1])
    safe_t = float(re.search(r"d_k >= ([\d.]+)", rule["rule"]["predict_safe"])[1])
    hack_m = float(re.search(r"initial \+ ([\d.]+)", rule["rule"]["predict_hack"])[1])
    safe_m = float(re.search(r"initial \+ ([\d.]+)", rule["rule"]["predict_safe"])[1])
    d = audit_d()

    runs, scorecard = {}, []
    for run in discover(ART / "grpo_v2" if args.corrected else ART, corrected=args.corrected):
        name = f"{run['group']}/{run['lp']}/{run['reward']}{'_guarded' if run['guarded'] else ''}_seed{run['seed']}"
        s = summarize(run)
        runs[name] = {k: run[k] for k in ("group", "lp", "reward", "guarded", "seed")} | s
        # Only unguarded runs on rewards whose audit was frozen test the rule.
        # The en-es pilot and replication fixed the thresholds, so they are fit, not tests.
        if run["guarded"] or run["reward"] not in d or run["lp"] == "en_zh":
            continue  # en-zh is scored against the policy-relative rule below
        is_test = run["group"] == "rule_test"
        for dtype, rate in rule["type_to_indicator"].items():
            dk = d[run["reward"]][dtype]
            delta = s["final"][rate] - s["initial"][rate]
            if dk < hack_t:
                pred, ok = "hack", delta >= hack_m
            elif dk >= safe_t:
                pred, ok = "safe", delta <= safe_m
            else:
                pred, ok = "none", None
            scorecard.append({"run": name, "is_rule_test": is_test, "type": dtype, "indicator": rate,
                              "d_k": dk, "prediction": pred, "delta": delta, "correct": ok})

    tests = [r for r in scorecard if r["is_rule_test"] and r["prediction"] != "none"]
    tally = {p: {"n": sum(r["prediction"] == p for r in tests),
                 "correct": sum(r["prediction"] == p and r["correct"] for r in tests)} for p in ("hack", "safe")}
    expected = {f"rule_test/en_de/{r}_seed{s}" for r in REWARDS for s in range(3)} | {
        f"rule_test/en_es/metricx_24_qe_seed{s}" for s in range(3)}
    result = {
        "rule_sha256_note": "rule thresholds read from results/grpo_audit_rule.json",
        "thresholds": {"hack_d_below": hack_t, "safe_d_at_or_above": safe_t, "hack_margin": hack_m, "safe_margin": safe_m},
        "audit_d_k": d,
        "scorecard": scorecard,
        "rule_test_tally": tally,
        "rule_test_failures": [r for r in tests if not r["correct"]],
        "rule_tests_missing": sorted(expected - set(runs)),
        "refusal_indicator_note": "exploratory, post hoc (2026-11-07); regex in src/analyze_grpo_rule_tests.py",
    }
    # Policy-relative rule (results/policy_relative_rule.json), en-zh only.
    preds = json.loads((ROOT / "results/grpo_enzh_predictions.json").read_text())
    pr_cells = []
    for name, v in runs.items():
        if v["lp"] != "en_zh":
            continue
        for kind, cell in preds["predictions"][v["reward"]].items():
            rate = cell["indicator"]
            delta = v["final"][rate] - v["initial"][rate]
            ok = (delta >= 0.10 if cell["prediction"] == "hack" else
                  delta <= 0.02 if cell["prediction"] == "safe" else None)
            pr_cells.append({"run": name, "type": kind, "indicator": rate, "p": cell["p"],
                             "prediction": cell["prediction"], "delta": delta, "correct": ok})
    scored = [c for c in pr_cells if c["prediction"] != "none"]
    expected_zh = {f"rule_test/en_zh/{r}_seed{s}" for r in REWARDS for s in range(3)}
    policy_relative = {
        "predictions_sha256": preds.get("audit_sha256") and hashlib.sha256(
            (ROOT / "results/grpo_enzh_predictions.json").read_bytes()).hexdigest(),
        "cells": pr_cells,
        "tally": {p: {"n": sum(c["prediction"] == p for c in scored),
                      "correct": sum(c["prediction"] == p and c["correct"] for c in scored)} for p in ("hack", "safe")},
        "failures": [c for c in scored if not c["correct"]],
        "missing": sorted(expected_zh - set(runs)),
    }
    path = ROOT / "results/paper_stats.json"
    root = json.loads(path.read_text(encoding="utf-8"))
    suffix = "_corrected" if args.corrected else ""
    root[f"policy_relative_scorecard{suffix}"] = policy_relative
    root[f"grpo_runs{suffix}"] = runs
    root[f"grpo_rule_scorecard{suffix}"] = result
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    print("policy-relative (en-zh):", json.dumps({"tally": policy_relative["tally"], "missing": len(policy_relative["missing"]),
          "failures": [(c["run"], c["type"], round(c["delta"], 3)) for c in policy_relative["failures"]]}))
    print(json.dumps({"tally": tally, "failures": [(r["run"], r["type"], round(r["delta"], 3)) for r in result["rule_test_failures"]],
                      "missing": result["rule_tests_missing"],
                      "refusals_final": {k: v["final"]["refusal_count"] for k, v in runs.items() if v["final"]["refusal_count"]}}, indent=1))


if __name__ == "__main__":
    main()
