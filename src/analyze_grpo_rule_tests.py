"""Score every GRPO run against the frozen audit rule (results/grpo_audit_rule.json).

Rates are recomputed from per-output records in eval_outputs.jsonl and checked
against each run's eval_summary.json. Refusal counts are an exploratory
post-hoc indicator (added 2026-11-07 after the en-de CometKiwi-23-XL seed-0
collapse); they are not part of the frozen rule.

Merges keys "grpo_runs" and "grpo_rule_scorecard" into results/paper_stats.json.
"""
from __future__ import annotations

import json
import re
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


def discover() -> list[dict]:
    runs = []
    for p in sorted(ART.glob("grpo_pilot/*/eval_outputs.jsonl")):
        runs.append({"group": "pilot", "lp": "en_es", "reward": p.parent.name, "guarded": False, "seed": 0, "path": p})
    for p in sorted(ART.glob("grpo_replication/*/eval_outputs.jsonl")):
        m = re.fullmatch(r"(.+?)(_guarded)?_seed(\d+)", p.parent.name)
        runs.append({"group": "replication", "lp": "en_es", "reward": m[1], "guarded": bool(m[2]), "seed": int(m[3]), "path": p})
    for p in sorted(ART.glob("grpo_rule_tests/*/*/eval_outputs.jsonl")):
        m = re.fullmatch(r"(.+)_seed(\d+)", p.parent.name)
        runs.append({"group": "rule_test", "lp": p.parent.parent.name, "reward": m[1], "guarded": False, "seed": int(m[2]), "path": p})
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
        out[key][f"mean_{run['reward']}"] = summary[str(step)]["mean_reward"][run["reward"]]
    out["refusal_count_by_step"] = {
        str(s): sum(refusal(r["output"]) for r in v) for s, v in sorted(by_step.items())
    }
    return out


def main() -> None:
    rule = json.loads((ROOT / "results/grpo_audit_rule.json").read_text())
    hack_t = float(re.search(r"d_k < ([\d.]+)", rule["rule"]["predict_hack"])[1])
    safe_t = float(re.search(r"d_k >= ([\d.]+)", rule["rule"]["predict_safe"])[1])
    hack_m = float(re.search(r"initial \+ ([\d.]+)", rule["rule"]["predict_hack"])[1])
    safe_m = float(re.search(r"initial \+ ([\d.]+)", rule["rule"]["predict_safe"])[1])
    d = audit_d()

    runs, scorecard = {}, []
    for run in discover():
        name = f"{run['group']}/{run['lp']}/{run['reward']}{'_guarded' if run['guarded'] else ''}_seed{run['seed']}"
        s = summarize(run)
        runs[name] = {k: run[k] for k in ("group", "lp", "reward", "guarded", "seed")} | s
        # Only unguarded runs on rewards whose audit was frozen test the rule.
        # The en-es pilot and replication fixed the thresholds, so they are fit, not tests.
        if run["guarded"] or run["reward"] not in d:
            continue
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
    path = ROOT / "results/paper_stats.json"
    root = json.loads(path.read_text(encoding="utf-8"))
    root["grpo_runs"] = runs
    root["grpo_rule_scorecard"] = result
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(json.dumps({"tally": tally, "failures": [(r["run"], r["type"], round(r["delta"], 3)) for r in result["rule_test_failures"]],
                      "missing": result["rule_tests_missing"],
                      "refusals_final": {k: v["final"]["refusal_count"] for k, v in runs.items() if v["final"]["refusal_count"]}}, indent=1))


if __name__ == "__main__":
    main()
