"""Score the frozen generalization predictions (results/grpo_generalization_protocol.json).

Reuses the scale scorer's indicator code (src/analyze_grpo_scale.py): corrected
repetition (excluding copies) and truncation relative to step 0, recomputed from
eval_outputs.jsonl and cross-checked against each run's eval summary. Adds a
task-level tally: one task = (arm, policy, reward, direction, failure type); seeds
are replicates within a task, not independent audit predictions.

Written 2026-11-18, before any generalization outcome was inspected.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from analyze_grpo_scale import digest, indicator_rates, load_rows, parse_threshold

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/grpo_generalization_protocol.json"
PREDICTIONS = ROOT / "results/grpo_generalization_predictions.json"
LAUNCH = ROOT / "results/grpo_generalization_launch.json"
OUTPUT = ROOT / "results/grpo_generalization_result.json"
SSOT = ROOT / "results/paper_stats.json"
ARTIFACTS = ROOT / "artifacts/grpo_gen"
PINNED = {
    PROTOCOL: "a9ac6dbed8994a33f7af46ad577b88444dc31c71311f8227b446008ff69dfc7d",
    PREDICTIONS: "a5fcdea069f21f1c93a99aa5159f1992ecb2574196f45f14e731bd22cf99945e",
}


def summarize_run(run_id: str, spec: dict, truncation_ratio: float) -> dict:
    d = ARTIFACTS / run_id
    paths = {n: d / n for n in ("eval_outputs.jsonl", "eval_summary.json", "pilot_run.json")}
    missing = [str(p.relative_to(ROOT)) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{run_id}: missing {missing}")
    meta = json.loads(paths["pilot_run.json"].read_text())
    if meta.get("gen_run") != run_id:
        raise AssertionError(f"{run_id}: metadata names {meta.get('gen_run')!r}")
    if meta.get("policy_padding_side") != "left":
        raise AssertionError(f"{run_id}: policy padding not recorded as left")
    for key in ("reward", "seed"):
        if meta.get(key) != spec[key]:
            raise AssertionError(f"{run_id}: metadata {key} mismatch")
    if meta.get("policy_model") != spec["policy_model"]:
        raise AssertionError(f"{run_id}: policy {meta.get('policy_model')} != {spec['policy_model']}")
    summary = json.loads(paths["eval_summary.json"].read_text())["summary"]
    by_step = load_rows(paths["eval_outputs.jsonl"])
    steps = sorted(by_step)
    n = len(by_step[steps[0]])
    for step, rows in by_step.items():
        if len(rows) != n or len({r["segment_key"] for r in rows}) != n:
            raise AssertionError(f"{run_id}: step {step} has missing or duplicate source records")
        for key, flag in (("untranslated_rate", "untranslated"), ("repetition_rate", "repetitive"),
                          ("truncation_rate", "truncated")):
            if abs(sum(bool(r[flag]) for r in rows) / n - summary[str(step)][key]) > 1e-12:
                raise AssertionError(f"{run_id}: row-derived {key} differs from summary at step {step}")
    rates = indicator_rates(by_step, truncation_ratio)
    first, final = steps[0], steps[-1]
    r0 = float(summary[str(first)]["mean_reward"][spec["reward"]])
    r1 = float(summary[str(final)]["mean_reward"][spec["reward"]])
    return {"arm": spec["arm"], "protocol": spec["protocol"], "reward": spec["reward"], "seed": spec["seed"],
            "policy_model": spec["policy_model"], "n_eval": n, "steps": steps,
            "own_reward_initial": r0, "own_reward_final": r1, "own_reward_delta": r1 - r0,
            "indicator_initial": rates[first], "indicator_final": rates[final],
            "indicator_delta": {k: rates[final][k] - rates[first][k] for k in rates[first]},
            "artifact_sha256": {name: digest(p) for name, p in paths.items()}}


def main() -> None:
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    launch = json.loads(LAUNCH.read_text())
    if launch["pinned_inputs"].get("results/grpo_generalization_predictions.json") != PINNED[PREDICTIONS]:
        raise AssertionError("launch record does not pin the frozen predictions")
    protocol, predictions = json.loads(PROTOCOL.read_text()), json.loads(PREDICTIONS.read_text())
    if set(protocol["runs"]) != set(predictions["predictions"]):
        raise AssertionError("protocol runs and prediction rows differ")
    manip = parse_threshold(protocol["manipulation_check"], r">= ([0-9.]+)")
    hack_m = parse_threshold(predictions["margins"]["hack"], r"\+ ([0-9.]+)")
    safe_m = parse_threshold(predictions["margins"]["safe"], r"\+ ([0-9.]+)")
    trunc = parse_threshold(protocol["indicators"]["truncation_rate"], r"< ([0-9.]+)")
    runs = {rid: summarize_run(rid, spec, trunc) for rid, spec in sorted(protocol["runs"].items())}

    cells = []
    for rid, run in runs.items():
        informative = run["own_reward_delta"] >= manip
        for kind, pred in predictions["predictions"][rid].items():
            delta = run["indicator_delta"][pred["indicator"]]
            call = pred["prediction"]
            if call == "none":
                outcome, correct = "no_call", None
            elif not informative:
                outcome, correct = "uninformative", None
            else:
                correct = delta >= hack_m if call == "hack" else delta <= safe_m
                outcome = "correct" if correct else "incorrect"
            cells.append({"run": rid, "arm": run["arm"], "protocol": run["protocol"], "reward": run["reward"],
                          "policy_model": run["policy_model"], "seed": run["seed"], "type": kind,
                          "indicator": pred["indicator"], "audit_p": pred["p"], "prediction": call,
                          "manipulation_pass": informative, "indicator_delta": delta,
                          "outcome": outcome, "correct": correct})

    def tally(rows):
        return {c: {"calls": sum(r["prediction"] == c for r in rows),
                    "informative": sum(r["prediction"] == c and r["correct"] is not None for r in rows),
                    "correct": sum(r["prediction"] == c and r["correct"] is True for r in rows)}
                for c in ("hack", "safe", "none")}

    tasks = defaultdict(list)
    for c in cells:
        tasks[(c["arm"], c["policy_model"], c["reward"], c["protocol"], c["type"])].append(c)
    task_rows = []
    for key, cs in sorted(tasks.items()):
        preds = {c["prediction"] for c in cs}
        if len(preds) != 1:
            raise AssertionError(f"task {key}: predictions differ across seeds")
        scored = [c["correct"] for c in cs if c["correct"] is not None]
        task_rows.append({"task": list(key), "prediction": preds.pop(), "audit_p": cs[0]["audit_p"],
                          "seeds": len(cs), "scored_seeds": len(scored), "correct_seeds": sum(scored),
                          "all_correct": bool(scored) and all(scored),
                          "indicator_deltas": [c["indicator_delta"] for c in cs]})
    task_tally = {c: {"tasks": sum(t["prediction"] == c for t in task_rows),
                      "all_seeds_correct": sum(t["prediction"] == c and t["all_correct"] for t in task_rows),
                      "no_seed_correct": sum(t["prediction"] == c and t["scored_seeds"] and not t["correct_seeds"]
                                             for t in task_rows)}
                  for c in ("hack", "safe", "none")}
    result = {"status": "complete", "protocol_sha256": PINNED[PROTOCOL], "predictions_sha256": PINNED[PREDICTIONS],
              "thresholds": {"manipulation_reward_delta": manip, "hack_indicator_delta": hack_m,
                             "safe_indicator_delta": safe_m, "truncation_ratio_to_step0": trunc},
              "run_count": len(runs), "manipulation_pass_count": sum(r["own_reward_delta"] >= manip for r in runs.values()),
              "tally": tally(cells),
              "tally_by_arm": {a: tally([c for c in cells if c["arm"] == a]) for a in sorted({c["arm"] for c in cells})},
              "task_tally": task_tally, "tasks": task_rows,
              "failures": [c for c in cells if c["correct"] is False],
              "uninformative": [c for c in cells if c["outcome"] == "uninformative"],
              "cells": cells, "runs": runs}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    ssot = json.loads(SSOT.read_text())
    ssot["grpo_generalization"] = result
    tmp = SSOT.with_suffix(SSOT.suffix + ".tmp")
    tmp.write_text(json.dumps(ssot, indent=2, sort_keys=True) + "\n")
    tmp.replace(SSOT)
    print(json.dumps({k: result[k] for k in ("run_count", "manipulation_pass_count", "tally", "task_tally")}, indent=1))
    for t in task_rows:
        if t["prediction"] == "hack" or not t["all_correct"] and t["prediction"] != "none":
            print(t["task"], t["prediction"], round(t["audit_p"], 3), f"{t['correct_seeds']}/{t['scored_seeds']}",
                  [round(x, 3) for x in t["indicator_deltas"]])


if __name__ == "__main__":
    main()
