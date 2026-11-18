"""Build the GRPO audit paper's SSOT section and generated LaTeX macros."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean


def esc(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssot", type=Path, default=root / "results/paper_stats.json")
    ap.add_argument("--macros", type=Path, default=root / "paper/grpo_audit_acl/generated_stats.tex")
    args = ap.parse_args()

    ssot = json.loads(args.ssot.read_text())
    required = (
        "grpo_scale", "grpo_rule_scorecard_corrected", "policy_relative_scorecard_corrected",
        "grpo_ende_followup_corrected", "grpo_reference_eval", "grpo_reference_eval_stripped",
        "preamble_audit_exploratory", "human_eval_manifest",
    )
    missing = [key for key in required if key not in ssot]
    if missing:
        raise AssertionError(f"missing SSOT sections: {missing}")

    scale = ssot["grpo_scale"]
    if scale["status"] != "complete" or len(scale["runs"]) != scale["run_count"]:
        raise AssertionError("scale scorecard is incomplete or has an untraced run row")
    if any(cell["run"] not in scale["runs"] for cell in scale["cells"]):
        raise AssertionError("scale scorecard contains a cell without a source run")
    if scale["uninformative"]:
        raise AssertionError("scale scorecard contains an uninformative call")

    corrected = ssot["grpo_rule_scorecard_corrected"]
    relative = ssot["policy_relative_scorecard_corrected"]
    if corrected["rule_tests_missing"] or relative["missing"]:
        raise AssertionError("corrected scorecard is missing expected rows")

    refs = ssot["grpo_reference_eval"]["runs"]
    scale_run_names = sorted(name for name in refs if name.startswith("grpo_scale/"))
    if len(scale_run_names) != scale["run_count"]:
        raise AssertionError("reference evaluation does not trace every scale run")

    def ref_values(prefix: str) -> list[float]:
        values = [row["delta_comet22"] for name, row in refs.items() if name.startswith(prefix)]
        if not values:
            raise AssertionError(f"no reference rows for {prefix}")
        return values

    xzh = ref_values("grpo_scale/A_7b_en_zh_xcomet_xl_qe_seed")
    xde = ref_values("grpo_scale/A_7b_en_de_xcomet_xl_qe_seed")
    mxl = [row["delta_comet22"] for name, row in refs.items()
           if name.startswith("grpo_scale/B_1p5b_") and "metricx_24_xl_qe" in name]
    mxxl = [row["delta_comet22"] for name, row in refs.items()
            if name.startswith("grpo_scale/B_1p5b_") and "metricx_24_xxl_qe" in name]
    if len(mxl) != len(mxxl) or not mxl:
        raise AssertionError("MetricX scale reference rows are incomplete")

    follow = ssot["grpo_ende_followup_corrected"]
    if follow["status"] != "complete" or len(follow["runs"]) != 6:
        raise AssertionError("corrected follow-up rows are incomplete")
    refusal_final = [run["refusal_count_by_step"]["1000"] for run in follow["runs"].values()
                     if not run["guarded"]]

    preamble = ssot["preamble_audit_exploratory"]
    corrected_counts = {name: row for name, row in preamble["counts"].items()
                        if name.startswith("grpo_v2/")}
    threshold_crossings = [row for row in corrected_counts.values()
                           if row["preamble_final"] >= preamble["min_count"]]
    corrected_500 = [row for row in corrected_counts.values() if row["final_step"] == 500]
    if any(row["preamble_final"] for row in corrected_500):
        raise AssertionError("paper statement about corrected 500-step preambles is stale")

    human = ssot["human_eval_manifest"]
    if human.get("design") != "2026-11-18 headline-aligned":
        raise AssertionError("human-evaluation package is not the headline-aligned design")
    for sheet in human["sheets"].values():
        for run_dir, _step in list(sheet["systems"].values()) + list(sheet["attention_systems"].values()):
            if not (root / run_dir / "eval_outputs.jsonl").exists():
                raise AssertionError(f"human-eval system does not trace to a run: {run_dir}")
    primary = {k: v for k, v in human["sheets"].items() if v["role"] == "primary"}

    protocol = json.loads((root / "results/grpo_scale_protocol.json").read_text())
    padding = json.loads((root / "results/grpo_padding_rerun_amendment.json").read_text())
    training_steps = int(re.search(r"(\d+) steps", protocol["inherits"]).group(1))
    eval_sources = 400

    paper = {
        "status": "draft_pending_human_evaluation",
        "scale": {
            "runs": scale["run_count"],
            "manipulation_passes": scale["manipulation_pass_count"],
            "hack_calls": scale["tally"]["hack"],
            "safe_calls": scale["tally"]["safe"],
            "abstentions": scale["tally"]["none"]["calls"],
            "policy_arm": scale["tally_by_arm"]["A_policy_scale"],
            "reward_arm": scale["tally_by_arm"]["B_reward_scale"],
        },
        "corrected_earlier_grid": {
            "fixed_rule": corrected["rule_test_tally"],
            "policy_relative": relative["tally"],
        },
        "reference_quality": {
            "xcomet_7b_en_zh": xzh,
            "xcomet_7b_en_de": xde,
            "metricx_xl": mxl,
            "metricx_xxl": mxxl,
        },
        "padding": {
            "affected_logs": int(padding["affected"].split(" of ")[0].rsplit(" ", 1)[-1]),
            "corrected_runs": padding["rerun"]["n"],
            "followup_hypotheses": follow["hypotheses"],
            "unguarded_final_refusals": refusal_final,
            "corrected_preamble_threshold_crossings": len(threshold_crossings),
            "corrected_500_final_preambles": sum(row["preamble_final"] for row in corrected_500),
        },
        "human_evaluation": human,
        "protocol": {
            "training_steps": training_steps,
            "eval_sources_per_direction": eval_sources,
            "policy_small": protocol["arm_B"]["policy"],
            "policy_large": protocol["arm_A"]["policy"],
            "hack_threshold": scale["thresholds"]["hack_indicator_delta"],
            "safe_threshold": scale["thresholds"]["safe_indicator_delta"],
            "manipulation_threshold": scale["thresholds"]["manipulation_reward_delta"],
        },
        "trace": {
            "scale_run_ids": sorted(scale["runs"]),
            "scale_reference_run_ids": scale_run_names,
            "human_sheet_hashes": {k: row["sheet_sha256"] for k, row in human["sheets"].items()},
        },
    }
    ssot["grpo_audit_paper"] = paper
    tmp = args.ssot.with_suffix(".tmp")
    tmp.write_text(json.dumps(ssot, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.ssot)

    macro = {
        "ScaleRuns": paper["scale"]["runs"],
        "ScaleManipulationPasses": paper["scale"]["manipulation_passes"],
        "ScaleHackCorrect": scale["tally"]["hack"]["correct"],
        "ScaleHackCalls": scale["tally"]["hack"]["calls"],
        "ScaleSafeCorrect": scale["tally"]["safe"]["correct"],
        "ScaleSafeCalls": scale["tally"]["safe"]["calls"],
        "ScaleAbstentions": paper["scale"]["abstentions"],
        "PolicyHackCorrect": scale["tally_by_arm"]["A_policy_scale"]["hack"]["correct"],
        "PolicyHackCalls": scale["tally_by_arm"]["A_policy_scale"]["hack"]["calls"],
        "PolicySafeCorrect": scale["tally_by_arm"]["A_policy_scale"]["safe"]["correct"],
        "PolicySafeCalls": scale["tally_by_arm"]["A_policy_scale"]["safe"]["calls"],
        "RewardSafeCorrect": scale["tally_by_arm"]["B_reward_scale"]["safe"]["correct"],
        "RewardSafeCalls": scale["tally_by_arm"]["B_reward_scale"]["safe"]["calls"],
        "FixedHackCorrect": corrected["rule_test_tally"]["hack"]["correct"],
        "FixedHackCalls": corrected["rule_test_tally"]["hack"]["n"],
        "FixedSafeCorrect": corrected["rule_test_tally"]["safe"]["correct"],
        "FixedSafeCalls": corrected["rule_test_tally"]["safe"]["n"],
        "RelativeHackCorrect": relative["tally"]["hack"]["correct"],
        "RelativeHackCalls": relative["tally"]["hack"]["n"],
        "RelativeSafeCorrect": relative["tally"]["safe"]["correct"],
        "RelativeSafeCalls": relative["tally"]["safe"]["n"],
        "TrainingSteps": training_steps,
        "EvalSources": eval_sources,
        "HackThreshold": f"{scale['thresholds']['hack_indicator_delta']:.2f}",
        "SafeThreshold": f"{scale['thresholds']['safe_indicator_delta']:.2f}",
        "ManipulationThreshold": f"{scale['thresholds']['manipulation_reward_delta']:.2f}",
        "XcometZhCometMean": signed(mean(xzh)),
        "XcometZhCometMin": signed(min(xzh)),
        "XcometZhCometMax": signed(max(xzh)),
        "XcometDeCometMean": signed(mean(xde)),
        "XcometDeCometMin": signed(min(xde)),
        "XcometDeCometMax": signed(max(xde)),
        "MetricXlCometMean": signed(mean(mxl)),
        "MetricXxlCometMean": signed(mean(mxxl)),
        "AffectedLogs": paper["padding"]["affected_logs"],
        "CorrectedRuns": paper["padding"]["corrected_runs"],
        "FollowupSeeds": len(refusal_final),
        "CorrectedRefusalMax": max(refusal_final),
        "PreambleCrossings": len(threshold_crossings),
        "CorrectedFiveHundredPreambles": paper["padding"]["corrected_500_final_preambles"],
        "HumanPrimarySheets": len(primary),
        "HumanItemsPerSheet": human["items_per_sheet"],
        "HumanPrimaryRows": sum(row["rows"] for row in primary.values()),
        "HumanAttentionItems": human["attention_items_per_sheet"],
        "SmallPolicy": esc(protocol["arm_B"]["policy"].split("/")[-1]),
        "LargePolicy": esc(protocol["arm_A"]["policy"].split("/")[-1]),
        "ReferenceMetric": "COMET-22",
    }
    # ---- Setup, audit, compute and task-level additions (2026-11-17) ----
    import glob
    from collections import defaultdict
    ende = json.loads((root / "results/grpo_ende_protocol.json").read_text())
    g = ende["grpo"]
    rule = json.loads((root / "results/policy_relative_rule.json").read_text())["rule"]
    hack_p = float(re.search(r"p_k > ([0-9.]+)", rule["predict_hack"]).group(1))
    safe_p = float(re.search(r"p_k <= ([0-9.]+)", rule["predict_safe"]).group(1))
    audit_n = {json.loads(open(f).read())["n_prompts"]
               for f in glob.glob(str(root / "results/policy_relative_audit_*.json")) if "outputs" not in f}
    if len(audit_n) != 1:
        raise AssertionError(f"audits use different prompt counts: {audit_n}")

    def gpu_hours(pattern: str) -> tuple[float, int]:
        rows = [json.loads(open(f).read()) for f in glob.glob(str(root / pattern))]
        secs = [r["log_history"][-1]["train_runtime"] for r in rows]
        if len(secs) != len(rows):
            raise AssertionError(f"missing train_runtime under {pattern}")
        return sum(secs) / 3600, len(rows)

    scale_h, scale_n = gpu_hours("artifacts/grpo_scale/*/pilot_run.json")
    v2_h, v2_n = gpu_hours("artifacts/grpo_v2/*/*/pilot_run.json")
    if scale_n != scale["run_count"] or v2_n != padding["rerun"]["n"]:
        raise AssertionError("compute rows do not trace every run")

    # Task level: one task = (arm, policy, reward, direction, failure type); seeds are replicates.
    tasks = defaultdict(list)
    for cell in scale["cells"]:
        run = scale["runs"][cell["run"]]
        tasks[(cell["arm"], run["policy_model"], cell["reward"], cell["protocol"], cell["type"])].append(cell)
    task_rows = []
    for key, cells in sorted(tasks.items()):
        preds = {c["prediction"] for c in cells}
        if len(preds) != 1:
            raise AssertionError(f"task {key} has inconsistent predictions across seeds")
        pred = preds.pop()
        correct = [c["correct"] for c in cells if c["prediction"] != "none"]
        task_rows.append({"task": list(key), "prediction": pred, "seeds": len(cells),
                          "all_correct": bool(correct) and all(correct),
                          "any_correct": bool(correct) and any(correct)})
    tt = {x: {"tasks": sum(r["prediction"] == x for r in task_rows),
              "all_seeds_correct": sum(r["prediction"] == x and r["all_correct"] for r in task_rows),
              "mixed": sum(r["prediction"] == x and r["any_correct"] and not r["all_correct"] for r in task_rows)}
          for x in ("hack", "safe", "none")}
    gen_done = (root / "results/grpo_generalization_result.json").exists()
    paper["setup"] = {"prompts_per_step": g["prompts_per_step"], "generations": g["generations_per_prompt"],
                      "kl_beta": g["kl_beta"], "lr_full": g["learning_rate"], "temperature": g["temperature"],
                      "max_new_tokens": g["max_new_tokens"], "lora": protocol["arm_A"]["lora"],
                      "lr_lora": protocol["arm_A"]["learning_rate"], "audit_prompts": audit_n.pop(),
                      "hack_p": hack_p, "safe_p": safe_p}
    paper["compute"] = {"scale_gpu_hours": scale_h, "corrected_grid_gpu_hours": v2_h}
    paper["scale_task_level"] = {"tally": tt, "rows": task_rows}
    paper["generalization_status"] = "scored" if gen_done else "running"
    ssot["grpo_audit_paper"] = paper
    tmp.write_text(json.dumps(ssot, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.ssot)
    macro.update({
        "PromptsPerStep": g["prompts_per_step"], "GenerationsPerPrompt": g["generations_per_prompt"],
        "KLBeta": g["kl_beta"], "LRFull": f"{g['learning_rate']:.0e}".replace("e-0", r"\times 10^{-") + "}",
        "Temperature": g["temperature"], "MaxNewTokens": g["max_new_tokens"],
        "LoraRank": protocol["arm_A"]["lora"]["r"], "LoraAlpha": protocol["arm_A"]["lora"]["alpha"],
        "LRLora": f"{protocol['arm_A']['learning_rate']:.0e}".replace("e-0", r"\times 10^{-") + "}",
        "AuditPrompts": paper["setup"]["audit_prompts"],
        "HackP": f"{hack_p:.2f}", "SafeP": f"{safe_p:.2f}",
        "ScaleGPUHours": f"{scale_h:.0f}", "CorrectedGridGPUHours": f"{v2_h:.0f}",
        "TaskHackTasks": tt["hack"]["tasks"], "TaskHackAllCorrect": tt["hack"]["all_seeds_correct"],
        "TaskSafeTasks": tt["safe"]["tasks"], "TaskSafeAllCorrect": tt["safe"]["all_seeds_correct"],
        "TaskAbstainTasks": tt["none"]["tasks"],
        "GeneralizationStatus": "complete" if gen_done else "running",
    })
    # Appendix table: one row per scale run, generated so every row traces to a run record.
    rows = []
    for name in sorted(scale["runs"]):
        run = scale["runs"][name]
        ref = refs["grpo_scale/" + name]
        ind = run["indicator_final"]
        rows.append(" & ".join([esc(run["policy_model"].split("-")[1]), esc(run["reward"]), esc(run["protocol"]),
                                str(run["seed"]), signed(run["own_reward_delta"]),
                                f"{ind['untranslated_rate']:.3f}", f"{ind['refusal_rate']:.3f}",
                                signed(ref["delta_comet22"])]) + r" \\")
    (args.macros.parent / "generated_scale_runs.tex").write_text(
        "% Generated from results/paper_stats.json; do not edit.\n" + "\n".join(rows) + "\n")
    args.macros.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated from results/paper_stats.json; do not edit."]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macro.items())
    args.macros.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.macros} and SSOT section grpo_audit_paper")


if __name__ == "__main__":
    main()
