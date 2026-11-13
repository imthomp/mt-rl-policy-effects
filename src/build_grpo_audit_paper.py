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
    if set(human["pairs"]) != {"en_de", "en_es", "en_zh"}:
        raise AssertionError("human-evaluation package has unexpected language pairs")

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
            "human_sheet_hashes": {lp: row["sheet_sha256"] for lp, row in human["pairs"].items()},
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
        "HumanPairs": len(human["pairs"]),
        "HumanItemsPerPair": human["n_per_pair"],
        "HumanRows": sum(row["rows"] for row in human["pairs"].values()),
        "SmallPolicy": esc(protocol["arm_B"]["policy"].split("/")[-1]),
        "LargePolicy": esc(protocol["arm_A"]["policy"].split("/")[-1]),
        "ReferenceMetric": "COMET-22",
    }
    args.macros.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated from results/paper_stats.json; do not edit."]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macro.items())
    args.macros.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.macros} and SSOT section grpo_audit_paper")


if __name__ == "__main__":
    main()
