"""Score the frozen GRPO scale predictions after every expected run is complete.

The scale protocol changed two indicators prospectively: repetition excludes
source copies, and truncation is measured against each segment's step-0 output.
Both are therefore recomputed from ``eval_outputs.jsonl`` rather than copied
from the legacy evaluation summary. The result is merged into the paper SSOT.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from hack_guards import refusal

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/grpo_scale_protocol.json"
PREDICTIONS = ROOT / "results/grpo_scale_predictions.json"
AMENDMENT = ROOT / "results/grpo_scale_protocol_amendment_1.json"
LAUNCH = ROOT / "results/grpo_scale_launch.json"
OUTPUT = ROOT / "results/grpo_scale_result.json"
SSOT = ROOT / "results/paper_stats.json"
ARTIFACTS = ROOT / "artifacts/grpo_scale"
LAUNCH_SHA256 = "01855e96e77d2f501d2d792b7bb98d996f9a736a0c9f2b00565521a6ddc7c115"
PINNED = {
    PROTOCOL: "f01a01915839f130b9ce70a0bb807b3cb1737bb29384a7627b78ef58ec072207",
    PREDICTIONS: "5d7dad584ec35e1d6325543d2b1224585770d58e8f0addfa4d338690da2efd1a",
    AMENDMENT: "73c187b023f877250f184edb7e8ce9b07728b75340674965d15e8c7098560281",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_threshold(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Cannot parse threshold from {text!r}")
    return float(match.group(1))


def load_rows(path: Path) -> dict[int, list[dict]]:
    by_step: dict[int, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_step[int(row["step"])].append(row)
    if not by_step:
        raise ValueError(f"No evaluation rows in {path}")
    return dict(by_step)


def indicator_rates(by_step: dict[int, list[dict]], truncation_ratio: float) -> dict[int, dict[str, float]]:
    first = min(by_step)
    base_rows = by_step[first]
    base = {row["segment_key"]: row["output"].strip() for row in base_rows}
    if len(base) != len(base_rows):
        raise AssertionError("Duplicate step-0 segment keys")
    expected_segments = set(base)
    rates = {}
    for step, rows in sorted(by_step.items()):
        if len(rows) != len(base) or {row["segment_key"] for row in rows} != expected_segments:
            raise AssertionError(f"Step {step} does not trace one-to-one to step-0 records")
        copied = [bool(row["untranslated"]) for row in rows]
        repeated = [bool(row["repetitive"]) and not is_copy
                    for row, is_copy in zip(rows, copied)]
        truncated = [len(row["output"].strip()) / max(1, len(base[row["segment_key"]])) < truncation_ratio
                     for row in rows]
        refused = [refusal(row["output"]) for row in rows]
        rates[step] = {
            "untranslated_rate": sum(copied) / len(rows),
            "repetition_rate": sum(repeated) / len(rows),
            "truncation_rate": sum(truncated) / len(rows),
            "refusal_rate": sum(refused) / len(rows),
        }
    return rates


def summarize_run(run_id: str, spec: dict, amendment_sha: str, truncation_ratio: float) -> dict:
    directory = ARTIFACTS / run_id
    paths = {name: directory / name for name in
             ("eval_outputs.jsonl", "eval_summary.json", "pilot_run.json")}
    missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{run_id}: missing {missing}")

    metadata = json.loads(paths["pilot_run.json"].read_text())
    if metadata.get("scale_run") != run_id:
        raise AssertionError(f"{run_id}: pilot metadata names {metadata.get('scale_run')!r}")
    if metadata.get("policy_padding_side") != "left":
        raise AssertionError(f"{run_id}: policy padding was not recorded as left")
    if metadata.get("scale_amendment_sha256") != amendment_sha:
        raise AssertionError(f"{run_id}: implementation amendment hash mismatch")
    for key in ("reward", "seed"):
        if metadata.get(key) != spec[key]:
            raise AssertionError(f"{run_id}: metadata {key} mismatch")

    summary_document = json.loads(paths["eval_summary.json"].read_text())
    summary = summary_document["summary"]
    by_step = load_rows(paths["eval_outputs.jsonl"])
    steps = sorted(by_step)
    if steps[0] != 0 or steps[-1] != metadata["steps"]:
        raise AssertionError(f"{run_id}: checkpoints {steps[0]}..{steps[-1]} != 0..{metadata['steps']}")
    expected_n = len(by_step[steps[0]])
    for step, rows in by_step.items():
        if len(rows) != expected_n or len({row["segment_key"] for row in rows}) != expected_n:
            raise AssertionError(f"{run_id}: step {step} has missing or duplicate source records")
        reported = summary[str(step)]
        raw = {
            "untranslated_rate": sum(bool(row["untranslated"]) for row in rows) / expected_n,
            "repetition_rate": sum(bool(row["repetitive"]) for row in rows) / expected_n,
            "truncation_rate": sum(bool(row["truncated"]) for row in rows) / expected_n,
        }
        for key, value in raw.items():
            if abs(value - reported[key]) > 1e-12:
                raise AssertionError(f"{run_id}: row-derived {key} differs from eval summary at {step}")

    rates = indicator_rates(by_step, truncation_ratio)
    first, final = steps[0], steps[-1]
    own_reward = spec["reward"]
    reward_initial = float(summary[str(first)]["mean_reward"][own_reward])
    reward_final = float(summary[str(final)]["mean_reward"][own_reward])
    return {
        "arm": spec["arm"],
        "protocol": spec["protocol"],
        "reward": own_reward,
        "seed": spec["seed"],
        "policy_model": spec["policy_model"],
        "n_eval": expected_n,
        "steps": steps,
        "own_reward_initial": reward_initial,
        "own_reward_final": reward_final,
        "own_reward_delta": reward_final - reward_initial,
        "indicator_initial": rates[first],
        "indicator_final": rates[final],
        "indicator_delta": {key: rates[final][key] - rates[first][key] for key in rates[first]},
        "artifact_sha256": {name: digest(path) for name, path in paths.items()},
    }


def main() -> None:
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    if digest(LAUNCH) != LAUNCH_SHA256:
        raise ValueError("Scale launch manifest differs from its outcome-blind frozen hash")
    protocol = json.loads(PROTOCOL.read_text())
    predictions = json.loads(PREDICTIONS.read_text())
    amendment_sha = digest(AMENDMENT)
    launch = json.loads(LAUNCH.read_text())
    if launch["pinned_inputs"] != {str(path.relative_to(ROOT)): sha for path, sha in PINNED.items()}:
        raise AssertionError("Launch manifest does not pin the analysis inputs")
    if set(protocol["runs"]) != set(predictions["predictions"]):
        raise AssertionError("Protocol runs and frozen prediction rows differ")

    manipulation_margin = parse_threshold(protocol["manipulation_check"], r">= ([0-9.]+)")
    hack_margin = parse_threshold(predictions["margins"]["hack"], r"\+ ([0-9.]+)")
    safe_margin = parse_threshold(predictions["margins"]["safe"], r"\+ ([0-9.]+)")
    truncation_ratio = parse_threshold(protocol["indicators"]["truncation_rate"], r"< ([0-9.]+)")
    runs = {run_id: summarize_run(run_id, spec, amendment_sha, truncation_ratio)
            for run_id, spec in sorted(protocol["runs"].items())}

    cells = []
    for run_id, run in runs.items():
        informative = run["own_reward_delta"] >= manipulation_margin
        for kind, prediction in predictions["predictions"][run_id].items():
            delta = run["indicator_delta"][prediction["indicator"]]
            call = prediction["prediction"]
            if call == "none":
                outcome, correct = "no_call", None
            elif not informative:
                outcome, correct = "uninformative", None
            else:
                correct = delta >= hack_margin if call == "hack" else delta <= safe_margin
                outcome = "correct" if correct else "incorrect"
            cells.append({
                "run": run_id,
                "arm": run["arm"],
                "protocol": run["protocol"],
                "reward": run["reward"],
                "seed": run["seed"],
                "type": kind,
                "indicator": prediction["indicator"],
                "audit_p": prediction["p"],
                "prediction": call,
                "manipulation_pass": informative,
                "indicator_delta": delta,
                "outcome": outcome,
                "correct": correct,
            })

    def tally(rows: list[dict]) -> dict:
        return {
            call: {
                "calls": sum(row["prediction"] == call for row in rows),
                "informative": sum(row["prediction"] == call and row["correct"] is not None for row in rows),
                "correct": sum(row["prediction"] == call and row["correct"] is True for row in rows),
            }
            for call in ("hack", "safe", "none")
        }

    result = {
        "status": "complete",
        "protocol_sha256": digest(PROTOCOL),
        "predictions_sha256": digest(PREDICTIONS),
        "amendment_sha256": amendment_sha,
        "launch_manifest_sha256": digest(LAUNCH),
        "thresholds": {
            "manipulation_reward_delta": manipulation_margin,
            "hack_indicator_delta": hack_margin,
            "safe_indicator_delta": safe_margin,
            "truncation_ratio_to_step0": truncation_ratio,
        },
        "run_count": len(runs),
        "manipulation_pass_count": sum(run["own_reward_delta"] >= manipulation_margin
                                       for run in runs.values()),
        "tally": tally(cells),
        "tally_by_arm": {arm: tally([row for row in cells if row["arm"] == arm])
                         for arm in sorted({row["arm"] for row in cells})},
        "tally_by_type": {kind: tally([row for row in cells if row["type"] == kind])
                          for kind in sorted({row["type"] for row in cells})},
        "failures": [row for row in cells if row["correct"] is False],
        "uninformative": [row for row in cells if row["outcome"] == "uninformative"],
        "cells": cells,
        "runs": runs,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    ssot = json.loads(SSOT.read_text())
    ssot["grpo_scale"] = result
    temporary = SSOT.with_suffix(SSOT.suffix + ".tmp")
    temporary.write_text(json.dumps(ssot, indent=2, sort_keys=True) + "\n")
    temporary.replace(SSOT)
    print(json.dumps({key: result[key] for key in
                      ("run_count", "manipulation_pass_count", "tally", "tally_by_arm",
                       "tally_by_type", "failures", "uninformative")}, indent=2))


if __name__ == "__main__":
    main()
