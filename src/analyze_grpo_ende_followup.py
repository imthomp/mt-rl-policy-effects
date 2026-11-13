"""Score corrected en-de amendment-2 follow-ups against frozen H1--H3.

Reads the six left-padded reruns in artifacts/grpo_v2/ende_followup, verifies
their recorded padding and amendment hash, and merges the result into the SSOT.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from hack_guards import refusal

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "results/grpo_ende_protocol_amendment_2.json"
ART = ROOT / "artifacts/grpo_v2/ende_followup"
OUTPUT = ROOT / "results/grpo_ende_followup_corrected.json"
SSOT = ROOT / "results/paper_stats.json"
REWARDS = ("xcomet_xl_qe", "cometkiwi_22", "metricx_24_qe")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(path: Path) -> dict:
    metadata = json.loads((path / "pilot_run.json").read_text())
    if metadata["policy_padding_side"] != "left":
        raise AssertionError(f"{path}: policy padding was not left")
    if metadata["ende_amendment_2_sha256"] != digest(AMENDMENT):
        raise AssertionError(f"{path}: amendment hash mismatch")
    summary = json.loads((path / "eval_summary.json").read_text())
    summary = summary.get("summary", summary)
    by_step: dict[int, list[dict]] = {}
    for line in (path / "eval_outputs.jsonl").open():
        row = json.loads(line)
        by_step.setdefault(row["step"], []).append(row)
    if sorted(by_step) != list(range(0, 1001, 100)):
        raise AssertionError(f"{path}: unexpected checkpoints {sorted(by_step)}")
    if {len(rows) for rows in by_step.values()} != {400}:
        raise AssertionError(f"{path}: expected 400 outputs per checkpoint")
    return {
        "initial": summary["0"],
        "final": summary["1000"],
        "refusal_count_by_step": {
            str(step): sum(refusal(row["output"]) for row in rows)
            for step, rows in sorted(by_step.items())
        },
    }


def main() -> None:
    amendment = json.loads(AMENDMENT.read_text())
    h1 = amendment["predictions"]["H1_horizon"]
    h2 = amendment["predictions"]["H2_guard"]
    refusal_threshold, required_seeds = map(
        int, re.search(r">= (\d+)/400 in at least (\d) of 3", h1).groups()
    )
    off_target_margin = float(re.search(r"initial \+ ([\d.]+)", h2).group(1))
    own_reward_floor = float(re.search(r">= initial \(([\d.]+)\)", h2).group(1))

    runs = {}
    for spec in amendment["runs"]:
        guarded = spec["guard"] == "langid"
        name = f"cometkiwi_23_xl{'_langid' if guarded else ''}_steps1000_seed{spec['seed']}"
        runs[name] = {"guarded": guarded, "seed": spec["seed"], **load_run(ART / name)}
    unguarded = [run for run in runs.values() if not run["guarded"]]
    guarded = [run for run in runs.values() if run["guarded"]]

    h1_passes = [run["refusal_count_by_step"]["1000"] >= refusal_threshold for run in unguarded]
    h2_cells = [{
        "off_target": run["final"]["off_target_rate"]
        <= run["initial"]["off_target_rate"] + off_target_margin,
        "own_reward": run["final"]["mean_reward"]["cometkiwi_23_xl"] >= own_reward_floor,
    } for run in guarded]
    h3_cells = [{reward: run["final"]["mean_reward"][reward] >= run["initial"]["mean_reward"][reward]
                 for reward in REWARDS} for run in guarded]
    result = {
        "status": "complete",
        "condition": "corrected_left_padded",
        "amendment_sha256": digest(AMENDMENT),
        "runs": runs,
        "hypotheses": {
            "H1_horizon": {"pass": sum(h1_passes) >= required_seeds, "seed_passes": h1_passes,
                           "threshold": refusal_threshold, "required_seeds": required_seeds},
            "H2_guard": {"pass": all(all(cell.values()) for cell in h2_cells), "seed_cells": h2_cells,
                         "off_target_margin": off_target_margin, "own_reward_floor": own_reward_floor},
            "H3_quality_crosscheck": {"pass": all(all(cell.values()) for cell in h3_cells),
                                      "seed_cells": h3_cells},
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    root = json.loads(SSOT.read_text())
    root["grpo_ende_followup_corrected"] = result
    tmp = SSOT.with_suffix(".tmp")
    tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n")
    tmp.replace(SSOT)
    print(json.dumps(result["hypotheses"], indent=2))


if __name__ == "__main__":
    main()
