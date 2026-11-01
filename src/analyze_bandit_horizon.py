#!/usr/bin/env python3
"""Summarize human quality against optimization horizon and merge it into the SSOT."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN_STEPS = 200
REFERENCE = "global_random_2"


def main() -> None:
    raw = json.loads((ROOT / "results/bandit_horizon_raw.json").read_text(encoding="utf-8"))
    run = raw["contextual_bandit"]
    trajectories = {c: v["trajectories"]["exact"] for c, v in run["conditions"].items()}
    at = {c: {row["step"]: row["test"] for row in t} for c, t in trajectories.items()}
    final = run["protocol"]["steps"]
    # The most diffuse of the other rewards at the frozen horizon sets the matched entropy.
    entropy = at[REFERENCE][FROZEN_STEPS]["entropy"]
    conditions = {}
    for condition, rows in trajectories.items():
        matched = next(row for row in rows if row["test"]["entropy"] <= entropy)
        conditions[condition] = {
            "frozen_human_cesa": at[condition][FROZEN_STEPS]["human_cesa"],
            "frozen_entropy": at[condition][FROZEN_STEPS]["entropy"],
            "final_human_cesa": at[condition][final]["human_cesa"],
            "final_entropy": at[condition][final]["entropy"],
            "matched_step": matched["step"],
            "matched_human_cesa": matched["test"]["human_cesa"],
            "proxy_best_human_cesa": run["conditions"][condition]["baselines"]["test"]["proxy_best"]["human_cesa"],
        }
    summary = {"final_step": final, "frozen_step": FROZEN_STEPS, "matched_entropy": entropy,
               "reward_seeds": 1, "conditions": conditions}
    path = ROOT / "results/paper_stats.json"
    root = json.loads(path.read_text(encoding="utf-8"))
    root["bandit_horizon"] = summary
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
