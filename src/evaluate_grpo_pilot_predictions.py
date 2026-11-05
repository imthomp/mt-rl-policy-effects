"""Score the frozen GRPO pilot predictions P1-P3 and merge into the SSOT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/grpo_pilot_protocol.json"
PROTOCOL_SHA256 = "61d89eaab72a7a9d10fd8ec2cedb28b752293faae354d4a6d5d4f1c69323343c"
RUNS = ("xcomet_xl_qe", "cometkiwi_22", "cometkiwi_23_xl")
OUTPUT = ROOT / "results/grpo_pilot_result.json"
SSOT = ROOT / "results/paper_stats.json"


def main() -> None:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("GRPO pilot protocol hash mismatch")
    runs = {}
    for run in RUNS:
        path = ROOT / "artifacts/grpo_pilot" / run / "eval_summary.json"
        data = json.loads(path.read_text())
        if data["protocol_sha256"] != PROTOCOL_SHA256:
            raise ValueError(f"{run}: evaluated under a different protocol")
        runs[run] = {"summary": data["summary"],
                     "eval_summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    initial = {r: runs[r]["summary"]["0"]["untranslated_rate"] for r in RUNS}
    if len(set(initial.values())) != 1:
        raise ValueError("Step-0 evaluation differs across runs (should be the same base model)")
    base = initial[RUNS[0]]
    final = {r: runs[r]["summary"]["500"]["untranslated_rate"] for r in RUNS}
    p1 = final["xcomet_xl_qe"] >= base + 0.10 and final["xcomet_xl_qe"] > max(
        final["cometkiwi_22"], final["cometkiwi_23_xl"])
    p2 = final["cometkiwi_23_xl"] <= base + 0.02
    p3 = final["xcomet_xl_qe"] > final["cometkiwi_22"] > final["cometkiwi_23_xl"]
    first_step = {r: min((int(s) for s, v in runs[r]["summary"].items()
                          if int(s) > 0 and v["untranslated_rate"] >= base + 0.10), default=None)
                  for r in RUNS}
    result = {
        "protocol_sha256": PROTOCOL_SHA256,
        "initial_untranslated_rate": base,
        "final_untranslated_rate": final,
        "first_checkpoint_with_untranslated_rise_ge_0p10": first_step,
        "final_mean_reward": {r: runs[r]["summary"]["500"]["mean_reward"] for r in RUNS},
        "initial_mean_reward": runs[RUNS[0]]["summary"]["0"]["mean_reward"],
        "predictions": {"P1": p1, "P2": p2, "P3": p3},
        "caveats": [
            "single seed per reward",
            "P3 ordering between the two CometKiwi runs rests on a small difference in final untranslated rate",
            "no human evaluation; quality of the non-hacking runs is not established",
        ],
        "runs": runs,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    root = json.loads(SSOT.read_text(encoding="utf-8"))
    root["grpo_pilot"] = {k: v for k, v in result.items() if k != "runs"}
    tmp = SSOT.with_suffix(SSOT.suffix + ".tmp")
    tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(SSOT)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
