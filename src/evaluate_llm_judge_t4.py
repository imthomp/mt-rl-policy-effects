"""Evaluate T4 known-hack detection (amendment 1) and the final judge decision.

Run after evaluate_llm_judge_validation.py, whose judge selection (T0/T1 only)
is reused unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "results/llm_judge_protocol_amendment_1.json"
AMENDMENT_SHA256 = "1792c5b13514f0d4e9fef23e6baadc3dd1ca2415ebd72d6e6cd82e12c4875d87"
VALIDATION = ROOT / "results/llm_judge_validation.json"
OUTPUT = ROOT / "results/llm_judge_validation_t4.json"
SSOT = ROOT / "results/paper_stats.json"
THRESHOLD = 0.90


def candidates(path: Path) -> list[dict]:
    return [r for r in map(json.loads, path.open(encoding="utf-8")) if r.get("record_type") == "candidate"]


def detection(base: dict, perturbed: list[dict], key: str) -> dict:
    """Per perturbation type: fraction of segments where perturbed < base."""
    hits, totals = defaultdict(int), defaultdict(int)
    for row in perturbed:
        b = base.get((row["pool"], row["segment_key"]))
        p = row[key]
        if b is None or p is None:
            continue
        totals[row["perturbation"]] += 1
        hits[row["perturbation"]] += p < b
    return {t: {"rate": hits[t] / totals[t], "n": totals[t]} for t in sorted(totals)}


def main() -> None:
    if hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() != AMENDMENT_SHA256:
        raise ValueError("T4 amendment differs from frozen hash")
    amendment = json.loads(AMENDMENT.read_text())
    validation = json.loads(VALIDATION.read_text())
    items = [json.loads(l) for l in (ROOT / amendment["items_path"]).open(encoding="utf-8")]
    expected_types = sorted({i["perturbation"] for i in items})

    judges = {}
    for path in sorted((ROOT / "artifacts/llm_judge/t4").glob("*.jsonl")):
        judge = path.stem
        original = ROOT / f"artifacts/llm_judge/validation/{judge}.jsonl"
        base = {
            (r["pool"], r["segment_key"], r["system_id"]): r["judge_quality"]
            for r in candidates(original)
        }
        rows = candidates(path)
        base_by_segment = {
            (r["pool"], r["segment_key"]): base[(r["pool"], r["segment_key"], r["system_id"].split("::")[0])]
            for r in rows
        }
        judges[judge] = {
            "parse_rate": sum(r["judge_quality"] is not None for r in rows) / len(rows),
            "detection": detection(base_by_segment, rows, "judge_quality"),
        }

    rewards = {}
    for path in sorted((ROOT / "artifacts/llm_judge/t4_qe").glob("*.jsonl")):
        pool_dir = ROOT / "artifacts/reward_models/confirmation/external/predictions"
        base = {}
        for pool in ("wmt23_zh_en", "wmt24_ja_zh", "wmt24_en_es_mqm"):
            source = pool_dir / pool / f"{path.stem}.jsonl"
            for r in candidates(source):
                base[(pool, r["segment_key"], r["system_id"])] = r["predicted_reward"]
        rows = candidates(path)
        base_by_segment = {
            (r["pool"], r["segment_key"]): base[(r["pool"], r["segment_key"], r["base_system"])]
            for r in rows
        }
        rewards[path.stem] = detection(base_by_segment, rows, "predicted_reward")

    selected = validation["decision"].get("selected_judge")
    decision = dict(validation["decision"])
    if selected in judges:
        det = judges[selected]["detection"]
        a5 = sorted(det) == expected_types and all(det[t]["rate"] >= THRESHOLD for t in det)
        decision["A5"] = a5
        decision["accepted"] = bool(decision.get("accepted") and a5)
    else:
        decision["A5"] = None
        decision["accepted"] = False
    result = {
        "amendment_sha256": AMENDMENT_SHA256,
        "threshold": THRESHOLD,
        "judges": judges,
        "external_qe_rewards": rewards,
        "final_decision": decision,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if decision.get("complete") and len(judges) == 3:
        root = json.loads(SSOT.read_text(encoding="utf-8"))
        root["llm_judge_validation_t4"] = result
        tmp = SSOT.with_suffix(SSOT.suffix + ".tmp")
        tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(SSOT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
