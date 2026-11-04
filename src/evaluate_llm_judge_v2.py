"""Evaluate LLM-judge validation v2 (results/llm_judge_protocol_v2.json)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from evaluate_llm_judge_validation import (
    BOOTSTRAP_SEED, MQM_ROOT, acc_eq, bootstrap_ci, gains, matrix, read_scores, reward_sources,
)
from hack_guards import flags
from mqm_human_scores import load_mqm_quality
from score_mqm_confirmation_candidates import load_candidates

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/llm_judge_protocol_v2.json"
PROTOCOL_SHA256 = "c56dd6a38e4545b471d4ab9cc63504999137de6bdb004b4d1d29bf0655fd75c5"
JUDGE_ROOT = ROOT / "artifacts/llm_judge/v2"
OUTPUT = ROOT / "results/llm_judge_validation_v2.json"
SSOT = ROOT / "results/paper_stats.json"
GUARD_PENALTY = 25.0


def main() -> None:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("v2 protocol hash mismatch")
    protocol = json.loads(PROTOCOL.read_text())
    judges = [j for j in protocol["judges"] if (JUDGE_ROOT / "validation" / f"{j}.jsonl").exists()]
    missing = sorted(set(protocol["judges"]) - set(judges))

    pools: dict[str, Any] = {}
    data: dict[str, Any] = {}
    for pool_name, pool in protocol["pools"].items():
        segments, systems = pool["sampled_segments"], pool["systems"]
        lang = pool["target_lang"]
        human, human_stats = load_mqm_quality(MQM_ROOT / pool["source"], segments, systems)
        human = human.astype(float)
        _, rows = load_candidates(MQM_ROOT / pool["source"], ROOT / pool["manifest"], pool_name)
        text = {(r["segment_key"], r["system_id"]): r for r in rows}
        guard = np.array([[any(flags(text[(s, y)]["hypothesis"], text[(s, y)]["source"], lang).values())
                           for y in systems] for s in segments])
        candidates = {}
        for j in judges:
            q = matrix(read_scores(JUDGE_ROOT / "validation" / f"{j}.jsonl", "judge_quality"),
                       segments, systems)
            candidates[f"{j}"] = q
            candidates[f"{j}+guards"] = q - GUARD_PENALTY * guard
        rewards = {n: matrix(read_scores(p, "predicted_reward"), segments, systems)
                   for n, p in reward_sources(pool_name).items()}
        if any(not np.isfinite(r).all() for r in rewards.values()):
            raise ValueError(f"{pool_name}: incomplete reward coverage")
        good = human >= np.median(human)
        data[pool_name] = (human, candidates, rewards)
        pools[pool_name] = {
            "human_mqm": human_stats,
            "T0_parse_rate": {c: float(np.isfinite(q).mean()) for c, q in candidates.items()},
            "T1_acc_eq": {**{c: acc_eq(q, human) for c, q in candidates.items()},
                          **{n: acc_eq(r, human) for n, r in rewards.items() if n.startswith("external:")}},
            "T2_system_pearson": {c: float(pearsonr(np.nanmean(q, axis=0), human.mean(axis=0))[0])
                                  for c, q in candidates.items()},
            "G1_guard_false_positive": float(guard[good].mean()),
            "guard_fire_rate_all": float(guard.mean()),
        }

    names = [c for c in pools[next(iter(pools))]["T0_parse_rate"]]
    eligible = [c for c in names if all(pools[p]["T0_parse_rate"][c] >= 0.95 for p in pools)]
    mean_t1 = {c: float(np.mean([pools[p]["T1_acc_eq"][c] for p in pools])) for c in names}
    selected = max(eligible, key=lambda c: mean_t1[c]) if eligible else None

    cells = []
    if selected is not None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for pool_name, (human, candidates, rewards) in data.items():
            judge = candidates[selected]
            ok = np.isfinite(judge).all(axis=1)
            h_z, j_z = human / human.std(), judge / np.nanstd(judge)
            for reward_name, reward in rewards.items():
                for rule in ("top1", "best_of_4"):
                    hg, jg = gains(reward[ok], h_z[ok], rule), gains(reward[ok], j_z[ok], rule)
                    cells.append({"pool": pool_name, "reward": reward_name, "rule": rule,
                                  "segments": int(ok.sum()), "human_gain": float(hg.mean()),
                                  "human_gain_ci95": bootstrap_ci(hg, rng),
                                  "judge_gain": float(jg.mean()), "judge_gain_ci95": bootstrap_ci(jg, rng)})

    # T4 on v2 items, for judge-only and hybrid candidates.
    t4: dict[str, Any] = {}
    items = [json.loads(l) for l in (ROOT / protocol["t4"]["items_path"]).open(encoding="utf-8")]
    langs = {p: protocol["pools"][p]["target_lang"] for p in protocol["pools"]}
    item_guard = {(i["pool"], i["segment_key"], i["perturbation"]):
                  any(flags(i["hypothesis"], i["source"], langs[i["pool"]]).values()) for i in items}
    for j in judges:
        t4_path = JUDGE_ROOT / "t4" / f"{j}.jsonl"
        if not t4_path.exists():
            continue
        base_q = read_scores(JUDGE_ROOT / "validation" / f"{j}.jsonl", "judge_quality")
        out = {j: defaultdict(lambda: [0, 0]), f"{j}+guards": defaultdict(lambda: [0, 0])}
        for line in t4_path.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("record_type") != "candidate":
                continue
            base_sys, kind = r["system_id"].split("::")
            b, p = base_q.get((r["segment_key"], base_sys)), r["judge_quality"]
            if b is None or p is None:
                continue
            fired = item_guard[(r["pool"], r["segment_key"], kind)]
            for name, pq in ((j, p), (f"{j}+guards", p - GUARD_PENALTY * fired)):
                out[name][kind][0] += pq < b
                out[name][kind][1] += 1
        for name, per in out.items():
            t4[name] = {k: {"rate": h / n, "n": n} for k, (h, n) in sorted(per.items())}

    decision: dict[str, Any] = {"selected": selected, "judges_missing": missing}
    t3: dict[str, Any] = {"cells": cells}
    if cells:
        h = np.array([c["human_gain"] for c in cells]); jv = np.array([c["judge_gain"] for c in cells])
        sig = [c for c in cells if c["human_gain_ci95"][0] > 0 or c["human_gain_ci95"][1] < 0]
        neg = [c for c in cells if c["human_gain_ci95"][1] < 0]
        sign = float(np.mean([np.sign(c["judge_gain"]) == np.sign(c["human_gain"]) for c in sig])) if sig else None
        t3.update({"cells_total": len(cells), "cells_human_significant": len(sig),
                   "cells_human_significant_negative": len(neg), "sign_agreement_significant": sign,
                   "spearman_all_cells": float(spearmanr(h, jv)[0]),
                   "negative_detection_rate": float(np.mean([c["judge_gain"] < 0 for c in neg])) if neg else None})
        det = t4.get(selected, {})
        a5 = bool(det) and len(det) == len(protocol["t4"]["perturbations"]) and all(v["rate"] >= 0.90 for v in det.values())
        g1 = all(pools[p]["G1_guard_false_positive"] <= 0.02 for p in pools) if selected.endswith("+guards") else None
        a4 = "underpowered" if len(neg) < 5 else ("pass" if t3["negative_detection_rate"] >= 0.80 else "fail")
        decision.update({
            "A1": True, "A2": sign is not None and sign >= 0.80, "A3": t3["spearman_all_cells"] >= 0.60,
            "A4": a4, "A5": a5, "G1": g1, "complete": not missing and len(t4) == 2 * len(protocol["judges"]),
        })
        decision["accepted"] = bool(decision["A2"] and decision["A3"] and a5 and a4 != "fail" and g1 is not False)
    result = {"protocol_sha256": PROTOCOL_SHA256, "candidates_mean_T1": mean_t1, "eligible": eligible,
              "pools": pools, "T3": t3, "T4": t4, "decision": decision}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if decision.get("complete"):
        root = json.loads(SSOT.read_text(encoding="utf-8"))
        root["llm_judge_validation_v2"] = result
        tmp = SSOT.with_suffix(SSOT.suffix + ".tmp")
        tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(SSOT)
    print(json.dumps({"mean_T1": mean_t1, "eligible": eligible,
                      "T3": {k: v for k, v in t3.items() if k != "cells"},
                      "T4": {k: {t: round(x["rate"], 3) for t, x in v.items()} for k, v in t4.items()},
                      "G1": {p: round(v["G1_guard_false_positive"], 4) for p, v in pools.items()},
                      "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
