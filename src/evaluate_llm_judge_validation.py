"""Evaluate frozen LLM-judge validation tests T0-T3 and criteria A1-A4.

Implements results/llm_judge_protocol.json.  Judge selection uses T0/T1 only;
acceptance uses T3 on the selected judge.  Results are merged into the SSOT
under ``llm_judge_validation``.
"""

from __future__ import annotations

import hashlib
import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from mqm_human_scores import load_mqm_quality

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/llm_judge_protocol.json"
PROTOCOL_SHA256 = "172360f175fe106d44d4c666582e9a56a5eceb3ba6ff4d785b9618a0a9786013"
MQM_ROOT = Path.home() / "projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation"
JUDGE_ROOT = ROOT / "artifacts/llm_judge/validation"
EXTERNAL_ROOT = ROOT / "artifacts/reward_models/confirmation/external/predictions"
TRAINED_ROOTS = {
    "wmt23_zh_en": ROOT / "artifacts/reward_models/confirmation",
    "wmt24_ja_zh": ROOT / "artifacts/reward_models/confirmation",
    "wmt24_en_es_mqm": ROOT / "artifacts/reward_models/safe_pressure",
}
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261104
OUTPUT = ROOT / "results/llm_judge_validation.json"
SSOT = ROOT / "results/paper_stats.json"


def read_scores(path: Path, key: str) -> dict[tuple[str, str], float | None]:
    scores = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") != "candidate":
                continue
            scores[(row["segment_key"], row["system_id"])] = row[key]
    return scores


def matrix(scores: dict, segments: list[str], systems: list[str]) -> np.ndarray:
    out = np.full((len(segments), len(systems)), np.nan)
    for i, segment in enumerate(segments):
        for j, system in enumerate(systems):
            value = scores.get((segment, system))
            if value is not None:
                out[i, j] = value
    return out


def acc_eq(metric: np.ndarray, human: np.ndarray) -> float:
    """Tie-calibrated within-source pairwise accuracy (Deutsch et al. 2023)."""
    m_diffs, h_diffs = [], []
    for m_row, h_row in zip(metric, human):
        ok = np.isfinite(m_row) & np.isfinite(h_row)
        m, h = m_row[ok], h_row[ok]
        i, j = np.triu_indices(len(m), k=1)
        m_diffs.append(m[i] - m[j])
        h_diffs.append(h[i] - h[j])
    md, hd = np.concatenate(m_diffs), np.concatenate(h_diffs)
    h_sign = np.sign(np.where(np.abs(hd) < 1e-9, 0.0, hd))
    candidates = np.unique(np.concatenate([[0.0], np.quantile(np.abs(md), np.linspace(0, 0.5, 51))]))
    best = 0.0
    for eps in candidates:
        m_sign = np.where(np.abs(md) <= eps, 0.0, np.sign(md))
        best = max(best, float(np.mean(m_sign == h_sign)))
    return best


def tie_averaged(reward: np.ndarray, quality: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by reward ascending; replace quality by its mean within reward ties."""
    order = np.argsort(reward, kind="stable")
    r, q = reward[order], quality[order].astype(float)
    start = 0
    while start < len(r):
        end = start
        while end + 1 < len(r) and r[end + 1] == r[start]:
            end += 1
        q[start:end + 1] = q[start:end + 1].mean()
        start = end + 1
    return r, q


def selection_values(reward_row: np.ndarray, quality_row: np.ndarray, rule: str) -> float:
    """Selected quality minus uniform mean for one segment."""
    _, q = tie_averaged(reward_row, quality_row)
    n = len(q)
    if rule == "top1":
        selected = q[-1]
    elif rule == "best_of_4":
        k = min(4, n)
        weights = np.array([comb(i, k - 1) for i in range(n)], dtype=float) / comb(n, k)
        selected = float(weights @ q)
    else:
        raise ValueError(rule)
    return float(selected - q.mean())


def gains(reward: np.ndarray, quality: np.ndarray, rule: str) -> np.ndarray:
    return np.array([selection_values(r, q, rule) for r, q in zip(reward, quality)])


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def reward_sources(pool: str) -> dict[str, Path]:
    sources = {}
    for seed_dir in sorted(TRAINED_ROOTS[pool].glob("2*")):
        for path in sorted((seed_dir / "predictions" / pool).glob("*.jsonl")):
            sources[f"trained:{seed_dir.name}:{path.stem}"] = path
    for path in sorted((EXTERNAL_ROOT / pool).glob("*.jsonl")):
        sources[f"external:{path.stem}"] = path
    return sources


def merge_ssot(result: dict[str, Any]) -> None:
    root = json.loads(SSOT.read_text(encoding="utf-8"))
    root["llm_judge_validation"] = result
    temporary = SSOT.with_suffix(SSOT.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(SSOT)


def main() -> None:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("LLM judge protocol differs from frozen hash")
    protocol = json.loads(PROTOCOL.read_text())
    criteria = protocol["acceptance_criteria"]
    judges = [j for j in protocol["judges"] if (JUDGE_ROOT / f"{j}.jsonl").exists()]
    missing = sorted(set(protocol["judges"]) - set(judges))
    if not judges:
        raise FileNotFoundError("No judge outputs found")

    pools: dict[str, Any] = {}
    data: dict[str, Any] = {}
    for pool_name, pool in protocol["pools"].items():
        segments, systems = pool["sampled_segments"], pool["systems"]
        human, human_stats = load_mqm_quality(MQM_ROOT / pool["source"], segments, systems)
        human = human.astype(float)
        judge_q = {
            j: matrix(read_scores(JUDGE_ROOT / f"{j}.jsonl", "judge_quality"), segments, systems)
            for j in judges
        }
        rewards = {
            name: matrix(read_scores(path, "predicted_reward"), segments, systems)
            for name, path in reward_sources(pool_name).items()
        }
        incomplete = [n for n, r in rewards.items() if not np.isfinite(r).all()]
        if incomplete:
            raise ValueError(f"{pool_name}: incomplete reward coverage for {incomplete}")
        data[pool_name] = (human, judge_q, rewards)
        pools[pool_name] = {
            "human_mqm": human_stats,
            "T0_parse_rate": {j: float(np.isfinite(q).mean()) for j, q in judge_q.items()},
            "T1_acc_eq": {
                **{f"judge:{j}": acc_eq(q, human) for j, q in judge_q.items()},
                **{n: acc_eq(r, human) for n, r in rewards.items() if n.startswith("external:")},
            },
            "T2_system_pearson": {
                j: float(pearsonr(np.nanmean(q, axis=0), human.mean(axis=0))[0])
                for j, q in judge_q.items()
            },
        }

    eligible = [
        j for j in judges
        if all(pools[p]["T0_parse_rate"][j] >= 0.95 for p in pools)
    ]
    mean_t1 = {j: float(np.mean([pools[p]["T1_acc_eq"][f"judge:{j}"] for p in pools])) for j in judges}
    selected = max(eligible, key=lambda j: mean_t1[j]) if eligible else None

    cells = []
    if selected is not None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for pool_name, (human, judge_q, rewards) in data.items():
            judge = judge_q[selected]
            complete = np.isfinite(judge).all(axis=1)
            h_z = human / human.std()
            j_z = judge / np.nanstd(judge)
            for reward_name, reward in rewards.items():
                for rule in ("top1", "best_of_4"):
                    hg = gains(reward[complete], h_z[complete], rule)
                    jg = gains(reward[complete], j_z[complete], rule)
                    cells.append(
                        {
                            "pool": pool_name,
                            "reward": reward_name,
                            "rule": rule,
                            "segments": int(complete.sum()),
                            "human_gain": float(hg.mean()),
                            "human_gain_ci95": bootstrap_ci(hg, rng),
                            "judge_gain": float(jg.mean()),
                            "judge_gain_ci95": bootstrap_ci(jg, rng),
                        }
                    )

    t3: dict[str, Any] = {"cells": cells}
    decision: dict[str, Any] = {"selected_judge": selected, "judges_missing": missing}
    if cells:
        h = np.array([c["human_gain"] for c in cells])
        jv = np.array([c["judge_gain"] for c in cells])
        sig = [c for c in cells if c["human_gain_ci95"][0] > 0 or c["human_gain_ci95"][1] < 0]
        neg = [c for c in cells if c["human_gain_ci95"][1] < 0]
        sign_agree = float(np.mean([np.sign(c["judge_gain"]) == np.sign(c["human_gain"]) for c in sig])) if sig else None
        neg_detect = float(np.mean([c["judge_gain"] < 0 for c in neg])) if neg else None
        t3.update(
            {
                "cells_total": len(cells),
                "cells_human_significant": len(sig),
                "cells_human_significant_negative": len(neg),
                "sign_agreement_significant": sign_agree,
                "spearman_all_cells": float(spearmanr(h, jv)[0]),
                "negative_detection_rate": neg_detect,
            }
        )
        a1 = all(pools[p]["T0_parse_rate"][selected] >= 0.95 for p in pools)
        a2 = sign_agree is not None and sign_agree >= 0.80
        a3 = t3["spearman_all_cells"] >= 0.60
        if len(neg) < 5:
            a4 = "underpowered"
        else:
            a4 = "pass" if neg_detect >= 0.80 else "fail"
        decision.update(
            {
                "A1": a1, "A2": a2, "A3": a3, "A4": a4,
                "accepted": bool(a1 and a2 and a3 and a4 != "fail"),
                "complete": not missing,
            }
        )
    result = {
        "protocol_sha256": PROTOCOL_SHA256,
        "criteria_text": criteria,
        "judges_evaluated": judges,
        "judge_mean_T1_acc_eq": mean_t1,
        "judges_eligible_T0": eligible,
        "pools": pools,
        "T3": t3,
        "decision": decision,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not missing:
        merge_ssot(result)
    summary = {k: v for k, v in t3.items() if k != "cells"}
    print(json.dumps({"mean_T1": mean_t1, "T3": summary, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
