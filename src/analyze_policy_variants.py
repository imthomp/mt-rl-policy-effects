"""Paired bootstrap analysis of decoder-policy variant calibration."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


INPUT = Path("artifacts/diagnostics/nllb600m_policy_variants.json")
OUTPUT = Path("artifacts/diagnostics/nllb600m_policy_variants_analysis.json")
EOS_ID = 2
BOOTSTRAPS = 10_000
SEED = 17


def chrf_proxy(hypothesis: str, reference: str) -> float:
    orders = []
    for n in range(1, 7):
        hyp = Counter(hypothesis[i : i + n] for i in range(max(0, len(hypothesis) - n + 1)))
        ref = Counter(reference[i : i + n] for i in range(max(0, len(reference) - n + 1)))
        overlap = sum((hyp & ref).values())
        precision = overlap / max(1, sum(hyp.values()))
        recall = overlap / max(1, sum(ref.values()))
        orders.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return 100 * sum(orders) / len(orders)


def main() -> None:
    artifact = json.loads(INPUT.read_text(encoding="utf-8"))
    conditions = {row["name"]: row for row in artifact["conditions"]}
    baseline = conditions["baseline"]["rows"]
    rng = random.Random(SEED)
    results = []

    for name, treatment in conditions.items():
        if name == "baseline":
            continue
        paired = {metric: [] for metric in ("chrf_proxy", "tokens_per_word", "characters", "words")}
        for base_row, row in zip(baseline, treatment["rows"], strict=True):
            base_eos = bool(base_row["sampled_token_ids"] and base_row["sampled_token_ids"][-1] == EOS_ID)
            row_eos = bool(row["sampled_token_ids"] and row["sampled_token_ids"][-1] == EOS_ID)
            base_words = max(1, len(base_row["hypothesis"].split()))
            row_words = max(1, len(row["hypothesis"].split()))
            base_tokens = len(base_row["sampled_token_ids"]) - int(base_eos)
            row_tokens = len(row["sampled_token_ids"]) - int(row_eos)
            paired["chrf_proxy"].append(
                chrf_proxy(row["hypothesis"], row["reference"])
                - chrf_proxy(base_row["hypothesis"], base_row["reference"])
            )
            paired["tokens_per_word"].append(row_tokens / row_words - base_tokens / base_words)
            paired["characters"].append(len(row["hypothesis"]) - len(base_row["hypothesis"]))
            paired["words"].append(row_words - base_words)

        n = len(baseline)
        metrics = {}
        for metric, deltas in paired.items():
            boot = sorted(
                sum(deltas[rng.randrange(n)] for _ in range(n)) / n
                for _ in range(BOOTSTRAPS)
            )
            metrics[metric] = {
                "mean_paired_delta": sum(deltas) / n,
                "bootstrap_95_percentile_interval": [boot[249], boot[9749]],
            }
        results.append({"name": name, "beta": treatment["beta"], "mixture": treatment["mixture"], "feature": treatment["feature"], "metrics": metrics})

    result = {
        "input": str(INPUT),
        "calibration_examples": len(baseline),
        "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "pairing": "same source/reference row across separately sampled conditions",
        "quality_metric": "custom character n-gram F-score proxy, orders 1..6; not official chrF++",
        "results_vs_baseline": results,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
