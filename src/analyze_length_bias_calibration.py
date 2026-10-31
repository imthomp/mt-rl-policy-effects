"""Paired bootstrap analysis for the NLLB length-bias calibration artifact."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


INPUT = Path("artifacts/diagnostics/nllb600m_length_bias_dose_calibration.json")
OUTPUT = Path("artifacts/diagnostics/nllb600m_length_bias_dose_analysis.json")
EOS_ID = 2  # facebook/nllb-200-distilled-600M tokenizer EOS id
BOOTSTRAPS = 10_000
SEED = 5


def chrf_proxy(hypothesis: str, reference: str) -> float:
    """Match the calibration's character n-gram F-score proxy (orders 1..6)."""
    scores = []
    for order in range(1, 7):
        hyp = Counter(
            hypothesis[i : i + order]
            for i in range(max(0, len(hypothesis) - order + 1))
        )
        ref = Counter(
            reference[i : i + order]
            for i in range(max(0, len(reference) - order + 1))
        )
        overlap = sum((hyp & ref).values())
        precision = overlap / max(1, sum(hyp.values()))
        recall = overlap / max(1, sum(ref.values()))
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return 100 * sum(scores) / len(scores)


def percentile(sorted_values: list[float], probability: float) -> float:
    return sorted_values[int(probability * (len(sorted_values) - 1))]


def main() -> None:
    artifact = json.loads(INPUT.read_text(encoding="utf-8"))
    by_beta = {condition["beta"]: condition for condition in artifact["conditions"]}
    baseline = by_beta[0.0]["rows"]
    rng = random.Random(SEED)
    results = []

    for beta in (0.2, 0.3):
        treatment = by_beta[beta]["rows"]
        paired = {name: [] for name in ("chrf_proxy", "tokens_per_word", "characters", "words")}
        for base_row, treatment_row in zip(baseline, treatment, strict=True):
            base_tokens = len(base_row["sampled_token_ids"]) - int(
                bool(base_row["sampled_token_ids"])
                and base_row["sampled_token_ids"][-1] == EOS_ID
            )
            treatment_tokens = len(treatment_row["sampled_token_ids"]) - int(
                bool(treatment_row["sampled_token_ids"])
                and treatment_row["sampled_token_ids"][-1] == EOS_ID
            )
            base_words = max(1, len(base_row["hypothesis"].split()))
            treatment_words = max(1, len(treatment_row["hypothesis"].split()))
            paired["chrf_proxy"].append(
                chrf_proxy(treatment_row["hypothesis"], treatment_row["reference"])
                - chrf_proxy(base_row["hypothesis"], base_row["reference"])
            )
            paired["tokens_per_word"].append(
                treatment_tokens / treatment_words - base_tokens / base_words
            )
            paired["characters"].append(
                len(treatment_row["hypothesis"]) - len(base_row["hypothesis"])
            )
            paired["words"].append(treatment_words - base_words)

        n = len(baseline)
        metrics = {}
        for name, differences in paired.items():
            means = [
                sum(differences[rng.randrange(n)] for _ in range(n)) / n
                for _ in range(BOOTSTRAPS)
            ]
            means.sort()
            metrics[name] = {
                "mean_paired_delta": sum(differences) / n,
                "bootstrap_95_percentile_interval": [
                    percentile(means, 0.025),
                    percentile(means, 0.975),
                ],
            }

        metrics["eos_rate"] = sum(
            bool(row["sampled_token_ids"])
            and row["sampled_token_ids"][-1] == EOS_ID
            for row in treatment
        ) / len(treatment)
        metrics["max_length_without_eos_count"] = sum(
            len(row["sampled_token_ids"]) == 256
            and (
                not row["sampled_token_ids"]
                or row["sampled_token_ids"][-1] != EOS_ID
            )
            for row in treatment
        )
        results.append({"beta": beta, "metrics": metrics})

    output = {
        "input": str(INPUT),
        "calibration_examples": len(baseline),
        "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "pairing": "same source/reference row across separately sampled policy conditions",
        "eos_token_id": EOS_ID,
        "quality_metric": "custom character n-gram F-score proxy over orders 1..6; not official chrF++",
        "results_vs_beta_0": results,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
