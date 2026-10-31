"""Compare parallel policy-dose results with the same-seed baseline screen."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


INPUTS = [
    Path(f"artifacts/diagnostics/nllb600m_policy_variant_matched_cs_{index}.json")
    for index in range(5)
]
OUTPUT = Path("artifacts/diagnostics/nllb600m_policy_variant_array_analysis.json")
EOS_ID = 2
BOOTSTRAPS = 10_000
SEED = 29


def chrf_proxy(hypothesis: str, reference: str) -> float:
    scores = []
    for order in range(1, 7):
        hyp = Counter(hypothesis[i : i + order] for i in range(max(0, len(hypothesis) - order + 1)))
        ref = Counter(reference[i : i + order] for i in range(max(0, len(reference) - order + 1)))
        overlap = sum((hyp & ref).values())
        precision = overlap / max(1, sum(hyp.values()))
        recall = overlap / max(1, sum(ref.values()))
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return 100 * sum(scores) / len(scores)


def main() -> None:
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in INPUTS]
    baseline_artifact = next(
        artifact for artifact in artifacts
        if artifact["conditions"][0]["name"] == "baseline_a100"
    )
    baseline = baseline_artifact["conditions"][0]["rows"]
    rng = random.Random(SEED)
    results = []

    for input_path, artifact in zip(INPUTS, artifacts, strict=True):
        treatment = artifact["conditions"][0]
        if treatment["name"] == "baseline_a100":
            continue
        rows = treatment["rows"]
        if len(rows) != len(baseline):
            raise ValueError(f"Mismatched row count in {input_path}")
        paired = {key: [] for key in ("chrf_proxy", "tokens_per_word", "characters", "words")}
        for base_row, row in zip(baseline, rows, strict=True):
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

        metrics = {}
        n = len(rows)
        for metric, deltas in paired.items():
            bootstrap = sorted(
                sum(deltas[rng.randrange(n)] for _ in range(n)) / n
                for _ in range(BOOTSTRAPS)
            )
            metrics[metric] = {
                "mean_paired_delta": sum(deltas) / n,
                "bootstrap_95_percentile_interval": [bootstrap[249], bootstrap[9749]],
            }
        metrics["mean_abs_sampling_vs_rescore_logprob_error"] = treatment[
            "mean_abs_sampling_vs_rescore_logprob_error"
        ]
        metrics["sampled_eos_rate"] = treatment["sampled_eos_rate"]
        metrics["max_length_without_eos_count"] = treatment["max_length_without_eos_count"]
        results.append(
            {
                "name": treatment["name"],
                "feature": treatment["feature"],
                "beta": treatment["beta"],
                "mixture": treatment["mixture"],
                "device": artifact["device"],
                "metrics": metrics,
            }
        )

    output = {
        "matched_baseline_input": str(INPUTS[4]),
        "inputs": [str(path) for path in INPUTS],
        "calibration_examples": len(baseline),
        "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "pairing": "same WMT23 source/reference rows; treatment and baseline sampled separately with the same seed",
        "quality_metric": "custom character n-gram F-score proxy, orders 1..6; not official chrF++",
        "results_vs_baseline": results,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
