"""Merge the sole prespecified second rollout sample for Tatoeba calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from evaluate_nllb_policy import group_rows_by_source, read_rows, validate_protocol_split
from sacrebleu.metrics import BLEU


INITIAL_SEED = 812701
EXTENSION_SEED = 812801
BOOTSTRAP_SEED = 812702
BOOTSTRAP_RESAMPLES = 2000
BLEU = BLEU(tokenize="13a", lowercase=False, smooth_method="exp",
            smooth_value=None, max_ngram_order=4, effective_order=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_cluster_interval(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty source set")
    generator = random.Random(BOOTSTRAP_SEED)
    means = sorted(
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "lower_95": means[round(0.025 * (len(means) - 1))],
        "upper_95": means[round(0.975 * (len(means) - 1))],
    }


def assert_compatible(first: dict[str, Any], second: dict[str, Any]) -> None:
    for key in ("data_sha256", "model_snapshot_hash", "tilt_beta", "tilt_temperature"):
        if first.get(key) != second.get(key):
            raise ValueError(f"calibration artifacts disagree on {key}")
    if first.get("base_seed") != INITIAL_SEED or second.get("base_seed") != EXTENSION_SEED:
        raise ValueError("artifacts must use the frozen initial and extension rollout seeds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    first, second = read_json(args.initial), read_json(args.extension)
    assert_compatible(first, second)
    if not first.get("entropy_match_pass") or not second.get("entropy_match_pass"):
        raise ValueError("the entropy-match gate must pass before the rollout extension is analyzed")
    if first.get("screening_decisions", {}).get("overall_calibration_screen") != "inconclusive_one_extension_allowed":
        raise ValueError("the bounded extension is only allowed after an inconclusive first calibration")
    split_name, _ = validate_protocol_split(args.data, {"calibration"})
    rows = group_rows_by_source(read_rows(args.data, None))
    if first.get("data_split") != split_name or second.get("data_split") != split_name:
        raise ValueError("artifacts do not identify the frozen calibration split")
    if first.get("source_count") != len(rows) or second.get("source_count") != len(rows):
        raise ValueError("calibration source count does not match the frozen split")

    paired: dict[str, dict[str, Any]] = {}
    for policy in ("control", "length_tilt"):
        records_by_run = [
            artifact["sampled_outputs"][policy]
            for artifact in (first, second)
        ]
        for records in records_by_run:
            if [record["source_sentence_id"] for record in records] != [
                row.get("source_sentence_id") for row in rows
            ]:
                raise ValueError(f"{policy} outputs are not aligned to the frozen sources")
        paired[policy] = {"records_by_run": records_by_run}

    fields = {
        "mean_reference_chrfpp_reward": "chrF++_reward",
        "fertility_tokens_per_word": "fertility_tokens_per_word",
        "action_count_including_eos": "actions_including_eos",
        "word_count": "decoded_word_count",
    }
    differences: dict[str, list[float]] = {}
    for field, output_name in fields.items():
        deltas = []
        for index in range(len(rows)):
            left_values = [run[index][field] for run in paired["length_tilt"]["records_by_run"]]
            right_values = [run[index][field] for run in paired["control"]["records_by_run"]]
            left_values = [float(value) for value in left_values if value is not None]
            right_values = [float(value) for value in right_values if value is not None]
            if left_values and right_values:
                deltas.append(sum(left_values) / len(left_values) - sum(right_values) / len(right_values))
        differences[output_name] = deltas

    paired_differences = {
        name: {
            "n_source_groups": len(values),
            "mean_length_tilt_minus_control": sum(values) / len(values),
            "paired_source_bootstrap_95_interval": source_cluster_interval(values),
        }
        for name, values in differences.items()
    }

    # For corpus BLEU, average the two replicate-level corpus-score differences.
    replicate_bleu_deltas = []
    for run_index in range(2):
        left = paired["length_tilt"]["records_by_run"][run_index]
        right = paired["control"]["records_by_run"][run_index]
        max_references = max(len(row["references"]) for row in rows)
        reference_streams = [
            [
                row["references"][reference_index]["text"]
                if reference_index < len(row["references"]) else ""
                for row in rows
            ]
            for reference_index in range(max_references)
        ]
        replicate_bleu_deltas.append(
            BLEU.corpus_score([record["hypothesis"] for record in left], reference_streams).score
            - BLEU.corpus_score([record["hypothesis"] for record in right], reference_streams).score
        )

    generator = random.Random(BOOTSTRAP_SEED)
    bootstrap_bleu_deltas = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = [generator.randrange(len(rows)) for _ in rows]
        max_references = max(len(rows[index]["references"]) for index in indices)
        reference_streams = [
            [
                rows[index]["references"][reference_index]["text"]
                if reference_index < len(rows[index]["references"]) else ""
                for index in indices
            ]
            for reference_index in range(max_references)
        ]
        run_deltas = []
        for run_index in range(2):
            left = paired["length_tilt"]["records_by_run"][run_index]
            right = paired["control"]["records_by_run"][run_index]
            run_deltas.append(
                BLEU.corpus_score([left[index]["hypothesis"] for index in indices], reference_streams).score
                - BLEU.corpus_score([right[index]["hypothesis"] for index in indices], reference_streams).score
            )
        bootstrap_bleu_deltas.append(sum(run_deltas) / len(run_deltas))
    bootstrap_bleu_deltas.sort()
    paired_differences["corpus_BLEU_points"] = {
        "n_source_groups": len(rows),
        "mean_length_tilt_minus_control": sum(replicate_bleu_deltas) / len(replicate_bleu_deltas),
        "paired_source_bootstrap_95_interval": {
            "lower_95": bootstrap_bleu_deltas[round(0.025 * (len(bootstrap_bleu_deltas) - 1))],
            "upper_95": bootstrap_bleu_deltas[round(0.975 * (len(bootstrap_bleu_deltas) - 1))],
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
    }

    def classify(interval: dict[str, float], margin: float) -> str:
        if interval["lower_95"] >= -margin:
            return "pass"
        if interval["upper_95"] < -margin:
            return "fail"
        return "inconclusive"

    quality_decisions = {
        "chrF++_reward": classify(
            paired_differences["chrF++_reward"]["paired_source_bootstrap_95_interval"],
            0.03,
        ),
        "corpus_BLEU_points": classify(
            paired_differences["corpus_BLEU_points"]["paired_source_bootstrap_95_interval"],
            2.0,
        ),
    }
    treatment_records = [
        record
        for run in paired["length_tilt"]["records_by_run"]
        for record in run
    ]
    output_count = len(treatment_records)
    validity_rates = {
        "eos_rate": sum(record["eos_emitted"] for record in treatment_records) / output_count,
        "truncation_rate": sum(record["cap_hit"] for record in treatment_records) / output_count,
        "empty_output_rate": sum(record["empty_output"] for record in treatment_records) / output_count,
        "non_eos_special_token_emission_rate": sum(
            record["non_eos_special_token_count"] > 0 for record in treatment_records
        ) / output_count,
    }
    validity_decisions = {
        "eos_rate": "pass" if validity_rates["eos_rate"] >= 0.99 else "fail",
        "truncation_rate": "pass" if validity_rates["truncation_rate"] <= 0.01 else "fail",
        "empty_output_rate": "pass" if validity_rates["empty_output_rate"] <= 0.005 else "fail",
        "non_eos_special_token_emission_rate": (
            "pass" if validity_rates["non_eos_special_token_emission_rate"] <= 0.01 else "fail"
        ),
    }
    if "fail" in (*quality_decisions.values(), *validity_decisions.values()):
        overall = "fail"
    elif "inconclusive" in quality_decisions.values():
        overall = "inconclusive_after_single_extension"
    else:
        overall = "pass"

    output = {
        "status": "single-prespecified-calibration-extension",
        "initial_artifact": str(args.initial),
        "extension_artifact": str(args.extension),
        "data_path": str(args.data),
        "data_sha256": first["data_sha256"],
        "model_snapshot_hash": first["model_snapshot_hash"],
        "rollout_seeds": [INITIAL_SEED, EXTENSION_SEED],
        "source_count": len(rows),
        "paired_differences": paired_differences,
        "screening_decisions": {
            "quality": quality_decisions,
            "validity_rates": validity_rates,
            "validity_counts": {
                "generated_output_count": output_count,
                "eos_output_count": sum(record["eos_emitted"] for record in treatment_records),
                "truncated_output_count": sum(record["cap_hit"] for record in treatment_records),
                "empty_output_count": sum(record["empty_output"] for record in treatment_records),
                "non_eos_special_token_output_count": sum(
                    record["non_eos_special_token_count"] > 0 for record in treatment_records
                ),
                "non_eos_special_token_id_count": sum(
                    record["non_eos_special_token_count"] for record in treatment_records
                ),
            },
            "validity": validity_decisions,
            "overall_calibration_screen": overall,
        },
        "sampling_scope": "one output per policy, source, and seed; source is the bootstrap unit, and replicate-level corpus BLEU differences are averaged",
        "interpretation": "Use only to resolve an inconclusive first calibration. This is the sole permitted increase in calibration sampling.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
