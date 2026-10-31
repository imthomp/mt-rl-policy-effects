"""Align NLLB evaluation outputs and calculate paired seed contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import os
from pathlib import Path
from typing import Any

from sacrebleu.metrics import BLEU


BLEU = BLEU(
    tokenize="13a", lowercase=False, smooth_method="exp", smooth_value=None,
    max_ngram_order=4, effective_order=False,
)


def read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"empty predictions file: {path}")
    return rows


def row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("row_index"), row.get("source_sentence_id"),
        row["source"],
        tuple((ref.get("sentence_id"), ref.get("text")) for ref in row["references"]),
    )


def source_clusters(rows: list[dict[str, Any]]) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        key = row.get("source_sentence_id")
        if key is None:
            key = row["source"]
        groups.setdefault(str(key), []).append(index)
    return list(groups.values())


def corpus_bleu(hypotheses: list[str], reference_rows: list[list[dict[str, Any]]]) -> float:
    max_refs = max(len(row) for row in reference_rows)
    streams = [
        [row[index]["text"] if index < len(row) else "" for row in reference_rows]
        for index in range(max_refs)
    ]
    return BLEU.corpus_score(hypotheses, streams).score


def bootstrap_paired_bleu(
    control: list[dict[str, Any]], treatment: list[dict[str, Any]],
    replicates: int, seed: int,
) -> dict[str, float]:
    clusters = source_clusters(control)
    rng = random.Random(seed)
    deltas = []
    for _ in range(replicates):
        chosen = [rng.randrange(len(clusters)) for _ in clusters]
        indices = [index for cluster_index in chosen for index in clusters[cluster_index]]
        sampled_refs = [control[index]["references"] for index in indices]
        control_hyps = [control[index]["common_beam4"]["hypothesis"] for index in indices]
        treatment_hyps = [treatment[index]["common_beam4"]["hypothesis"] for index in indices]
        deltas.append(corpus_bleu(treatment_hyps, sampled_refs) - corpus_bleu(control_hyps, sampled_refs))
    deltas.sort()
    return {
        "replicates": replicates,
        "seed": seed,
        "source_clusters": len(clusters),
        "ci_95_low": deltas[int(0.025 * replicates)],
        "ci_95_high": deltas[min(replicates - 1, int(0.975 * replicates))],
    }


def mean_reward_by_source(rows: list[dict[str, Any]], beta_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        key = str(row.get("source_sentence_id") or row["source"])
        sample_rewards = [sample["sentence_chrf_reward"] for sample in row["stochastic"][beta_key]]
        grouped.setdefault(key, []).append(sum(sample_rewards) / len(sample_rewards))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def bootstrap_mean_reward_difference(
    control: list[dict[str, Any]], treatment: list[dict[str, Any]],
    control_beta: str, treatment_beta: str, replicates: int, seed: int,
) -> dict[str, float]:
    control_values = mean_reward_by_source(control, control_beta)
    treatment_values = mean_reward_by_source(treatment, treatment_beta)
    keys = sorted(set(control_values) & set(treatment_values))
    if len(keys) != len(control_values) or len(keys) != len(treatment_values):
        raise ValueError("source clusters do not align for stochastic reward bootstrap")
    differences = [treatment_values[key] - control_values[key] for key in keys]
    rng = random.Random(seed)
    samples = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(replicates)
    )
    return {
        "source_clusters": len(keys),
        "mean_sentence_chrf_reward_difference": sum(differences) / len(differences),
        "ci_95_low": samples[int(0.025 * replicates)],
        "ci_95_high": samples[min(replicates - 1, int(0.975 * replicates))],
        "replicates": replicates,
        "seed": seed,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", action="append", required=True,
                        help="Evaluation metadata: LABEL=metrics_LABEL.json; repeat for each checkpoint")
    parser.add_argument("--paired-seed", action="append", required=True,
                        help="Paired run: SEED:CONTROL_LABEL:TREATMENT_LABEL; repeat for seeds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=582019)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("use at least 100 bootstrap replicates")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)

    metadata: dict[str, dict[str, Any]] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    for spec in args.metrics:
        if "=" not in spec:
            raise ValueError(f"bad --metrics value {spec!r}")
        label, raw_path = spec.split("=", 1)
        if label in metadata:
            raise ValueError(f"duplicate evaluation label {label}")
        meta_path = Path(raw_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("checkpoint_label") != label:
            raise ValueError(f"label mismatch in {meta_path}")
        pred_path = Path(meta["prediction_path"])
        if hashlib.sha256(pred_path.read_bytes()).hexdigest() != meta["prediction_sha256"]:
            raise ValueError(f"prediction SHA-256 mismatch for {label}")
        metadata[label] = meta
        predictions[label] = read_predictions(pred_path)

    if len({meta["data_sha256"] for meta in metadata.values()}) != 1:
        raise ValueError("checkpoint evaluations used different input data")
    first_label = next(iter(predictions))
    canonical = predictions[first_label]
    for label, rows in predictions.items():
        if len(rows) != len(canonical):
            raise ValueError(f"row count mismatch for {label}")
        if [row_identity(row) for row in rows] != [row_identity(row) for row in canonical]:
            raise ValueError(f"source/reference alignment mismatch for {label}")

    paired_results = []
    for spec in args.paired_seed:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad paired-seed specification {spec!r}")
        seed_text, control_label, treatment_label = parts
        seed = int(seed_text)
        if control_label not in predictions or treatment_label not in predictions:
            raise ValueError(f"unknown labels in {spec}")
        control, treatment = predictions[control_label], predictions[treatment_label]
        control_hypotheses = [row["common_beam4"]["hypothesis"] for row in control]
        treatment_hypotheses = [row["common_beam4"]["hypothesis"] for row in treatment]
        refs = [row["references"] for row in control]
        delta = corpus_bleu(treatment_hypotheses, refs) - corpus_bleu(control_hypotheses, refs)
        paired_results.append({
            "seed": seed,
            "control_label": control_label,
            "treatment_label": treatment_label,
            "control_common_bleu": corpus_bleu(control_hypotheses, refs),
            "treatment_common_bleu": corpus_bleu(treatment_hypotheses, refs),
            "common_decoder_bleu_difference": delta,
            "common_decoder_source_cluster_bootstrap": bootstrap_paired_bleu(
                control, treatment, args.bootstrap_replicates, args.bootstrap_seed + seed,
            ),
            "assigned_stochastic_reward_difference": bootstrap_mean_reward_difference(
                control, treatment, "beta_0.00", "beta_0.20",
                args.bootstrap_replicates, args.bootstrap_seed + 10_000 + seed,
            ),
        })

    summary = {
        "data_sha256": metadata[first_label]["data_sha256"],
        "source_count": len(canonical),
        "pair_count": metadata[first_label].get("pair_count"),
        "input_pair_count": metadata[first_label].get("input_pair_count"),
        "metric_protocol": metadata[first_label]["metric_signatures"],
        "checkpoint_labels": list(metadata),
        "paired_seed_results": paired_results,
        "interpretation_note": (
            "Source-cluster bootstrap intervals quantify test-source uncertainty within each run; "
            "they do not replace variation across the three paired training seeds."
        ),
    }
    atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
