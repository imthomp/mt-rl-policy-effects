"""Evaluate frozen predictor decisions after policy artifacts are fixed.

This is the first confirmation stage allowed to read evaluation-document MQM
outcomes.  It validates and hashes every policy artifact before loading those
outcomes, reports every checkpoint, and merges the result into the paper SSOT.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from mqm_human_scores import load_mqm_quality


DATASETS = ("wmt24_ja_zh", "wmt23_zh_en")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values: list[float], level: float) -> dict[str, float]:
    alpha = (1.0 - level) / 2.0
    return {
        "low": float(np.quantile(values, alpha)),
        "high": float(np.quantile(values, 1.0 - alpha)),
    }


def validate_policy_artifacts(
    root: Path, protocol: dict[str, Any], calibration: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    policies: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    expected_checkpoints = {
        row["checkpoint"]
        for dataset in calibration["datasets"].values()
        for row in dataset["rows"]
    }
    for dataset_name in DATASETS:
        result_path = root / f"results/confirmation_policy_{dataset_name}.json"
        probability_path = (
            root
            / f"artifacts/bandit/confirmation/{dataset_name}_policy_probabilities.npz"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] != "policy_trajectories_fixed_evaluation_outcomes_sealed":
            raise ValueError(f"Policy artifact is not frozen: {dataset_name}")
        if result["evaluation_outcomes_accessed"]:
            raise ValueError(f"Policy artifact reports outcome access: {dataset_name}")
        checkpoints = {row["checkpoint"] for row in result["rows"]}
        if checkpoints != expected_checkpoints:
            raise ValueError(f"Checkpoint mismatch: {dataset_name}")
        calibration_rows = {
            row["checkpoint"]: row
            for row in calibration["datasets"][dataset_name]["rows"]
        }
        arrays = np.load(probability_path, allow_pickle=False)
        for row in result["rows"]:
            decision = calibration_rows[row["checkpoint"]]
            for key in ("sign_screen_decision", "best_of_n_screen_decision"):
                if row[key] != decision[key]:
                    raise ValueError(
                        f"Frozen decision mismatch: {dataset_name}/{row['checkpoint']}/{key}"
                    )
            for target, key in row["probability_artifact_keys"].items():
                if key not in arrays:
                    raise ValueError(
                        f"Missing probability array: {dataset_name}/{row['checkpoint']}/{target}"
                    )
                probabilities = arrays[key]
                if not np.isfinite(probabilities).all():
                    raise ValueError(f"Nonfinite probabilities: {dataset_name}/{key}")
                if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
                    raise ValueError(f"Unnormalized probabilities: {dataset_name}/{key}")
        policies[dataset_name] = result
        hashes[str(result_path.relative_to(root))] = sha256(result_path)
        hashes[str(probability_path.relative_to(root))] = sha256(probability_path)
    hashes["results/text_policy_predictor_protocol.json"] = sha256(
        root / "results/text_policy_predictor_protocol.json"
    )
    hashes["results/text_policy_predictor_confirmation_calibration.json"] = sha256(
        root / "results/text_policy_predictor_confirmation_calibration.json"
    )
    return policies, hashes


def evaluate_dataset(
    root: Path,
    dataset_name: str,
    dataset: dict[str, Any],
    policy: dict[str, Any],
    primary_target: str,
) -> dict[str, Any]:
    split = dataset["calibration_evaluation_split"]
    systems = dataset["policy_system_split"]["heldout_systems"]
    contexts = split["evaluation_segment_ids"]
    human, human_metadata = load_mqm_quality(
        Path(dataset["source_path"]), contexts, systems
    )
    probability_path = root / policy["probabilities_output"]
    arrays = np.load(probability_path, allow_pickle=False)
    if arrays["evaluation_contexts"].tolist() != contexts:
        raise ValueError(f"Evaluation context mismatch: {dataset_name}")
    if arrays["heldout_systems"].tolist() != systems:
        raise ValueError(f"Held-out system mismatch: {dataset_name}")
    uniform_per_context = human.mean(axis=1)
    rows = []
    context_changes: dict[str, dict[str, list[float]]] = {}
    for row in policy["rows"]:
        pressure_rows = {}
        context_changes[row["checkpoint"]] = {}
        for target, key in row["probability_artifact_keys"].items():
            probabilities = arrays[key]
            if probabilities.shape != human.shape:
                raise ValueError(
                    f"Probability/human shape mismatch: {dataset_name}/{row['checkpoint']}"
                )
            optimized = (probabilities * human).sum(axis=1)
            change = optimized - uniform_per_context
            pressure_rows[target] = {
                "expected_human_quality": float(optimized.mean()),
                "uniform_human_quality": float(uniform_per_context.mean()),
                "human_change_from_uniform": float(change.mean()),
            }
            context_changes[row["checkpoint"]][target] = change.astype(float).tolist()
        if primary_target not in pressure_rows:
            rows.append(
                {
                    "checkpoint": row["checkpoint"],
                    "kind": row["kind"],
                    "seed": row["seed"],
                    "condition": row["condition"],
                    "sign_screen_decision": row["sign_screen_decision"],
                    "best_of_n_screen_decision": row["best_of_n_screen_decision"],
                    "pressure": pressure_rows,
                    "primary_target_reached": False,
                    "primary_human_change": None,
                    "primary_failure": None,
                    "decision_values": None,
                    "decision_regret": None,
                    "policy_trace": f"results/confirmation_policy_{dataset_name}.json",
                    "probability_trace": policy["probabilities_output"],
                }
            )
            continue
        primary_change = pressure_rows[primary_target]["human_change_from_uniform"]
        oracle = max(0.0, primary_change)
        sign_value = (
            primary_change if row["sign_screen_decision"] == "optimize" else 0.0
        )
        best_value = (
            primary_change if row["best_of_n_screen_decision"] == "optimize" else 0.0
        )
        rows.append(
            {
                "checkpoint": row["checkpoint"],
                "kind": row["kind"],
                "seed": row["seed"],
                "condition": row["condition"],
                "sign_screen_decision": row["sign_screen_decision"],
                "best_of_n_screen_decision": row["best_of_n_screen_decision"],
                "pressure": pressure_rows,
                "primary_target_reached": True,
                "primary_human_change": primary_change,
                "primary_failure": primary_change < 0.0,
                "decision_values": {
                    "sign_screen": sign_value,
                    "best_of_n_screen": best_value,
                    "always_optimize": primary_change,
                    "always_keep_initial": 0.0,
                    "oracle": oracle,
                },
                "decision_regret": {
                    "sign_screen": oracle - sign_value,
                    "best_of_n_screen": oracle - best_value,
                    "always_optimize": oracle - primary_change,
                    "always_keep_initial": oracle,
                },
                "policy_trace": f"results/confirmation_policy_{dataset_name}.json",
                "probability_trace": policy["probabilities_output"],
            }
        )
    return {
        "dataset": dataset_name,
        "evaluation_outcomes_accessed": True,
        "evaluation_segments": len(contexts),
        "evaluation_documents": len({context.split("|", 1)[0] for context in contexts}),
        "heldout_systems": systems,
        "human_score_metadata": human_metadata,
        "human_source": dataset["source_path"],
        "contexts": contexts,
        "rows": rows,
        "context_changes": context_changes,
    }


def summarize_rows(
    dataset_results: dict[str, dict[str, Any]], target: str
) -> dict[str, Any]:
    learned = [
        (dataset_name, row)
        for dataset_name, result in dataset_results.items()
        for row in result["rows"]
        if row["kind"] == "new_learned_reward"
    ]
    reachable = [
        (dataset_name, row)
        for dataset_name, row in learned
        if target in row["pressure"]
    ]
    methods = (
        "sign_screen",
        "best_of_n_screen",
        "always_optimize",
        "always_keep_initial",
    )
    summary: dict[str, Any] = {
        "target_kl_nats": float(target),
        "learned_reward_cells": len(learned),
        "reachable_cells": len(reachable),
        "unreachable_cells": len(learned) - len(reachable),
        "failures": sum(row["pressure"][target]["human_change_from_uniform"] < 0 for _, row in reachable),
        "beneficial": sum(row["pressure"][target]["human_change_from_uniform"] >= 0 for _, row in reachable),
        "methods": {},
    }
    for method in methods:
        regrets = []
        values = []
        for _, row in reachable:
            change = row["pressure"][target]["human_change_from_uniform"]
            oracle = max(0.0, change)
            value = {
                "sign_screen": change if row["sign_screen_decision"] == "optimize" else 0.0,
                "best_of_n_screen": change if row["best_of_n_screen_decision"] == "optimize" else 0.0,
                "always_optimize": change,
                "always_keep_initial": 0.0,
            }[method]
            regrets.append(oracle - value)
            values.append(value)
        summary["methods"][method] = {
            "mean_decision_regret": float(np.mean(regrets)),
            "mean_selected_human_change": float(np.mean(values)),
        }
    failures = [
        row
        for _, row in reachable
        if row["pressure"][target]["human_change_from_uniform"] < 0
    ]
    summary["sign_screen_false_optimize_rate"] = (
        float(np.mean([row["sign_screen_decision"] == "optimize" for row in failures]))
        if failures
        else None
    )
    beneficial = [
        row
        for _, row in reachable
        if row["pressure"][target]["human_change_from_uniform"] >= 0
    ]
    summary["sign_screen_false_keep_rate"] = (
        float(np.mean([row["sign_screen_decision"] != "optimize" for row in beneficial]))
        if beneficial
        else math.nan
    )
    return summary


def bootstrap_summary(
    dataset_results: dict[str, dict[str, Any]], protocol: dict[str, Any], target: str
) -> dict[str, Any]:
    # These fallbacks were present in freeze_text_predictor_protocol.py before
    # calibration and outcome access, but the generated protocol JSON had not
    # been refreshed.  Preserve the hashed JSON and record this provenance.
    bootstrap_seed = int(protocol["analysis"].get("bootstrap_seed", 20261206))
    resamples = int(protocol["analysis"].get("bootstrap_resamples", 10_000))
    level = float(protocol["analysis"].get("interval_level", 0.95))
    rng = np.random.default_rng(bootstrap_seed)
    seeds = protocol["reward_training"]["new_seeds"]
    conditions = protocol["reward_training"]["conditions"]
    rows = {
        (dataset_name, row["seed"], row["condition"]): row
        for dataset_name, result in dataset_results.items()
        for row in result["rows"]
        if row["kind"] == "new_learned_reward"
    }
    document_indices: dict[str, dict[str, list[int]]] = {}
    for dataset_name, result in dataset_results.items():
        grouped: dict[str, list[int]] = {}
        for index, context in enumerate(result["contexts"]):
            grouped.setdefault(context.split("|", 1)[0], []).append(index)
        document_indices[dataset_name] = grouped
    methods = ("sign_screen", "best_of_n_screen", "always_optimize", "always_keep_initial")
    draws = {method: [] for method in methods}
    for _ in range(resamples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True).tolist()
        sampled_indices = {}
        for dataset_name, grouped in document_indices.items():
            documents = list(grouped)
            sampled_documents = rng.choice(
                documents, size=len(documents), replace=True
            ).tolist()
            sampled_indices[dataset_name] = np.asarray(
                [index for document in sampled_documents for index in grouped[document]]
            )
        regrets = {method: [] for method in methods}
        for seed in sampled_seeds:
            for condition in conditions:
                for dataset_name, result in dataset_results.items():
                    row = rows[(dataset_name, seed, condition)]
                    if target not in result["context_changes"][row["checkpoint"]]:
                        continue
                    changes = np.asarray(
                        result["context_changes"][row["checkpoint"]][target]
                    )
                    change = float(changes[sampled_indices[dataset_name]].mean())
                    oracle = max(0.0, change)
                    values = {
                        "sign_screen": change
                        if row["sign_screen_decision"] == "optimize"
                        else 0.0,
                        "best_of_n_screen": change
                        if row["best_of_n_screen_decision"] == "optimize"
                        else 0.0,
                        "always_optimize": change,
                        "always_keep_initial": 0.0,
                    }
                    for method in methods:
                        regrets[method].append(oracle - values[method])
        for method in methods:
            draws[method].append(float(np.mean(regrets[method])))
    result: dict[str, Any] = {
        "target_kl_nats": float(target),
        "seed": bootstrap_seed,
        "resamples": resamples,
        "interval_level": level,
        "specification_source": (
            "results/text_policy_predictor_protocol.json when present; otherwise "
            "pre-outcome constants in src/freeze_text_predictor_protocol.py"
        ),
        "methods": {},
        "contrasts": {},
    }
    for method in methods:
        result["methods"][method] = interval(draws[method], level)
    for baseline in ("best_of_n_screen", "always_optimize", "always_keep_initial"):
        difference = [
            sign - base for sign, base in zip(draws["sign_screen"], draws[baseline])
        ]
        result["contrasts"][f"sign_screen_minus_{baseline}"] = {
            "mean": float(np.mean(difference)),
            **interval(difference, level),
        }
    return result


def build(root: Path) -> dict[str, Any]:
    protocol = json.loads(
        (root / "results/text_policy_predictor_protocol.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        (root / "results/text_policy_predictor_confirmation_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (root / "results/predictor_confirmation_manifest.json").read_text(encoding="utf-8")
    )
    if calibration["status"] != "calibration_decisions_frozen_evaluation_outcomes_sealed":
        raise ValueError("Calibration decisions are not frozen")
    policies, hashes = validate_policy_artifacts(root, protocol, calibration)
    primary_target = str(protocol["optimization"]["primary_pressure_nats"])
    datasets = {
        name: evaluate_dataset(
            root, name, manifest["datasets"][name], policies[name], primary_target
        )
        for name in DATASETS
    }
    summary = summarize_rows(datasets, primary_target)
    bootstrap = bootstrap_summary(datasets, protocol, primary_target)
    secondary_target = "0.1"
    secondary_summary = summarize_rows(datasets, secondary_target)
    secondary_bootstrap = bootstrap_summary(datasets, protocol, secondary_target)
    primary_fully_evaluable = summary["unreachable_cells"] == 0
    success = None
    if primary_fully_evaluable:
        success = (
            summary["methods"]["sign_screen"]["mean_decision_regret"]
            < summary["methods"]["always_optimize"]["mean_decision_regret"]
            and summary["methods"]["sign_screen"]["mean_decision_regret"]
            < summary["methods"]["best_of_n_screen"]["mean_decision_regret"]
        )
    for result in datasets.values():
        result.pop("context_changes")
        result.pop("contexts")
    return {
        "status": "prospective_confirmation_complete",
        "evaluation_outcomes_accessed": True,
        "protocol": "results/text_policy_predictor_protocol.json",
        "calibration": "results/text_policy_predictor_confirmation_calibration.json",
        "policy_artifact_hashes_before_outcome_access": hashes,
        "primary_pressure_nats": protocol["optimization"]["primary_pressure_nats"],
        "datasets": datasets,
        "learned_reward_summary": summary,
        "hierarchical_bootstrap": bootstrap,
        "complete_secondary_summary": secondary_summary,
        "complete_secondary_hierarchical_bootstrap": secondary_bootstrap,
        "frozen_success_criterion": protocol["analysis"]["success_criterion"],
        "frozen_success_criterion_fully_evaluable": primary_fully_evaluable,
        "frozen_success_criterion_met": success,
        "claim_boundary": (
            "Prospective confirmation for fixed-pool source-candidate-text selection; "
            "not a safeguard claim for generative RL."
        ),
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["text_policy_predictor_confirmation"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results/text_policy_predictor_confirmation_v2.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite prospective confirmation results")
    result = build(root)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    merge_ssot(root / "results/paper_stats.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "frozen_success_criterion_met": result[
                    "frozen_success_criterion_met"
                ],
                "learned_reward_summary": result["learned_reward_summary"],
                "hierarchical_bootstrap": result["hierarchical_bootstrap"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
