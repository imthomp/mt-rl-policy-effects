"""Open sealed evaluation MQM outcomes after every decision is frozen."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_confirmation_calibration import read_reward
from mqm_human_scores import load_mqm_quality


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    calibration_path = root / "results/safe_pressure_calibration.json"
    policies_path = root / "results/safe_pressure_policies.json"
    probability_path = root / "artifacts/bandit/safe_pressure/policy_probabilities.npz"
    calibration = json.loads(calibration_path.read_text())
    policies = json.loads(policies_path.read_text())
    manifest = json.loads((root / "results/safe_pressure_manifest.json").read_text())
    if calibration["status"] != "safe_pressure_calibration_decisions_frozen_evaluation_outcomes_sealed":
        raise ValueError("Calibration decisions are not sealed")
    if calibration["evaluation_outcomes_accessed"]:
        raise ValueError("Calibration artifact reports evaluation outcome access")
    expected = calibration["pre_outcome_artifact_hashes"]
    for relative, digest in expected.items():
        if sha256(root / relative) != digest:
            raise ValueError(f"Pre-outcome artifact changed: {relative}")
    if sha256(policies_path) != expected["results/safe_pressure_policies.json"]:
        raise ValueError("Policy JSON changed after calibration")
    if sha256(probability_path) != expected[
        "artifacts/bandit/safe_pressure/policy_probabilities.npz"
    ]:
        raise ValueError("Policy probabilities changed after calibration")
    policy_checkpoints = {row["checkpoint"] for row in policies["rows"]}
    calibration_checkpoints = {row["checkpoint"] for row in calibration["rows"]}
    if policy_checkpoints != calibration_checkpoints:
        raise ValueError("Calibration decisions do not cover every policy checkpoint")
    arrays = np.load(probability_path, allow_pickle=False)
    if arrays["evaluation_contexts"].tolist() != manifest["split"]["evaluation_segments"]:
        raise ValueError("Evaluation context order differs from manifest")
    if arrays["heldout_systems"].tolist() != manifest["split"]["heldout_systems"]:
        raise ValueError("Evaluation system order differs from manifest")
    hashes = {
        "results/safe_pressure_calibration.json": sha256(calibration_path),
        "results/safe_pressure_policies.json": sha256(policies_path),
        "artifacts/bandit/safe_pressure/policy_probabilities.npz": sha256(probability_path),
    }
    return calibration, policies, manifest, {"arrays": arrays, "hashes": hashes}


def per_context_best_of_n(
    reward: np.ndarray, human: np.ndarray, subset_size: int
) -> np.ndarray:
    if subset_size == 1:
        return human.mean(1)
    combinations = list(itertools.combinations(range(reward.shape[1]), subset_size))
    values = np.empty(len(reward), dtype=float)
    for context in range(len(reward)):
        selected = []
        for subset in combinations:
            winner = max(subset, key=lambda action: (reward[context, action], -action))
            selected.append(human[context, winner])
        values[context] = float(np.mean(selected))
    return values


def aggregate_documents(values: np.ndarray, contexts: list[str]) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for context, value in zip(contexts, values, strict=True):
        grouped[context.split("|", 1)[0]].append(float(value))
    documents = sorted(grouped)
    return documents, np.asarray([np.mean(grouped[document]) for document in documents])


def bootstrap(
    rows: list[dict[str, Any]], methods: list[str], protocol: dict[str, Any]
) -> dict[str, Any]:
    seeds = sorted({row["seed"] for row in rows})
    documents = rows[0]["documents"]
    if any(row["documents"] != documents for row in rows):
        raise ValueError("Evaluation document ordering differs across rows")
    by_seed = {seed: [row for row in rows if row["seed"] == seed] for seed in seeds}
    generator = np.random.default_rng(int(protocol["analysis"]["bootstrap_seed"]))
    resamples = int(protocol["analysis"]["bootstrap_resamples"])
    estimates = {method: np.empty(resamples) for method in methods}
    for replicate in range(resamples):
        selected_seeds = generator.choice(seeds, len(seeds), replace=True)
        document_indices = generator.integers(0, len(documents), size=len(documents))
        values = {method: [] for method in methods}
        for seed in selected_seeds:
            for row in by_seed[int(seed)]:
                pressure = row["pressure_document_quality"]
                oracle_target = max(
                    pressure,
                    key=lambda target: float(np.mean(np.asarray(pressure[target])[document_indices])),
                )
                oracle = np.asarray(pressure[oracle_target])[document_indices]
                for method in methods:
                    selected = np.asarray(row["method_document_quality"][method])[document_indices]
                    values[method].append(float(np.mean(oracle - selected)))
        for method in methods:
            estimates[method][replicate] = float(np.mean(values[method]))
    level = float(protocol["analysis"]["interval_level"])
    tail = (1.0 - level) / 2.0
    intervals = {
        method: {
            "low": float(np.quantile(values, tail)),
            "high": float(np.quantile(values, 1.0 - tail)),
        }
        for method, values in estimates.items()
    }
    contrasts = {}
    for method in methods:
        if method == "primary_selector":
            continue
        values = estimates["primary_selector"] - estimates[method]
        contrasts[f"primary_selector_minus_{method}"] = {
            "mean": float(values.mean()),
            "low": float(np.quantile(values, tail)),
            "high": float(np.quantile(values, 1.0 - tail)),
        }
    return {
        "seed": protocol["analysis"]["bootstrap_seed"],
        "resamples": resamples,
        "interval_level": level,
        "method_regret_intervals": intervals,
        "contrasts": contrasts,
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["safe_pressure_confirmation"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results/safe_pressure_confirmation.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    calibration, policies, manifest, validated = validate(root)
    protocol = json.loads((root / "results/safe_pressure_protocol.json").read_text())
    arrays = validated["arrays"]
    split = manifest["split"]
    contexts = arrays["evaluation_contexts"].tolist()
    systems = arrays["heldout_systems"].tolist()

    source = Path(
        "/home/it238/projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation/"
        "generalMT2024/mqm_generalMT2024_enes.tsv"
    )
    human, human_metadata = load_mqm_quality(source, contexts, systems)
    uniform_context = human.mean(1)
    documents, uniform_document = aggregate_documents(uniform_context, contexts)
    calibration_rows = {row["checkpoint"]: row for row in calibration["rows"]}
    embedding = np.load(root / policies["embedding_artifact"], allow_pickle=False)
    all_contexts = embedding["contexts"].tolist()
    all_actions = embedding["actions"].tolist()
    full_context_index = {value: index for index, value in enumerate(all_contexts)}
    full_action_index = {value: index for index, value in enumerate(all_actions)}
    evaluation_context_indices = [full_context_index[value] for value in contexts]
    heldout_action_indices = [full_action_index[value] for value in systems]
    jointly_reachable = [str(value) for value in calibration["jointly_reachable_pressures"]]
    largest_pressure = max(jointly_reachable, key=float)
    fixed = [
        str(value)
        for value in protocol["baselines"]["fixed_normalized_kl_targets"]
        if str(value) in jointly_reachable
    ]
    missing_fixed = sorted(
        set(map(str, protocol["baselines"]["fixed_normalized_kl_targets"])) - set(fixed)
    )
    methods = [
        "primary_selector",
        "pessimistic_calibration",
        "always_keep_initial",
        "always_optimize",
        "best_of_n",
        *[f"fixed_{target}" for target in fixed],
    ]
    rows = []
    for policy_row in policies["rows"]:
        checkpoint = policy_row["checkpoint"]
        decision = calibration_rows[checkpoint]
        pressure_document_quality = {}
        pressure_mean_change = {}
        for target in jointly_reachable:
            key = policy_row["probability_artifact_keys"][target]["evaluation"]
            expected = (arrays[key] * human).sum(1)
            observed_documents, document_values = aggregate_documents(expected, contexts)
            if observed_documents != documents:
                raise ValueError("Pressure document order mismatch")
            pressure_document_quality[target] = document_values.tolist()
            pressure_mean_change[target] = float(document_values.mean() - uniform_document.mean())
        oracle_target = max(
            jointly_reachable, key=lambda target: pressure_mean_change[target]
        )

        reward_full = read_reward(root / policy_row["reward_trace"], all_contexts, all_actions)
        reward = reward_full[np.ix_(evaluation_context_indices, heldout_action_indices)]
        selected_n = int(decision["best_of_n_selected_n"])
        best_context = per_context_best_of_n(reward, human, selected_n)
        best_documents, best_document = aggregate_documents(best_context, contexts)
        if best_documents != documents:
            raise ValueError("Best-of-N document order mismatch")
        method_document_quality = {
            "primary_selector": pressure_document_quality[
                str(decision["primary_selected_pressure"])
            ],
            "pessimistic_calibration": pressure_document_quality[
                str(decision["pessimistic_selected_pressure"])
            ],
            "always_keep_initial": uniform_document.tolist(),
            "always_optimize": pressure_document_quality[largest_pressure],
            "best_of_n": best_document.tolist(),
            **{
                f"fixed_{target}": pressure_document_quality[target] for target in fixed
            },
        }
        oracle_document = np.asarray(pressure_document_quality[oracle_target])
        method_regret = {
            method: float(
                np.mean(oracle_document - np.asarray(method_document_quality[method]))
            )
            for method in methods
        }
        rows.append(
            {
                "checkpoint": checkpoint,
                "seed": policy_row["seed"],
                "condition": policy_row["training_condition"],
                "documents": documents,
                "uniform_human_quality": float(uniform_document.mean()),
                "pressure_mean_human_change": pressure_mean_change,
                "oracle_pressure": float(oracle_target),
                "primary_selected_pressure": decision["primary_selected_pressure"],
                "pessimistic_selected_pressure": decision[
                    "pessimistic_selected_pressure"
                ],
                "best_of_n_selected_n": selected_n,
                "method_mean_decision_regret": method_regret,
                "pressure_document_quality": pressure_document_quality,
                "method_document_quality": method_document_quality,
                "policy_trace": "results/safe_pressure_policies.json",
                "calibration_trace": "results/safe_pressure_calibration.json",
            }
        )
    summary = {
        method: {
            "mean_decision_regret": float(
                np.mean([row["method_mean_decision_regret"][method] for row in rows])
            ),
            "mean_human_quality": float(
                np.mean(
                    [np.mean(row["method_document_quality"][method]) for row in rows]
                )
            ),
        }
        for method in methods
    }
    bootstrap_result = bootstrap(rows, methods, protocol)
    competitors = [method for method in methods if method != "primary_selector"]
    strongest = min(competitors, key=lambda method: summary[method]["mean_decision_regret"])
    strongest_contrast = bootstrap_result["contrasts"][
        f"primary_selector_minus_{strongest}"
    ]
    criterion_evaluable = not missing_fixed
    success = (
        all(
            summary["primary_selector"]["mean_decision_regret"]
            < summary[method]["mean_decision_regret"]
            for method in competitors
        )
        and strongest_contrast["high"] < 0
        if criterion_evaluable
        else None
    )
    result = {
        "status": "safe_pressure_prospective_confirmation_complete",
        "dataset": manifest["dataset"],
        "protocol": "results/safe_pressure_protocol.json",
        "manifest": "results/safe_pressure_manifest.json",
        "calibration": "results/safe_pressure_calibration.json",
        "pre_evaluation_artifact_hashes": validated["hashes"],
        "evaluation_outcomes_accessed": True,
        "human_score_metadata": human_metadata,
        "jointly_reachable_pressures": list(map(float, jointly_reachable)),
        "missing_fixed_pressure_baselines": list(map(float, missing_fixed)),
        "methods": summary,
        "strongest_non_oracle_baseline": strongest,
        "bootstrap": bootstrap_result,
        "frozen_success_criterion_evaluable": criterion_evaluable,
        "frozen_success_criterion_met": success,
        "rows": rows,
        "claim_boundary": (
            "Prospective pressure selection for a fixed WMT24 English-Spanish candidate "
            "pool; not a safeguard claim for generative RL."
        ),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    merge_ssot(root / "results/paper_stats.json", result)
    print(json.dumps({
        "status": result["status"],
        "criterion_evaluable": criterion_evaluable,
        "criterion_met": success,
        "strongest_baseline": strongest,
        "methods": summary,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
