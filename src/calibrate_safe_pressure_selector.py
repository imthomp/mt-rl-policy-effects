"""Validate sealed policies, then open calibration-only MQM outcomes."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_confirmation_calibration import exact_best_of_n_human_change, read_reward
from mqm_human_scores import load_mqm_quality


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_hash(root: Path, relative: str, expected: str) -> None:
    observed = sha256(root / relative)
    if observed != expected:
        raise ValueError(f"Hash mismatch: {relative}")


def validate_before_outcome_access(root: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    policy_path = root / "results/safe_pressure_policies.json"
    probability_path = root / "artifacts/bandit/safe_pressure/policy_probabilities.npz"
    policies = json.loads(policy_path.read_text(encoding="utf-8"))
    manifest = json.loads((root / "results/safe_pressure_manifest.json").read_text())
    plan = json.loads((root / "results/safe_pressure_reward_plan.json").read_text())
    if policies["status"] != "safe_pressure_policy_trajectories_fixed_human_outcomes_sealed":
        raise ValueError("Policy trajectories are not sealed")
    if policies["human_outcomes_accessed"] or manifest["human_outcomes_accessed"]:
        raise ValueError("A sealed artifact reports outcome access")
    require_hash(root, policies["protocol"], policies["protocol_sha256"])
    require_hash(root, policies["protocol_amendment"], policies["protocol_amendment_sha256"])
    require_hash(root, policies["manifest"], policies["manifest_sha256"])
    require_hash(root, policies["reward_plan"], policies["reward_plan_sha256"])
    require_hash(root, policies["embedding_artifact"], policies["embedding_artifact_sha256"])
    if sha256(probability_path) != policies["probabilities_output_sha256"]:
        raise ValueError("Probability artifact hash mismatch")
    expected_checkpoints = {
        f"{row['seed']}/{row['training_condition']}" for row in plan["tasks"]
    }
    if {row["checkpoint"] for row in policies["rows"]} != expected_checkpoints:
        raise ValueError("Policy rows differ from reward plan")
    arrays = np.load(probability_path, allow_pickle=False)
    for row in policies["rows"]:
        require_hash(root, row["reward_trace"], row["reward_trace_sha256"])
        for target, keys in row["probability_artifact_keys"].items():
            if not row["pressure"][target]["reached"]:
                raise ValueError("Unreached pressure has probability keys")
            for split, key in keys.items():
                values = arrays[key]
                if not np.isfinite(values).all() or not np.allclose(values.sum(1), 1.0, atol=1e-5):
                    raise ValueError(f"Invalid {split} probabilities: {row['checkpoint']}/{target}")
                expected_actions = len(
                    manifest["split"]["train_systems" if split == "calibration" else "heldout_systems"]
                )
                if values.shape[1] != expected_actions:
                    raise ValueError(f"Probability action mismatch: {row['checkpoint']}/{target}")
    hashes = {
        "results/safe_pressure_policies.json": sha256(policy_path),
        "artifacts/bandit/safe_pressure/policy_probabilities.npz": sha256(probability_path),
        "results/safe_pressure_manifest.json": sha256(root / "results/safe_pressure_manifest.json"),
        "results/safe_pressure_reward_plan.json": sha256(root / "results/safe_pressure_reward_plan.json"),
        "results/safe_pressure_protocol.json": sha256(root / "results/safe_pressure_protocol.json"),
        "results/safe_pressure_protocol_amendment_1.json": sha256(
            root / "results/safe_pressure_protocol_amendment_1.json"
        ),
        "results/safe_pressure_protocol_amendment_2.json": sha256(
            root / "results/safe_pressure_protocol_amendment_2.json"
        ),
        "results/safe_pressure_protocol_amendment_3.json": sha256(
            root / "results/safe_pressure_protocol_amendment_3.json"
        ),
    }
    return policies, manifest, {"arrays": arrays, "hashes": hashes}


def standardized_human(values: np.ndarray) -> np.ndarray:
    means = values.mean(1, keepdims=True)
    std = values.std(1, keepdims=True)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Calibration segment has zero or invalid human-score variation")
    return (values - means) / std


def lower_bound(
    changes: np.ndarray, seed: int, resamples: int, level: float
) -> float:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(changes), size=(resamples, len(changes)))
    means = changes[indices].mean(1)
    return float(np.quantile(means, (1.0 - level) / 2.0))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results/safe_pressure_calibration.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    policies, manifest, validated = validate_before_outcome_access(root)
    protocol = json.loads((root / "results/safe_pressure_protocol.json").read_text())
    amendment_2 = json.loads(
        (root / "results/safe_pressure_protocol_amendment_2.json").read_text()
    )
    if amendment_2["human_outcomes_accessed"]:
        raise ValueError("Best-of-N amendment reports outcome access")

    split = manifest["split"]
    source = Path(
        "/home/it238/projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation/"
        "generalMT2024/mqm_generalMT2024_enes.tsv"
    )
    human_raw, human_metadata = load_mqm_quality(
        source, split["calibration_human_label_segments"], split["train_systems"]
    )
    human = standardized_human(human_raw)
    arrays = validated["arrays"]
    targets = [str(float(value)) for value in protocol["pressure"]["targets"]]
    jointly_reachable = [
        target
        for target in targets
        if all(row["pressure"][target]["reached"] for row in policies["rows"])
    ]
    if "0.0" not in jointly_reachable:
        raise ValueError("Uniform pressure is unavailable")
    valid_n = [
        value
        for value in protocol["baselines"]["best_of_n_subset_sizes"]
        if value <= len(split["train_systems"])
    ]
    invalid_n = sorted(
        set(protocol["baselines"]["best_of_n_subset_sizes"]) - set(valid_n)
    )
    if arrays["calibration_contexts"].tolist() != split["calibration_human_label_segments"]:
        raise ValueError("Calibration context order differs from manifest")
    if arrays["train_systems"].tolist() != split["train_systems"]:
        raise ValueError("Calibration action order differs from manifest")
    embedding = np.load(root / policies["embedding_artifact"], allow_pickle=False)
    all_contexts = embedding["contexts"].tolist()
    all_actions = embedding["actions"].tolist()
    full_context_index = {value: index for index, value in enumerate(all_contexts)}
    full_action_index = {value: index for index, value in enumerate(all_actions)}
    calibration_context_indices = [
        full_context_index[value] for value in split["calibration_human_label_segments"]
    ]
    calibration_action_indices = [
        full_action_index[value] for value in split["train_systems"]
    ]

    rows = []
    for row_index, policy_row in enumerate(policies["rows"]):
        pressure_changes = {}
        per_document = {}
        for target in jointly_reachable:
            key = policy_row["probability_artifact_keys"][target]["calibration"]
            probability = arrays[key]
            changes = (probability * human).sum(1) - human.mean(1)
            pressure_changes[target] = float(changes.mean())
            per_document[target] = changes
        selected = min(
            jointly_reachable,
            key=lambda target: (-pressure_changes[target], float(target)),
        )
        pessimistic_bounds = {
            target: lower_bound(
                per_document[target],
                int(protocol["analysis"]["bootstrap_seed"]) + row_index,
                int(protocol["analysis"]["bootstrap_resamples"]),
                float(protocol["analysis"]["interval_level"]),
            )
            for target in jointly_reachable
        }
        pessimistic = min(
            jointly_reachable,
            key=lambda target: (-pessimistic_bounds[target], float(target)),
        )
        if pessimistic_bounds[pessimistic] <= 0:
            pessimistic = "0.0"

        reward_path = root / policy_row["reward_trace"]
        reward_full = read_reward(reward_path, all_contexts, all_actions)
        reward = reward_full[np.ix_(
            calibration_context_indices,
            calibration_action_indices,
        )]
        best_of_n = {
            str(value): exact_best_of_n_human_change(reward, human, value)
            for value in valid_n
        }
        best_n = min(valid_n, key=lambda value: (-best_of_n[str(value)], value))
        selected_n = best_n if best_of_n[str(best_n)] > 0 else 1
        rows.append(
            {
                "checkpoint": policy_row["checkpoint"],
                "primary_selected_pressure": float(selected),
                "calibration_human_change_by_pressure": pressure_changes,
                "pessimistic_selected_pressure": float(pessimistic),
                "pessimistic_lower_bound_by_pressure": pessimistic_bounds,
                "best_of_n_human_change": best_of_n,
                "best_of_n_selected_n": selected_n,
                "best_of_n_decision": (
                    "optimize" if best_of_n[str(best_n)] > 0 else "keep_initial"
                ),
            }
        )
    result = {
        "status": "safe_pressure_calibration_decisions_frozen_evaluation_outcomes_sealed",
        "protocol": "results/safe_pressure_protocol.json",
        "manifest": "results/safe_pressure_manifest.json",
        "policies": "results/safe_pressure_policies.json",
        "pre_outcome_artifact_hashes": validated["hashes"],
        "calibration_outcomes_accessed": True,
        "evaluation_outcomes_accessed": False,
        "human_score_metadata": human_metadata,
        "jointly_reachable_pressures": [float(value) for value in jointly_reachable],
        "valid_best_of_n_subset_sizes": valid_n,
        "unavailable_best_of_n_subset_sizes": invalid_n,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
