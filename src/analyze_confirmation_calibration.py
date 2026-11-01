"""Apply the frozen predictor using calibration labels only.

This script intentionally refuses to load MQM labels outside the exact frozen
calibration segment allowlist. Evaluation-document outcomes remain sealed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_text_policy_predictors import text_parameter_gradient
from mqm_human_scores import load_mqm_quality


def read_reward(path: Path, contexts: list[str], actions: list[str]) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0])
    rows = [json.loads(line) for line in lines[1:]]
    if metadata.get("rows") != len(rows):
        raise ValueError(f"Reward row count differs from metadata: {path}")
    keyed = {(row["segment_key"], row["system_id"]): row for row in rows}
    expected = {(context, action) for context in contexts for action in actions}
    if set(keyed) != expected:
        raise ValueError(f"Reward candidate environment mismatch: {path}")
    values = np.asarray(
        [
            [keyed[(context, action)]["predicted_reward"] for action in actions]
            for context in contexts
        ],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite reward: {path}")
    return values


def exact_best_of_n_human_change(
    reward: np.ndarray, human: np.ndarray, subset_size: int
) -> float:
    action_count = reward.shape[1]
    if not 1 <= subset_size <= action_count:
        raise ValueError("Invalid best-of-N subset size")
    combinations = list(itertools.combinations(range(action_count), subset_size))
    selected = []
    for context in range(len(reward)):
        values = []
        for subset in combinations:
            winner = max(subset, key=lambda action: (reward[context, action], -action))
            values.append(human[context, winner])
        selected.append(float(np.mean(values)))
    return float(np.mean(selected) - np.mean(human))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right)) / norm if norm else math.nan


def reward_specs(protocol: dict[str, Any], root: Path, dataset: str) -> list[dict[str, Any]]:
    specs = []
    for seed in protocol["reward_training"]["new_seeds"]:
        for condition in protocol["reward_training"]["conditions"]:
            specs.append(
                {
                    "checkpoint": f"{seed}/{condition}",
                    "kind": "new_learned_reward",
                    "seed": seed,
                    "condition": condition,
                    "path": root
                    / f"artifacts/reward_models/confirmation/{seed}/predictions/{dataset}/{condition}.jsonl",
                }
            )
    for condition in protocol["reward_training"]["external_positive_controls"]:
        specs.append(
            {
                "checkpoint": f"external/{condition}",
                "kind": "external_single_checkpoint",
                "seed": None,
                "condition": condition,
                "path": root
                / f"artifacts/reward_models/confirmation/external/predictions/{dataset}/{condition}.jsonl",
            }
        )
    return specs


def analyze_dataset(
    root: Path,
    dataset_name: str,
    dataset: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    split = dataset["calibration_evaluation_split"]
    policy_split = dataset["policy_system_split"]
    embedding_path = root / f"artifacts/bandit/confirmation/{dataset_name}_text_embeddings.npz"
    artifact = np.load(embedding_path, allow_pickle=False)
    contexts = artifact["contexts"].tolist()
    actions = artifact["actions"].tolist()
    embeddings = artifact["embeddings"].astype(np.float32)
    if embeddings.shape[:2] != (len(contexts), len(actions)):
        raise ValueError(f"Embedding shape mismatch: {dataset_name}")
    context_index = {context: index for index, context in enumerate(contexts)}
    action_index = {action: index for index, action in enumerate(actions)}
    optimization_indices = np.asarray(
        [context_index[context] for context in split["optimization_segment_ids"]]
    )
    calibration_contexts = split["calibration_segment_ids"]
    calibration_indices = np.asarray(
        [context_index[context] for context in calibration_contexts]
    )
    train_actions = policy_split["train_systems"]
    train_indices = np.asarray([action_index[action] for action in train_actions])
    human, human_metadata = load_mqm_quality(
        Path(dataset["source_path"]), calibration_contexts, train_actions
    )
    calibration_embeddings = embeddings[np.ix_(calibration_indices, train_indices)]
    optimization_embeddings = embeddings[np.ix_(optimization_indices, train_indices)]
    human_gradient = text_parameter_gradient(human, calibration_embeddings)
    best_of_n_grid = protocol["baselines"]["best_of_n_grid"]
    primary_n = str(protocol["baselines"]["best_of_n_primary"])
    rows = []
    trace = {}
    for spec in reward_specs(protocol, root, dataset_name):
        reward = read_reward(spec["path"], contexts, actions)
        optimization_reward = reward[np.ix_(optimization_indices, train_indices)]
        mean, std = float(optimization_reward.mean()), float(optimization_reward.std())
        if not math.isfinite(std) or std <= 0:
            alignment = math.nan
        else:
            standardized = (optimization_reward - mean) / std
            reward_gradient = text_parameter_gradient(
                standardized, optimization_embeddings
            )
            alignment = cosine(reward_gradient, human_gradient)
        calibration_reward = reward[np.ix_(calibration_indices, train_indices)]
        best_of_n = {
            str(subset_size): exact_best_of_n_human_change(
                calibration_reward, human, subset_size
            )
            for subset_size in best_of_n_grid
        }
        row = {
            "checkpoint": spec["checkpoint"],
            "kind": spec["kind"],
            "seed": spec["seed"],
            "condition": spec["condition"],
            "gradient_cosine": alignment,
            "sign_screen_decision": (
                "optimize" if math.isfinite(alignment) and alignment > 0 else "keep_initial"
            ),
            "best_of_n_human_change": best_of_n,
            "best_of_n_screen_decision": (
                "optimize" if best_of_n[primary_n] > 0 else "keep_initial"
            ),
        }
        rows.append(row)
        trace[spec["checkpoint"]] = str(spec["path"])
    return {
        "dataset": dataset_name,
        "calibration_outcomes_opened": True,
        "evaluation_outcomes_accessed": False,
        "calibration_segments": len(calibration_contexts),
        "optimization_segments": len(optimization_indices),
        "train_systems": train_actions,
        "human_score_metadata": human_metadata,
        "embedding_path": str(embedding_path),
        "rows": rows,
        "table_row_trace": trace,
    }


def build(root: Path) -> dict[str, Any]:
    protocol = json.loads(
        (root / "results/text_policy_predictor_protocol.json").read_text(encoding="utf-8")
    )
    if protocol["status"] != "protocol_frozen_confirmation_outcomes_sealed":
        raise ValueError("Predictor protocol is not frozen")
    manifest = json.loads(
        (root / "results/predictor_confirmation_manifest.json").read_text(encoding="utf-8")
    )
    datasets = {
        name: analyze_dataset(root, name, dataset, protocol)
        for name, dataset in manifest["datasets"].items()
    }
    return {
        "status": "calibration_decisions_frozen_evaluation_outcomes_sealed",
        "protocol": "results/text_policy_predictor_protocol.json",
        "manifest": "results/predictor_confirmation_manifest.json",
        "datasets": datasets,
        "evaluation_outcomes_accessed": False,
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["text_policy_predictor_confirmation_calibration"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = build(root)
    output = root / "results/text_policy_predictor_confirmation_calibration.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    merge_ssot(root / "results/paper_stats.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
