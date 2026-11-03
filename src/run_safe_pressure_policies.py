"""Run sealed safe-pressure policy trajectories without reading human outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_confirmation_calibration import read_reward
from analyze_preoptimization_predictors import system_predictors
from analyze_text_policy_predictors import text_parameter_gradient
from run_confirmation_text_policies import optimize_to_kl_grid, probabilities


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reward_specs(root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "checkpoint": f"{row['seed']}/{row['training_condition']}",
            "path": root
            / f"artifacts/reward_models/safe_pressure/{row['seed']}/predictions/"
            f"wmt24_en_es_mqm/{row['training_condition']}.jsonl",
        }
        for row in plan["tasks"]
    ]


def one_step_normalized_kl(
    embeddings: "torch.Tensor", reward: "torch.Tensor", learning_rate: float
) -> float:
    import torch

    action_count = reward.shape[1]
    initial = torch.full_like(reward, 1.0 / action_count)
    expected = (initial * reward).sum(1, keepdims=True)
    gradient = torch.einsum(
        "ca,cah->h", initial * (reward - expected), embeddings
    ) / len(reward)
    weights = learning_rate * gradient
    policy = probabilities(embeddings, weights)
    entropy = float((-(policy * policy.clamp_min(1e-12).log()).sum(1)).mean())
    return (math.log(action_count) - entropy) / math.log(action_count)


def run(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    protocol_path = root / "results/safe_pressure_protocol.json"
    amendment_path = root / "results/safe_pressure_protocol_amendment_1.json"
    manifest_path = root / "results/safe_pressure_manifest.json"
    plan_path = root / "results/safe_pressure_reward_plan.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if manifest["human_outcomes_accessed"]:
        raise ValueError("Manifest reports human outcome access")
    if amendment["human_outcomes_accessed"]:
        raise ValueError("Protocol amendment reports human outcome access")
    if plan["human_outcomes_accessed"]:
        raise ValueError("Reward plan reports human outcome access")

    embedding_path = root / "artifacts/bandit/safe_pressure/wmt24_en_es_mqm_text_embeddings.npz"
    artifact = np.load(embedding_path, allow_pickle=False)
    contexts = artifact["contexts"].tolist()
    actions = artifact["actions"].tolist()
    embeddings_np = artifact["embeddings"].astype(np.float32)
    context_index = {context: index for index, context in enumerate(contexts)}
    action_index = {action: index for index, action in enumerate(actions)}
    split = manifest["split"]
    optimization_indices = np.asarray(
        [context_index[value] for value in split["calibration_optimization_segments"]]
    )
    label_indices = np.asarray(
        [context_index[value] for value in split["calibration_human_label_segments"]]
    )
    evaluation_indices = np.asarray(
        [context_index[value] for value in split["evaluation_segments"]]
    )
    train_indices = np.asarray([action_index[value] for value in split["train_systems"]])
    heldout_indices = np.asarray(
        [action_index[value] for value in split["heldout_systems"]]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_embeddings_np = embeddings_np[np.ix_(optimization_indices, train_indices)]
    train_embeddings = torch.tensor(train_embeddings_np, device=device)
    label_embeddings = torch.tensor(
        embeddings_np[np.ix_(label_indices, train_indices)], device=device
    )
    evaluation_embeddings = torch.tensor(
        embeddings_np[np.ix_(evaluation_indices, heldout_indices)], device=device
    )
    normalized_targets = [float(value) for value in protocol["pressure"]["targets"]]
    raw_targets = [value * math.log(len(train_indices)) for value in normalized_targets]
    specs = reward_specs(root, plan)

    loaded = []
    for spec in specs:
        reward_raw = read_reward(spec["path"], contexts, actions)
        train_values = reward_raw[np.ix_(optimization_indices, train_indices)]
        mean, std = float(train_values.mean()), float(train_values.std())
        if not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid reward scale: {spec['checkpoint']}")
        standardized = (train_values - mean) / std
        loaded.append({**spec, "reward": standardized, "mean": mean, "std": std})

    disagreement: dict[str, float] = {}
    for condition in sorted({row["training_condition"] for row in loaded}):
        matrices = np.stack(
            [row["reward"] for row in loaded if row["training_condition"] == condition]
        )
        disagreement[condition] = float(np.var(matrices, axis=0, ddof=1).mean())

    rows = []
    arrays: dict[str, np.ndarray] = {
        "calibration_contexts": np.asarray(split["calibration_human_label_segments"]),
        "evaluation_contexts": np.asarray(split["evaluation_segments"]),
        "train_systems": np.asarray(split["train_systems"]),
        "heldout_systems": np.asarray(split["heldout_systems"]),
    }
    for row_index, spec in enumerate(loaded):
        reward_np = spec["reward"]
        reward = torch.tensor(reward_np, device=device)
        checkpoints, raw_pressure = optimize_to_kl_grid(
            train_embeddings,
            reward,
            raw_targets,
            float(protocol["policy"]["learning_rate"]),
            int(protocol["policy"]["maximum_steps"]),
        )
        pressure = {}
        artifact_keys = {}
        for normalized, raw in zip(normalized_targets, raw_targets, strict=True):
            record = dict(raw_pressure[str(raw)])
            record["target_normalized_kl"] = normalized
            if record["reached"]:
                record["achieved_normalized_kl"] = (
                    record["achieved_kl_nats"] / math.log(len(train_indices))
                )
                weights = checkpoints[raw]
                calibration_key = f"cal_r{row_index:03d}_k{normalized_targets.index(normalized):03d}"
                evaluation_key = f"eval_r{row_index:03d}_k{normalized_targets.index(normalized):03d}"
                arrays[calibration_key] = probabilities(label_embeddings, weights).cpu().numpy()
                arrays[evaluation_key] = probabilities(evaluation_embeddings, weights).cpu().numpy()
                artifact_keys[str(normalized)] = {
                    "calibration": calibration_key,
                    "evaluation": evaluation_key,
                }
            else:
                record["final_normalized_kl"] = (
                    record["final_kl_nats"] / math.log(len(train_indices))
                )
            pressure[str(normalized)] = record

        gradient = text_parameter_gradient(reward_np, train_embeddings_np)
        centered = reward_np - reward_np.mean(1, keepdims=True)
        sorted_values = np.sort(centered, axis=1)
        diagnostics = {
            **system_predictors(reward_np),
            "initial_text_gradient_norm": float(np.linalg.norm(gradient)),
            "one_step_normalized_kl": one_step_normalized_kl(
                train_embeddings, reward, float(protocol["policy"]["learning_rate"])
            ),
            "reward_tail_gap": float((sorted_values[:, -1] - sorted_values[:, -2]).mean()),
            "reward_seed_disagreement": disagreement[spec["training_condition"]],
        }
        rows.append(
            {
                "checkpoint": spec["checkpoint"],
                "seed": spec["seed"],
                "protocol_condition": spec["protocol_condition"],
                "training_condition": spec["training_condition"],
                "reward_train_mean": spec["mean"],
                "reward_train_std": spec["std"],
                "reward_trace": str(spec["path"]),
                "reward_trace_sha256": sha256(spec["path"]),
                "label_free_diagnostics": diagnostics,
                "pressure": pressure,
                "probability_artifact_keys": artifact_keys,
            }
        )
    result = {
        "status": "safe_pressure_policy_trajectories_fixed_human_outcomes_sealed",
        "protocol": str(protocol_path.relative_to(root)),
        "protocol_sha256": sha256(protocol_path),
        "protocol_amendment": str(amendment_path.relative_to(root)),
        "protocol_amendment_sha256": sha256(amendment_path),
        "manifest": str(manifest_path.relative_to(root)),
        "manifest_sha256": sha256(manifest_path),
        "reward_plan": str(plan_path.relative_to(root)),
        "reward_plan_sha256": sha256(plan_path),
        "embedding_artifact": str(embedding_path.relative_to(root)),
        "embedding_artifact_sha256": sha256(embedding_path),
        "human_outcomes_accessed": False,
        "optimization_segments": len(optimization_indices),
        "calibration_label_segments": len(label_indices),
        "evaluation_segments": len(evaluation_indices),
        "train_systems": split["train_systems"],
        "heldout_systems": split["heldout_systems"],
        "rows": rows,
    }
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/safe_pressure_policies.json")
    )
    parser.add_argument(
        "--probabilities-output",
        type=Path,
        default=Path("artifacts/bandit/safe_pressure/policy_probabilities.npz"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    probabilities_output = root / args.probabilities_output
    if output.exists() or probabilities_output.exists():
        raise FileExistsError("Refusing to overwrite safe-pressure policy artifacts")
    result, arrays = run(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    probabilities_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(probabilities_output, **arrays)
    result["probabilities_output"] = str(probabilities_output.relative_to(root))
    result["probabilities_output_sha256"] = sha256(probabilities_output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
