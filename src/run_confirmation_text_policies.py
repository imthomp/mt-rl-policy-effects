"""Run frozen confirmation policies without reading evaluation outcomes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_confirmation_calibration import read_reward, reward_specs


def probabilities(embeddings: "torch.Tensor", weights: "torch.Tensor") -> "torch.Tensor":
    import torch

    return torch.softmax(torch.einsum("cah,h->ca", embeddings, weights), dim=-1)


def pressure_metrics(
    embeddings: "torch.Tensor", weights: "torch.Tensor", reward: "torch.Tensor"
) -> tuple[float, float, float]:
    import torch

    policy = probabilities(embeddings, weights)
    entropy = float((-(policy * policy.clamp_min(1e-12).log()).sum(1)).mean())
    kl = math.log(policy.shape[1]) - entropy
    proxy = float((policy * reward).sum(1).mean())
    return kl, entropy, proxy


def optimize_to_kl_grid(
    embeddings: "torch.Tensor",
    reward: "torch.Tensor",
    targets: list[float],
    learning_rate: float,
    maximum_steps: int,
) -> tuple[dict[float, "torch.Tensor"], dict[str, dict[str, Any]]]:
    import torch

    weights = torch.zeros(embeddings.shape[-1], device=embeddings.device)
    checkpoints: dict[float, torch.Tensor] = {}
    records: dict[str, dict[str, Any]] = {}
    ordered_targets = sorted(targets)
    for step in range(maximum_steps + 1):
        policy = probabilities(embeddings, weights)
        entropy_tensor = (-(policy * policy.clamp_min(1e-12).log()).sum(1)).mean()
        kl = math.log(policy.shape[1]) - float(entropy_tensor)
        for target in ordered_targets:
            reached = (step == 0 and target == 0.0) or kl + 1e-7 >= target
            if target not in checkpoints and reached:
                proxy = float((policy * reward).sum(1).mean())
                checkpoints[target] = weights.detach().clone()
                records[str(target)] = {
                    "target_kl_nats": target,
                    "reached": True,
                    "step": step,
                    "achieved_kl_nats": kl,
                    "train_entropy": float(entropy_tensor),
                    "train_standardized_proxy": proxy,
                }
        if len(checkpoints) == len(ordered_targets):
            break
        if step == maximum_steps:
            break
        expected = (policy * reward).sum(1, keepdims=True)
        gradient = torch.einsum(
            "ca,cah->h", policy * (reward - expected), embeddings
        ) / len(reward)
        weights = weights + learning_rate * gradient
    for target in ordered_targets:
        if target not in checkpoints:
            records[str(target)] = {
                "target_kl_nats": target,
                "reached": False,
                "maximum_steps": maximum_steps,
                "final_kl_nats": kl,
            }
    return checkpoints, records


def run(root: Path, dataset_name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    protocol = json.loads(
        (root / "results/text_policy_predictor_protocol.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        (root / "results/text_policy_predictor_confirmation_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    if calibration["status"] != "calibration_decisions_frozen_evaluation_outcomes_sealed":
        raise ValueError("Calibration decisions are not frozen")
    if calibration["evaluation_outcomes_accessed"]:
        raise ValueError("Calibration artifact reports evaluation outcome access")
    manifest = json.loads(
        (root / "results/predictor_confirmation_manifest.json").read_text(encoding="utf-8")
    )
    dataset = manifest["datasets"][dataset_name]
    split = dataset["calibration_evaluation_split"]
    system_split = dataset["policy_system_split"]
    embedding_path = root / f"artifacts/bandit/confirmation/{dataset_name}_text_embeddings.npz"
    artifact = np.load(embedding_path, allow_pickle=False)
    contexts = artifact["contexts"].tolist()
    actions = artifact["actions"].tolist()
    embeddings_np = artifact["embeddings"].astype(np.float32)
    context_index = {context: index for index, context in enumerate(contexts)}
    action_index = {action: index for index, action in enumerate(actions)}
    optimization_indices = np.asarray(
        [context_index[context] for context in split["optimization_segment_ids"]]
    )
    evaluation_indices = np.asarray(
        [context_index[context] for context in split["evaluation_segment_ids"]]
    )
    train_indices = np.asarray(
        [action_index[action] for action in system_split["train_systems"]]
    )
    heldout_indices = np.asarray(
        [action_index[action] for action in system_split["heldout_systems"]]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_embeddings = torch.tensor(
        embeddings_np[np.ix_(optimization_indices, train_indices)], device=device
    )
    evaluation_embeddings = torch.tensor(
        embeddings_np[np.ix_(evaluation_indices, heldout_indices)], device=device
    )
    targets = [float(value) for value in protocol["optimization"]["pressure_grid_nats"]]
    decisions = {
        row["checkpoint"]: row
        for row in calibration["datasets"][dataset_name]["rows"]
    }
    rows = []
    arrays: dict[str, np.ndarray] = {
        "evaluation_contexts": np.asarray(split["evaluation_segment_ids"]),
        "heldout_systems": np.asarray(system_split["heldout_systems"]),
    }
    for row_index, spec in enumerate(reward_specs(protocol, root, dataset_name)):
        reward_raw = read_reward(spec["path"], contexts, actions)
        train_values = reward_raw[np.ix_(optimization_indices, train_indices)]
        mean, std = float(train_values.mean()), float(train_values.std())
        if not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid reward scale: {dataset_name}/{spec['checkpoint']}")
        train_reward = torch.tensor((train_values - mean) / std, device=device)
        checkpoints, pressure = optimize_to_kl_grid(
            train_embeddings,
            train_reward,
            targets,
            float(protocol["optimization"]["learning_rate"]),
            int(protocol["optimization"]["maximum_steps"]),
        )
        artifact_keys = {}
        for target, weights in checkpoints.items():
            key = f"r{row_index:03d}_k{targets.index(target):03d}"
            artifact_keys[str(target)] = key
            arrays[key] = probabilities(evaluation_embeddings, weights).cpu().numpy()
        decision = decisions[spec["checkpoint"]]
        rows.append(
            {
                "checkpoint": spec["checkpoint"],
                "kind": spec["kind"],
                "seed": spec["seed"],
                "condition": spec["condition"],
                "sign_screen_decision": decision["sign_screen_decision"],
                "best_of_n_screen_decision": decision["best_of_n_screen_decision"],
                "reward_train_mean": mean,
                "reward_train_std": std,
                "pressure": pressure,
                "probability_artifact_keys": artifact_keys,
                "reward_trace": str(spec["path"]),
            }
        )
    result = {
        "status": "policy_trajectories_fixed_evaluation_outcomes_sealed",
        "dataset": dataset_name,
        "optimization_segments": len(optimization_indices),
        "evaluation_segments": len(evaluation_indices),
        "train_systems": system_split["train_systems"],
        "heldout_systems": system_split["heldout_systems"],
        "rows": rows,
        "evaluation_outcomes_accessed": False,
    }
    return result, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probabilities-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.probabilities_output.exists():
        raise FileExistsError("Refusing to overwrite confirmation policy artifacts")
    result, arrays = run(root, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.probabilities_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.probabilities_output, **arrays)
    result["probabilities_output"] = str(args.probabilities_output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
