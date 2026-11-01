"""Independently verify bandit outcomes and analyze routing mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = (
    "all_data", "diagnostic_filtered", "global_random_2", "direction_matched_earlier_year"
)
BOOTSTRAP_SEED = 20261120
BOOTSTRAP_RESAMPLES = 10_000


def load_candidates(root: Path, contexts: list[str], actions: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    human = None
    rewards: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        lines = [json.loads(line) for line in (root / f"{condition}.jsonl").read_text().splitlines()[1:]]
        rows = {(row["segment_key"], row["system_id"]): row for row in lines}
        rewards[condition] = np.asarray(
            [[rows[(context, action)]["predicted_reward"] for action in actions] for context in contexts]
        )
        current_human = np.asarray(
            [[rows[(context, action)]["human_cesa"] for action in actions] for context in contexts]
        )
        if human is None:
            human = current_human
        elif not np.array_equal(human, current_human):
            raise ValueError("Human outcomes differ between reward files")
    assert human is not None
    return human, rewards


def cluster_interval(values: np.ndarray, documents: np.ndarray, seed_offset: int) -> dict[str, float]:
    unique = sorted(set(documents.tolist()))
    grouped = {doc: values[documents == doc] for doc in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    samples = np.empty(BOOTSTRAP_RESAMPLES)
    for index in range(BOOTSTRAP_RESAMPLES):
        selected = rng.choice(unique, len(unique), replace=True)
        samples[index] = np.concatenate([grouped[doc] for doc in selected]).mean()
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "document_clusters": len(unique),
        "segments": len(values),
    }


def policy_statistics(probabilities: np.ndarray, reward: np.ndarray, human: np.ndarray) -> dict[str, Any]:
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-30, None))).sum(1)
    return {
        "raw_proxy": float((probabilities * reward).sum(1).mean()),
        "human_cesa": float((probabilities * human).sum(1).mean()),
        "entropy": float(entropy.mean()),
        "kl_from_uniform": float(math.log(probabilities.shape[1]) - entropy.mean()),
    }


def reward_tilt(alpha: float, standardized_reward: np.ndarray) -> np.ndarray:
    logits = alpha * standardized_reward
    logits -= logits.max(1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(1, keepdims=True)
    return probabilities


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    artifact = np.load(args.probabilities, allow_pickle=False)
    contexts = artifact["contexts"].tolist()
    actions = artifact["actions"].tolist()
    train, test = artifact["train_indices"], artifact["test_indices"]
    human, rewards = load_candidates(args.prediction_root, contexts, actions)
    test_docs = np.asarray([contexts[index].rsplit("_###_", 1)[0] for index in test])
    test_domains = np.asarray([contexts[index].split("_###_", 1)[0] for index in test])
    train_domains = np.asarray([contexts[index].split("_###_", 1)[0] for index in train])
    uniform = np.full((len(test), len(actions)), 1 / len(actions))
    uniform_human = (uniform * human[test]).sum(1)
    result: dict[str, Any] = {
        "status": "complete",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "uniform_human": cluster_interval(uniform_human, test_docs, 0),
        "conditions": {},
        "paired_exact_condition_differences": {},
    }
    exact_human_by_condition: dict[str, np.ndarray] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        reward = rewards[condition]
        train_mean, train_std = reward[train].mean(), reward[train].std()
        standardized = (reward - train_mean) / train_std
        key = f"{condition}__exact"
        probabilities = artifact[key][test]
        if not np.allclose(probabilities.sum(1), 1.0, atol=1e-6):
            raise ValueError(f"Invalid saved probabilities: {key}")
        expected_human = (probabilities * human[test]).sum(1)
        exact_human_by_condition[condition] = expected_human
        expected_proxy = (probabilities * standardized[test]).sum(1)
        policy = policy_statistics(probabilities, reward[test], human[test])
        policy["standardized_proxy"] = float(expected_proxy.mean())
        policy["human_change_from_uniform"] = cluster_interval(
            expected_human - uniform_human, test_docs, 100 + condition_index
        )

        fixed_action = int(standardized[train].mean(0).argmax())
        fixed_human = human[test, fixed_action]
        domain_actions = {}
        domain_human = np.empty(len(test))
        for domain in sorted(set(train_domains)):
            action = int(standardized[train[train_domains == domain]].mean(0).argmax())
            domain_actions[domain] = actions[action]
            domain_human[test_domains == domain] = human[test[test_domains == domain], action]

        alphas = np.concatenate(([0.0], np.geomspace(0.001, 100.0, 400)))
        tilt_rows = []
        for alpha in alphas:
            tilted = reward_tilt(float(alpha), standardized[test])
            entropy = float(-(tilted * np.log(np.clip(tilted, 1e-30, None))).sum(1).mean())
            tilt_rows.append(
                {
                    "alpha": float(alpha),
                    "standardized_proxy": float((tilted * standardized[test]).sum(1).mean()),
                    "human_cesa": float((tilted * human[test]).sum(1).mean()),
                    "entropy": entropy,
                    "kl_from_uniform": float(math.log(len(actions)) - entropy),
                }
            )
        matched_proxy = min(
            tilt_rows, key=lambda row: abs(row["standardized_proxy"] - policy["standardized_proxy"])
        )
        matched_entropy = min(tilt_rows, key=lambda row: abs(row["entropy"] - policy["entropy"]))

        def matched_tilt_result(matched: dict[str, float], seed_offset: int) -> dict[str, Any]:
            tilted = reward_tilt(matched["alpha"], standardized[test])
            tilted_human = (tilted * human[test]).sum(1)
            return {
                **matched,
                "exact_minus_tilt_human": cluster_interval(
                    expected_human - tilted_human, test_docs, seed_offset
                ),
            }
        mean_action_probability = probabilities.mean(0)
        top = np.argsort(mean_action_probability)[::-1][:5]
        permutation = np.random.default_rng(BOOTSTRAP_SEED).permutation(len(actions))
        original = (probabilities * human[test]).sum(1)
        permuted = (probabilities[:, permutation] * human[test][:, permutation]).sum(1)
        if not np.allclose(original, permuted, atol=1e-10):
            raise AssertionError("Consistent action permutation changed expected quality")

        reinforce = {}
        for artifact_key in artifact.files:
            prefix = f"{condition}__reinforce__"
            if artifact_key.startswith(prefix):
                seed_probabilities = artifact[artifact_key][test]
                seed_human = (seed_probabilities * human[test]).sum(1)
                reinforce[artifact_key.removeprefix(prefix)] = {
                    **policy_statistics(seed_probabilities, reward[test], human[test]),
                    "human_change_from_uniform": cluster_interval(
                        seed_human - uniform_human,
                        test_docs,
                        1000 + condition_index * 10 + len(reinforce),
                    ),
                }
        result["conditions"][condition] = {
            "exact_policy": policy,
            "reinforce_seeds": reinforce,
            "best_fixed_system": {
                "system": actions[fixed_action],
                "human_cesa": cluster_interval(fixed_human, test_docs, 200 + condition_index),
                "exact_minus_fixed_human": cluster_interval(
                    expected_human - fixed_human, test_docs, 300 + condition_index
                ),
            },
            "domain_routed_systems": domain_actions,
            "domain_routed_human": cluster_interval(domain_human, test_docs, 400 + condition_index),
            "reward_tilt_matched_proxy": matched_tilt_result(
                matched_proxy, 600 + condition_index
            ),
            "reward_tilt_matched_entropy_and_kl": matched_tilt_result(
                matched_entropy, 700 + condition_index
            ),
            "top_mean_action_probabilities": [
                {"system": actions[index], "probability": float(mean_action_probability[index])}
                for index in top
            ],
            "consistent_action_permutation_invariance": True,
        }
    for left_index, left in enumerate(CONDITIONS):
        for right in CONDITIONS[left_index + 1 :]:
            result["paired_exact_condition_differences"][f"{left}_minus_{right}"] = cluster_interval(
                exact_human_by_condition[left] - exact_human_by_condition[right],
                test_docs,
                5000 + len(result["paired_exact_condition_differences"]),
            )
    return result


def validate_reconstruction(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text())
    recorded_conditions = root["contextual_bandit"]["conditions"]
    fields = ("human_cesa", "raw_proxy", "standardized_proxy", "entropy")
    for condition in CONDITIONS:
        recorded = recorded_conditions[condition]["trajectories"]["exact"][-1]["test"]
        reconstructed = result["conditions"][condition]["exact_policy"]
        for field in fields:
            if not math.isclose(recorded[field], reconstructed[field], abs_tol=1e-5):
                raise AssertionError(
                    f"Independent reconstruction mismatch for {condition}.{field}: "
                    f"{reconstructed[field]} versus {recorded[field]}"
                )
    result["independent_reconstruction_matches_recorded_metrics"] = True


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text())
    root["bandit_mechanism"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probabilities", type=Path, default=root / "artifacts/bandit/final_action_probabilities.npz")
    parser.add_argument("--prediction-root", type=Path, default=root / "artifacts/reward_models/wmt26_predictions")
    parser.add_argument("--output", type=Path, default=root / "results/paper_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args)
    validate_reconstruction(args.output, result)
    merge_ssot(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
