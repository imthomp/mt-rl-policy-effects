"""Simulate human-label budgets for the text-policy first-order sign rule.

All outcomes are already-viewed WMT26 development outcomes. Reward gradients
use the full unlabeled training pool; human gradients use document-disjoint
calibration samples. Frozen WMT23/24 outcomes are never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_bandit_seeds import spearman
from analyze_predictor_label_budget import one_segment_per_document_samples
from analyze_preoptimization_predictors import STRONG_CONDITIONS, read_reward_matrix
from analyze_text_policy_predictors import text_parameter_gradient
from analyze_wmt25_best_of_n import CONDITIONS
from run_contextual_bandit import document_split
from run_text_policy_bandit import load_environment, system_split


SAMPLE_SEED = 20261202
REPLICATES = 500
SEGMENT_BUDGETS = (16, 32, 64, 128)
DECISION_THRESHOLD = 0.0


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    development = json.loads(args.development_results.read_text(encoding="utf-8"))
    outcome_by_key = {
        (row["seed"], row["condition"]): row["final_human_change_from_uniform"]
        for row in development["rows"]
    }
    seed_files = sorted(args.seed_results.glob("seed_*.json"))
    reference = None
    instances: list[tuple[str, str, np.ndarray]] = []
    for seed_file in seed_files:
        seed = seed_file.stem.removeprefix("seed_")
        prediction_root = args.seed_models / seed / "predictions"
        if not prediction_root.exists() and seed == "20261102":
            prediction_root = args.seed_models.parent / "wmt26_predictions"
        environment = load_environment(prediction_root, args.campaign_archive)
        contexts, actions, human, rewards, sources, candidates = environment
        if reference is None:
            reference = (contexts, actions, human, sources, candidates)
        elif contexts != reference[0] or actions != reference[1] or not np.array_equal(human, reference[2]):
            raise ValueError(f"Development environment mismatch: {seed}")
        for condition in CONDITIONS:
            instances.append((seed, condition, rewards[condition]))
    if reference is None:
        raise ValueError("No development environments found")
    contexts, actions, human, _, _ = reference
    for condition in STRONG_CONDITIONS:
        instances.append(
            (
                f"external_{condition}",
                condition,
                read_reward_matrix(
                    args.strong_predictions / f"{condition}.jsonl",
                    contexts,
                    actions,
                    human,
                ),
            )
        )
    embeddings = np.load(args.embedding_cache, allow_pickle=False)["embeddings"].astype(np.float32)
    train_contexts, _, _ = document_split(contexts)
    train_systems, _ = system_split(actions)
    train_embeddings = embeddings[np.ix_(train_contexts, train_systems)]
    train_human = human[np.ix_(train_contexts, train_systems)]
    keys = [f"{seed}/{condition}" for seed, condition, _ in instances]
    outcomes = np.asarray([outcome_by_key[(seed, condition)] for seed, condition, _ in instances])
    reward_gradients = []
    for _, _, reward in instances:
        values = reward[np.ix_(train_contexts, train_systems)]
        standardized = (values - float(values.mean())) / float(values.std())
        reward_gradients.append(text_parameter_gradient(standardized, train_embeddings))
    reward_gradients = np.asarray(reward_gradients)
    reward_unit = reward_gradients / np.linalg.norm(reward_gradients, axis=1, keepdims=True)
    full_human_gradient = text_parameter_gradient(train_human, train_embeddings)
    full_cosines = reward_unit @ (full_human_gradient / np.linalg.norm(full_human_gradient))
    full_decisions = full_cosines > DECISION_THRESHOLD
    budgets = {}
    for budget in SEGMENT_BUDGETS:
        samples = one_segment_per_document_samples(
            contexts, train_contexts, budget, REPLICATES, SAMPLE_SEED + budget
        )
        cosines = np.empty((REPLICATES, len(instances)))
        for replicate, global_indices in enumerate(samples):
            local_lookup = {int(global_index): local for local, global_index in enumerate(train_contexts)}
            local_indices = np.asarray([local_lookup[int(index)] for index in global_indices])
            human_gradient = text_parameter_gradient(
                train_human[local_indices], train_embeddings[local_indices]
            )
            human_norm = np.linalg.norm(human_gradient)
            cosines[replicate] = reward_unit @ (human_gradient / human_norm)
        decisions = cosines > DECISION_THRESHOLD
        safe = outcomes >= 0.0
        correlations = np.asarray([spearman(row, outcomes) for row in cosines])
        budgets[str(budget)] = {
            "human_scored_candidates": budget * len(train_systems),
            "spearman_with_final_human_change": interval(correlations),
            "decision_agreement_with_full_development_predictor": interval(
                np.mean(decisions == full_decisions[None, :], axis=1)
            ),
            "false_optimize_rate_on_development_failures": interval(
                np.mean(decisions[:, ~safe], axis=1)
            ),
            "optimize_rate_on_development_nonfailures": interval(
                np.mean(decisions[:, safe], axis=1)
            ),
            "instance_cosine_intervals": {
                key: interval(cosines[:, index]) for index, key in enumerate(keys)
            },
        }
    return {
        "status": "exploratory_development_complete",
        "claim_boundary": "Already-viewed WMT26 development outcomes under the text policy; not confirmation.",
        "sample_seed": SAMPLE_SEED,
        "replicates": REPLICATES,
        "sampling_rule": "one segment per sampled document; full unlabeled train pool for reward gradient and sampled human labels for human gradient",
        "decision_rule": "optimize iff the estimated text-policy reward/human gradient cosine is strictly positive; otherwise keep the initial policy",
        "decision_threshold": DECISION_THRESHOLD,
        "train_system_count": len(train_systems),
        "training_document_count": len({contexts[int(i)].rsplit('_###_', 1)[0] for i in train_contexts}),
        "reward_instance_count": len(instances),
        "development_nonfailure_count": int(np.sum(outcomes >= 0.0)),
        "instance_order": keys,
        "full_development_cosines": full_cosines.tolist(),
        "full_development_outcomes": outcomes.tolist(),
        "budgets": budgets,
        "frozen_confirmation_outcomes_accessed": False,
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["text_policy_predictor_label_budget"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-models", type=Path, default=root / "artifacts/reward_models/seeds")
    parser.add_argument("--seed-results", type=Path, default=root / "results/bandit_seeds")
    parser.add_argument("--campaign-archive", type=Path, default=root / "data/reward_da/wmt26-campaign_sources.zip")
    parser.add_argument("--embedding-cache", type=Path, default=root / "artifacts/bandit/text_pair_embeddings.npz")
    parser.add_argument("--strong-predictions", type=Path, default=Path("/home/it238/projects/research_lab/project_16_reward_selection/strong"))
    parser.add_argument("--development-results", type=Path, default=root / "results/text_policy_predictor_development.json")
    parser.add_argument("--output", type=Path, default=root / "results/text_policy_predictor_label_budget.json")
    parser.add_argument("--ssot", type=Path, default=root / "results/paper_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    merge_ssot(args.ssot, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
