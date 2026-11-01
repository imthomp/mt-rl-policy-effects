"""Simulate bounded human-label budgets for the first-order predictor.

This is development-only analysis on already-viewed WMT26 outcomes. It samples
at most one segment per document, uses the same sampled contexts for every
reward checkpoint, and never reads the frozen WMT23/24 confirmation outcomes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_bandit_seeds import spearman
from analyze_preoptimization_predictors import (
    STRONG_CONDITIONS,
    first_order_alignment,
    parameter_gradient,
    read_reward_matrix,
)
from analyze_wmt25_best_of_n import CONDITIONS
from run_contextual_bandit import contextual_features, document_split, read_environment


SAMPLE_SEED = 20261130
REPLICATES = 500
SEGMENT_BUDGETS = (16, 32, 64, 128)
DEVELOPMENT_COSINE_THRESHOLD = 0.60


def cosine_with_sampled_human_gradient(
    reward_gradient: np.ndarray, human: np.ndarray, features: np.ndarray
) -> float:
    """Compare the full-pool reward update with a labeled-sample human update."""
    human_gradient = parameter_gradient(human, features)
    norm = float(np.linalg.norm(reward_gradient) * np.linalg.norm(human_gradient))
    return float(np.dot(reward_gradient, human_gradient)) / norm if norm else 0.0


def one_segment_per_document_samples(
    contexts: list[str], train: np.ndarray, budget: int, replicates: int, seed: int
) -> list[np.ndarray]:
    """Draw broad document-clustered samples with no repeated document."""
    by_document: dict[str, list[int]] = defaultdict(list)
    for index in train:
        by_document[contexts[int(index)].rsplit("_###_", 1)[0]].append(int(index))
    documents = sorted(by_document)
    if budget > len(documents):
        raise ValueError(f"Budget {budget} exceeds {len(documents)} training documents")
    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        selected_documents = generator.choice(documents, budget, replace=False)
        samples.append(
            np.asarray(
                [generator.choice(by_document[document]) for document in selected_documents],
                dtype=int,
            )
        )
    return samples


def summarize_budget(
    cosines: np.ndarray, outcomes: np.ndarray, full_decisions: np.ndarray
) -> dict[str, Any]:
    correlations = np.asarray(
        [spearman(cosines[index], outcomes) for index in range(len(cosines))]
    )
    decisions = cosines >= DEVELOPMENT_COSINE_THRESHOLD
    safe = outcomes >= 0.0
    false_positive_rate = np.mean(decisions[:, ~safe], axis=1)
    true_positive_rate = np.mean(decisions[:, safe], axis=1)
    agreement = np.mean(decisions == full_decisions[None, :], axis=1)

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.median(values)),
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }

    return {
        "spearman_with_final_human_change": interval(correlations),
        "decision_agreement_with_full_development_predictor": interval(agreement),
        "false_optimize_rate_on_development_failures": interval(false_positive_rate),
        "optimize_rate_on_development_nonfailures": interval(true_positive_rate),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    development = json.loads(args.development_results.read_text(encoding="utf-8"))
    outcome_by_key = {
        (row["seed"], row["condition"]): row["final_human_change_from_uniform"]
        for row in development["rows"]
    }
    seed_files = sorted(args.seed_results.glob("seed_*.json"))
    instances: list[tuple[str, str, np.ndarray]] = []
    reference = None
    for seed_file in seed_files:
        seed = seed_file.stem.removeprefix("seed_")
        prediction_root = args.seed_models / seed / "predictions"
        if not prediction_root.exists() and seed == "20261102":
            prediction_root = args.seed_models.parent / "wmt26_predictions"
        contexts, actions, human, rewards, source_texts = read_environment(
            prediction_root, args.campaign_archive
        )
        signature = (contexts, actions, human, source_texts)
        if reference is None:
            reference = signature
        elif contexts != reference[0] or actions != reference[1] or not np.array_equal(human, reference[2]):
            raise ValueError(f"Development environment mismatch: {seed}")
        for condition in CONDITIONS:
            instances.append((seed, condition, rewards[condition]))
    if reference is None:
        raise ValueError("No development environments found")
    contexts, actions, human, source_texts = reference
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
    train, _, _ = document_split(contexts)
    features = contextual_features(contexts, source_texts)
    standardized = []
    outcomes = []
    keys = []
    for seed, condition, reward in instances:
        scale = float(reward[train].std())
        if scale <= 0:
            raise ValueError(f"Invalid reward scale: {seed}/{condition}")
        standardized.append((reward - float(reward[train].mean())) / scale)
        outcomes.append(outcome_by_key[(seed, condition)])
        keys.append(f"{seed}/{condition}")
    outcomes_array = np.asarray(outcomes)
    full_cosines = np.asarray(
        [
            first_order_alignment(reward[train], human[train], features[train])[
                "human_reward_gradient_cosine"
            ]
            for reward in standardized
        ]
    )
    full_reward_gradients = [
        parameter_gradient(reward[train], features[train]) for reward in standardized
    ]
    full_decisions = full_cosines >= DEVELOPMENT_COSINE_THRESHOLD
    budgets = {}
    for budget in SEGMENT_BUDGETS:
        samples = one_segment_per_document_samples(
            contexts, train, budget, REPLICATES, SAMPLE_SEED + budget
        )
        cosines = np.empty((REPLICATES, len(instances)))
        for replicate, indices in enumerate(samples):
            for instance, reward in enumerate(standardized):
                cosines[replicate, instance] = cosine_with_sampled_human_gradient(
                    full_reward_gradients[instance], human[indices], features[indices]
                )
        budgets[str(budget)] = {
            "human_scored_candidates": budget * len(actions),
            "instance_cosine_intervals": {
                key: {
                    "median": float(np.median(cosines[:, index])),
                    "low": float(np.quantile(cosines[:, index], 0.025)),
                    "high": float(np.quantile(cosines[:, index], 0.975)),
                }
                for index, key in enumerate(keys)
            },
            **summarize_budget(cosines, outcomes_array, full_decisions),
        }
    return {
        "status": "exploratory_development_complete",
        "claim_boundary": "WMT26 development outcomes were already viewed; no confirmation outcomes were accessed.",
        "sample_seed": SAMPLE_SEED,
        "replicates": REPLICATES,
        "sampling_rule": "one uniformly sampled segment from each of uniformly sampled distinct training documents; full unlabeled pool for reward gradient and sampled human labels for human gradient",
        "action_count": len(actions),
        "training_document_count": len({contexts[int(i)].rsplit('_###_', 1)[0] for i in train}),
        "development_cosine_threshold": DEVELOPMENT_COSINE_THRESHOLD,
        "threshold_rationale": "Round threshold chosen between the largest failing and smallest nonfailing full-development cosine; exploratory and not an uncertainty guarantee.",
        "reward_instance_count": len(instances),
        "development_nonfailure_count": int(np.sum(outcomes_array >= 0.0)),
        "instance_order": keys,
        "full_development_cosines": full_cosines.tolist(),
        "full_development_outcomes": outcomes_array.tolist(),
        "budgets": budgets,
        "frozen_confirmation_outcomes_accessed": False,
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["preoptimization_predictor_label_budget"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-models", type=Path, default=root / "artifacts/reward_models/seeds")
    parser.add_argument("--seed-results", type=Path, default=root / "results/bandit_seeds")
    parser.add_argument("--campaign-archive", type=Path, default=root / "data/reward_da/wmt26-campaign_sources.zip")
    parser.add_argument("--development-results", type=Path, default=root / "results/preoptimization_predictor_development.json")
    parser.add_argument("--strong-predictions", type=Path, default=Path("/home/it238/projects/research_lab/project_16_reward_selection/strong"))
    parser.add_argument("--output", type=Path, default=root / "results/preoptimization_predictor_label_budget.json")
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
