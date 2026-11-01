"""Develop pre-optimization predictors for the source--candidate-text policy.

The analysis uses already-viewed WMT26 outcomes only. It reuses the frozen text
embeddings and evaluates every available reward checkpoint under the same
held-out-document/held-out-system policy protocol. Frozen WMT23/24 outcomes are
never read.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import run_text_policy_bandit as text_bandit
from analyze_preoptimization_predictors import (
    STRONG_CONDITIONS,
    read_reward_matrix,
    robustness_summary,
    system_predictors,
)
from analyze_wmt25_best_of_n import CONDITIONS
from run_contextual_bandit import document_split
from run_text_policy_bandit import load_environment, policy_probabilities, system_split, train_policy


def text_parameter_gradient(values: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Exact shared text-scorer gradient at the uniform initial policy."""
    action_count = values.shape[1]
    centered = values - values.mean(1, keepdims=True)
    return np.einsum("ca,cah->h", centered / action_count, embeddings) / len(values)


def text_first_order_alignment(
    reward: np.ndarray, human: np.ndarray, embeddings: np.ndarray
) -> dict[str, float]:
    reward_gradient = text_parameter_gradient(reward, embeddings)
    human_gradient = text_parameter_gradient(human, embeddings)
    dot = float(np.dot(reward_gradient, human_gradient))
    norm = float(np.linalg.norm(reward_gradient) * np.linalg.norm(human_gradient))
    return {
        "human_gradient_dot_reward_gradient": dot,
        "human_reward_gradient_cosine": dot / norm if norm else 0.0,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    global torch
    import torch

    text_bandit.torch = torch
    seed_files = sorted(args.seed_results.glob("seed_*.json"))
    if len(seed_files) < 2:
        raise ValueError("Text-policy development requires multiple reward seeds")
    embedding_artifact = np.load(args.embedding_cache, allow_pickle=False)
    embeddings_np = embedding_artifact["embeddings"].astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = torch.tensor(embeddings_np, device=device)
    rows = []
    traces = {}
    reference = None
    instances: list[tuple[str, str, np.ndarray, str, str]] = []
    for seed_file in seed_files:
        seed = seed_file.stem.removeprefix("seed_")
        prediction_root = args.seed_models / seed / "predictions"
        if not prediction_root.exists() and seed == "20261102":
            prediction_root = args.seed_models.parent / "wmt26_predictions"
        environment = load_environment(prediction_root, args.campaign_archive)
        contexts, actions, human, rewards, sources, candidates = environment
        if reference is None:
            reference = (contexts, actions, human, sources, candidates)
            if embeddings_np.shape[:2] != human.shape:
                raise ValueError("Cached embedding shape does not match reward environment")
        elif contexts != reference[0] or actions != reference[1] or not np.array_equal(human, reference[2]):
            raise ValueError(f"Development environment mismatch: {seed}")
        for condition in CONDITIONS:
            instances.append(
                (
                    seed,
                    condition,
                    rewards[condition],
                    "project_multi_seed",
                    str(prediction_root / f"{condition}.jsonl"),
                )
            )
    if reference is None:
        raise AssertionError("No development environment was loaded")
    contexts, actions, human_np, _, _ = reference
    for condition in STRONG_CONDITIONS:
        path = args.strong_predictions / f"{condition}.jsonl"
        instances.append(
            (
                f"external_{condition}",
                condition,
                read_reward_matrix(path, contexts, actions, human_np),
                "external_single_checkpoint",
                str(path),
            )
        )
    train_contexts, test_contexts, document_summary = document_split(contexts)
    train_systems, heldout_systems = system_split(actions)
    train_embeddings_np = embeddings_np[np.ix_(train_contexts, train_systems)]
    heldout_embeddings = embeddings[test_contexts][:, heldout_systems]
    heldout_human = human_np[np.ix_(test_contexts, heldout_systems)]
    uniform_human = float(heldout_human.mean())
    for seed, condition, reward_raw, kind, trace in instances:
        scale_values = reward_raw[np.ix_(train_contexts, train_systems)]
        mean, std = float(scale_values.mean()), float(scale_values.std())
        if not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid reward scale: {seed}/{condition}")
        standardized = (reward_raw - mean) / std
        train_reward = standardized[np.ix_(train_contexts, train_systems)]
        train_human = human_np[np.ix_(train_contexts, train_systems)]
        checkpoints = train_policy(
            torch.tensor(train_embeddings_np, device=device),
            torch.tensor(train_reward, device=device),
            sampled=False,
            seed=None,
        )
        final_weights = checkpoints[text_bandit.STEPS]
        probabilities = policy_probabilities(heldout_embeddings, final_weights).detach().cpu().numpy()
        final_human = float((probabilities * heldout_human).sum(1).mean())
        final_entropy = float(
            (-(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(1)).mean()
        )
        row = {
            "seed": seed,
            "reward_checkpoint_kind": kind,
            "condition": condition,
            **system_predictors(train_reward),
            **text_first_order_alignment(train_reward, train_human, train_embeddings_np),
            "final_human_cesa": final_human,
            "final_human_change_from_uniform": final_human - uniform_human,
            "final_entropy": final_entropy,
        }
        rows.append(row)
        traces[f"{seed}/{condition}"] = {
            "predictors": trace,
            "outcome": "computed by this script under the frozen text-policy protocol",
        }
    predictor_names = (
        "top_system_margin",
        "between_system_eta_squared",
        "initial_system_logit_gradient_norm",
        "system_mean_range",
        "human_gradient_dot_reward_gradient",
        "human_reward_gradient_cosine",
    )
    target = [row["final_human_change_from_uniform"] for row in rows]
    from analyze_bandit_seeds import spearman

    return {
        "status": "exploratory_development_complete",
        "claim_boundary": "Post hoc WMT26 development analysis under the source-candidate-text policy; not confirmation.",
        "policy": "frozen XLM-R source-candidate embeddings with one shared linear scorer and no system identity",
        "reward_seed_count": len(seed_files),
        "external_single_checkpoint_count": len(STRONG_CONDITIONS),
        "reward_instance_count": len(rows),
        "optimization_steps": text_bandit.STEPS,
        "learning_rate": text_bandit.LEARNING_RATE,
        "train_systems": [actions[index] for index in train_systems],
        "heldout_systems": [actions[index] for index in heldout_systems],
        "document_split_by_domain": document_summary,
        "uniform_heldout_human_cesa": uniform_human,
        "rows": rows,
        "spearman_with_final_human_change": {
            name: spearman([row[name] for row in rows], target) for name in predictor_names
        },
        "robustness": robustness_summary(rows, predictor_names),
        "table_row_trace": traces,
        "frozen_confirmation_outcomes_accessed": False,
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["text_policy_predictor_development"] = result
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
    parser.add_argument("--output", type=Path, default=root / "results/text_policy_predictor_development.json")
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
