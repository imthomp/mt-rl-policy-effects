"""Develop cheap pre-optimization failure predictors on already-viewed WMT26 data.

This is exploratory development only. It must not read the frozen WMT23/24 MQM
outcomes. Predictors use bandit-training contexts; outcomes use the existing
WMT26 test contexts and are already known.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_bandit_seeds import spearman
from analyze_wmt25_best_of_n import CONDITIONS
from run_contextual_bandit import (
    contextual_features,
    document_split,
    read_environment,
)
from wmt26_candidate_pool import EXCLUDED_NON_MACHINE_SYSTEMS


BOOTSTRAP_SEED = 20261129
BOOTSTRAP_RESAMPLES = 10_000
STRONG_CONDITIONS = ("wmt22-cometkiwi-da", "wmt23-cometkiwi-da-xl")


def system_predictors(reward: np.ndarray) -> dict[str, float]:
    """Outcome-free diagnostics on a contexts-by-systems standardized matrix."""
    source_centered = reward - reward.mean(1, keepdims=True)
    system_means = source_centered.mean(0)
    total = float(np.square(source_centered).sum())
    between = float(reward.shape[0] * np.square(system_means).sum())
    action_count = reward.shape[1]
    uniform_logit_gradient = system_means / action_count
    return {
        "top_system_margin": float(system_means.max()),
        "between_system_eta_squared": between / total if total else 0.0,
        "initial_system_logit_gradient_norm": float(np.linalg.norm(uniform_logit_gradient)),
        "system_mean_range": float(system_means.max() - system_means.min()),
    }


def parameter_gradient(
    values: np.ndarray, features: np.ndarray
) -> np.ndarray:
    """Exact shared-policy gradient at uniform action probabilities."""
    action_count = values.shape[1]
    logit_gradient = (values - values.mean(1, keepdims=True)) / action_count
    weight_gradient = features.T @ logit_gradient / len(values)
    bias_gradient = logit_gradient.mean(0)
    return np.concatenate((weight_gradient.ravel(), bias_gradient))


def first_order_alignment(
    reward: np.ndarray, human: np.ndarray, features: np.ndarray
) -> dict[str, float]:
    reward_gradient = parameter_gradient(reward, features)
    human_gradient = parameter_gradient(human, features)
    dot = float(np.dot(reward_gradient, human_gradient))
    norms = float(np.linalg.norm(reward_gradient) * np.linalg.norm(human_gradient))
    return {
        "human_gradient_dot_reward_gradient": dot,
        "human_reward_gradient_cosine": dot / norms if norms else 0.0,
    }


def read_reward_matrix(
    path: Path, contexts: list[str], actions: list[str], human: np.ndarray
) -> np.ndarray:
    """Read one external reward while asserting the shared candidate environment."""
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0])
    rows = [json.loads(line) for line in lines[1:]]
    if metadata.get("record_type") != "metadata" or metadata.get("rows") != len(rows):
        raise ValueError(f"Invalid reward metadata: {path}")
    keyed = {(row["segment_key"], row["system_id"]): row for row in rows}
    expected = {(context, action) for context in contexts for action in actions}
    observed = set(keyed)
    extras = observed - expected
    if not expected <= observed or {action for _, action in extras} - EXCLUDED_NON_MACHINE_SYSTEMS:
        raise ValueError(f"External reward candidate mismatch: {path}")
    observed_human = np.asarray(
        [[keyed[(context, action)]["human_cesa"] for action in actions] for context in contexts],
        dtype=np.float32,
    )
    if not np.array_equal(observed_human, human):
        raise ValueError(f"External reward human-score mismatch: {path}")
    return np.asarray(
        [
            [keyed[(context, action)]["predicted_reward"] for action in actions]
            for context in contexts
        ],
        dtype=np.float32,
    )


def robustness_summary(
    rows: list[dict[str, Any]], predictor_names: tuple[str, ...]
) -> dict[str, Any]:
    seeds = sorted({row["seed"] for row in rows})
    by_seed = {seed: [row for row in rows if row["seed"] == seed] for seed in seeds}
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = {name: np.empty(BOOTSTRAP_RESAMPLES) for name in predictor_names}
    for replicate in range(BOOTSTRAP_RESAMPLES):
        selected = generator.choice(seeds, len(seeds), replace=True)
        sample = [row for seed in selected for row in by_seed[seed]]
        target = [row["final_human_change_from_uniform"] for row in sample]
        for name in predictor_names:
            bootstrap[name][replicate] = spearman([row[name] for row in sample], target)
    leave_one_out = {}
    for heldout in seeds:
        sample = [row for row in rows if row["seed"] != heldout]
        target = [row["final_human_change_from_uniform"] for row in sample]
        leave_one_out[heldout] = {
            name: spearman([row[name] for row in sample], target) for name in predictor_names
        }
    return {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "seed_clustered_intervals": {
            name: {
                "low": float(np.quantile(values, 0.025)),
                "high": float(np.quantile(values, 0.975)),
            }
            for name, values in bootstrap.items()
        },
        "leave_one_reward_seed_out": leave_one_out,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    seed_files = sorted(args.seed_results.glob("seed_*.json"))
    if len(seed_files) < 2:
        raise ValueError("Development analysis requires multiple reward seed files")
    rows = []
    traces = {}
    uniform_values = set()
    reference_environment = None
    for seed_file in seed_files:
        seed = seed_file.stem.removeprefix("seed_")
        prediction_root = args.seed_models / seed / "predictions"
        if not prediction_root.exists() and seed == "20261102":
            prediction_root = args.seed_models.parent / "wmt26_predictions"
        contexts, actions, human, rewards, source_texts = read_environment(
            prediction_root, args.campaign_archive
        )
        if reference_environment is None:
            reference_environment = (contexts, actions, human, source_texts)
        train, test, _ = document_split(contexts)
        features = contextual_features(contexts, source_texts)[train]
        run = json.loads(seed_file.read_text(encoding="utf-8"))["contextual_bandit"]
        if set(run["conditions"]) != set(CONDITIONS):
            raise ValueError(f"Incomplete conditions in {seed_file}")
        for condition in CONDITIONS:
            reward = rewards[condition]
            train_mean = float(reward[train].mean())
            train_std = float(reward[train].std())
            if not math.isfinite(train_std) or train_std <= 0:
                raise ValueError(f"Invalid reward scale: {seed}/{condition}")
            standardized = (reward[train] - train_mean) / train_std
            final = run["conditions"][condition]["trajectories"]["exact"][-1]["test"]
            uniform = run["conditions"][condition]["baselines"]["test"]["uniform"][
                "human_cesa"
            ]
            uniform_values.add(round(float(uniform), 8))
            row = {
                "seed": seed,
                "reward_checkpoint_kind": "project_multi_seed",
                "condition": condition,
                **system_predictors(standardized),
                **first_order_alignment(standardized, human[train], features),
                "final_human_cesa": final["human_cesa"],
                "final_human_change_from_uniform": final["human_cesa"] - uniform,
                "final_entropy": final["entropy"],
                "final_standardized_proxy": final["standardized_proxy"],
            }
            rows.append(row)
            traces[f"{seed}/{condition}"] = {
                "predictors": str(prediction_root / f"{condition}.jsonl"),
                "outcome": f"results/bandit_seeds/seed_{seed}.json",
            }
    if reference_environment is None:
        raise AssertionError("No development environment was loaded")
    contexts, actions, human, source_texts = reference_environment
    train, _, _ = document_split(contexts)
    features = contextual_features(contexts, source_texts)[train]
    strong_result = json.loads(args.strong_results.read_text(encoding="utf-8"))[
        "contextual_bandit"
    ]["conditions"]
    for condition in STRONG_CONDITIONS:
        reward_path = args.strong_predictions / f"{condition}.jsonl"
        reward = read_reward_matrix(reward_path, contexts, actions, human)
        train_mean = float(reward[train].mean())
        train_std = float(reward[train].std())
        if not math.isfinite(train_std) or train_std <= 0:
            raise ValueError(f"Invalid external reward scale: {condition}")
        standardized = (reward[train] - train_mean) / train_std
        condition_result = strong_result[condition]
        final = condition_result["trajectories"]["exact"][-1]["test"]
        uniform = condition_result["baselines"]["test"]["uniform"]["human_cesa"]
        uniform_values.add(round(float(uniform), 8))
        row = {
            "seed": f"external_{condition}",
            "reward_checkpoint_kind": "external_single_checkpoint",
            "condition": condition,
            **system_predictors(standardized),
            **first_order_alignment(standardized, human[train], features),
            "final_human_cesa": final["human_cesa"],
            "final_human_change_from_uniform": final["human_cesa"] - uniform,
            "final_entropy": final["entropy"],
            "final_standardized_proxy": final["standardized_proxy"],
        }
        rows.append(row)
        traces[f"external/{condition}"] = {
            "predictors": str(reward_path),
            "outcome": str(args.strong_results),
        }
    if len(uniform_values) != 1:
        raise AssertionError("Uniform human baseline differs across development runs")
    predictor_names = (
        "top_system_margin",
        "between_system_eta_squared",
        "initial_system_logit_gradient_norm",
        "system_mean_range",
        "human_gradient_dot_reward_gradient",
        "human_reward_gradient_cosine",
    )
    target = [row["final_human_change_from_uniform"] for row in rows]
    correlations = {
        name: spearman([row[name] for row in rows], target) for name in predictor_names
    }
    return {
        "status": "exploratory_development_complete",
        "claim_boundary": (
            "Post hoc predictor development on already-viewed WMT26 outcomes; not confirmation."
        ),
        "reward_seed_count": len(seed_files),
        "external_single_checkpoint_count": len(STRONG_CONDITIONS),
        "reward_instance_count": len(rows),
        "predictor_context_split": "WMT26 training documents only",
        "outcome_context_split": "WMT26 test documents only",
        "uniform_human_cesa": next(iter(uniform_values)),
        "rows": rows,
        "spearman_with_final_human_change": correlations,
        "robustness": robustness_summary(rows, predictor_names),
        "table_row_trace": traces,
        "frozen_confirmation_outcomes_accessed": False,
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["preoptimization_predictor_development"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-models", type=Path, default=root / "artifacts/reward_models/seeds"
    )
    parser.add_argument(
        "--seed-results", type=Path, default=root / "results/bandit_seeds"
    )
    parser.add_argument(
        "--campaign-archive", type=Path, default=root / "data/reward_da/wmt26-campaign_sources.zip"
    )
    parser.add_argument(
        "--strong-predictions",
        type=Path,
        default=Path(
            "/home/it238/projects/research_lab/project_16_reward_selection/strong"
        ),
    )
    parser.add_argument(
        "--strong-results",
        type=Path,
        default=Path(
            "/home/it238/projects/research_lab/project_16_reward_selection/results/strong.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=root / "results/preoptimization_predictor_development.json"
    )
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
