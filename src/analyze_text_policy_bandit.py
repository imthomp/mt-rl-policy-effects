"""Independently reconstruct held-out-system text-policy outcomes and mechanisms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_wmt25_best_of_n import CONDITIONS
from run_contextual_bandit import document_split
from run_text_policy_bandit import load_environment, system_split


def policy_statistics(
    probabilities: np.ndarray, reward: np.ndarray, human: np.ndarray
) -> dict[str, float]:
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-30, None))).sum(1)
    return {
        "raw_proxy": float((probabilities * reward).sum(1).mean()),
        "human_cesa": float((probabilities * human).sum(1).mean()),
        "entropy": float(entropy.mean()),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    artifact = np.load(args.probabilities, allow_pickle=False)
    contexts, actions, human, rewards, _, candidate_texts = load_environment(
        args.prediction_root, args.campaign_archive
    )
    _, test_contexts, _ = document_split(contexts)
    _, heldout_systems = system_split(actions)
    expected_contexts = [contexts[index] for index in test_contexts]
    expected_systems = [actions[index] for index in heldout_systems]
    if artifact["contexts"].tolist() != expected_contexts:
        raise ValueError("Probability artifact context order does not match frozen split")
    if artifact["heldout_systems"].tolist() != expected_systems:
        raise ValueError("Probability artifact system order does not match frozen split")

    human_eval = human[np.ix_(test_contexts, heldout_systems)]
    uniform_human = human_eval.mean(1)
    text_lengths = np.asarray(
        [[len(candidate_texts[context][system]) for system in heldout_systems]
         for context in test_contexts],
        dtype=np.float64,
    )
    result: dict[str, Any] = {
        "status": "complete",
        "contexts": len(test_contexts),
        "systems": len(heldout_systems),
        "independent_reconstruction_matches_recorded_metrics": True,
        "conditions": {},
        "table_row_trace": {},
    }
    root = json.loads(args.output.read_text(encoding="utf-8"))
    recorded = root["text_policy_bandit"]["conditions"]
    for condition in CONDITIONS:
        probabilities = artifact[f"{condition}__exact__heldout"]
        if probabilities.shape != human_eval.shape:
            raise ValueError(f"Probability shape mismatch for {condition}")
        if not np.allclose(probabilities.sum(1), 1.0, atol=1e-6):
            raise ValueError(f"Invalid probabilities for {condition}")
        reward_eval = rewards[condition][np.ix_(test_contexts, heldout_systems)]
        stats = policy_statistics(probabilities, reward_eval, human_eval)
        for field in ("raw_proxy", "human_cesa", "entropy"):
            expected = recorded[condition]["primary_exact"][field]
            if not math.isclose(stats[field], expected, abs_tol=1e-5):
                result["independent_reconstruction_matches_recorded_metrics"] = False
                raise AssertionError(
                    f"Reconstruction mismatch for {condition}.{field}: {stats[field]} vs {expected}"
                )
        mean_probability = probabilities.mean(0)
        top = np.argsort(mean_probability)[::-1]
        system_rows = [
            {
                "system": expected_systems[index],
                "mean_probability": float(mean_probability[index]),
                "mean_human_cesa": float(human_eval[:, index].mean()),
                "mean_reward": float(reward_eval[:, index].mean()),
            }
            for index in top
        ]
        result["conditions"][condition] = {
            **stats,
            "human_change_from_uniform": float(
                ((probabilities * human_eval).sum(1) - uniform_human).mean()
            ),
            "expected_candidate_characters": float((probabilities * text_lengths).sum(1).mean()),
            "uniform_candidate_characters": float(text_lengths.mean()),
            "system_rows": system_rows,
        }
        result["table_row_trace"][condition] = {
            row["system"]: (
                f"artifacts/reward_models/wmt26_predictions/{condition}.jsonl; "
                "artifacts/bandit/text_policy_heldout_probabilities.npz"
            )
            for row in system_rows
        }
    return result


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8"))
    root["text_policy_mechanism"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probabilities",
        type=Path,
        default=root / "artifacts/bandit/text_policy_heldout_probabilities.npz",
    )
    parser.add_argument(
        "--prediction-root", type=Path, default=root / "artifacts/reward_models/wmt26_predictions"
    )
    parser.add_argument(
        "--campaign-archive", type=Path, default=root / "data/reward_da/wmt26-campaign_sources.zip"
    )
    parser.add_argument("--output", type=Path, default=root / "results/paper_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args)
    merge_ssot(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
