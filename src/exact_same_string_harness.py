"""Exact finite-policy harness for same-string segmentation invariance.

This is a CPU-only estimator check, not a translation model. It samples a
parameterized string first, then a segmentation path from q(phi | string).
Ordinary REINFORCE uses the exact joint probability p_theta(string) q_phi(path
| string); token/position-weighted variants are explicitly different
estimators and are not silently treated as the joint-policy score.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import dataclass
from typing import Iterable


STRINGS = ("A", "B", "C")
REWARDS = {"A": 1.0, "B": 0.2, "C": -0.7}
SHORT_PATHS = {
    "A": ("A",),
    "C": ("C1", "C2"),
}
LONG_PATHS = {
    "A": ("A1", "A2", "A3", "A4"),
    "C": ("C1", "C2", "C3", "C4", "C5"),
}
FIXED_PATHS = {"B": ("B1", "B2")}


@dataclass(frozen=True)
class Outcome:
    string: str
    path: tuple[str, ...]
    probability: float
    behavior_log_probability: float
    reward: float
    score_theta: tuple[float, ...]
    score_phi: float

    @property
    def token_count(self) -> int:
        return len(self.path)


def softmax(values: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(values)
    exps = tuple(math.exp(value - maximum) for value in values)
    total = sum(exps)
    return tuple(value / total for value in exps)


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def enumerate_outcomes(theta: tuple[float, ...], phi: float) -> tuple[Outcome, ...]:
    """Enumerate P_theta(s) q_phi(z|s), with exact joint scores."""
    if len(theta) != len(STRINGS):
        raise ValueError(f"theta must have {len(STRINGS)} logits")
    string_probs = softmax(theta)
    q_short = sigmoid(phi)
    outcomes = []
    for index, string in enumerate(STRINGS):
        p_string = string_probs[index]
        score_theta = tuple(
            (1.0 if coordinate == index else 0.0) - string_probs[coordinate]
            for coordinate in range(len(STRINGS))
        )
        reward = REWARDS[string]
        if string in FIXED_PATHS:
            path_options = ((FIXED_PATHS[string], 1.0, 0.0),)
        else:
            path_options = (
                (SHORT_PATHS[string], q_short, 1.0 - q_short),
                (LONG_PATHS[string], 1.0 - q_short, -q_short),
            )
        for path, path_prob, score_phi in path_options:
            probability = p_string * path_prob
            outcomes.append(
                Outcome(
                    string=string,
                    path=path,
                    probability=probability,
                    behavior_log_probability=math.log(p_string) + math.log(path_prob),
                    reward=reward,
                    score_theta=score_theta,
                    score_phi=score_phi,
                )
            )
    return tuple(outcomes)


def exact_string_reward(theta: tuple[float, ...]) -> float:
    return sum(p * REWARDS[s] for p, s in zip(softmax(theta), STRINGS, strict=True))


def exact_string_gradient(theta: tuple[float, ...]) -> tuple[float, ...]:
    """Gradient of sum_s p_theta(s) R(s) for a categorical-logit policy."""
    probs = softmax(theta)
    expected_reward = exact_string_reward(theta)
    return tuple(p * (REWARDS[s] - expected_reward) for p, s in zip(probs, STRINGS, strict=True))


def exact_response_mean_gradient(
    theta: tuple[float, ...], phi: float
) -> tuple[tuple[float, ...], float]:
    """Expected ordinary score-function gradient for response-mean reduction."""
    grad_theta = [0.0] * len(theta)
    grad_phi = 0.0
    for outcome in enumerate_outcomes(theta, phi):
        weight = outcome.probability * outcome.reward
        for index, score in enumerate(outcome.score_theta):
            grad_theta[index] += weight * score
        grad_phi += weight * outcome.score_phi
    return tuple(grad_theta), grad_phi


def exact_per_response_token_mean_gradient(
    theta: tuple[float, ...], phi: float
) -> tuple[tuple[float, ...], float]:
    """Expected estimator that divides each response's score by its path length."""
    grad_theta = [0.0] * len(theta)
    grad_phi = 0.0
    for outcome in enumerate_outcomes(theta, phi):
        weight = outcome.probability * outcome.reward / outcome.token_count
        for index, score in enumerate(outcome.score_theta):
            grad_theta[index] += weight * score
        grad_phi += weight * outcome.score_phi
    return tuple(grad_theta), grad_phi


def exact_batch_gradient(
    theta: tuple[float, ...], phi: float, batch_size: int, token_mean: bool
) -> tuple[tuple[float, ...], float]:
    """Exactly average the response-mean or global token-mean batch estimator.

    The global token mean divides the sum of response score contributions by
    the *random batch's* total token count. Every ordered batch is enumerated.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    outcomes = enumerate_outcomes(theta, phi)
    grad_theta = [0.0] * len(theta)
    grad_phi = 0.0
    for batch in itertools.product(outcomes, repeat=batch_size):
        batch_probability = math.prod(outcome.probability for outcome in batch)
        denominator = (
            sum(outcome.token_count for outcome in batch) if token_mean else batch_size
        )
        for outcome in batch:
            weight = batch_probability * outcome.reward / denominator
            for index, score in enumerate(outcome.score_theta):
                grad_theta[index] += weight * score
            grad_phi += weight * outcome.score_phi
    return tuple(grad_theta), grad_phi


def sample_outcome(
    theta: tuple[float, ...], phi: float, rng: random.Random
) -> Outcome:
    outcomes = enumerate_outcomes(theta, phi)
    return _sample_from_outcomes(outcomes, rng)


def _sample_from_outcomes(
    outcomes: tuple[Outcome, ...], rng: random.Random
) -> Outcome:
    draw = rng.random()
    cumulative = 0.0
    for outcome in outcomes:
        cumulative += outcome.probability
        if draw < cumulative:
            return outcome
    return outcomes[-1]


def score_sample(
    string: str, path: tuple[str, ...], theta: tuple[float, ...], phi: float
) -> tuple[float, tuple[float, ...], float]:
    """Recompute the exact joint log probability and scores for one sample."""
    for outcome in enumerate_outcomes(theta, phi):
        if outcome.string == string and outcome.path == path:
            return (
                outcome.behavior_log_probability,
                outcome.score_theta,
                outcome.score_phi,
            )
    raise ValueError("sampled string/path pair is unsupported")


def monte_carlo_batch_gradients(
    theta: tuple[float, ...],
    phi: float,
    batches: int,
    batch_size: int,
    seed: int,
) -> dict[str, tuple[tuple[float, ...], float]]:
    """Estimate response-mean and global token-mean gradients from joint batches."""
    if batches < 1 or batch_size < 1:
        raise ValueError("batches and batch_size must be positive")
    rng = random.Random(seed)
    outcomes = enumerate_outcomes(theta, phi)
    totals = {
        "response_mean": ([0.0] * len(theta), 0.0),
        "global_token_mean": ([0.0] * len(theta), 0.0),
    }
    for _ in range(batches):
        batch = [_sample_from_outcomes(outcomes, rng) for _ in range(batch_size)]
        token_count = sum(outcome.token_count for outcome in batch)
        for key, denominator in (
            ("response_mean", batch_size),
            ("global_token_mean", token_count),
        ):
            grad_theta, grad_phi = totals[key]
            for outcome in batch:
                weight = outcome.reward / denominator
                for index, score in enumerate(outcome.score_theta):
                    grad_theta[index] += weight * score
                grad_phi += weight * outcome.score_phi
            totals[key] = (grad_theta, grad_phi)
    return {
        key: (tuple(grad_theta / batches for grad_theta in values[0]), values[1] / batches)
        for key, values in totals.items()
    }


def exact_position_weighted_gradient(
    theta: tuple[float, ...], phi: float, gamma: float
) -> tuple[float, ...]:
    """Apply explicit position weights to an equal-share score decomposition.

    For each sampled path, the string score is split equally over its token
    positions, so the unweighted token scores sum to grad log p_theta(s).
    Weighting position t by gamma**t changes the estimator when path lengths
    differ. This is an estimator/credit-assignment intervention, not the score
    of p_theta(s) q_phi(z|s) unless a model's actual autoregressive factorization
    supplies that decomposition.
    """
    result = [0.0] * len(theta)
    for outcome in enumerate_outcomes(theta, phi):
        length = outcome.token_count
        position_factor = sum(gamma**position for position in range(length)) / length
        weight = outcome.probability * outcome.reward * position_factor
        for index, score in enumerate(outcome.score_theta):
            result[index] += weight * score
    return tuple(result)


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left, right = tuple(left), tuple(right)
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def report(
    theta: tuple[float, ...] = (0.3, -0.2, 0.1),
    phi: float = math.log(0.25 / 0.75),
    batches: int = 200_000,
    batch_size: int = 2,
    seed: int = 41,
) -> dict:
    outcomes = enumerate_outcomes(theta, phi)
    exact_response_theta, exact_response_phi = exact_response_mean_gradient(theta, phi)
    exact_batch_response = exact_batch_gradient(theta, phi, batch_size, token_mean=False)
    exact_batch_token = exact_batch_gradient(theta, phi, batch_size, token_mean=True)
    per_response_token = exact_per_response_token_mean_gradient(theta, phi)
    position_weighted = exact_position_weighted_gradient(theta, phi, gamma=0.5)
    monte_carlo = monte_carlo_batch_gradients(theta, phi, batches, batch_size, seed)

    q_sensitivity = []
    for short_probability in (0.25, 0.75):
        q_phi = math.log(short_probability / (1 - short_probability))
        q_response = exact_response_mean_gradient(theta, q_phi)
        q_token = exact_batch_gradient(theta, q_phi, batch_size, token_mean=True)
        q_sensitivity.append(
            {
                "q_short": short_probability,
                "response_mean_theta": q_response[0],
                "response_mean_phi": q_response[1],
                "global_token_mean_theta": q_token[0],
                "global_token_mean_phi": q_token[1],
            }
        )

    return {
        "scope": "exact finite synthetic policy; not an MT result",
        "strings": STRINGS,
        "rewards": REWARDS,
        "theta_logits": theta,
        "phi_logit": phi,
        "q_short": sigmoid(phi),
        "path_probabilities_sum": sum(outcome.probability for outcome in outcomes),
        "string_probabilities": {
            string: sum(o.probability for o in outcomes if o.string == string)
            for string in STRINGS
        },
        "expected_string_reward": exact_string_reward(theta),
        "exact_string_gradient_theta": exact_string_gradient(theta),
        "exact_response_mean_theta": exact_response_theta,
        "exact_response_mean_phi": exact_response_phi,
        "exact_batch_response_mean_theta": exact_batch_response[0],
        "exact_batch_token_mean_theta": exact_batch_token[0],
        "exact_batch_token_mean_phi": exact_batch_token[1],
        "per_response_token_mean_theta": per_response_token[0],
        "per_response_token_mean_phi": per_response_token[1],
        "position_weighted_gamma_1_theta": exact_position_weighted_gradient(theta, phi, 1.0),
        "position_weighted_gamma_0_5_theta": position_weighted,
        "token_mean_cosine_vs_response_mean": cosine_similarity(
            exact_batch_response[0], exact_batch_token[0]
        ),
        "position_weighted_cosine_vs_response_mean": cosine_similarity(
            exact_response_theta, position_weighted
        ),
        "q_sampler_sensitivity": q_sensitivity,
        "monte_carlo_batch_gradients": monte_carlo,
        "batches": batches,
        "batch_size": batch_size,
        "seed": seed,
        "notes": [
            "Joint probability is p_theta(string) times q_phi(path|string).",
            "Ordinary response-mean REINFORCE is invariant to q_phi for string-only reward.",
            "Global token mean uses each sampled batch's total token count, so its expected sample weighting depends on q_phi.",
            "Position weighting is an explicit alternative estimator and must not be confused with the joint behavior score.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    print(json.dumps(report(batches=args.batches, batch_size=args.batch_size, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
