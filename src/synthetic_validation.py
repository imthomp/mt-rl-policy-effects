"""Finite-policy checks for segmentation and policy-gradient estimators.

This is a CPU-only correctness example, not a translation model or training loop.
The policy chooses string A with a Bernoulli probability and string B otherwise.
String A has two token paths with different lengths; the path choice is
independent of the policy parameters and both paths decode to the same string.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Path:
    string: str
    probability: float
    reward: float
    score: float
    token_count: int


def sigmoid(theta: float) -> float:
    if theta >= 0:
        z = math.exp(-theta)
        return 1.0 / (1.0 + z)
    z = math.exp(theta)
    return z / (1.0 + z)


def paths(theta: float, short_path_probability: float = 0.25) -> tuple[Path, ...]:
    """Enumerate joint string/path outcomes and their d log probability/d theta."""
    p = sigmoid(theta)
    q = short_path_probability
    # Path choice is ancillary: q does not depend on theta.
    return (
        Path("A", p * q, 1.0, 1.0 - p, 1),
        Path("A", p * (1.0 - q), 1.0, 1.0 - p, 3),
        Path("B", 1.0 - p, 0.0, -p, 1),
    )


def exact_reward(theta: float) -> float:
    return sigmoid(theta)


def exact_gradient(theta: float) -> float:
    p = sigmoid(theta)
    return p * (1.0 - p)


def exact_reinforce_gradient(theta: float) -> float:
    """Expected joint-path REINFORCE gradient with state baseline b=E[R]."""
    baseline = exact_reward(theta)
    return sum(
        path.probability * (path.reward - baseline) * path.score
        for path in paths(theta)
    )


def exact_per_response_token_mean_gradient(theta: float) -> float:
    """A deliberately different estimator, dividing each trajectory by length."""
    baseline = exact_reward(theta)
    return sum(
        path.probability
        * (path.reward - baseline)
        * path.score
        / path.token_count
        for path in paths(theta)
    )


def monte_carlo_gradient(theta: float, samples: int, seed: int) -> float:
    """Sample full joint trajectories and return the mean REINFORCE estimate."""
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    p = sigmoid(theta)
    baseline = p
    total = 0.0
    for _ in range(samples):
        if rng.random() < p:
            # Sample the segmentation path, although it has no theta score.
            rng.random() < 0.25
            reward, score = 1.0, 1.0 - p
        else:
            reward, score = 0.0, -p
        total += (reward - baseline) * score
    return total / samples


def batch_gradients(theta: float, selected_paths: tuple[Path, ...]) -> tuple[float, float]:
    """Return fixed-denominator and token-mean gradients on identical paths."""
    baseline = exact_reward(theta)
    contributions = [
        (path.reward - baseline) * path.score for path in selected_paths
    ]
    fixed_denominator = sum(contributions) / len(selected_paths)
    token_mean = sum(contributions) / sum(p.token_count for p in selected_paths)
    return fixed_denominator, token_mean


def run(samples: int = 200_000, seed: int = 7, theta: float = 0.4) -> dict[str, float]:
    p = sigmoid(theta)
    exact = exact_gradient(theta)
    joint = exact_reinforce_gradient(theta)
    token_mean = exact_per_response_token_mean_gradient(theta)
    mc = monte_carlo_gradient(theta, samples, seed)
    # This identical-path batch check isolates scalar reduction differences.
    batch = (paths(theta)[0], paths(theta)[1], paths(theta)[2])
    fixed_batch, token_batch = batch_gradients(theta, batch)
    return {
        "theta": theta,
        "p_string_A": p,
        "expected_reward": exact_reward(theta),
        "exact_gradient": exact,
        "exact_joint_path_gradient": joint,
        "monte_carlo_gradient": mc,
        "exact_per_response_token_mean_gradient": token_mean,
        "identical_batch_fixed_denominator_gradient": fixed_batch,
        "identical_batch_token_mean_gradient": token_batch,
        "samples": float(samples),
        "seed": float(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--theta", type=float, default=0.4)
    args = parser.parse_args()
    result = run(samples=args.samples, seed=args.seed, theta=args.theta)
    for name, value in result.items():
        print(f"{name}: {value:.9f}" if isinstance(value, float) else f"{name}: {value}")


if __name__ == "__main__":
    main()
