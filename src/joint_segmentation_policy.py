"""Joint string/segmentation policy for synthetic sampling/likelihood checks.

The policy emits a string first and then samples one of its token paths as a
stochastic terminal action. The segmentation choice changes the token path and
its behavior probability without changing the induced probability of the
decoded string. This isolates policy/estimator mechanics; it is not an MT model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def sigmoid(value: float) -> float:
    if value >= 0:
        e = math.exp(-value)
        return 1.0 / (1.0 + e)
    e = math.exp(value)
    return e / (1.0 + e)


@dataclass(frozen=True)
class Trajectory:
    string: str
    tokens: tuple[str, ...]
    reward: float
    behavior_log_probability: float
    score_theta: float
    score_phi: float
    action_mask: tuple[int, ...]


def enumerate_trajectories(theta: float, phi: float) -> tuple[Trajectory, ...]:
    """Enumerate joint outcomes and exact scores for string and path parameters."""
    p = sigmoid(theta)
    q = sigmoid(phi)
    return (
        Trajectory("A", ("A",), 1.0, math.log(p) + math.log(q), 1.0 - p, 1.0 - q, (1, 0)),
        Trajectory("A", ("A1", "A2", "A3"), 1.0, math.log(p) + math.log1p(-q), 1.0 - p, -q, (1, 1, 1, 0)),
        Trajectory("B", ("B",), 0.0, math.log1p(-p), -p, 0.0, (1, 0)),
    )


def sample_trajectory(theta: float, phi: float, seed: int) -> Trajectory:
    """Draw a trajectory from the joint policy and attach its exact behavior score."""
    rng = random.Random(seed)
    p, q = sigmoid(theta), sigmoid(phi)
    if rng.random() < p:
        if rng.random() < q:
            return Trajectory("A", ("A",), 1.0, math.log(p) + math.log(q), 1.0 - p, 1.0 - q, (1, 0))
        return Trajectory("A", ("A1", "A2", "A3"), 1.0, math.log(p) + math.log1p(-q), 1.0 - p, -q, (1, 1, 1, 0))
    return Trajectory("B", ("B",), 0.0, math.log1p(-p), -p, 0.0, (1, 0))


def score_trajectory(trajectory: Trajectory, theta: float, phi: float) -> tuple[float, float, float]:
    """Recompute joint log probability and its scores for a sampled path."""
    p, q = sigmoid(theta), sigmoid(phi)
    if trajectory.string == "A" and trajectory.tokens == ("A",):
        return math.log(p) + math.log(q), 1.0 - p, 1.0 - q
    if trajectory.string == "A" and trajectory.tokens == ("A1", "A2", "A3"):
        return math.log(p) + math.log1p(-q), 1.0 - p, -q
    if trajectory.string == "B" and trajectory.tokens == ("B",):
        return math.log1p(-p), -p, 0.0
    raise ValueError("trajectory is not supported by this behavior policy")


def exact_reward(theta: float) -> float:
    return sigmoid(theta)


def exact_gradient(theta: float) -> float:
    p = sigmoid(theta)
    return p * (1.0 - p)


def exact_reinforce_gradient(theta: float, phi: float) -> tuple[float, float]:
    baseline = exact_reward(theta)
    outcomes = enumerate_trajectories(theta, phi)
    grad_theta = sum(math.exp(t.behavior_log_probability) * (t.reward - baseline) * t.score_theta for t in outcomes)
    grad_phi = sum(math.exp(t.behavior_log_probability) * (t.reward - baseline) * t.score_phi for t in outcomes)
    return grad_theta, grad_phi


def monte_carlo_gradient(theta: float, phi: float, samples: int, seed: int = 7) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    p = sigmoid(theta)
    g_theta = g_phi = 0.0
    baseline = p  # Detached state baseline; its derivative is not included.
    for _ in range(samples):
        if rng.random() < p:
            q = sigmoid(phi)
            if rng.random() < q:
                reward, score_theta, score_phi = 1.0, 1.0 - p, 1.0 - q
            else:
                reward, score_theta, score_phi = 1.0, 1.0 - p, -q
        else:
            reward, score_theta, score_phi = 0.0, -p, 0.0
        advantage = reward - baseline
        g_theta += advantage * score_theta
        g_phi += advantage * score_phi
    return g_theta / samples, g_phi / samples


def exact_token_mean_gradient(theta: float, phi: float) -> tuple[float, float]:
    """Deliberately length-weight each trajectory to expose normalization effects."""
    baseline = exact_reward(theta)
    total_theta = total_phi = 0.0
    for trajectory in enumerate_trajectories(theta, phi):
        probability = math.exp(trajectory.behavior_log_probability)
        valid_actions = sum(trajectory.action_mask)
        weight = probability * (trajectory.reward - baseline) / valid_actions
        total_theta += weight * trajectory.score_theta
        total_phi += weight * trajectory.score_phi
    return total_theta, total_phi


def validate_behavior_scoring(theta: float, phi: float) -> None:
    """Assert sampled log probabilities/scores equal independently recomputed values."""
    for seed in range(128):
        trajectory = sample_trajectory(theta, phi, seed)
        lp, st, sf = score_trajectory(trajectory, theta, phi)
        if not math.isclose(lp, trajectory.behavior_log_probability, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError("sampled and scored behavior log probabilities differ")
        if not math.isclose(st, trajectory.score_theta, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError("sampled and scored string-policy scores differ")
        if not math.isclose(sf, trajectory.score_phi, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError("sampled and scored segmentation-policy scores differ")


def report(theta: float = 0.4, phi: float = -0.2, samples: int = 200_000) -> dict[str, float]:
    validate_behavior_scoring(theta, phi)
    p = sigmoid(theta)
    string_mass_a = sum(math.exp(t.behavior_log_probability) for t in enumerate_trajectories(theta, phi) if t.string == "A")
    exact_t, exact_f = exact_reinforce_gradient(theta, phi)
    mc_t, mc_f = monte_carlo_gradient(theta, phi, samples)
    tm_t, tm_f = exact_token_mean_gradient(theta, phi)
    return {
        "theta": theta,
        "phi": phi,
        "string_A_probability": p,
        "joint_path_probability_mass_for_A": string_mass_a,
        "exact_string_gradient": exact_gradient(theta),
        "exact_joint_gradient_theta": exact_t,
        "exact_joint_gradient_phi": exact_f,
        "monte_carlo_gradient_theta": mc_t,
        "monte_carlo_gradient_phi": mc_f,
        "token_mean_gradient_theta": tm_t,
        "token_mean_gradient_phi": tm_f,
        "samples": float(samples),
    }


if __name__ == "__main__":
    for key, value in report().items():
        print(f"{key}: {value:.9f}")
