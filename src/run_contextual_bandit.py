"""Run the frozen finite-action WMT26 contextual-bandit experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_wmt25_best_of_n import CONDITIONS
from score_wmt26_cesa_candidates import CAMPAIGN_MEMBER
from audit_reward_holdouts import sha256_file


HASH_FEATURES = 512
SPLIT_SEED = 20261109
TRAIN_FRACTION = 0.70
STEPS = 200
LEARNING_RATE = 0.5
RECORD_EVERY = 10
POLICY_SEEDS = tuple(range(20261110, 20261115))
AUDIT_STEPS = (0, 50, 100, 200)
AUDIT_DRAWS = 256


def hash_features(text: str, dimensions: int = HASH_FEATURES) -> np.ndarray:
    features = np.zeros(dimensions, dtype=np.float32)
    normalized = " ".join(text.split())
    for width in (3, 4, 5):
        for start in range(max(0, len(normalized) - width + 1)):
            gram = normalized[start : start + width].encode("utf-8")
            digest = hashlib.sha256(gram).digest()[:8]
            value = int.from_bytes(digest)
            index = value % dimensions
            features[index] += -1.0 if value & (1 << 63) else 1.0
    norm = float(np.linalg.norm(features))
    if norm:
        features /= norm
    return features


def load_sources(campaign_archive: Path) -> dict[str, str]:
    with zipfile.ZipFile(campaign_archive) as archive:
        campaign = json.load(archive.open(CAMPAIGN_MEMBER))
    return {
        str(item["item_id"]): str(item.get("src_text", item["src"]))
        for document in campaign["data"]
        for item in document
    }


def read_environment(
    prediction_root: Path, campaign_archive: Path
) -> tuple[list[str], list[str], np.ndarray, dict[str, np.ndarray], list[str]]:
    sources = load_sources(campaign_archive)
    by_condition: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = prediction_root / f"{condition}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[1:]]
        by_condition[condition] = {(row["segment_key"], row["system_id"]): row for row in rows}
    keys = set(by_condition[CONDITIONS[0]])
    if any(set(by_condition[condition]) != keys for condition in CONDITIONS[1:]):
        raise ValueError("Reward conditions do not share the same candidate environment")
    contexts = sorted({segment for segment, _ in keys})
    actions = sorted({system for _, system in keys})
    if any((context, action) not in keys for context in contexts for action in actions):
        raise ValueError("Candidate environment is not rectangular")
    human = np.asarray(
        [[by_condition[CONDITIONS[0]][(context, action)]["human_cesa"] for action in actions] for context in contexts],
        dtype=np.float32,
    )
    rewards = {
        condition: np.asarray(
            [[by_condition[condition][(context, action)]["predicted_reward"] for action in actions] for context in contexts],
            dtype=np.float32,
        )
        for condition in CONDITIONS
    }
    source_texts = [sources[context] for context in contexts]
    return contexts, actions, human, rewards, source_texts


def document_split(contexts: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    by_domain: dict[str, set[str]] = defaultdict(set)
    document_for_context: list[str] = []
    for context in contexts:
        document = context.rsplit("_###_", 1)[0]
        domain = document.split("_###_", 1)[0]
        document_for_context.append(document)
        by_domain[domain].add(document)
    train_documents: set[str] = set()
    split_summary: dict[str, Any] = {}
    for domain, documents in sorted(by_domain.items()):
        ordered = sorted(
            documents,
            key=lambda doc: hashlib.sha256(f"{SPLIT_SEED}|{doc}".encode()).digest(),
        )
        count = min(len(ordered) - 1, max(1, round(TRAIN_FRACTION * len(ordered))))
        train_documents.update(ordered[:count])
        split_summary[domain] = {"train_documents": count, "test_documents": len(ordered) - count}
    train = np.asarray([index for index, doc in enumerate(document_for_context) if doc in train_documents])
    test = np.asarray([index for index, doc in enumerate(document_for_context) if doc not in train_documents])
    if set(train) & set(test) or len(train) + len(test) != len(contexts):
        raise AssertionError("Invalid document split")
    return train, test, split_summary


def contextual_features(contexts: list[str], source_texts: list[str]) -> np.ndarray:
    domains = sorted({context.split("_###_", 1)[0] for context in contexts})
    domain_index = {domain: index for index, domain in enumerate(domains)}
    result = np.zeros((len(contexts), HASH_FEATURES + len(domains)), dtype=np.float32)
    for index, (context, text) in enumerate(zip(contexts, source_texts, strict=True)):
        result[index, :HASH_FEATURES] = hash_features(text)
        result[index, HASH_FEATURES + domain_index[context.split("_###_", 1)[0]]] = 1.0
    return result


def policy_metrics(
    features: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    reward_std: torch.Tensor,
    reward_raw: torch.Tensor,
    human: torch.Tensor,
    indices: torch.Tensor,
) -> dict[str, float]:
    probabilities = torch.softmax(features[indices] @ weights + bias, dim=-1)
    return {
        "standardized_proxy": float((probabilities * reward_std[indices]).sum(-1).mean()),
        "raw_proxy": float((probabilities * reward_raw[indices]).sum(-1).mean()),
        "human_cesa": float((probabilities * human[indices]).sum(-1).mean()),
        "entropy": float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)).mean()),
    }


def train_shared_policy(
    features: torch.Tensor,
    train_indices: torch.Tensor,
    reward_std: torch.Tensor,
    *,
    sampled: bool,
    seed: int | None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    weights = torch.zeros(
        (features.shape[1], reward_std.shape[1]), device=features.device, requires_grad=True
    )
    bias = torch.zeros(reward_std.shape[1], device=features.device, requires_grad=True)
    generator = torch.Generator(device=features.device).manual_seed(seed or 0)
    checkpoints: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    x = features[train_indices]
    rewards = reward_std[train_indices]
    for step in range(STEPS + 1):
        if step % RECORD_EVERY == 0 or step in AUDIT_STEPS:
            checkpoints[step] = (weights.detach().clone(), bias.detach().clone())
        if step == STEPS:
            break
        probabilities = torch.softmax(x @ weights + bias, dim=-1)
        if sampled:
            actions = torch.multinomial(probabilities.detach(), 1, generator=generator).squeeze(-1)
            selected = rewards.gather(1, actions[:, None]).squeeze(1)
            objective = (selected * probabilities.gather(1, actions[:, None]).squeeze(1).log()).mean()
        else:
            objective = (probabilities * rewards).sum(-1).mean()
        gradients = torch.autograd.grad(objective, (weights, bias))
        with torch.no_grad():
            weights += LEARNING_RATE * gradients[0]
            bias += LEARNING_RATE * gradients[1]
    return checkpoints


def train_tabular_policy(
    rewards: torch.Tensor, *, sampled: bool, seed: int | None
) -> dict[int, torch.Tensor]:
    logits = torch.zeros_like(rewards, requires_grad=True)
    generator = torch.Generator(device=rewards.device).manual_seed(seed or 0)
    checkpoints: dict[int, torch.Tensor] = {}
    for step in range(STEPS + 1):
        if step % RECORD_EVERY == 0:
            checkpoints[step] = logits.detach().clone()
        if step == STEPS:
            break
        probabilities = torch.softmax(logits, dim=-1)
        if sampled:
            actions = torch.multinomial(probabilities.detach(), 1, generator=generator).squeeze(-1)
            selected = rewards.gather(1, actions[:, None]).squeeze(1)
            objective = (selected * probabilities.gather(1, actions[:, None]).squeeze(1).log()).sum()
        else:
            objective = (probabilities * rewards).sum()
        (gradient,) = torch.autograd.grad(objective, (logits,))
        with torch.no_grad():
            logits += LEARNING_RATE * gradient
    return checkpoints


def tabular_metrics(
    logits: torch.Tensor,
    reward_std: torch.Tensor,
    reward_raw: torch.Tensor,
    human: torch.Tensor,
) -> dict[str, float]:
    probabilities = torch.softmax(logits, dim=-1)
    return {
        "standardized_proxy": float((probabilities * reward_std).sum(-1).mean()),
        "raw_proxy": float((probabilities * reward_raw).sum(-1).mean()),
        "human_cesa": float((probabilities * human).sum(-1).mean()),
        "entropy": float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)).mean()),
    }


def gradient_audit(
    rewards: torch.Tensor,
    probabilities: torch.Tensor,
    features: torch.Tensor,
    seed: int,
) -> dict[str, float]:
    expected = (probabilities * rewards).sum(-1, keepdim=True)
    exact_logits = probabilities * (rewards - expected)
    generator = torch.Generator(device=probabilities.device).manual_seed(seed)
    actions = torch.multinomial(probabilities, AUDIT_DRAWS, replacement=True, generator=generator).T
    selected = rewards.gather(1, actions.T).T
    one_hot = torch.nn.functional.one_hot(actions, probabilities.shape[1]).float()
    logit_estimates = selected[..., None] * (one_hot - probabilities[None, ...])
    context_count = features.shape[0]
    exact_weight = features.T @ exact_logits / context_count
    exact_bias = exact_logits.mean(0)
    exact = torch.cat((exact_weight.flatten(), exact_bias))
    weight_estimates = torch.einsum("cf,rca->rfa", features, logit_estimates) / context_count
    bias_estimates = logit_estimates.mean(1)
    estimates = torch.cat((weight_estimates.flatten(1), bias_estimates), dim=1)
    mean_estimate = estimates.mean(0)
    bias_norm = torch.linalg.vector_norm(mean_estimate - exact)
    exact_norm = torch.linalg.vector_norm(exact)
    cosine = torch.nn.functional.cosine_similarity(mean_estimate, exact, dim=0)
    variance = ((estimates - mean_estimate) ** 2).sum(-1).mean()
    return {
        "mean_gradient_bias_norm": float(bias_norm),
        "relative_bias_norm": float(bias_norm / exact_norm.clamp_min(1e-12)),
        "estimator_variance": float(variance),
        "mean_gradient_cosine": float(cosine),
        "exact_gradient_norm": float(exact_norm),
        "gradient_space": "shared contextual policy parameters",
    }


def evaluation_baselines(reward: np.ndarray, human: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    r, h = reward[indices], human[indices]
    proxy_actions = r.argmax(axis=1)
    human_actions = h.argmax(axis=1)
    rows = np.arange(len(indices))
    return {
        "uniform": {"raw_proxy": float(r.mean()), "human_cesa": float(h.mean())},
        "proxy_best": {
            "raw_proxy": float(r[rows, proxy_actions].mean()),
            "human_cesa": float(h[rows, proxy_actions].mean()),
        },
        "human_best_ceiling": {
            "raw_proxy": float(r[rows, human_actions].mean()),
            "human_cesa": float(h[rows, human_actions].mean()),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    global torch
    import torch

    contexts, actions, human_np, rewards_np, source_texts = read_environment(
        args.prediction_root, args.campaign_archive
    )
    train_np, test_np, split_summary = document_split(contexts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.tensor(contextual_features(contexts, source_texts), device=device)
    human = torch.tensor(human_np, device=device)
    train_indices = torch.tensor(train_np, device=device)
    test_indices = torch.tensor(test_np, device=device)
    result: dict[str, Any] = {
        "status": "complete",
        "protocol": {
            "hash_features": HASH_FEATURES,
            "split_seed": SPLIT_SEED,
            "train_fraction": TRAIN_FRACTION,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "record_every": RECORD_EVERY,
            "policy_seeds": list(POLICY_SEEDS),
            "audit_steps": list(AUDIT_STEPS),
            "audit_draws": AUDIT_DRAWS,
        },
        "environment": {
            "contexts": len(contexts), "actions": len(actions),
            "train_contexts": len(train_np), "test_contexts": len(test_np),
            "split_by_domain": split_summary,
        },
        "conditions": {},
    }
    probability_arrays: dict[str, np.ndarray] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        reward_raw_np = rewards_np[condition]
        mean = float(reward_raw_np[train_np].mean())
        std = float(reward_raw_np[train_np].std())
        if not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid training reward scale: {condition}")
        reward_raw = torch.tensor(reward_raw_np, device=device)
        reward_std = (reward_raw - mean) / std
        exact = train_shared_policy(features, train_indices, reward_std, sampled=False, seed=None)
        trajectories: dict[str, Any] = {"exact": [], "reinforce": {}}
        for step, (weights, bias) in sorted(exact.items()):
            trajectories["exact"].append({
                "step": step,
                "train": policy_metrics(features, weights, bias, reward_std, reward_raw, human, train_indices),
                "test": policy_metrics(features, weights, bias, reward_std, reward_raw, human, test_indices),
            })
        for seed in POLICY_SEEDS:
            checkpoints = train_shared_policy(features, train_indices, reward_std, sampled=True, seed=seed)
            trajectories["reinforce"][str(seed)] = [
                {
                    "step": step,
                    "train": policy_metrics(features, weights, bias, reward_std, reward_raw, human, train_indices),
                    "test": policy_metrics(features, weights, bias, reward_std, reward_raw, human, test_indices),
                }
                for step, (weights, bias) in sorted(checkpoints.items())
            ]
            final_weights, final_bias = checkpoints[STEPS]
            probability_arrays[f"{condition}__reinforce__{seed}"] = torch.softmax(
                features @ final_weights + final_bias, dim=-1
            ).cpu().numpy()
        final_weights, final_bias = exact[STEPS]
        probability_arrays[f"{condition}__exact"] = torch.softmax(
            features @ final_weights + final_bias, dim=-1
        ).cpu().numpy()
        audits = {}
        for step in AUDIT_STEPS:
            weights, bias = exact[step]
            probabilities = torch.softmax(features[train_indices] @ weights + bias, dim=-1)
            audits[str(step)] = gradient_audit(
                reward_std[train_indices],
                probabilities,
                features[train_indices],
                SPLIT_SEED + condition_index * 1000 + step,
            )
        tabular_exact = train_tabular_policy(
            reward_std[train_indices], sampled=False, seed=None
        )
        tabular: dict[str, Any] = {
            "scope": "known training contexts only; no source generalization",
            "exact": [
                {
                    "step": step,
                    "train": tabular_metrics(
                        logits,
                        reward_std[train_indices],
                        reward_raw[train_indices],
                        human[train_indices],
                    ),
                }
                for step, logits in sorted(tabular_exact.items())
            ],
            "reinforce": {},
        }
        for seed in POLICY_SEEDS:
            checkpoints = train_tabular_policy(
                reward_std[train_indices], sampled=True, seed=seed
            )
            tabular["reinforce"][str(seed)] = [
                {
                    "step": step,
                    "train": tabular_metrics(
                        logits,
                        reward_std[train_indices],
                        reward_raw[train_indices],
                        human[train_indices],
                    ),
                }
                for step, logits in sorted(checkpoints.items())
            ]
        result["conditions"][condition] = {
            "reward_train_mean": mean,
            "reward_train_std": std,
            "baselines": {
                "train": evaluation_baselines(reward_raw_np, human_np, train_np),
                "test": evaluation_baselines(reward_raw_np, human_np, test_np),
            },
            "gradient_audit": audits,
            "trajectories": trajectories,
            "tabular_trajectories": tabular,
        }
    args.probabilities_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.probabilities_output,
        contexts=np.asarray(contexts),
        actions=np.asarray(actions),
        train_indices=train_np,
        test_indices=test_np,
        **probability_arrays,
    )
    result["probability_artifact"] = {
        "path": str(
            args.probabilities_output.resolve().relative_to(Path(__file__).resolve().parents[1])
        ),
        "sha256": sha256_file(args.probabilities_output),
        "arrays": len(probability_arrays),
    }
    return result


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing["contextual_bandit"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=root / "artifacts/reward_models/wmt26_predictions")
    parser.add_argument("--campaign-archive", type=Path, default=root / "data/reward_da/wmt26-campaign_sources.zip")
    parser.add_argument("--output", type=Path, default=root / "results/paper_stats.json")
    parser.add_argument(
        "--probabilities-output",
        type=Path,
        default=root / "artifacts" / "bandit" / "final_action_probabilities.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    merge_ssot(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
