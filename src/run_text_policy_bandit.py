"""Train a shared source--candidate-text policy and evaluate on held-out systems.

The policy never receives system identity.  A frozen multilingual encoder maps
each source/candidate pair to a vector, and one shared linear scorer defines a
softmax policy over the candidates available in a context.  Both documents and
systems are disjoint between the training and primary evaluation environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from analyze_wmt25_best_of_n import CONDITIONS
from audit_reward_holdouts import sha256_file
from run_contextual_bandit import (
    AUDIT_DRAWS,
    AUDIT_STEPS,
    LEARNING_RATE,
    POLICY_SEEDS,
    RECORD_EVERY,
    STEPS,
    document_split,
    machine_candidate_keys,
)
from score_wmt26_cesa_candidates import CAMPAIGN_MEMBER


SYSTEM_SPLIT_SEED = 20261127
TRAIN_SYSTEM_FRACTION = 2 / 3
ENCODER_NAME = "FacebookAI/xlm-roberta-base"
ENCODER_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
MAX_LENGTH = 256
ENCODER_BATCH_SIZE = 64
BOOTSTRAP_SEED = 20261128
BOOTSTRAP_RESAMPLES = 10_000


def system_split(actions: list[str]) -> tuple[list[int], list[int]]:
    """Return a deterministic outcome-blind split of system identities."""
    ordered = sorted(
        range(len(actions)),
        key=lambda index: hashlib.sha256(
            f"{SYSTEM_SPLIT_SEED}|{actions[index]}".encode("utf-8")
        ).digest(),
    )
    count = min(len(ordered) - 1, max(1, round(TRAIN_SYSTEM_FRACTION * len(ordered))))
    train = sorted(ordered[:count])
    heldout = sorted(ordered[count:])
    if set(train) & set(heldout) or sorted(train + heldout) != list(range(len(actions))):
        raise AssertionError("Invalid system split")
    return train, heldout


def load_environment(
    prediction_root: Path, campaign_archive: Path
) -> tuple[list[str], list[str], np.ndarray, dict[str, np.ndarray], list[str], list[list[str]]]:
    with zipfile.ZipFile(campaign_archive) as archive:
        campaign = json.load(archive.open(CAMPAIGN_MEMBER))
    items = {
        str(item["item_id"]): item
        for document in campaign["data"]
        for item in document
    }
    by_condition: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = prediction_root / f"{condition}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[1:]]
        by_condition[condition] = {(row["segment_key"], row["system_id"]): row for row in rows}
    all_keys = set(by_condition[CONDITIONS[0]])
    if any(set(by_condition[condition]) != all_keys for condition in CONDITIONS[1:]):
        raise ValueError("Reward conditions do not share the same candidate environment")
    keys = machine_candidate_keys(all_keys)
    contexts = sorted({context for context, _ in keys})
    actions = sorted({action for _, action in keys})
    if any((context, action) not in keys for context in contexts for action in actions):
        raise ValueError("Candidate environment is not rectangular")
    source_texts = [str(items[context].get("src_text", items[context]["src"])) for context in contexts]
    candidate_texts = [
        [str(items[context]["tgt"][action]) for action in actions] for context in contexts
    ]
    human = np.asarray(
        [[by_condition[CONDITIONS[0]][(context, action)]["human_cesa"] for action in actions]
         for context in contexts],
        dtype=np.float32,
    )
    rewards = {
        condition: np.asarray(
            [[by_condition[condition][(context, action)]["predicted_reward"] for action in actions]
             for context in contexts],
            dtype=np.float32,
        )
        for condition in CONDITIONS
    }
    return contexts, actions, human, rewards, source_texts, candidate_texts


def mean_pool(last_hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp_min(1)


def encode_pairs(
    source_texts: list[str],
    candidate_texts: list[list[str]],
    *,
    model_name: str,
    revision: str,
    max_length: int,
    batch_size: int,
    device: "torch.device",
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, revision=revision, local_files_only=True).to(device)
    model.eval()
    action_count = len(candidate_texts[0])
    flat_sources = [source for source in source_texts for _ in range(action_count)]
    flat_candidates = [candidate for rows in candidate_texts for candidate in rows]
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(flat_sources), batch_size):
            batch = tokenizer(
                flat_sources[start : start + batch_size],
                flat_candidates[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            pooled = mean_pool(output.last_hidden_state, batch["attention_mask"])
            chunks.append(pooled.float().cpu().numpy())
    embeddings = np.concatenate(chunks).reshape(len(source_texts), action_count, -1)
    del model
    return embeddings.astype(np.float32)


def load_or_encode(
    path: Path,
    source_texts: list[str],
    candidate_texts: list[list[str]],
    device: "torch.device",
) -> tuple[np.ndarray, str]:
    text_digest = hashlib.sha256()
    for source, candidates in zip(source_texts, candidate_texts, strict=True):
        text_digest.update(source.encode("utf-8"))
        for candidate in candidates:
            text_digest.update(b"\0")
            text_digest.update(candidate.encode("utf-8"))
    digest = text_digest.hexdigest()
    if path.exists():
        artifact = np.load(path, allow_pickle=False)
        if str(artifact["text_sha256"].item()) != digest:
            raise ValueError("Cached text embeddings do not match the candidate environment")
        return artifact["embeddings"], digest
    embeddings = encode_pairs(
        source_texts,
        candidate_texts,
        model_name=ENCODER_NAME,
        revision=ENCODER_REVISION,
        max_length=MAX_LENGTH,
        batch_size=ENCODER_BATCH_SIZE,
        device=device,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings, text_sha256=np.asarray(digest))
    return embeddings, digest


def policy_probabilities(embeddings: "torch.Tensor", weights: "torch.Tensor") -> "torch.Tensor":
    return torch.softmax(torch.einsum("cah,h->ca", embeddings, weights), dim=-1)


def metrics(
    embeddings: "torch.Tensor",
    weights: "torch.Tensor",
    reward_std: "torch.Tensor",
    reward_raw: "torch.Tensor",
    human: "torch.Tensor",
) -> dict[str, float]:
    probabilities = policy_probabilities(embeddings, weights)
    return {
        "standardized_proxy": float((probabilities * reward_std).sum(-1).mean()),
        "raw_proxy": float((probabilities * reward_raw).sum(-1).mean()),
        "human_cesa": float((probabilities * human).sum(-1).mean()),
        "entropy": float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)).mean()),
    }


def train_policy(
    embeddings: "torch.Tensor",
    rewards: "torch.Tensor",
    *,
    sampled: bool,
    seed: int | None,
) -> dict[int, "torch.Tensor"]:
    weights = torch.zeros(embeddings.shape[-1], device=embeddings.device, requires_grad=True)
    generator = torch.Generator(device=embeddings.device).manual_seed(seed or 0)
    checkpoints: dict[int, torch.Tensor] = {}
    for step in range(STEPS + 1):
        if step % RECORD_EVERY == 0 or step in AUDIT_STEPS:
            checkpoints[step] = weights.detach().clone()
        if step == STEPS:
            break
        probabilities = policy_probabilities(embeddings, weights)
        if sampled:
            actions = torch.multinomial(probabilities.detach(), 1, generator=generator).squeeze(-1)
            selected = rewards.gather(1, actions[:, None]).squeeze(1)
            log_probability = probabilities.gather(1, actions[:, None]).squeeze(1).log()
            objective = (selected * log_probability).mean()
        else:
            objective = (probabilities * rewards).sum(-1).mean()
        (gradient,) = torch.autograd.grad(objective, (weights,))
        with torch.no_grad():
            weights += LEARNING_RATE * gradient
    return checkpoints


def gradient_audit(
    embeddings: "torch.Tensor", rewards: "torch.Tensor", weights: "torch.Tensor", seed: int
) -> dict[str, float]:
    reference = weights.detach().clone().requires_grad_(True)
    probabilities = policy_probabilities(embeddings, reference)
    objective = (probabilities * rewards).sum(-1).mean()
    (exact,) = torch.autograd.grad(objective, (reference,))
    generator = torch.Generator(device=embeddings.device).manual_seed(seed)
    estimates = []
    for _ in range(AUDIT_DRAWS):
        draw_weights = weights.detach().clone().requires_grad_(True)
        draw_probabilities = policy_probabilities(embeddings, draw_weights)
        actions = torch.multinomial(
            draw_probabilities.detach(), 1, generator=generator
        ).squeeze(-1)
        selected = rewards.gather(1, actions[:, None]).squeeze(1)
        objective = (
            selected * draw_probabilities.gather(1, actions[:, None]).squeeze(1).log()
        ).mean()
        (estimate,) = torch.autograd.grad(objective, (draw_weights,))
        estimates.append(estimate)
    stacked = torch.stack(estimates)
    mean = stacked.mean(0)
    exact_norm = torch.linalg.vector_norm(exact)
    return {
        "mean_gradient_bias_norm": float(torch.linalg.vector_norm(mean - exact)),
        "relative_bias_norm": float(torch.linalg.vector_norm(mean - exact) / exact_norm.clamp_min(1e-12)),
        "estimator_variance": float(((stacked - mean) ** 2).sum(-1).mean()),
        "mean_gradient_cosine": float(torch.nn.functional.cosine_similarity(mean, exact, dim=0)),
        "exact_gradient_norm": float(exact_norm),
        "gradient_space": "shared source-candidate text scorer",
    }


def cluster_interval(values: np.ndarray, documents: np.ndarray, seed_offset: int) -> dict[str, float]:
    unique = sorted(set(documents.tolist()))
    grouped = {document: values[documents == document] for document in unique}
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    samples = np.empty(BOOTSTRAP_RESAMPLES)
    for index in range(BOOTSTRAP_RESAMPLES):
        selected = generator.choice(unique, len(unique), replace=True)
        samples[index] = np.concatenate([grouped[document] for document in selected]).mean()
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "document_clusters": len(unique),
        "segments": len(values),
    }


def evaluation_baselines(reward: np.ndarray, human: np.ndarray) -> dict[str, float]:
    rows = np.arange(len(reward))
    return {
        "uniform_human_cesa": float(human.mean()),
        "proxy_best_human_cesa": float(human[rows, reward.argmax(1)].mean()),
        "human_best_ceiling_cesa": float(human.max(1).mean()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    global torch
    import torch

    contexts, actions, human_np, rewards_np, sources, candidates = load_environment(
        args.prediction_root, args.campaign_archive
    )
    train_contexts, test_contexts, document_summary = document_split(contexts)
    train_systems, heldout_systems = system_split(actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings_np, text_sha256 = load_or_encode(args.embedding_cache, sources, candidates, device)
    embeddings = torch.tensor(embeddings_np, device=device)
    human = torch.tensor(human_np, device=device)
    train_context_index = torch.tensor(train_contexts, device=device)
    test_context_index = torch.tensor(test_contexts, device=device)
    train_system_index = torch.tensor(train_systems, device=device)
    heldout_system_index = torch.tensor(heldout_systems, device=device)
    train_embeddings = embeddings[train_context_index][:, train_system_index]
    seen_test_embeddings = embeddings[test_context_index][:, train_system_index]
    heldout_test_embeddings = embeddings[test_context_index][:, heldout_system_index]
    test_documents = np.asarray([contexts[index].rsplit("_###_", 1)[0] for index in test_contexts])

    result: dict[str, Any] = {
        "status": "complete",
        "claim_boundary": (
            "Tests generalization of reward optimization across unseen systems using a shared "
            "source-candidate text scorer; it does not evaluate generated translations."
        ),
        "protocol": {
            "encoder_name": ENCODER_NAME,
            "encoder_revision": ENCODER_REVISION,
            "encoder_frozen": True,
            "pooling": "attention-mask mean pooling",
            "policy": "single shared linear scorer; no system identity features or biases",
            "max_length": MAX_LENGTH,
            "system_split_seed": SYSTEM_SPLIT_SEED,
            "train_system_fraction": TRAIN_SYSTEM_FRACTION,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "policy_seeds": list(POLICY_SEEDS),
            "audit_steps": list(AUDIT_STEPS),
            "audit_draws": AUDIT_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "environment": {
            "contexts": len(contexts),
            "systems": len(actions),
            "train_contexts": len(train_contexts),
            "test_contexts": len(test_contexts),
            "train_systems": [actions[index] for index in train_systems],
            "heldout_systems": [actions[index] for index in heldout_systems],
            "document_split_by_domain": document_summary,
            "primary_evaluation": "held-out documents crossed with held-out systems",
        },
        "embedding_artifact": {
            "path": str(args.embedding_cache.resolve().relative_to(Path(__file__).resolve().parents[1])),
            "sha256": sha256_file(args.embedding_cache),
            "text_sha256": text_sha256,
            "shape": list(embeddings_np.shape),
        },
        "conditions": {},
    }
    probability_arrays: dict[str, np.ndarray] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        reward_raw_np = rewards_np[condition]
        scale_values = reward_raw_np[np.ix_(train_contexts, train_systems)]
        mean, std = float(scale_values.mean()), float(scale_values.std())
        if not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid reward scale for {condition}")
        reward_raw = torch.tensor(reward_raw_np, device=device)
        reward_std = (reward_raw - mean) / std
        train_rewards = reward_std[train_context_index][:, train_system_index]
        exact = train_policy(train_embeddings, train_rewards, sampled=False, seed=None)
        trajectories: dict[str, Any] = {"exact": [], "reinforce": {}}
        for step, weights in sorted(exact.items()):
            trajectories["exact"].append(
                {
                    "step": step,
                    "train_seen_systems": metrics(
                        train_embeddings,
                        weights,
                        train_rewards,
                        reward_raw[train_context_index][:, train_system_index],
                        human[train_context_index][:, train_system_index],
                    ),
                    "test_seen_systems": metrics(
                        seen_test_embeddings,
                        weights,
                        reward_std[test_context_index][:, train_system_index],
                        reward_raw[test_context_index][:, train_system_index],
                        human[test_context_index][:, train_system_index],
                    ),
                    "test_heldout_systems": metrics(
                        heldout_test_embeddings,
                        weights,
                        reward_std[test_context_index][:, heldout_system_index],
                        reward_raw[test_context_index][:, heldout_system_index],
                        human[test_context_index][:, heldout_system_index],
                    ),
                }
            )
        for seed in POLICY_SEEDS:
            checkpoints = train_policy(train_embeddings, train_rewards, sampled=True, seed=seed)
            trajectories["reinforce"][str(seed)] = [
                {
                    "step": step,
                    "test_heldout_systems": metrics(
                        heldout_test_embeddings,
                        weights,
                        reward_std[test_context_index][:, heldout_system_index],
                        reward_raw[test_context_index][:, heldout_system_index],
                        human[test_context_index][:, heldout_system_index],
                    ),
                }
                for step, weights in sorted(checkpoints.items())
            ]
        audits = {
            str(step): gradient_audit(
                train_embeddings,
                train_rewards,
                exact[step],
                SYSTEM_SPLIT_SEED + condition_index * 1000 + step,
            )
            for step in AUDIT_STEPS
        }
        final = exact[STEPS]
        heldout_probabilities = policy_probabilities(heldout_test_embeddings, final).detach().cpu().numpy()
        heldout_human = human_np[np.ix_(test_contexts, heldout_systems)]
        uniform_human = heldout_human.mean(1)
        policy_human = (heldout_probabilities * heldout_human).sum(1)
        probability_arrays[f"{condition}__exact__heldout"] = heldout_probabilities
        result["conditions"][condition] = {
            "reward_train_mean": mean,
            "reward_train_std": std,
            "baselines": evaluation_baselines(
                reward_raw_np[np.ix_(test_contexts, heldout_systems)], heldout_human
            ),
            "gradient_audit": audits,
            "trajectories": trajectories,
            "primary_exact": {
                **metrics(
                    heldout_test_embeddings,
                    final,
                    reward_std[test_context_index][:, heldout_system_index],
                    reward_raw[test_context_index][:, heldout_system_index],
                    human[test_context_index][:, heldout_system_index],
                ),
                "human_change_from_uniform": cluster_interval(
                    policy_human - uniform_human, test_documents, condition_index
                ),
            },
        }

    args.probabilities_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.probabilities_output,
        contexts=np.asarray([contexts[index] for index in test_contexts]),
        heldout_systems=np.asarray([actions[index] for index in heldout_systems]),
        **probability_arrays,
    )
    result["probability_artifact"] = {
        "path": str(args.probabilities_output.resolve().relative_to(Path(__file__).resolve().parents[1])),
        "sha256": sha256_file(args.probabilities_output),
        "arrays": len(probability_arrays),
    }
    return result


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["text_policy_bandit"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root", type=Path, default=root / "artifacts/reward_models/wmt26_predictions"
    )
    parser.add_argument(
        "--campaign-archive", type=Path, default=root / "data/reward_da/wmt26-campaign_sources.zip"
    )
    parser.add_argument(
        "--embedding-cache", type=Path, default=root / "artifacts/bandit/text_pair_embeddings.npz"
    )
    parser.add_argument(
        "--probabilities-output",
        type=Path,
        default=root / "artifacts/bandit/text_policy_heldout_probabilities.npz",
    )
    parser.add_argument("--output", type=Path, default=root / "results/paper_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    merge_ssot(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
