"""GRPO pilot: Qwen2.5-1.5B-Instruct en->es against one COMET-family QE reward.

All settings come from results/grpo_pilot_protocol.json (frozen before this
script was written).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/grpo_pilot_protocol.json"
PROTOCOL_SHA256 = "61d89eaab72a7a9d10fd8ec2cedb28b752293faae354d4a6d5d4f1c69323343c"
REPLICATION = ROOT / "results/grpo_replication_protocol.json"
REPLICATION_SHA256 = "64b2d092765be5994b0f2ae7718720cb04a7261fe964fb0932bc66af0ebc2277"
REWARD_CHECKPOINTS = {
    "xcomet_xl_qe": "XCOMET-XL",
    "cometkiwi_22": "wmt22-cometkiwi-da",
    "cometkiwi_23_xl": "wmt23-cometkiwi-da-xl",
}


def load_protocol() -> dict:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("GRPO pilot protocol differs from frozen hash")
    return json.loads(PROTOCOL.read_text())


def load_comet(name: str):
    from comet import load_from_checkpoint

    matches = glob.glob(str(Path.home() / ".cache/huggingface/hub" / f"models--Unbabel--{name}"
                            / "snapshots/*/checkpoints/model.ckpt"))
    if len(matches) != 1:
        raise ValueError(f"Expected one cached checkpoint for {name}, found {len(matches)}")
    return load_from_checkpoint(matches[0])


def train_prompts(protocol: dict) -> list[str]:
    spec = protocol["train_prompts"]
    path = ROOT / spec["source"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["source_sha256"]:
        raise ValueError("Training source differs from frozen hash")
    texts = []
    for line in path.open(encoding="utf-8"):
        text = json.loads(line)["source"].strip()
        if 5 <= len(text.split()) <= 60:
            texts.append(text)
    if hashlib.sha256("\n".join(texts).encode()).hexdigest() != spec["content_sha256"]:
        raise ValueError("Filtered training prompts differ from frozen content hash")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", required=True, choices=sorted(REWARD_CHECKPOINTS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="2-step plumbing test; not a result")
    parser.add_argument("--seed", type=int, default=None, help="replication protocol only")
    parser.add_argument("--guarded", action="store_true", help="replication protocol only")
    args = parser.parse_args()
    protocol = load_protocol()
    g = dict(protocol["grpo"])
    if args.seed is not None or args.guarded:
        if hashlib.sha256(REPLICATION.read_bytes()).hexdigest() != REPLICATION_SHA256:
            raise ValueError("Replication protocol differs from frozen hash")
        rep = json.loads(REPLICATION.read_text())
        allowed = rep["runs"]["replication"] + rep["runs"]["mitigation"]
        run = {"reward": args.reward, "seed": g["seed"] if args.seed is None else args.seed,
               "guarded": args.guarded}
        if run not in allowed:
            raise ValueError(f"Run not in frozen replication protocol: {run}")
        g["seed"] = run["seed"]

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    sources = train_prompts(protocol)
    dataset = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": protocol["prompt"].format(source=s)}] for s in sources],
        "source": sources,
    })
    comet = load_comet(REWARD_CHECKPOINTS[args.reward])

    def reward_fn(prompts, completions, source, **_):
        texts = [c[0]["content"].strip() if isinstance(c, list) else str(c).strip() for c in completions]
        out = comet.predict([{"src": s, "mt": t} for s, t in zip(source, texts)],
                            batch_size=32, gpus=1, progress_bar=False)
        scores = [float(x) for x in out.scores]
        if args.guarded:
            from hack_guards import flags
            scores = [x - 1.0 if any(flags(t, s, "Spanish").values()) else x
                      for x, s, t in zip(scores, source, texts)]
        return scores

    reward_fn.__name__ = args.reward
    accumulation = 8
    config = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=2 if args.smoke else g["steps"],
        per_device_train_batch_size=g["prompts_per_step"] * g["generations_per_prompt"] // accumulation,
        gradient_accumulation_steps=accumulation,
        num_generations=g["generations_per_prompt"],
        learning_rate=g["learning_rate"],
        beta=g["kl_beta"],
        temperature=g["temperature"],
        max_completion_length=g["max_new_tokens"],
        seed=g["seed"],
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=1 if args.smoke else 100,
        save_only_model=True,
        report_to="none",
        log_completions=False,
    )
    # 16 prompts x 8 generations = 128 completions per optimizer step.
    if config.per_device_train_batch_size * accumulation != g["prompts_per_step"] * g["generations_per_prompt"]:
        raise ValueError("Batch arithmetic does not match the frozen protocol")
    trainer = GRPOTrainer(model=protocol["policy_model"], reward_funcs=reward_fn,
                          args=config, train_dataset=dataset)
    trainer.train()
    (args.output_dir / "pilot_run.json").write_text(json.dumps({
        "reward": args.reward, "protocol_sha256": PROTOCOL_SHA256,
        "seed": g["seed"], "guarded": args.guarded,
        "replication_sha256": REPLICATION_SHA256 if (args.seed is not None or args.guarded) else None,
        "trl_config": config.to_dict(), "log_history": trainer.state.log_history,
    }, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
