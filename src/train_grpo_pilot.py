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
RULE = ROOT / "results/grpo_audit_rule.json"
RULE_SHA256 = "0dc190f79589987af21f2a8476787e68260e25db26b21bd02e44c4837437ca4c"
ENDE_AMENDMENT = ROOT / "results/grpo_ende_protocol_amendment_1.json"
ENDE_AMENDMENT_SHA256 = "289acb8713b7b444935641c9b4b9dd8ed9d387f491c9c506ceec25fa01b240dd"
PROTOCOLS = {
    "pilot": (PROTOCOL, PROTOCOL_SHA256),
    "ende": (ROOT / "results/grpo_ende_protocol.json",
             "a0bec8727d8bcb13f41074e51cc0e28086fa68c220bcf730c2d1e22052845779"),
}
REWARD_CHECKPOINTS = {
    "xcomet_xl_qe": "XCOMET-XL",
    "cometkiwi_22": "wmt22-cometkiwi-da",
    "cometkiwi_23_xl": "wmt23-cometkiwi-da-xl",
    "metricx_24_qe": None,  # src/metricx_qe.py; allowed only under the frozen audit rule
}


def load_protocol(name: str = "pilot") -> dict:
    path, digest = PROTOCOLS[name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"GRPO protocol {name!r} differs from frozen hash")
    return json.loads(path.read_text())


def make_scorer(reward: str):
    """Return fn(sources, texts) -> list[float], higher is better."""
    if reward == "metricx_24_qe":
        from metricx_qe import MetricXQE

        model = MetricXQE()
        return lambda sources, texts: model.reward(sources, texts)
    comet = load_comet(REWARD_CHECKPOINTS[reward])
    return lambda sources, texts: [float(x) for x in comet.predict(
        [{"src": s, "mt": t} for s, t in zip(sources, texts)],
        batch_size=32, gpus=1, progress_bar=False).scores]


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
    parser.add_argument("--protocol-name", default="pilot", choices=sorted(PROTOCOLS))
    args = parser.parse_args()
    protocol = load_protocol(args.protocol_name)
    g = dict(protocol["grpo"])
    target_lang = protocol.get("target_lang", "Spanish")
    rule_run = args.reward == "metricx_24_qe" or args.protocol_name != "pilot"
    if rule_run:
        # Runs that test the frozen audit rule: MetricX en-es seeds 0-2; en-de seeds 0-2 (amendment 1), any reward.
        if hashlib.sha256(RULE.read_bytes()).hexdigest() != RULE_SHA256:
            raise ValueError("Audit rule differs from frozen hash")
        seed = g["seed"] if args.seed is None else args.seed
        seeds = list(protocol.get("seeds", [0]))
        if args.protocol_name == "ende":
            if hashlib.sha256(ENDE_AMENDMENT.read_bytes()).hexdigest() != ENDE_AMENDMENT_SHA256:
                raise ValueError("en-de amendment differs from frozen hash")
            seeds += json.loads(ENDE_AMENDMENT.read_text())["seeds_added"]
        allowed = seed in seeds if args.protocol_name != "pilot" else seed in (0, 1, 2)
        if args.guarded or not allowed:
            raise ValueError(f"Run not covered by the frozen audit rule: {args.reward}, seed {seed}")
        g["seed"] = seed
    elif args.seed is not None or args.guarded:
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
    scorer = make_scorer(args.reward)

    def reward_fn(prompts, completions, source, **_):
        texts = [c[0]["content"].strip() if isinstance(c, list) else str(c).strip() for c in completions]
        scores = scorer(list(source), texts)
        if args.guarded:
            from hack_guards import flags
            scores = [x - 1.0 if any(flags(t, s, target_lang).values()) else x
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
        "reward": args.reward, "protocol_name": args.protocol_name,
        "protocol_sha256": PROTOCOLS[args.protocol_name][1],
        "audit_rule_sha256": RULE_SHA256 if rule_run else None,
        "seed": g["seed"], "guarded": args.guarded,
        "replication_sha256": REPLICATION_SHA256 if (args.seed is not None or args.guarded) else None,
        "trl_config": config.to_dict(), "log_history": trainer.state.log_history,
    }, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
