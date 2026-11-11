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
ENDE_AMENDMENT_2 = ROOT / "results/grpo_ende_protocol_amendment_2.json"
ENDE_AMENDMENT_2_SHA256 = "0062a8772e0fc299fa222988d21aaa0d51311919fbcdbb3d673cf20cd9c3304a"
PROTOCOLS = {
    "pilot": (PROTOCOL, PROTOCOL_SHA256),
    "ende": (ROOT / "results/grpo_ende_protocol.json",
             "a0bec8727d8bcb13f41074e51cc0e28086fa68c220bcf730c2d1e22052845779"),
    "enzh": (ROOT / "results/grpo_enzh_protocol.json",
             "d4e59612a39b56c841df729cb0bfcd433039ee39a83b948686729cde7634bb93"),
}
SCALE_PROTOCOL = ROOT / "results/grpo_scale_protocol.json"
SCALE_PREDICTIONS = ROOT / "results/grpo_scale_predictions.json"
SCALE_PREDICTIONS_SHA256 = "5d7dad584ec35e1d6325543d2b1224585770d58e8f0addfa4d338690da2efd1a"
ENZH_PREDICTIONS = ROOT / "results/grpo_enzh_predictions.json"
ENZH_PREDICTIONS_SHA256 = "59f6afd5683ac0ad1db90e45a7b6df596c57ceb8531769f20eedbc3766d93153"
REWARD_CHECKPOINTS = {
    "xcomet_xl_qe": "XCOMET-XL",
    "cometkiwi_22": "wmt22-cometkiwi-da",
    "cometkiwi_23_xl": "wmt23-cometkiwi-da-xl",
    "metricx_24_qe": None,  # src/metricx_qe.py; allowed only under the frozen audit rule
}
# Larger rewards (2026-11-11), cached under /nobackup. Kept out of REWARD_CHECKPOINTS so the
# standard eval (which scores every checkpoint with every base reward) is unchanged.
EXTRA_CACHE = "/nobackup/archive/usr/it238/hf_cache"
NEW_REWARDS = {
    "xcomet_xxl_qe": "XCOMET-XXL",
    "cometkiwi_23_xxl": "wmt23-cometkiwi-da-xxl",
    "metricx_24_xl_qe": "google/metricx-24-hybrid-xl-v2p6-bfloat16",
    "metricx_24_xxl_qe": "google/metricx-24-hybrid-xxl-v2p6-bfloat16",
}
ALL_REWARDS = {**REWARD_CHECKPOINTS, **NEW_REWARDS}


def load_protocol(name: str = "pilot") -> dict:
    path, digest = PROTOCOLS[name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"GRPO protocol {name!r} differs from frozen hash")
    return json.loads(path.read_text())


def make_scorer(reward: str):
    """Return fn(sources, texts) -> list[float], higher is better."""
    if reward.startswith("metricx_24"):
        from metricx_qe import MetricXQE

        model = (MetricXQE() if reward == "metricx_24_qe"
                 else MetricXQE(model=NEW_REWARDS[reward], cache_dir=EXTRA_CACHE))
        return lambda sources, texts: model.reward(sources, texts)
    comet = load_comet(ALL_REWARDS[reward])
    return lambda sources, texts: [float(x) for x in comet.predict(
        [{"src": s, "mt": t} for s, t in zip(sources, texts)],
        batch_size=32, gpus=1, progress_bar=False).scores]


def load_comet(name: str):
    from comet import load_from_checkpoint

    matches = [m for root in (Path.home() / ".cache/huggingface/hub", Path(EXTRA_CACHE))
               for m in glob.glob(str(root / f"models--Unbabel--{name}" / "snapshots/*/checkpoints/model.ckpt"))]
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
    parser.add_argument("--reward", required=True, choices=sorted(ALL_REWARDS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="2-step plumbing test; not a result")
    parser.add_argument("--seed", type=int, default=None, help="replication protocol only")
    parser.add_argument("--guarded", action="store_true", help="replication protocol only")
    parser.add_argument("--protocol-name", default="pilot", choices=sorted(PROTOCOLS))
    parser.add_argument("--steps", type=int, default=None, help="en-de amendment 2 follow-up only")
    parser.add_argument("--langid-guard", action="store_true", help="en-de amendment 2 follow-up only")
    parser.add_argument("--scale-run", default=None, help="run id from results/grpo_scale_protocol.json")
    parser.add_argument("--policy-model", default=None, help="--smoke only (resource tests)")
    parser.add_argument("--lora-r", type=int, default=0, help="--smoke only (resource tests)")
    parser.add_argument("--learning-rate", type=float, default=None, help="--smoke only")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol_name)
    g = dict(protocol["grpo"])
    target_lang = protocol.get("target_lang", "Spanish")
    scale = None
    if args.scale_run:
        # Frozen scale / new-reward runs: everything comes from the protocol; predictions must be frozen.
        if not SCALE_PREDICTIONS.exists() or \
                hashlib.sha256(SCALE_PREDICTIONS.read_bytes()).hexdigest() != SCALE_PREDICTIONS_SHA256:
            raise ValueError("scale predictions missing or differ from frozen hash")
        sp = json.loads(SCALE_PROTOCOL.read_text())
        if json.loads(SCALE_PREDICTIONS.read_text())["protocol_sha256"] != hashlib.sha256(SCALE_PROTOCOL.read_bytes()).hexdigest():
            raise ValueError("scale protocol differs from the one the predictions were frozen against")
        scale = sp["runs"][args.scale_run]
        if args.reward != scale["reward"] or args.protocol_name != scale["protocol"] or args.seed != scale["seed"]:
            raise ValueError(f"arguments do not match scale run {args.scale_run}: {scale}")
        args.policy_model = scale["policy_model"] if scale["policy_model"] != sp["base_policy"] else None
        args.lora_r = scale.get("lora_r", 0)
        args.learning_rate = scale.get("learning_rate")
    elif not args.smoke and (args.policy_model or args.lora_r or args.learning_rate
                             or args.reward in NEW_REWARDS):
        raise ValueError("--policy-model/--lora-r/--learning-rate and new rewards are smoke-only until a protocol covers them")
    if args.smoke:
        args.output_dir = Path("/tmp") / f"grpo_smoke_{args.reward}"
    followup = scale is None and (args.steps is not None or args.langid_guard)
    rule_run = scale is None and not followup and args.protocol_name != "enzh" and (
        args.reward == "metricx_24_qe" or args.protocol_name != "pilot")
    if followup:
        # Exploratory en-de follow-up runs (amendment 2); never scored against the audit rule.
        if hashlib.sha256(ENDE_AMENDMENT_2.read_bytes()).hexdigest() != ENDE_AMENDMENT_2_SHA256:
            raise ValueError("en-de amendment 2 differs from frozen hash")
        run = {"reward": args.reward, "seed": args.seed, "steps": args.steps,
               "guard": "langid" if args.langid_guard else "none"}
        if args.protocol_name != "ende" or args.guarded or run not in json.loads(ENDE_AMENDMENT_2.read_text())["runs"]:
            raise ValueError(f"Run not in en-de amendment 2: {run}")
        g["seed"], g["steps"] = args.seed, args.steps
    elif scale is not None:
        g["seed"] = args.seed
    elif args.protocol_name == "enzh":
        # Prospective test of results/policy_relative_rule.json; predictions must be frozen first.
        if not ENZH_PREDICTIONS.exists() or \
                hashlib.sha256(ENZH_PREDICTIONS.read_bytes()).hexdigest() != ENZH_PREDICTIONS_SHA256:
            raise ValueError("en-zh predictions missing or differ from frozen hash")
        seed = g["seed"] if args.seed is None else args.seed
        if args.guarded or seed not in protocol["seeds"]:
            raise ValueError(f"Run not covered by the en-zh protocol: {args.reward}, seed {seed}")
        g["seed"] = seed
    elif rule_run:
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
    elif not followup and (args.seed is not None or args.guarded):
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
        if args.langid_guard:
            from hack_guards import off_target
            scores = [x - 1.0 if off_target(t, target_lang) else x for x, t in zip(scores, texts)]
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
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    peft_config = None
    if args.lora_r:
        from peft import LoraConfig
        peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                                 target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=args.policy_model or protocol["policy_model"], reward_funcs=reward_fn,
                          args=config, train_dataset=dataset, peft_config=peft_config)
    trainer.train()
    (args.output_dir / "pilot_run.json").write_text(json.dumps({
        "reward": args.reward, "protocol_name": args.protocol_name,
        "policy_model": args.policy_model or protocol["policy_model"], "lora_r": args.lora_r,
        "learning_rate": config.learning_rate, "scale_run": args.scale_run,
        "protocol_sha256": PROTOCOLS[args.protocol_name][1],
        "audit_rule_sha256": RULE_SHA256 if rule_run else None,
        "seed": g["seed"], "guarded": args.guarded, "langid_guard": args.langid_guard,
        "steps": g["steps"], "ende_amendment_2_sha256": ENDE_AMENDMENT_2_SHA256 if followup else None,
        "replication_sha256": REPLICATION_SHA256 if (args.seed is not None or args.guarded) else None,
        "trl_config": config.to_dict(), "log_history": trainer.state.log_history,
    }, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
