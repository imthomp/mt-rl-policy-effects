"""Minimal NLLB sequence-level REINFORCE pilot for docs/RL_PROTOCOL.md.

The only supported smoke path uses eight built-in, non-research examples,
performs one rollout batch and a backward pass, and is guarded against optimizer
updates. Full runs require a frozen JSONL manifest of source/reference pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from sacrebleu.metrics import CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL = Path(os.environ.get(
    "NLLB_MODEL_DIR",
    str(Path.home() / ".cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d"),
))
TEMPERATURE = 0.7
BETA_TREATMENT = 0.20
MAX_SOURCE = 256
MAX_TARGET = 256
CHRF = CHRF(char_order=6, word_order=2, beta=2, lowercase=False,
            whitespace=False, eps_smoothing=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temp_name)
        with open(temp_name, "rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

# Deliberately synthetic smoke inputs: these are not research data and their
# scores must never be interpreted as model quality or pilot results.
SMOKE_PAIRS = [
    ("The train leaves at noon.", "Der Zug fährt um zwölf Uhr ab."),
    ("Please close the window.", "Bitte schließen Sie das Fenster."),
    ("We will meet again tomorrow.", "Wir treffen uns morgen wieder."),
    ("The report is on the table.", "Der Bericht liegt auf dem Tisch."),
    ("She bought fresh bread this morning.", "Sie hat heute Morgen frisches Brot gekauft."),
    ("Turn left after the bridge.", "Biegen Sie nach der Brücke links ab."),
    ("The children are playing outside.", "Die Kinder spielen draußen."),
    ("I sent you the new address.", "Ich habe Ihnen die neue Adresse geschickt."),
]


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def read_jsonl(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source, reference = normalize(row["source"]), normalize(row["reference"])
            if source and reference:
                pairs.append((source, reference))
            else:
                raise ValueError(f"empty source/reference on {path}:{line_no}")
    if not pairs:
        raise ValueError(f"no examples found in {path}")
    return pairs


def piece_bias(model_dir: Path, tokenizer: Any, vocab_size: int,
               device: torch.device) -> torch.Tensor:
    sp = spm.SentencePieceProcessor(model_file=str(model_dir / "sentencepiece.bpe.model"))
    values = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    specials = set(tokenizer.all_special_ids)
    for token_id in range(min(vocab_size, sp.get_piece_size())):
        if token_id not in specials:
            visible = sp.id_to_piece(token_id).replace("▁", " ")
            values[token_id] = 1.0 - len(visible) if visible else 0.0
    return values


def policy_log_probs(logits: torch.Tensor, bias: torch.Tensor, beta: float,
                     temperature: float = TEMPERATURE) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return ((logits.float() + beta * bias) / temperature).log_softmax(dim=-1)


def source_inputs(tokenizer: Any, source: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(source, return_tensors="pt", truncation=True, max_length=MAX_SOURCE)
    return {key: value.to(device) for key, value in encoded.items()}


def encode_source(model: Any, encoded: dict[str, torch.Tensor], autocast: bool) -> Any:
    """Compute encoder states once and reuse them across all target actions."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        return model.get_encoder()(**encoded, return_dict=True)


def decode_from_encoder(model: Any, encoded: dict[str, torch.Tensor], encoder_outputs: Any,
                        decoder_input_ids: torch.Tensor, past_key_values: Any = None) -> Any:
    return model(
        encoder_outputs=encoder_outputs,
        attention_mask=encoded.get("attention_mask"),
        decoder_input_ids=decoder_input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )


@torch.no_grad()
def greedy_baselines(model: Any, tokenizer: Any, sources: list[str], references: list[str],
                     device: torch.device) -> list[float]:
    rewards: list[float] = []
    target_language = tokenizer.convert_tokens_to_ids("deu_Latn")
    for source, reference in zip(sources, references, strict=True):
        encoded = source_inputs(tokenizer, source, device)
        generated = model.generate(
            **encoded, forced_bos_token_id=target_language,
            do_sample=False, num_beams=1, max_new_tokens=MAX_TARGET, max_length=None,
        )[0]
        hypothesis = tokenizer.decode(generated, skip_special_tokens=True)
        rewards.append(CHRF.sentence_score(hypothesis, [reference]).score / 100.0)
    return rewards


@torch.no_grad()
def sample_one(model: Any, tokenizer: Any, source: str, bias: torch.Tensor,
               beta: float, generator: torch.Generator, device: torch.device,
               autocast: bool, max_target: int,
               temperature: float = TEMPERATURE) -> tuple[list[int], float]:
    """Sample autoregressively from the exact transformed categorical policy."""
    eos, lang = tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("deu_Latn")
    encoded = source_inputs(tokenizer, source, device)
    first_input = torch.tensor([[eos]], dtype=torch.long, device=device)
    encoder_outputs = encode_source(model, encoded, autocast)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        first = decode_from_encoder(model, encoded, encoder_outputs, first_input)
    past = first.past_key_values
    step_input = torch.tensor([[lang]], dtype=torch.long, device=device)
    sampled: list[int] = []
    sampled_logp = 0.0
    for _ in range(max_target):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            output = decode_from_encoder(model, encoded, encoder_outputs, step_input, past)
        past = output.past_key_values
        logits = output.logits[:, -1, :]
        logp = policy_log_probs(logits, bias, beta, temperature=temperature)
        token = int(torch.multinomial(logp.exp(), 1, generator=generator).item())
        sampled.append(token)
        sampled_logp += float(logp[0, token].item())
        if token == eos:
            break
        step_input = torch.tensor([[token]], dtype=torch.long, device=device)
    return sampled, sampled_logp


def score_path(model: Any, tokenizer: Any, source: str, path: list[int],
               bias: torch.Tensor, beta: float, device: torch.device,
               autocast: bool, with_grad: bool) -> torch.Tensor:
    """Replay the exact cached autoregressive path; exclude forced language action."""
    eos, lang = tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("deu_Latn")
    if not path:
        return torch.zeros((), device=device)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        encoded = source_inputs(tokenizer, source, device)
        start = torch.tensor([[eos]], dtype=torch.long, device=device)
        encoder_outputs = encode_source(model, encoded, autocast)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            first = decode_from_encoder(model, encoded, encoder_outputs, start)
        past = first.past_key_values
        step_input = torch.tensor([[lang]], dtype=torch.long, device=device)
        terms: list[torch.Tensor] = []
        for token in path:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                output = decode_from_encoder(model, encoded, encoder_outputs, step_input, past)
            past = output.past_key_values
            logp = policy_log_probs(output.logits[:, -1, :], bias, beta)
            terms.append(logp[0, token])
            step_input = torch.tensor([[token]], dtype=torch.long, device=device)
        return torch.stack(terms).sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, help="Frozen JSONL; each row has source and reference fields")
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs/dryrun"))
    parser.add_argument("--beta", type=float, choices=(0.0, 0.20), default=0.0)
    parser.add_argument("--run-seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true",
                        help="One synthetic batch + backward pass; optimizer update is forbidden")
    parser.add_argument("--preflight", action="store_true",
                        help="One frozen-manifest batch + one throwaway AdamW step; saves no model")
    parser.add_argument("--dry-run-max-target", type=int, default=32)
    parser.add_argument("--max-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--subset-seed", type=int)
    parser.add_argument("--no-autocast", action="store_true")
    args = parser.parse_args()

    if args.dry_run and args.preflight:
        parser.error("choose either --dry-run or --preflight")
    if not args.dry_run and args.data is None:
        parser.error("training/preflight requires --data frozen.jsonl; --dry-run needs no research data")
    if args.dry_run and args.data is not None:
        parser.error("dry-run uses only built-in synthetic examples; omit --data")
    if not args.dry_run and not args.preflight and args.output.exists():
        if not args.output.is_dir() or any(args.output.iterdir()):
            raise FileExistsError(
                f"refusing to reuse non-empty run directory {args.output}; use a fresh directory for every run"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU allocation required")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    device = torch.device("cuda")
    allocator_limit_gib = float(os.environ.get("NLLB_ALLOCATOR_LIMIT_GIB", "0"))
    physical_gpu_memory_gib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if allocator_limit_gib > 0:
        if allocator_limit_gib >= physical_gpu_memory_gib:
            raise ValueError("NLLB_ALLOCATOR_LIMIT_GIB must be below physical GPU memory")
        torch.cuda.set_per_process_memory_fraction(
            allocator_limit_gib / physical_gpu_memory_gib, 0,
        )
    autocast = not args.no_autocast
    if args.subset_seed is None:
        args.subset_seed = 314159 + args.run_seed
    random.seed(args.run_seed)
    torch.manual_seed(args.run_seed)
    torch.cuda.manual_seed_all(args.run_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.src_lang, tokenizer.tgt_lang = "eng_Latn", "deu_Latn"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32,
    ).to(device).eval()
    model.config.use_cache = False
    bias = piece_bias(args.model, tokenizer, model.config.vocab_size, device)

    preflight_indices: list[int] | None = None
    if args.dry_run:
        pairs = SMOKE_PAIRS
        beta = args.beta
        max_target = args.dry_run_max_target
        batch_size = len(pairs)
    else:
        pairs = read_jsonl(args.data)
        beta = args.beta
        max_target = MAX_TARGET
        batch_size = args.batch_size
        if len(pairs) != 2048:
            raise ValueError(f"protocol requires exactly 2048 frozen rows; found {len(pairs)}")
        if batch_size != 8:
            raise ValueError("protocol fixes batch size at 8")
        if args.max_updates != 256:
            raise ValueError("protocol fixes training at 256 updates")
        if args.preflight:
            order_rng = random.Random(args.subset_seed)
            manifest_order = list(range(len(pairs)))
            order_rng.shuffle(manifest_order)
            preflight_indices = manifest_order[:batch_size]
            pairs = [pairs[index] for index in preflight_indices]

    sources = [s for s, _ in pairs]
    references = [r for _, r in pairs]
    if args.dry_run or args.preflight:
        training_indices = list(range(batch_size))
    else:
        order_rng = random.Random(args.subset_seed)
        training_indices = list(range(len(pairs)))
        order_rng.shuffle(training_indices)

    # Baseline rewards are frozen once per run. Preflight computes only its sampled batch.
    baseline_file = args.output / "baseline_rewards.json"
    if not args.dry_run and not args.preflight and baseline_file.exists():
        baseline_rows = json.loads(baseline_file.read_text(encoding="utf-8"))
        if baseline_rows.get("data_sha256") != __import__("hashlib").sha256(args.data.read_bytes()).hexdigest():
            raise ValueError("cached baselines belong to a different data file")
        baselines = baseline_rows["rewards"]
        if len(baselines) != len(pairs):
            raise ValueError("cached baseline count does not match frozen data")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        baselines = greedy_baselines(model, tokenizer, sources, references, device)
        if not args.dry_run and not args.preflight:
            atomic_json(baseline_file, {
                "data_sha256": __import__("hashlib").sha256(args.data.read_bytes()).hexdigest(),
                "reward": "SacreBLEU 2.6.0 sentence chrF++ / 100",
                "rewards": baselines,
            })
        baseline_seconds = time.perf_counter() - started

    sample_seed = 100 * args.run_seed + (0 if beta == 0 else 1)
    if not args.dry_run:
        import transformers
        import sacrebleu
        config = {
            "protocol": "docs/RL_PROTOCOL.md", "model_dir": str(args.model),
            "model_snapshot_revision": args.model.name,
            "model_config_sha256": file_sha256(args.model / "config.json"),
            "sentencepiece_sha256": file_sha256(args.model / "sentencepiece.bpe.model"),
            "data_path": str(args.data),
            "data_sha256": file_sha256(args.data),
            "beta": beta, "temperature": TEMPERATURE, "max_source_tokens": MAX_SOURCE,
            "max_target_actions": MAX_TARGET, "batch_size": batch_size,
            "updates": args.max_updates, "run_seed": args.run_seed,
            "subset_seed": args.subset_seed, "sample_seed": sample_seed,
            "learning_rate": 1e-6, "adamw_betas": [0.9, 0.999],
            "adamw_epsilon": 1e-8, "weight_decay": 0.0,
            "reward": "SacreBLEU 2.6.0 sentence chrF++ / 100",
            "sacrebleu": sacrebleu.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "autocast_bf16": autocast,
            "initial_checkpoint": "facebook/nllb-200-distilled-600M cached snapshot",
            "mode": "one-batch-resource-preflight" if args.preflight else "full-training",
            "physical_gpu_memory_gib": physical_gpu_memory_gib,
            "pytorch_allocator_limit_gib": allocator_limit_gib or None,
            "manifest_indices_used": preflight_indices,
        }
        atomic_json(args.output / "run_config.json", config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.0)
    generator = torch.Generator(device=device).manual_seed(sample_seed)
    log_path = args.output / "steps.jsonl"
    started_all = time.perf_counter()
    updates = 1 if args.dry_run or args.preflight else args.max_updates
    for update in range(updates):
        begin = update * batch_size
        batch_indices = training_indices[begin:begin + batch_size]
        if len(batch_indices) < batch_size:
            break
        batch_sources = [sources[i] for i in batch_indices]
        batch_refs = [references[i] for i in batch_indices]
        batch_baselines = torch.tensor([baselines[i] for i in batch_indices], device=device)
        if args.dry_run:
            max_target = args.dry_run_max_target
        torch.cuda.reset_peak_memory_stats(device)
        update_started = time.perf_counter()
        paths, sampled_logps, hypotheses, rewards = [], [], [], []
        for source, reference in zip(batch_sources, batch_refs, strict=True):
            path, sampled_logp = sample_one(
                model, tokenizer, source, bias, beta, generator, device, autocast, max_target,
            )
            paths.append(path)
            sampled_logps.append(sampled_logp)
            hypothesis = tokenizer.decode(path, skip_special_tokens=True)
            hypotheses.append(hypothesis)
            rewards.append(CHRF.sentence_score(hypothesis, [reference]).score / 100.0)

        if args.preflight:
            source_lengths = [
                len(tokenizer(source, truncation=True, max_length=MAX_SOURCE)["input_ids"])
                for source in batch_sources
            ]
            rollout_audit = {
                "phase": "rollouts_complete_before_rescoring_or_gradient",
                "manifest_indices": [preflight_indices[i] for i in batch_indices],
                "source_token_lengths": source_lengths,
                "target_action_lengths": [len(path) for path in paths],
                "cap_hits": [len(path) == max_target and path[-1] != tokenizer.eos_token_id
                             for path in paths],
                "sampled_logprobs": sampled_logps,
                "cuda_allocated_gib": torch.cuda.memory_allocated(device) / (1024 ** 3),
                "cuda_reserved_gib": torch.cuda.memory_reserved(device) / (1024 ** 3),
            }
            (args.output / "rollout_audit.json").write_text(
                json.dumps(rollout_audit, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(rollout_audit), flush=True)

        rescored = torch.stack([
            score_path(model, tokenizer, source, path, bias, beta, device, autocast, with_grad=False)
            for source, path in zip(batch_sources, paths, strict=True)
        ])
        score_errors = [abs(float(a) - float(b)) for a, b in zip(sampled_logps, rescored.tolist(), strict=True)]
        max_score_error = max(score_errors, default=0.0)
        if max_score_error > 1e-4:
            raise RuntimeError(f"behavior log-probability mismatch {max_score_error:.3g} > 1e-4")
        advantages = torch.tensor(rewards, dtype=torch.float32, device=device) - batch_baselines
        loss = -(advantages.detach() * rescored.detach()).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite REINFORCE loss")
        if not args.dry_run:
            optimizer.zero_grad(set_to_none=True)
        if args.dry_run:
            for i, (source, path) in enumerate(zip(batch_sources, paths, strict=True)):
                path_score = score_path(model, tokenizer, source, path, bias, beta,
                                        device, autocast, with_grad=True)
                (-(advantages[i].detach() * path_score) / batch_size).backward()
            grad_norm_sq = torch.zeros((), device=device)
            finite_grad_tensors = 0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("non-finite gradient")
                    grad_norm_sq += parameter.grad.float().square().sum()
                    finite_grad_tensors += 1
            grad_norm = float(grad_norm_sq.sqrt().item())
            optimizer.zero_grad(set_to_none=True)
        else:
            for i, (source, path) in enumerate(zip(batch_sources, paths, strict=True)):
                if args.preflight:
                    print(json.dumps({
                        "phase": "gradient_rescore_start", "response_index": i,
                        "target_actions": len(path),
                        "cuda_allocated_gib": torch.cuda.memory_allocated(device) / (1024 ** 3),
                        "cuda_reserved_gib": torch.cuda.memory_reserved(device) / (1024 ** 3),
                    }), flush=True)
                path_score = score_path(model, tokenizer, source, path, bias, beta,
                                        device, autocast, with_grad=True)
                (-(advantages[i].detach() * path_score) / batch_size).backward()
                if args.preflight:
                    print(json.dumps({
                        "phase": "gradient_rescore_done", "response_index": i,
                        "cuda_allocated_gib": torch.cuda.memory_allocated(device) / (1024 ** 3),
                        "cuda_reserved_gib": torch.cuda.memory_reserved(device) / (1024 ** 3),
                    }), flush=True)
                del path_score
            grad_norm_sq = torch.zeros((), device=device)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("non-finite gradient")
                    grad_norm_sq += parameter.grad.float().square().sum()
            grad_norm = float(grad_norm_sq.sqrt().item())
            optimizer.step()
            update_norm_sq = torch.zeros((), device=device)
            for parameter in model.parameters():
                state = optimizer.state.get(parameter)
                if not state:
                    continue
                step_number = int(state["step"].item())
                bias_correction1 = 1.0 - 0.9 ** step_number
                bias_correction2_sqrt = math.sqrt(1.0 - 0.999 ** step_number)
                delta = (state["exp_avg"] / bias_correction1) / (
                    state["exp_avg_sq"].sqrt() / bias_correction2_sqrt + 1e-8
                )
                update_norm_sq += delta.float().square().sum()
            update_norm = float((1e-6 * update_norm_sq.sqrt()).item())
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        record = {
            "update": update + 1, "beta": beta, "loss": float(loss.detach().item()),
            "manifest_indices": batch_indices,
            "rewards": rewards,
            "baseline_rewards": batch_baselines.detach().cpu().tolist(),
            "advantages": advantages.detach().cpu().tolist(),
            "mean_reward": sum(rewards) / len(rewards), "mean_baseline": float(batch_baselines.mean().item()),
            "advantage_mean": float(advantages.mean().item()), "gradient_norm": grad_norm,
            "update_norm": None if args.dry_run else update_norm,
            "mean_sample_logprob": sum(sampled_logps) / len(sampled_logps),
            "max_logprob_rescore_error": max_score_error,
            "mean_sampled_actions": sum(map(len, paths)) / len(paths),
            "mean_target_tokens_excluding_eos": sum(
                len(path) - int(bool(path) and path[-1] == tokenizer.eos_token_id)
                for path in paths
            ) / len(paths),
            "eos_rate": sum(bool(p and p[-1] == tokenizer.eos_token_id) for p in paths) / len(paths),
            "cap_rate": sum(len(p) == max_target and (not p or p[-1] != tokenizer.eos_token_id) for p in paths) / len(paths),
            "non_eos_special_emissions": sum(
                1 for p in paths for token in p
                if token in tokenizer.all_special_ids and token != tokenizer.eos_token_id
            ),
            "elapsed_seconds": time.perf_counter() - update_started,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
            "sampled_token_ids": paths,
            "sampled_logprobs": sampled_logps,
            "rescored_logprobs": rescored.tolist(),
            "hypotheses": hypotheses,
        }
        if not args.dry_run:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({key: value for key, value in record.items()
                          if key not in {"sampled_token_ids", "sampled_logprobs", "rescored_logprobs", "hypotheses"}},
                         ensure_ascii=False), flush=True)
        if args.dry_run:
            print(json.dumps({
                "dry_run": True, "optimizer_step_called": False,
                "finite_gradient_tensors": finite_grad_tensors,
                "gradient_norm": grad_norm,
                "baseline_seconds": baseline_seconds,
                "batch_seconds": time.perf_counter() - started_all,
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "sacrebleu": __import__("sacrebleu").__version__,
                "autocast_bf16": autocast,
                "physical_gpu_memory_gib": physical_gpu_memory_gib,
                "pytorch_allocator_limit_gib": allocator_limit_gib or None,
            }, indent=2), flush=True)
            return

        if args.preflight:
            preflight_record = {
                "preflight_complete": True,
                "optimizer_step_called": True,
                "throwaway_model_checkpoint_saved": False,
                "manifest_indices": preflight_indices,
                "baseline_seconds": baseline_seconds,
                "batch_seconds": time.perf_counter() - started_all,
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "sacrebleu": __import__("sacrebleu").__version__,
                "autocast_bf16": autocast,
            }
            (args.output / "preflight_summary.json").write_text(
                json.dumps(preflight_record, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(preflight_record, indent=2), flush=True)
            return

    atomic_torch_save(
        args.output / "final.pt",
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "beta": beta, "run_seed": args.run_seed, "protocol": "docs/RL_PROTOCOL.md"},
    )
    atomic_json(args.output / "final_checkpoint_integrity.json", {
        "path": str(args.output / "final.pt"),
        "sha256": file_sha256(args.output / "final.pt"),
        "size_bytes": (args.output / "final.pt").stat().st_size,
    })
    print(json.dumps({"training_complete": True, "updates": updates,
                      "elapsed_seconds": time.perf_counter() - started_all,
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
