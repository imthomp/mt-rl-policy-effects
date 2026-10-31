"""Calibration-only entropy-matched temperature diagnostic for Tatoeba.

Uses reference-prefix contexts to select a beta=0 temperature matching the
mean conditional entropy of beta=.20, then compares one sampled output per
source under the two procedures. Do not use this as a training arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluate_nllb_policy import group_rows_by_source, read_rows, validate_protocol_split
from nllb_policy_rl import (
    DEFAULT_MODEL,
    MAX_TARGET,
    decode_from_encoder,
    encode_source,
    piece_bias,
    sample_one,
    source_inputs,
)
from reproducibility import repository_metadata
from rights_clearance import clearance_metadata, require_rights_clearance
from sacrebleu.metrics import BLEU, CHRF


TEMPERATURES = tuple(round(0.5 + 0.1 * index, 1) for index in range(11))
BETA = 0.20
BASE_TEMPERATURE = 0.7
ENTROPY_MATCH_TOLERANCE_NATS = 0.05
PAIRED_BOOTSTRAP_RESAMPLES = 2000
PAIRED_BOOTSTRAP_SEED = 812702
CHRF = CHRF(char_order=6, word_order=2, beta=2, lowercase=False,
            whitespace=False, eps_smoothing=False)
BLEU = BLEU(tokenize="13a", lowercase=False, smooth_method="exp",
            smooth_value=None, max_ngram_order=4, effective_order=False)


def sha256(path: Path) -> str:
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


@torch.inference_mode()
def prefix_entropy_and_kl(
    model: Any,
    tokenizer: Any,
    bias: torch.Tensor,
    row: dict[str, Any],
    device: torch.device,
) -> tuple[float, list[float], list[float], int]:
    encoded = source_inputs(tokenizer, row["source"], device)
    target = tokenizer(
        text_target=row["reference"], return_tensors="pt",
        truncation=True, max_length=MAX_TARGET,
    )["input_ids"][0].tolist()
    if not target:
        raise ValueError("empty tokenized reference")
    eos, lang = tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("deu_Latn")
    if eos not in target:
        target.append(eos)
    target = target[:MAX_TARGET]
    if not target:
        raise ValueError("reference has no target actions")

    decoder_start = torch.tensor([[eos]], dtype=torch.long, device=device)
    encoder_outputs = encode_source(model, encoded, autocast=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        first = decode_from_encoder(model, encoded, encoder_outputs, decoder_start)
    past = first.past_key_values
    step_input = torch.tensor([[lang]], dtype=torch.long, device=device)
    tilt_entropies: list[float] = []
    temperature_entropies = [[] for _ in TEMPERATURES]
    divergences = [[] for _ in TEMPERATURES]
    for target_token in target:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = decode_from_encoder(model, encoded, encoder_outputs, step_input, past)
        past = output.past_key_values
        logits = output.logits[0, -1, :].float()
        tilted = ((logits + BETA * bias) / BASE_TEMPERATURE).log_softmax(dim=-1)
        p_tilted = tilted.exp()
        entropy = -(p_tilted * tilted).sum()
        tilt_entropies.append(float(entropy.item()))

        for index, temperature in enumerate(TEMPERATURES):
            temp_logp = (logits / temperature).log_softmax(dim=-1)
            p_temperature = temp_logp.exp()
            temperature_entropies[index].append(
                float((-(p_temperature * temp_logp).sum()).item())
            )
            divergences[index].append(
                float((p_tilted * (tilted - temp_logp)).sum().item())
            )
        step_input = torch.tensor([[target_token]], dtype=torch.long, device=device)
        if target_token == eos:
            break
    return (
        sum(tilt_entropies) / len(tilt_entropies),
        [sum(values) / len(values) for values in temperature_entropies],
        [sum(values) / len(values) for values in divergences],
        len(tilt_entropies),
    )


def seed_for(source_id: Any, base_seed: int, arm: int) -> int:
    digest = hashlib.sha256(str(source_id).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    return (base_seed + arm * 1_000_003 + offset) % (2**63 - 1)


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "p10": None, "median": None, "p90": None}
    ordered = sorted(values)
    def quantile(fraction: float) -> float:
        index = round(fraction * (len(ordered) - 1))
        return ordered[index]
    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "sd": (sum((value - sum(ordered) / len(ordered)) ** 2 for value in ordered)
               / max(1, len(ordered) - 1)) ** 0.5,
        "p10": quantile(0.10),
        "median": quantile(0.50),
        "p90": quantile(0.90),
    }


def paired_descriptive_differences(
    left: list[dict[str, Any]], right: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(left) != len(right) or [r["source_sentence_id"] for r in left] != [
        r["source_sentence_id"] for r in right
    ]:
        raise ValueError("calibration policy outputs are not source-aligned")
    fields = (
        "mean_reference_chrfpp_reward", "sentence_bleu", "action_count_including_eos",
        "word_count", "fertility_tokens_per_word", "eos_emitted", "cap_hit",
    )
    result = {}
    for field in fields:
        differences = [
            float(a[field]) - float(b[field])
            for a, b in zip(left, right, strict=True)
            if a[field] is not None and b[field] is not None
        ]
        mean = sum(differences) / len(differences) if differences else None
        interval = None
        if differences:
            generator = random.Random(PAIRED_BOOTSTRAP_SEED)
            bootstrap_means = sorted(
                sum(differences[index] for index in (
                    generator.randrange(len(differences))
                    for _ in range(len(differences))
                )) / len(differences)
                for _ in range(PAIRED_BOOTSTRAP_RESAMPLES)
            )
            interval = {
                "lower_95": bootstrap_means[round(0.025 * (len(bootstrap_means) - 1))],
                "upper_95": bootstrap_means[round(0.975 * (len(bootstrap_means) - 1))],
            }
        result[field] = {
            "n_source_groups": len(differences),
            "mean_left_minus_right": mean,
            "paired_source_bootstrap_95_interval": interval,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=812701)
    parser.add_argument("--max-sources", type=int, help="Optional engineering smoke limit")
    args = parser.parse_args()
    rights_record = require_rights_clearance()
    repo_metadata = repository_metadata()
    if repo_metadata["git_worktree_dirty"]:
        raise RuntimeError("calibration requires a clean Git worktree")
    if not repo_metadata["git_commit_sha"]:
        raise RuntimeError("calibration requires a Git commit")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU allocation required")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_sources is not None and args.max_sources <= 0:
        raise ValueError("--max-sources must be positive")

    split_name, split_metadata = validate_protocol_split(args.data, {"calibration"})
    pair_rows = read_rows(args.data, None)
    if len(pair_rows) != split_metadata["pair_count"]:
        raise ValueError("calibration pair count differs from the frozen split audit")
    rows = group_rows_by_source(pair_rows)
    if len(rows) != split_metadata["unique_source_count"]:
        raise ValueError("calibration source count differs from the frozen split audit")
    if args.max_sources is not None:
        rows = rows[:args.max_sources]
    if not rows:
        raise ValueError("no source groups selected")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.src_lang, tokenizer.tgt_lang = "eng_Latn", "deu_Latn"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.float32,
    ).to(device).eval()
    model.config.use_cache = True
    bias = piece_bias(args.model_dir, tokenizer, model.config.vocab_size, device)

    per_source_contexts = []
    for source_index, row in enumerate(rows):
        source_tilt_entropies = []
        source_temperature_entropies = [[] for _ in TEMPERATURES]
        source_kls = [0.0] * len(TEMPERATURES)
        source_prefix_position_counts = []
        for reference in row["references"]:
            tilt_entropy, temperature_entropies, prefix_kls, prefix_position_count = prefix_entropy_and_kl(
                model, tokenizer, bias,
                {"source": row["source"], "reference": reference["text"]}, device,
            )
            source_tilt_entropies.append(tilt_entropy)
            source_prefix_position_counts.append(prefix_position_count)
            for index, value in enumerate(temperature_entropies):
                source_temperature_entropies[index].append(value)
            for index, value in enumerate(prefix_kls):
                source_kls[index] += value
        if not source_tilt_entropies:
            raise ValueError(f"no reference-prefix positions for source {source_index}")
        mean_entropy = sum(source_tilt_entropies) / len(source_tilt_entropies)
        mean_temp_entropy = [sum(values) / len(values) for values in source_temperature_entropies]
        source_kls = [value / len(source_tilt_entropies) for value in source_kls]
        per_source_contexts.append({
            "source_sentence_id": row.get("source_sentence_id"),
            "source_mean_tilt_entropy_nats": mean_entropy,
            "source_mean_temperature_entropy_nats": mean_temp_entropy,
            "temperature_mean_kl_tilt_to_temperature_nats": source_kls,
            "reference_count": len(source_tilt_entropies),
            "reference_prefix_position_counts": source_prefix_position_counts,
        })
    mean_tilt_entropy = sum(
        row["source_mean_tilt_entropy_nats"] for row in per_source_contexts
    ) / len(per_source_contexts)
    mean_temperature_entropies = [
        sum(row["source_mean_temperature_entropy_nats"][index]
            for row in per_source_contexts) / len(per_source_contexts)
        for index in range(len(TEMPERATURES))
    ]
    mean_kls = [
        sum(row["temperature_mean_kl_tilt_to_temperature_nats"][index]
            for row in per_source_contexts) / len(per_source_contexts)
        for index in range(len(TEMPERATURES))
    ]
    matched_index = min(range(len(TEMPERATURES)),
                        key=lambda index: abs(mean_temperature_entropies[index] - mean_tilt_entropy))
    matched_temperature = TEMPERATURES[matched_index]
    entropy_residual = mean_temperature_entropies[matched_index] - mean_tilt_entropy

    samples = {"control": [], "length_tilt": []}
    summaries = {}
    sampling_conditions = [
        ("control", 0.0, BASE_TEMPERATURE),
        ("length_tilt", BETA, BASE_TEMPERATURE),
    ]
    # If entropy matching selects the control's existing temperature, the
    # comparator is the same policy. Do not sample or report it as another arm.
    temperature_comparator_aliases_control = matched_temperature == BASE_TEMPERATURE
    if not temperature_comparator_aliases_control:
        sampling_conditions.append(("temperature_only", 0.0, matched_temperature))
    for arm, (name, beta, temperature) in enumerate(sampling_conditions):
        records = []
        for row in rows:
            generator = torch.Generator(device=device).manual_seed(
                seed_for(row.get("source_sentence_id") or row["source"], args.seed, arm)
            )
            path, logprob = sample_one(
                model, tokenizer, row["source"], bias, beta, generator, device,
                autocast=True, max_target=MAX_TARGET, temperature=temperature,
            )
            hypothesis = tokenizer.decode(path, skip_special_tokens=True).strip()
            references = [ref["text"] for ref in row["references"]]
            eos = tokenizer.eos_token_id
            special_ids = set(tokenizer.all_special_ids)
            content_token_count = sum(
                token not in special_ids and token != eos for token in path
            )
            non_eos_special_token_count = sum(
                token in special_ids and token != eos for token in path
            )
            word_count = len(hypothesis.split())
            records.append({
                "source_sentence_id": row.get("source_sentence_id"),
                "sampled_token_ids": path,
                "sampled_logprob": logprob,
                "hypothesis": hypothesis,
                "mean_reference_chrfpp_reward": CHRF.sentence_score(hypothesis, references).score / 100.0,
                "sentence_bleu": BLEU.sentence_score(hypothesis, references).score,
                "action_count_including_eos": len(path),
                "content_token_count_excluding_eos_and_specials": content_token_count,
                "non_eos_special_token_count": non_eos_special_token_count,
                "word_count": word_count,
                "fertility_tokens_per_word": content_token_count / word_count if word_count else None,
                "empty_output": not hypothesis,
                "eos_emitted": bool(path and path[-1] == eos),
                "cap_hit": len(path) == MAX_TARGET and (not path or path[-1] != eos),
                "reference_count": len(references),
            })
        samples[name] = records
        summaries[name] = {
            "beta": beta,
            "temperature": temperature,
            "reward_distribution": distribution([r["mean_reference_chrfpp_reward"] for r in records]),
            "sentence_bleu_distribution": distribution([r["sentence_bleu"] for r in records]),
            "action_count_distribution": distribution([r["action_count_including_eos"] for r in records]),
            "word_count_distribution": distribution([r["word_count"] for r in records]),
            "fertility_distribution": distribution([
                r["fertility_tokens_per_word"] for r in records
                if r["fertility_tokens_per_word"] is not None
            ]),
            "empty_output_rate": sum(r["empty_output"] for r in records) / len(records),
            "non_eos_special_token_emission_rate": sum(
                r["non_eos_special_token_count"] > 0 for r in records
            ) / len(records),
            "non_eos_special_token_emitting_source_count": sum(
                r["non_eos_special_token_count"] > 0 for r in records
            ),
            "non_eos_special_token_count": sum(
                r["non_eos_special_token_count"] for r in records
            ),
            "eos_rate": sum(r["eos_emitted"] for r in records) / len(records),
            "cap_rate": sum(r["cap_hit"] for r in records) / len(records),
            "truncation_rate": sum(r["cap_hit"] for r in records) / len(records),
        }
        max_reference_count = max(len(row["references"]) for row in rows)
        reference_streams = [
            [
                row["references"][index]["text"]
                if index < len(row["references"]) else ""
                for row in rows
            ]
            for index in range(max_reference_count)
        ]
        summaries[name]["corpus_bleu"] = BLEU.corpus_score(
            [record["hypothesis"] for record in records], reference_streams,
        ).score

    included_pairs = sum(row["pair_count"] for row in rows)
    paired_differences = {
        "length_tilt_minus_control": paired_descriptive_differences(
            samples["length_tilt"], samples["control"],
        ),
    }
    if not temperature_comparator_aliases_control:
        paired_differences["temperature_only_minus_length_tilt"] = paired_descriptive_differences(
            samples["temperature_only"], samples["length_tilt"],
        )
    paired_differences["corpus_bleu_points_length_tilt_minus_control"] = (
        summaries["length_tilt"]["corpus_bleu"] - summaries["control"]["corpus_bleu"]
    )

    output = {
        **repo_metadata,
        "rights_clearance": clearance_metadata(rights_record),
        "status": "calibration-only-diagnostic",
        "data_path": str(args.data),
        "data_sha256": sha256(args.data),
        "data_split": split_name,
        "data_license": "CC-BY-2.0-FR",
        "pair_count": included_pairs,
        "source_count": len(rows),
        "model_id": "facebook/nllb-200-distilled-600M",
        "model_license": "CC-BY-NC-4.0",
        "source_snapshot": args.model_dir.name,
        "model_revision": args.model_dir.name,
        "model_snapshot_hash": args.model_dir.name,
        "model_config_sha256": sha256(args.model_dir / "config.json"),
        "sentencepiece_sha256": sha256(args.model_dir / "sentencepiece.bpe.model"),
        "base_seed": args.seed,
        "temperature_grid": list(TEMPERATURES),
        "tilt_beta": BETA,
        "tilt_temperature": BASE_TEMPERATURE,
        "mean_tilt_conditional_entropy_nats": mean_tilt_entropy,
        "temperature_only_mean_conditional_entropy_nats": dict(zip(TEMPERATURES, mean_temperature_entropies)),
        "mean_conditional_kl_tilt_to_temperature_nats": dict(zip(TEMPERATURES, mean_kls)),
        "matched_temperature": matched_temperature,
        "temperature_comparator_aliases_control": temperature_comparator_aliases_control,
        "entropy_match_tolerance_nats": ENTROPY_MATCH_TOLERANCE_NATS,
        "entropy_match_pass": abs(entropy_residual) <= ENTROPY_MATCH_TOLERANCE_NATS,
        "matched_temperature_entropy_residual_nats": entropy_residual,
        "entropy_conditioning": {
            "prefixes": "shared teacher-forced reference-token prefixes, identical across tilted and temperature-only policies",
            "averaging": f"average prefix-position entropy within each reference; average references within each unique English source; then equally average the {len(per_source_contexts)} source-level means",
            "generated_trajectory_entropy_used": False,
        },
        "quality_metric_signatures": {
            "training_reward": "SacreBLEU 2.6.0 sentence chrF++ / 100",
            "independent_quality": "SacreBLEU 2.6.0 corpus BLEU, tokenize=13a, lowercase=False, smooth=exp, effective_order=False",
        },
        "sampled_policy_summaries": summaries,
        "paired_descriptive_differences": paired_differences,
        "paired_source_bootstrap": {
            "resamples": PAIRED_BOOTSTRAP_RESAMPLES,
            "seed": PAIRED_BOOTSTRAP_SEED,
            "unit": "unique English source group",
            "limitation": "intervals reflect source resampling for one sampled output per policy and source; they do not capture within-source rollout sampling variability",
        },
        "sampled_outputs": samples,
        "source_prefix_diagnostics": per_source_contexts,
        "interpretation": "Calibration diagnostic only. The temperature comparator passes the prespecified entropy tolerance only if entropy_match_pass is true; equal conditional entropy and small KL do not establish policy equivalence or justify a third training arm. If the matched temperature equals the control temperature, the policies are identical and the comparator aliases control rather than being sampled as another condition.",
    }
    atomic_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key not in ("sampled_outputs", "source_prefix_diagnostics")}, indent=2))


if __name__ == "__main__":
    main()
