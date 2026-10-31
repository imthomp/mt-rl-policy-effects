"""Evaluate NLLB checkpoints under common and assigned stochastic decoders.

Input JSONL rows must contain `source` and `reference`; optional Tatoeba
sentence IDs and contributor fields are copied into the prediction audit.
This script never updates model parameters.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nllb_policy_rl import (
    DEFAULT_MODEL,
    MAX_SOURCE,
    MAX_TARGET,
    normalize,
    piece_bias,
    sample_one,
)


BETAS = (0.0, 0.20)
SAMPLES_PER_POLICY = 3
BLEU = BLEU(
    tokenize="13a", lowercase=False, smooth_method="exp", smooth_value=None,
    max_ngram_order=4, effective_order=False,
)
CHRF = CHRF(
    char_order=6, word_order=2, beta=2, lowercase=False,
    whitespace=False, eps_smoothing=False,
)


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


def read_rows(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "source" not in row or "reference" not in row:
                raise ValueError(f"row {line_no} needs source and reference fields")
            row["source"] = normalize(row["source"])
            row["reference"] = normalize(row["reference"])
            if not row["source"] or not row["reference"]:
                raise ValueError(f"row {line_no} has empty normalized text")
            pair_key = (
                row.get("source_sentence_id", row["source"]),
                row.get("reference_sentence_id", row["reference"]),
            )
            if pair_key in seen:
                raise ValueError(f"duplicate source/reference identity at row {line_no}")
            seen.add(pair_key)
            row["row_index"] = len(rows)
            rows.append(row)
            if max_examples and len(rows) >= max_examples:
                break
    if not rows:
        raise ValueError(f"no examples found in {path}")
    return rows


def group_rows_by_source(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Treat multiple reviewed translations of one Tatoeba source as references."""
    grouped: dict[str, dict[str, Any]] = {}
    for row in pair_rows:
        source_identity = row.get("source_sentence_id")
        if source_identity is None:
            source_identity = row["source"]
        key = str(source_identity)
        if key not in grouped:
            grouped[key] = {
                "source_sentence_id": row.get("source_sentence_id"),
                "source_contributor": row.get("source_contributor"),
                "source": row["source"],
                "references": [],
                "pair_count": 0,
            }
        group = grouped[key]
        if group["source"] != row["source"]:
            raise ValueError(f"one source sentence ID maps to differing text: {key}")
        group["pair_count"] += 1
        reference = {
            "sentence_id": row.get("reference_sentence_id"),
            "contributor": row.get("reference_contributor"),
            "license": row.get("reference_license", row.get("license")),
            "text": row["reference"],
        }
        if reference["text"] not in {item["text"] for item in group["references"]}:
            group["references"].append(reference)
    return list(grouped.values())


def reference_streams(rows: list[dict[str, Any]]) -> list[list[str]]:
    max_references = max(len(row["references"]) for row in rows)
    return [
        [
            # SacreBLEU expects every reference stream to have one real
            # reference per hypothesis. Repeat a real reference when a source
            # has fewer translations; an empty placeholder would distort the
            # closest-reference length used by BLEU.
            row["references"][index]["text"]
            if index < len(row["references"])
            else row["references"][0]["text"]
            for row in rows
        ]
        for index in range(max_references)
    ]


def load_checkpoint(
    model_dir: Path, checkpoint_spec: str, device: torch.device,
) -> tuple[str, Any, str | None]:
    if "=" not in checkpoint_spec:
        raise ValueError("--checkpoint must be LABEL=PATH or LABEL=base")
    label, value = checkpoint_spec.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"unsafe checkpoint label: {label!r}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32,
    ).to(device).eval()
    checkpoint_hash = None
    if value != "base":
        checkpoint_path = Path(value)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_hash = sha256(checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = state.get("model", state)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.config.use_cache = True
    return label, model, checkpoint_hash


def content_count(path: list[int], special_ids: set[int], eos_id: int) -> int:
    return sum(token not in special_ids for token in path if token != eos_id)


def generation_record(
    path: list[int], hypothesis: str, logprob: float | None,
    tokenizer: Any, references: list[str], sample_index: int | None,
) -> dict[str, Any]:
    eos_id = tokenizer.eos_token_id
    special_ids = set(tokenizer.all_special_ids)
    ended_eos = bool(path and path[-1] == eos_id)
    return {
        "sample_index": sample_index,
        "sampled_token_ids": path,
        "sampled_logprob": logprob,
        "hypothesis": normalize(hypothesis),
        "sentence_chrf_reward": CHRF.sentence_score(
            normalize(hypothesis), references,
        ).score / 100.0,
        "sampled_actions_including_eos": len(path),
        "target_tokens_excluding_eos": len(path) - int(ended_eos),
        "non_special_content_tokens_excluding_eos": content_count(path, special_ids, eos_id),
        "eos_emitted": ended_eos,
        "cap_hit": len(path) == MAX_TARGET and not ended_eos,
        "non_eos_special_emissions": sum(
            token in special_ids and token != eos_id for token in path
        ),
        "empty_output": not normalize(hypothesis),
        "whitespace_word_count": len(normalize(hypothesis).split()),
    }


def corpus_metrics(hypotheses: list[str], references: list[list[str]]) -> dict[str, float]:
    references = [stream for stream in references if len(stream) == len(hypotheses)]
    return {
        "bleu": BLEU.corpus_score(hypotheses, references).score,
        "chrfpp": CHRF.corpus_score(hypotheses, references).score,
        "mean_sentence_chrf_reward": sum(
            CHRF.sentence_score(
                hypothesis,
                [stream[index] for stream in references if stream[index]],
            ).score / 100.0
            for index, hypothesis in enumerate(hypotheses)
        ) / len(hypotheses),
    }


def summarize_generations(records: list[dict[str, Any]]) -> dict[str, float]:
    count = len(records)
    return {
        "mean_sampled_actions_including_eos": sum(r["sampled_actions_including_eos"] for r in records) / count,
        "mean_target_tokens_excluding_eos": sum(r["target_tokens_excluding_eos"] for r in records) / count,
        "mean_non_special_content_tokens_excluding_eos": sum(
            r["non_special_content_tokens_excluding_eos"] for r in records
        ) / count,
        "eos_rate": sum(r["eos_emitted"] for r in records) / count,
        "cap_rate": sum(r["cap_hit"] for r in records) / count,
        "empty_output_rate": sum(r["empty_output"] for r in records) / count,
        "non_eos_special_emissions_per_example": sum(r["non_eos_special_emissions"] for r in records) / count,
    }


@torch.inference_mode()
def common_beam_decode(
    model: Any, tokenizer: Any, sources: list[str], device: torch.device, batch_size: int,
) -> list[tuple[list[int], str]]:
    outputs: list[tuple[list[int], str]] = []
    target_language = tokenizer.convert_tokens_to_ids("deu_Latn")
    for start in range(0, len(sources), batch_size):
        encoded = tokenizer(
            sources[start:start + batch_size], return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SOURCE,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            forced_bos_token_id=target_language,
            do_sample=False,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=False,
            max_new_tokens=MAX_TARGET,
            no_repeat_ngram_size=0,
            repetition_penalty=1.0,
        )
        for sequence in generated.tolist():
            path = list(sequence)
            if path and path[0] == tokenizer.eos_token_id:
                path = path[1:]
            if path and path[0] == target_language:
                path = path[1:]
            hypothesis = tokenizer.decode(sequence, skip_special_tokens=True)
            outputs.append((path, normalize(hypothesis)))
    return outputs


def source_stream_seed(base_seed: int, row: dict[str, Any], beta_index: int, sample_index: int) -> int:
    identity = row.get("source_sentence_id", row["source"])
    digest = hashlib.sha256(str(identity).encode("utf-8")).digest()
    source_offset = int.from_bytes(digest[:4], "big")
    return (base_seed + 100_000 * beta_index + 1_000 * sample_index + source_offset) % (2**63 - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="Checkpoint label: path, repeat for multiple models; use LABEL=base for the original snapshot",
    )
    parser.add_argument("--seed-base", type=int, default=812700)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-examples", type=int,
        help="Smoke only: maximum source groups, retaining every linked reference; never use as a final evaluation",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU allocation required")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    pair_rows = read_rows(args.data, None)
    rows = group_rows_by_source(pair_rows)
    if args.max_examples is not None:
        # Smoke limits count source sentences and keep every reference for
        # each selected source, rather than splitting a multi-reference group.
        rows = rows[:args.max_examples]
    included_pair_count = sum(row["pair_count"] for row in rows)
    if not rows:
        raise ValueError("no source groups selected for evaluation")
    references = reference_streams(rows)
    source_hash = sha256(args.data)
    labels = [spec.split("=", 1)[0] for spec in args.checkpoint]
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint labels must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [args.output_dir / f"predictions_{label}.jsonl" for label in labels]
    final_paths.extend(args.output_dir / f"metrics_{label}.json" for label in labels)
    if not args.overwrite and any(path.exists() for path in final_paths):
        raise FileExistsError("evaluation outputs already exist; use a new directory or --overwrite")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.src_lang, tokenizer.tgt_lang = "eng_Latn", "deu_Latn"
    target_language = tokenizer.convert_tokens_to_ids("deu_Latn")
    if target_language == tokenizer.unk_token_id:
        raise ValueError("NLLB target language token deu_Latn is missing")
    if MAX_TARGET != 256:
        raise ValueError("protocol expects a 256-action cap")
    for model_index, checkpoint_spec in enumerate(args.checkpoint):
        label, model, checkpoint_hash = load_checkpoint(args.model_dir, checkpoint_spec, device)
        bias = piece_bias(args.model_dir, tokenizer, model.config.vocab_size, device)
        sources = [row["source"] for row in rows]

        beam_outputs = common_beam_decode(
            model, tokenizer, sources, device, args.batch_size,
        )
        common_records = [
            generation_record(
                path,
                hypothesis, None, tokenizer,
                [item["text"] for item in row["references"]], None,
            )
            for row, (path, hypothesis) in zip(rows, beam_outputs, strict=True)
        ]
        beam_hypotheses = [hypothesis for _, hypothesis in beam_outputs]
        conditions: dict[str, list[list[dict[str, Any]]]] = {}
        for beta_index, beta in enumerate(BETAS):
            condition_key = f"beta_{beta:.2f}"
            sample_records: list[list[dict[str, Any]]] = [
                [] for _ in range(SAMPLES_PER_POLICY)
            ]
            for sample_index in range(SAMPLES_PER_POLICY):
                for row in rows:
                    generator = torch.Generator(device=device).manual_seed(
                        source_stream_seed(args.seed_base, row, beta_index, sample_index)
                    )
                    path, logprob = sample_one(
                        model, tokenizer, row["source"], bias, beta, generator,
                        device, autocast=True, max_target=MAX_TARGET,
                    )
                    hypothesis = tokenizer.decode(path, skip_special_tokens=True)
                    sample_records[sample_index].append(
                        generation_record(
                            path, hypothesis, logprob, tokenizer,
                            [item["text"] for item in row["references"]], sample_index,
                        )
                    )
            conditions[condition_key] = sample_records

        aggregate: dict[str, Any] = {
            "common_beam4": {
                **corpus_metrics(beam_hypotheses, references),
                **summarize_generations(common_records),
            },
            "stochastic": {},
        }
        for beta in BETAS:
            condition_key = f"beta_{beta:.2f}"
            sample_metrics = []
            for sample_index, sample_records in enumerate(conditions[condition_key]):
                hypotheses = [record["hypothesis"] for record in sample_records]
                sample_metrics.append({
                    "sample_index": sample_index,
                    **corpus_metrics(hypotheses, references),
                    **summarize_generations(sample_records),
                })
            all_records = [record for sample in conditions[condition_key] for record in sample]
            aggregate["stochastic"][condition_key] = {
                "per_sample": sample_metrics,
                "mean_sentence_chrf_reward_across_samples": sum(
                    record["sentence_chrf_reward"] for record in all_records
                ) / len(all_records),
                "combined_diagnostics": summarize_generations(all_records),
            }

        prediction_path = args.output_dir / f"predictions_{label}.jsonl"
        fd, temp_name = tempfile.mkstemp(prefix=f".{prediction_path.name}.", dir=args.output_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                for index, row in enumerate(rows):
                    record = {
                        "row_index": index,
                        "source_sentence_id": row.get("source_sentence_id"),
                        "source_contributor": row.get("source_contributor"),
                        "source": row["source"],
                        "references": row["references"],
                        "pair_count": row["pair_count"],
                        "common_beam4": common_records[index],
                        "stochastic": {
                            key: [samples[index] for samples in sample_sets]
                            for key, sample_sets in conditions.items()
                        },
                    }
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, prediction_path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

        config = {
            "checkpoint_label": label,
            "checkpoint_spec": checkpoint_spec,
            "checkpoint_sha256": checkpoint_hash,
            "base_model_snapshot": args.model_dir.name,
            "model_config_sha256": sha256(args.model_dir / "config.json"),
            "data_path": str(args.data),
            "data_sha256": source_hash,
            "pair_count": included_pair_count,
            "input_pair_count": len(pair_rows),
            "source_count": len(rows),
            "truncated_smoke_run": bool(args.max_examples),
            "max_examples": args.max_examples,
            "source_max_tokens": MAX_SOURCE,
            "target_max_actions": MAX_TARGET,
            "common_decoder": {
                "num_beams": 4, "length_penalty": 1.0,
                "early_stopping": False, "do_sample": False,
                "max_new_tokens": MAX_TARGET, "beta": 0.0,
            },
            "stochastic_decoder": {
                "betas": list(BETAS), "temperature": 0.7,
                "samples_per_policy": SAMPLES_PER_POLICY,
                "seed_base": args.seed_base,
            },
            "metric_signatures": {
                "bleu": "SacreBLEU 2.6.0, tokenize=13a, lowercase=False, smooth=exp, effective_order=False",
                "chrfpp": "SacreBLEU 2.6.0, char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False, eps_smoothing=False",
            },
            "torch": torch.__version__,
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256(prediction_path),
            "summary_metrics": aggregate,
        }
        atomic_json(args.output_dir / f"metrics_{label}.json", config)
        print(json.dumps({
            "checkpoint": label, "source_count": len(rows),
            "pair_count": included_pair_count, "metrics": aggregate,
        }, indent=2), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
