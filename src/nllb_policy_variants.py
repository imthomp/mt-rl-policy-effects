"""Calibrate alternative valid decoder policies for NLLB token-length studies.

Every condition samples from and scores the exact same normalized policy. The
variants deliberately measure total decoder-policy effects, not fertility alone.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL = Path(os.environ["NLLB_MODEL_DIR"])
SOURCE = Path(os.environ["CALIBRATION_SOURCE"])
REFERENCE = Path(os.environ["CALIBRATION_REFERENCE"])
OUTPUT = Path(os.environ["DIAGNOSTIC_OUTPUT"])
COUNT = int(os.environ.get("CALIBRATION_COUNT", "128"))
SEED = int(os.environ.get("CALIBRATION_SEED", "920"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
TEMPERATURE = 0.7

CONDITIONS = [
    {"name": "baseline", "feature": "length", "beta": 0.0, "mixture": 0.0},
    {"name": "length_mixture_10pct", "feature": "length", "beta": 0.30, "mixture": 0.10},
    {"name": "length_mixture_25pct", "feature": "length", "beta": 0.30, "mixture": 0.25},
    {"name": "continuation_bias_005", "feature": "continuation", "beta": 0.05, "mixture": 1.0},
    {"name": "continuation_bias_010", "feature": "continuation", "beta": 0.10, "mixture": 1.0},
    {"name": "continuation_mixture_25pct", "feature": "continuation", "beta": 0.10, "mixture": 0.25},
]

ARRAY_CONDITIONS = [
    {"name": "length_mixture_15pct", "feature": "length", "beta": 0.30, "mixture": 0.15},
    {"name": "length_mixture_20pct", "feature": "length", "beta": 0.30, "mixture": 0.20},
    {"name": "continuation_bias_015", "feature": "continuation", "beta": 0.15, "mixture": 1.0},
    {"name": "continuation_bias_020", "feature": "continuation", "beta": 0.20, "mixture": 1.0},
    {"name": "baseline_a100", "feature": "length", "beta": 0.0, "mixture": 0.0},
]


def chrf_proxy(hypotheses: list[str], references: list[str]) -> float:
    from collections import Counter

    scores = []
    for hyp, ref in zip(hypotheses, references, strict=True):
        per_order = []
        for n in range(1, 7):
            hyp_counts = Counter(hyp[i : i + n] for i in range(max(0, len(hyp) - n + 1)))
            ref_counts = Counter(ref[i : i + n] for i in range(max(0, len(ref) - n + 1)))
            overlap = sum((hyp_counts & ref_counts).values())
            precision = overlap / max(1, sum(hyp_counts.values()))
            recall = overlap / max(1, sum(ref_counts.values()))
            per_order.append(
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        scores.append(sum(per_order) / len(per_order))
    return 100 * sum(scores) / max(1, len(scores))


def main() -> None:
    pairs = [
        (source, reference)
        for source, reference in zip(
            SOURCE.read_text(encoding="utf-8").splitlines(),
            REFERENCE.read_text(encoding="utf-8").splitlines(),
        )
        if source.strip() and reference.strip()
    ]
    random.Random(SEED).shuffle(pairs)
    pairs = pairs[:COUNT]
    sources, references = map(list, zip(*pairs, strict=True))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.src_lang, tokenizer.tgt_lang = "eng_Latn", "deu_Latn"
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, local_files_only=True).eval()
    device = torch.device("cuda")
    model.to(device)
    sp = spm.SentencePieceProcessor(model_file=str(MODEL / "sentencepiece.bpe.model"))
    special_ids = set(tokenizer.all_special_ids)
    vocab_size = model.config.vocab_size

    features = {}
    for name in ("length", "continuation"):
        values = torch.zeros(vocab_size, dtype=torch.float32, device=device)
        for token_id in range(min(vocab_size, sp.get_piece_size())):
            if token_id in special_ids:
                continue
            piece = sp.id_to_piece(token_id)
            if name == "length":
                visible = piece.replace("▁", " ")
                values[token_id] = 1.0 - len(visible) if visible else 0.0
            else:
                # A piece without SentencePiece's word-start marker is a continuation.
                values[token_id] = 1.0 if not piece.startswith("▁") else 0.0
        features[name] = values

    eos = tokenizer.eos_token_id
    target_language = tokenizer.convert_tokens_to_ids("deu_Latn")

    def policy_log_probs(logits: torch.Tensor, condition: dict) -> torch.Tensor:
        base = (logits / TEMPERATURE).log_softmax(dim=-1)
        mixture = condition["mixture"]
        if mixture <= 0 or condition["beta"] == 0:
            return base
        tilted = (
            (logits + condition["beta"] * features[condition["feature"]])
            / TEMPERATURE
        ).log_softmax(dim=-1)
        return torch.logaddexp(
            base + torch.log(torch.tensor(1 - mixture, device=device)),
            tilted + torch.log(torch.tensor(mixture, device=device)),
        )

    def encode_source(source: str):
        encoded = tokenizer(source, return_tensors="pt", truncation=True, max_length=256)
        return {key: value.to(device) for key, value in encoded.items()}

    def generate_one(source: str, condition: dict, seed: int):
        generator = torch.Generator(device=device).manual_seed(seed)
        encoded = encode_source(source)
        decoder_input = torch.tensor([[eos]], dtype=torch.long, device=device)
        with torch.inference_mode():
            first = model(**encoded, decoder_input_ids=decoder_input, use_cache=True)
            past = first.past_key_values
            step_input = torch.tensor([[target_language]], dtype=torch.long, device=device)
            generated, selected_logps = [], []
            for _ in range(MAX_NEW_TOKENS):
                output = model(
                    **encoded,
                    decoder_input_ids=step_input,
                    past_key_values=past,
                    use_cache=True,
                )
                past = output.past_key_values
                log_probs = policy_log_probs(output.logits[:, -1, :].float(), condition)
                selected = torch.multinomial(log_probs.exp(), 1, generator=generator)
                token_id = int(selected.item())
                generated.append(token_id)
                selected_logps.append(float(log_probs[0, token_id].item()))
                if token_id == eos:
                    break
                step_input = selected
        return generated, sum(selected_logps)

    def score_path(source: str, generated: list[int], condition: dict) -> float:
        encoded = encode_source(source)
        decoder_ids = [eos, target_language] + generated[:-1]
        decoder_input = torch.tensor([decoder_ids], dtype=torch.long, device=device)
        labels = torch.tensor([generated], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(**encoded, decoder_input_ids=decoder_input).logits.float()
            logits = logits[:, 1 : 1 + len(generated), :]
            log_probs = policy_log_probs(logits, condition)
            return float(log_probs.gather(-1, labels.unsqueeze(-1)).sum().item())

    if "POLICY_VARIANT_INDEX" in os.environ:
        conditions_to_run = [ARRAY_CONDITIONS[int(os.environ["POLICY_VARIANT_INDEX"])]]
    else:
        conditions_to_run = CONDITIONS

    conditions = []
    for condition in conditions_to_run:
        rows, hypotheses, token_lengths, score_errors = [], [], [], []
        for index, (source, reference) in enumerate(zip(sources, references, strict=True)):
            generated, sampled_logprob = generate_one(
                source, condition, seed=SEED + index
            )
            rescored_logprob = score_path(source, generated, condition)
            hypothesis = tokenizer.decode(generated, skip_special_tokens=True)
            hypotheses.append(hypothesis)
            token_count = len(generated) - int(bool(generated) and generated[-1] == eos)
            token_lengths.append(token_count)
            score_errors.append(abs(sampled_logprob - rescored_logprob))
            rows.append(
                {
                    "source": source,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "sampled_token_ids": generated,
                    "sampled_logprob": sampled_logprob,
                    "rescored_logprob": rescored_logprob,
                }
            )
        conditions.append(
            {
                **condition,
                "mean_generated_token_count_excluding_eos": sum(token_lengths) / len(rows),
                "mean_hypothesis_characters": sum(len(row["hypothesis"]) for row in rows) / len(rows),
                "mean_hypothesis_words": sum(len(row["hypothesis"].split()) for row in rows) / len(rows),
                "mean_generated_tokens_per_hypothesis_word": sum(
                    tokens / max(1, len(row["hypothesis"].split()))
                    for tokens, row in zip(token_lengths, rows, strict=True)
                ) / len(rows),
                "sampled_eos_rate": sum(
                    bool(row["sampled_token_ids"])
                    and row["sampled_token_ids"][-1] == eos
                    for row in rows
                ) / len(rows),
                "max_length_without_eos_count": sum(
                    len(row["sampled_token_ids"]) == MAX_NEW_TOKENS
                    and row["sampled_token_ids"][-1] != eos
                    for row in rows
                ),
                "mean_abs_sampling_vs_rescore_logprob_error": sum(score_errors) / len(rows),
                "chrf_proxy_1_to_6": chrf_proxy(hypotheses, references),
                "rows": rows,
            }
        )

    result = {
        "model": "facebook/nllb-200-distilled-600M",
        "direction": "eng_Latn->deu_Latn",
        "calibration_count": len(pairs),
        "calibration_source": "cached WMT23 en-de test, treated as calibration only",
        "heldout_reservation": "cached WMT24 en-de remains untouched",
        "device": torch.cuda.get_device_name(device),
        "temperature": TEMPERATURE,
        "interpretation": (
            "Valid normalized behavior policies. Mixture probabilities are marginalized "
            "exactly at each token. These are total decoder-policy interventions, not fertility-only effects."
        ),
        "conditions": conditions,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "conditions"}, indent=2))
    for condition in conditions:
        print({key: value for key, value in condition.items() if key != "rows"})


if __name__ == "__main__":
    main()
