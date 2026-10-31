"""On-policy NLLB diagnostic for a decoder-time token-length policy tilt.

For beta != 0, next-token logits receive beta * (1 - visible_piece_length)
for ordinary SentencePiece pieces, followed by a fresh softmax. EOS and model
special tokens receive zero bias. Sampling and likelihood scoring use the same
tilted distribution. This is a total policy-procedure intervention: it changes
string probabilities as well as token-path probabilities and does not isolate
fertility alone.
"""

from __future__ import annotations

import json
import math
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
COUNT = int(os.environ.get("CALIBRATION_COUNT", "16"))
SEED = int(os.environ.get("CALIBRATION_SEED", "206"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
BETAS = tuple(float(value) for value in os.environ.get("LENGTH_BIASES", "-0.03,0,0.03").split(","))
TEMPERATURE = 0.7


def chrf_proxy(hypotheses: list[str], references: list[str]) -> float:
    """Character n-gram F-score proxy, orders 1..6; not official chrF++."""
    from collections import Counter

    scores = []
    for hyp, ref in zip(hypotheses, references):
        f_scores = []
        for n in range(1, 7):
            h = Counter(hyp[i : i + n] for i in range(max(0, len(hyp) - n + 1)))
            r = Counter(ref[i : i + n] for i in range(max(0, len(ref) - n + 1)))
            match = sum((h & r).values())
            precision = match / max(1, sum(h.values()))
            recall = match / max(1, sum(r.values()))
            f_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        scores.append(sum(f_scores) / len(f_scores))
    return 100 * sum(scores) / max(1, len(scores))


def main() -> None:
    sources = SOURCE.read_text(encoding="utf-8").splitlines()
    references = REFERENCE.read_text(encoding="utf-8").splitlines()
    pairs = [(s, r) for s, r in zip(sources, references) if s.strip() and r.strip()]
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    pairs = pairs[:COUNT]
    sources, references = map(list, zip(*pairs))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.src_lang, tokenizer.tgt_lang = "eng_Latn", "deu_Latn"
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, local_files_only=True).eval()
    device = torch.device("cuda")
    model.to(device)
    sp = spm.SentencePieceProcessor(model_file=str(MODEL / "sentencepiece.bpe.model"))
    vocab_size = model.config.vocab_size
    special_ids = set(tokenizer.all_special_ids)
    piece_bias = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    for token_id in range(min(vocab_size, sp.get_piece_size())):
        if token_id not in special_ids:
            piece = sp.id_to_piece(token_id)
            visible = piece.replace("▁", " ")
            if visible:
                piece_bias[token_id] = 1.0 - len(visible)

    eos = tokenizer.eos_token_id
    lang = tokenizer.convert_tokens_to_ids("deu_Latn")
    pad = tokenizer.pad_token_id

    def encode_source(source: str):
        x = tokenizer(source, return_tensors="pt", truncation=True, max_length=256)
        return {k: v.to(device) for k, v in x.items()}

    def biased_logits(logits: torch.Tensor, beta: float) -> torch.Tensor:
        return logits + beta * piece_bias

    def generate_one(source: str, beta: float, seed: int):
        torch_generator = torch.Generator(device=device).manual_seed(seed)
        encoded = encode_source(source)
        decoder_input = torch.tensor([[eos]], dtype=torch.long, device=device)
        with torch.inference_mode():
            first = model(**encoded, decoder_input_ids=decoder_input, use_cache=True)
            past = first.past_key_values
            # NLLB's first decoder action is a forced target-language token.
            step_input = torch.tensor([[lang]], dtype=torch.long, device=device)
            generated: list[int] = []
            chosen_logps: list[float] = []
            for _ in range(MAX_NEW_TOKENS):
                out = model(
                    **encoded,
                    decoder_input_ids=step_input,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                scores = biased_logits(out.logits[:, -1, :].float(), beta) / TEMPERATURE
                log_probs = scores.log_softmax(dim=-1)
                next_id = torch.multinomial(log_probs.exp(), 1, generator=torch_generator)
                token_id = int(next_id.item())
                chosen_logps.append(float(log_probs[0, token_id].item()))
                generated.append(token_id)
                if token_id == eos:
                    break
                step_input = next_id
        return generated, sum(chosen_logps)

    def score_path(source: str, generated: list[int], beta: float) -> float:
        if not generated:
            return 0.0
        encoded = encode_source(source)
        decoder_ids = [eos, lang] + generated[:-1]
        decoder_input = torch.tensor([decoder_ids], dtype=torch.long, device=device)
        labels = torch.tensor([generated], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(**encoded, decoder_input_ids=decoder_input).logits.float()
            logits = logits[:, 1 : 1 + len(generated), :]
            log_probs = (biased_logits(logits, beta) / TEMPERATURE).log_softmax(dim=-1)
            return float(log_probs.gather(-1, labels.unsqueeze(-1)).sum().item())

    conditions = []
    for beta in BETAS:
        rows = []
        hypotheses = []
        token_lengths = []
        logprob_differences = []
        for i, (source, reference) in enumerate(zip(sources, references)):
            generated, sampled_lp = generate_one(source, beta, SEED + i)
            rescored_lp = score_path(source, generated, beta)
            visible = tokenizer.decode(generated, skip_special_tokens=True)
            hypotheses.append(visible)
            token_lengths.append(len(generated) - int(bool(generated and generated[-1] == eos)))
            logprob_differences.append(abs(sampled_lp - rescored_lp))
            rows.append(
                {
                    "source": source,
                    "reference": reference,
                    "hypothesis": visible,
                    "sampled_token_ids": generated,
                    "sampled_logprob": sampled_lp,
                    "rescored_logprob": rescored_lp,
                    "token_count_including_sampled_eos": len(generated),
                }
            )
        conditions.append(
            {
                "beta": beta,
                "mean_generated_token_count_excluding_eos": sum(token_lengths) / len(rows),
                "mean_hypothesis_characters": sum(len(row["hypothesis"]) for row in rows) / len(rows),
                "mean_hypothesis_words": sum(len(row["hypothesis"].split()) for row in rows) / len(rows),
                "mean_generated_tokens_per_hypothesis_word": sum(
                    tokens / max(1, len(row["hypothesis"].split()))
                    for tokens, row in zip(token_lengths, rows)
                ) / len(rows),
                "sampled_eos_rate": sum(
                    bool(row["sampled_token_ids"] and row["sampled_token_ids"][-1] == eos)
                    for row in rows
                ) / len(rows),
                "max_length_without_eos_count": sum(
                    len(row["sampled_token_ids"]) == MAX_NEW_TOKENS
                    and (not row["sampled_token_ids"] or row["sampled_token_ids"][-1] != eos)
                    for row in rows
                ),
                "mean_abs_sampling_vs_rescore_logprob_error": sum(logprob_differences) / len(rows),
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
        "policy_transform": f"softmax((logits + beta * (1 - visible_piece_unicode_codepoint_length)) / {TEMPERATURE}); normalizer recomputed per step; special tokens unshifted",
        "interpretation": "Valid transformed behavior policy; changes both string and token-path distributions, not a fertility-only intervention.",
        "conditions": conditions,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "conditions"}, indent=2))
    for condition in conditions:
        print({k: v for k, v in condition.items() if k != "rows"})


if __name__ == "__main__":
    main()
