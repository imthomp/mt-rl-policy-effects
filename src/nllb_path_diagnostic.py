"""Small calibration diagnostic for NLLB-200 and alternative target token paths.

Scores fixed reference strings under the ordinary NLLB decoder policy and
compares canonical SentencePiece paths with sampled BPE-dropout paths. This is
not an RL intervention: sampled paths are rescored after the string is fixed.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rights_clearance import require_rights_clearance


MODEL = Path(os.environ["NLLB_MODEL_DIR"])
SOURCE = Path(os.environ["CALIBRATION_SOURCE"])
REFERENCE = Path(os.environ["CALIBRATION_REFERENCE"])
OUTPUT = Path(os.environ["DIAGNOSTIC_OUTPUT"])
COUNT = int(os.environ.get("CALIBRATION_COUNT", "48"))
SEED = int(os.environ.get("CALIBRATION_SEED", "104"))


def chrf_proxy(hypotheses: list[str], references: list[str]) -> float:
    """Dependency-free character n-gram F-score proxy, order 1..6."""
    import collections

    scores: list[float] = []
    for hyp, ref in zip(hypotheses, references):
        per_order = []
        for n in range(1, 7):
            h = collections.Counter(hyp[i : i + n] for i in range(max(0, len(hyp) - n + 1)))
            r = collections.Counter(ref[i : i + n] for i in range(max(0, len(ref) - n + 1)))
            overlap = sum((h & r).values())
            precision = overlap / max(1, sum(h.values()))
            recall = overlap / max(1, sum(r.values()))
            per_order.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
        scores.append(sum(per_order) / len(per_order))
    return 100.0 * sum(scores) / max(1, len(scores))


def main() -> None:
    require_rights_clearance()
    sources = SOURCE.read_text(encoding="utf-8").splitlines()
    references = REFERENCE.read_text(encoding="utf-8").splitlines()
    pairs = [(s, r) for s, r in zip(sources, references) if s.strip() and r.strip()]
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    pairs = pairs[:COUNT]
    sources, references = map(list, zip(*pairs))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "deu_Latn"
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if device.type != "cuda":
        raise RuntimeError("This diagnostic is intended to run as a scheduled GPU job")

    batches = [sources[i : i + 8] for i in range(0, len(sources), 8)]
    generations: list[str] = []
    for batch in batches:
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            out = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("deu_Latn"),
                max_new_tokens=160,
                num_beams=4,
            )
        generations.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    sp_path = MODEL / "sentencepiece.bpe.model"
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))
    forced_id = tokenizer.convert_tokens_to_ids("deu_Latn")
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    def path_score(source: str, target_ids: list[int]) -> float:
        src = tokenizer(source, return_tensors="pt", truncation=True, max_length=256)
        src = {k: v.to(device) for k, v in src.items()}
        labels = torch.tensor([target_ids + [eos_id]], device=device)
        decoder_input = torch.tensor([[eos_id, forced_id] + target_ids], device=device)
        with torch.inference_mode():
            logits = model(
                **src,
                decoder_input_ids=decoder_input,
                use_cache=False,
            ).logits
            # Position 1 predicts the first non-forced target piece.
            log_probs = logits[:, 1 : 1 + labels.shape[1], :].log_softmax(-1)
            token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return float(token_log_probs.sum().item())

    records = []
    canonical_covered = 0
    alternate_covered = 0
    canonical_lengths = []
    sampled_lengths = []
    canonical_logps = []
    sampled_logps = []
    for source, reference, hypothesis in zip(sources, references, generations):
        canon = sp.encode(reference, out_type=int)
        if sp.decode(canon) != reference:
            # SentencePiece's unknown token destroys literal punctuation/characters.
            # Such examples cannot support a same-string path comparison.
            continue
        canonical_covered += 1
        alt = sp.SampleEncodeAsIds(reference, 64, 0.2)
        if sp.decode(alt) != reference:
            continue
        alternate_covered += 1
        canonical_lengths.append(len(canon))
        sampled_lengths.append(len(alt))
        c_lp = path_score(source, canon)
        a_lp = path_score(source, alt)
        canonical_logps.append(c_lp / max(1, len(canon)))
        sampled_logps.append(a_lp / max(1, len(alt)))
        records.append(
            {
                "source": source,
                "reference": reference,
                "hypothesis": hypothesis,
                "canonical_target_tokens": len(canon),
                "sampled_target_tokens": len(alt),
                "canonical_path_logprob": c_lp,
                "sampled_path_logprob": a_lp,
            }
        )

    result = {
        "model": "facebook/nllb-200-distilled-600M",
        "direction": "eng_Latn->deu_Latn",
        "calibration_count": len(pairs),
        "source_dataset": "cached WMT23 en-de test, used only as calibration here",
        "heldout_reservation": "cached WMT24 en-de remains untouched",
        "device": torch.cuda.get_device_name(device),
        "baseline_chrf_proxy_1_to_6": chrf_proxy(generations, references),
        "canonical_exact_coverage": canonical_covered / len(pairs),
        "alternate_exact_coverage": alternate_covered / len(pairs),
        "alternate_exact_given_canonical_exact": alternate_covered / max(1, canonical_covered),
        "mean_canonical_tokens": sum(canonical_lengths) / len(pairs),
        "mean_sampled_tokens": sum(sampled_lengths) / len(pairs),
        "mean_canonical_logprob_per_token": sum(canonical_logps) / len(pairs),
        "mean_sampled_logprob_per_token": sum(sampled_logps) / len(pairs),
        "interpretation": "Fixed-string likelihood diagnostic only; not on-policy path sampling.",
        "examples": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
