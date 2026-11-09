"""Deterministic degeneration guards shared by judge v2 and the GRPO pilot.

Definitions match results/grpo_pilot_protocol.json ``hack_indicators``.  For
targets written without spaces (Chinese, Japanese), repetition uses character
6-grams because whitespace 4-grams do not exist there.
"""

from __future__ import annotations

from collections import Counter

from sacrebleu.metrics import CHRF

CHARACTER_LANGUAGES = {"Chinese", "Japanese"}
UNTRANSLATED_CHRF = 60.0
TRUNCATION_RATIO = 0.5
REPEAT_COUNT = 3
_CHRF = CHRF()


def untranslated(output: str, source: str) -> bool:
    return _CHRF.sentence_score(output, [source]).score >= UNTRANSLATED_CHRF


def truncated(output: str, source: str) -> bool:
    return len(output.strip()) / max(1, len(source.strip())) < TRUNCATION_RATIO


def repetitive(output: str, target_lang: str = "Spanish") -> bool:
    if target_lang in CHARACTER_LANGUAGES:
        text = "".join(output.split())
        grams = [text[i:i + 6] for i in range(len(text) - 5)]
    else:
        words = output.split()
        grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return bool(grams) and max(Counter(grams).values()) >= REPEAT_COUNT


def flags(output: str, source: str, target_lang: str = "Spanish") -> dict[str, bool]:
    return {
        "untranslated": untranslated(output, source),
        "truncated": truncated(output, source),
        "repetitive": repetitive(output, target_lang),
    }


# Off-target-language guard (en-de amendment 2, 2026-11-09). Kept separate from
# ``flags`` so frozen indicator definitions and the existing guard are unchanged.
GLOTLID_CODES = {"German": "deu_Latn", "Spanish": "spa_Latn", "English": "eng_Latn", "Chinese": "cmn_Hani"}
OFF_TARGET_MIN_WORDS = 4
_GLOTLID = None


def _glotlid():
    global _GLOTLID
    if _GLOTLID is None:
        import glob
        from pathlib import Path

        import fasttext

        paths = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--cis-lmu--glotlid"
                              / "snapshots/*/model.bin"))
        if len(paths) != 1:
            raise ValueError(f"Expected one cached GlotLID model, found {len(paths)}")
        fasttext.FastText.eprint = lambda *_: None  # silence load_model deprecation notice
        _GLOTLID = fasttext.load_model(paths[0])
    return _GLOTLID


def off_target(output: str, target_lang: str) -> bool:
    """True if GlotLID's top label is not the target language (outputs >= 4 words only)."""
    text = " ".join(output.split())
    if len(text.split()) < OFF_TARGET_MIN_WORDS:
        return False
    label = _glotlid().predict(text)[0][0].removeprefix("__label__")
    return label != GLOTLID_CODES[target_lang]


# Refusal indicator (exploratory since 2026-11-07; Chinese pattern added 2026-11-09
# before any en-zh run). English pattern identical to the original scorecard regex.
import re as _re

REFUSAL_EN = _re.compile(r"(can'?t|cannot|unable to) (assist|help|comply)|^(i'?m )?sorry|as an ai", _re.I)
REFUSAL_ZH = _re.compile(r"^(很)?抱歉|^对不起|我(无法|不能)(翻译|帮助|协助|提供)|作为一个?(AI|人工智能)")


def refusal(output: str) -> bool:
    text = output.strip()
    return bool(REFUSAL_EN.search(text) or REFUSAL_ZH.search(text))
