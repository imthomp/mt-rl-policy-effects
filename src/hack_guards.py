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
