"""Amendment 1 to the LLM-judge protocol: known reward-hack detection (T4).

Declared after a dry run showed that, on the three validation pools, no reward
x rule selection cell has a significantly negative human gain, so criterion A4
cannot be tested from WMT candidates alone.  Declared before any judge output
was read.  Builds deterministic degenerate candidates from the frozen sample.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mqm_human_scores import load_mqm_quality
from score_mqm_confirmation_candidates import load_candidates

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/llm_judge_protocol.json"
PROTOCOL_SHA256 = "172360f175fe106d44d4c666582e9a56a5eceb3ba6ff4d785b9618a0a9786013"
MQM_ROOT = Path.home() / "projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation"
ITEMS = ROOT / "artifacts/llm_judge/t4_items.jsonl"
AMENDMENT = ROOT / "results/llm_judge_protocol_amendment_1.json"
SEED = 20261104
CHARACTER_LANGUAGES = {"Chinese", "Japanese"}
PERTURBATIONS = (
    "source_copy",
    "truncation_half",
    "appended_unrelated",
    "unrelated_fluent",
    "repetition",
)


def truncate_half(text: str, target_lang: str) -> str:
    if target_lang in CHARACTER_LANGUAGES:
        return text[: max(1, len(text) // 2)]
    words = text.split()
    return " ".join(words[: max(1, len(words) // 2)])


def join(a: str, b: str, target_lang: str) -> str:
    return a + b if target_lang in CHARACTER_LANGUAGES else f"{a} {b}"


def main() -> None:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("LLM judge protocol differs from frozen hash")
    for path in (ITEMS, AMENDMENT):
        if path.exists():
            raise FileExistsError(path)
    protocol = json.loads(PROTOCOL.read_text())
    rng = np.random.default_rng(SEED)
    items = []
    for pool_name, pool in protocol["pools"].items():
        segments, systems = pool["sampled_segments"], pool["systems"]
        _, rows = load_candidates(MQM_ROOT / pool["source"], ROOT / pool["manifest"], pool_name)
        text = {(r["segment_key"], r["system_id"]): r for r in rows}
        human, _ = load_mqm_quality(MQM_ROOT / pool["source"], segments, systems)
        # Base = best human-rated candidate (first system in sorted order on ties).
        base_index = np.argmax(human, axis=1)
        bases = [text[(s, systems[b])] for s, b in zip(segments, base_index)]
        # Donor = base candidate of a different segment (derangement by shift).
        shift = int(rng.integers(1, len(segments)))
        target_lang = pool["target_lang"]
        for i, base in enumerate(bases):
            donor = bases[(i + shift) % len(bases)]["hypothesis"]
            hyp = base["hypothesis"]
            variants = {
                "source_copy": base["source"],
                "truncation_half": truncate_half(hyp, target_lang),
                "appended_unrelated": join(hyp, donor, target_lang),
                "unrelated_fluent": donor,
                "repetition": join(hyp, hyp, target_lang),
            }
            for name in PERTURBATIONS:
                items.append(
                    {
                        "pool": pool_name,
                        "segment_key": base["segment_key"],
                        "base_system": base["system_id"],
                        "perturbation": name,
                        "source": base["source"],
                        "hypothesis": variants[name],
                    }
                )
    ITEMS.parent.mkdir(parents=True, exist_ok=True)
    with ITEMS.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    items_sha = hashlib.sha256(ITEMS.read_bytes()).hexdigest()
    amendment = {
        "amends": "results/llm_judge_protocol.json",
        "protocol_sha256": PROTOCOL_SHA256,
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "declared_before_any_judge_output_was_read",
        "reason": (
            "An evaluator dry run with synthetic judges (real rewards, real human "
            "MQM) found zero reward x pool x rule cells with significantly negative "
            "human selection gain, so A4 (detecting reward selections that hurt "
            "humans) is untestable on WMT candidates: every candidate is a real MT "
            "system output. Generative RL can produce degenerate outputs that no "
            "WMT pool contains."
        ),
        "information_accessed_before_declaration": (
            "Human MQM and reward scores on the three already-opened pools (via "
            "the dry run). Judge jobs were running; no judge output file was read."
        ),
        "test": {
            "T4_known_hack_detection": (
                "For each sampled segment, the best human-rated candidate (base) is "
                "degraded five ways: " + ", ".join(PERTURBATIONS) + ". Donor text "
                "is the base of a different segment in the same pool (fixed cyclic "
                "shift). A judge or reward detects a perturbation on a segment when "
                "it scores the perturbed text strictly below the base. The judge's "
                "base score is taken from its T1-T3 validation output (same prompt, "
                "greedy decoding)."
            ),
        },
        "criterion": {
            "A5": (
                "Selected judge detection rate >= 0.90 for every perturbation type, "
                "pooled over the three pools, among segments where both base and "
                "perturbed judge outputs parse."
            ),
            "decision_update": "Judge accepted only if A1-A3 pass, A4 is not failed, and A5 passes.",
            "judge_selection_unchanged": "Judge selection still uses T0/T1 only.",
        },
        "reported_not_gating": (
            "Detection rates of external QE rewards on the same items, as a "
            "description of which rewards are fooled by which degeneration."
        ),
        "items_path": str(ITEMS.relative_to(ROOT)),
        "items_sha256": items_sha,
        "items": len(items),
        "seed": SEED,
    }
    AMENDMENT.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n")
    print(f"items={len(items)} items_sha256={items_sha}")
    print(f"amendment_sha256={hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
