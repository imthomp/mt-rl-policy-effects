"""Freeze the en-de GRPO protocol.

Eval sources are read outcome-blind from the WMT24 en-de MQM release: only the
doc, docSegId, globalSegId and source columns are accessed.  MQM category,
severity, rater and target columns are never read, so the pool's human
outcomes stay sealed for later confirmation use.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from score_mqm_confirmation_candidates import clean_text

ROOT = Path(__file__).resolve().parents[1]
MQM = Path.home() / "projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation/generalMT2024/mqm_generalMT2024_ende.tsv"
PILOT = ROOT / "results/grpo_pilot_protocol.json"
PILOT_SHA256 = "61d89eaab72a7a9d10fd8ec2cedb28b752293faae354d4a6d5d4f1c69323343c"
RULE_SHA256 = "0dc190f79589987af21f2a8476787e68260e25db26b21bd02e44c4837437ca4c"
OUTPUT = ROOT / "results/grpo_ende_protocol.json"
EVAL_SEGMENTS = 400
SEED = 20261105


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if hashlib.sha256(PILOT.read_bytes()).hexdigest() != PILOT_SHA256:
        raise ValueError("pilot protocol hash mismatch")
    pilot = json.loads(PILOT.read_text())
    sources: dict[str, set[str]] = {}
    accessed = ("doc", "docSegId", "globalSegId", "source")
    with MQM.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        idx = {k: header.index(k) for k in accessed}
        for row in reader:
            key = "|".join(row[idx[k]] for k in ("doc", "docSegId", "globalSegId"))
            sources.setdefault(key, set()).add(clean_text(row[idx["source"]]).strip())
    clean = sorted(k for k, v in sources.items()
                   if len(v) == 1 and 5 <= len(next(iter(v)).split()) <= 120)
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(len(clean), size=EVAL_SEGMENTS, replace=False).tolist())
    segments = [clean[i] for i in chosen]
    eval_sources = [next(iter(sources[s])) for s in segments]

    protocol = {
        "status": "frozen_before_any_en_de_grpo_run",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "inherits": {"path": "results/grpo_pilot_protocol.json", "sha256": PILOT_SHA256,
                     "unchanged": "policy model, training prompts, GRPO hyperparameters, checkpoints, indicators"},
        "direction": "en-de",
        "target_lang": "German",
        "policy_model": pilot["policy_model"],
        "prompt": "Translate the following English text into German. Output only the translation.\n\nEnglish: {source}\nGerman:",
        "train_prompts": pilot["train_prompts"],
        "grpo": pilot["grpo"],
        "hack_indicators": pilot["hack_indicators"],
        "eval_set": {
            "pool": "wmt24_en_de_mqm (sources only; outcomes sealed)",
            "source_file_sha256": hashlib.sha256(MQM.read_bytes()).hexdigest(),
            "columns_accessed": list(accessed),
            "markup_disclosure": ("The source column embeds MQM span markup (<v>...</v>), which is "
                                  "outcome-correlated; it was stripped with clean_text and neither its "
                                  "presence nor its position was recorded or used."),
            "rule": "unambiguous-source segments with 5-120 words, 400 sampled with seed 20261105",
            "segments": segments,
            "sources": eval_sources,
            "count": len(segments),
            "sources_sha256": hashlib.sha256("\n".join(eval_sources).encode()).hexdigest(),
            "note": "WMT24 en-xx tasks share English sources, so some may also appear in the en-es eval set.",
        },
        "rewards": {**pilot["rewards"], "metricx_24_qe": "google/metricx-24-hybrid-large-v2p6-bfloat16 (QE, src/metricx_qe.py)"},
        "predictions": {"rule": "results/grpo_audit_rule.json", "rule_sha256": RULE_SHA256,
                        "note": "per-reward predictions follow from the frozen audit statistics; no en-de-specific predictions"},
        "seeds": [0],
    }
    OUTPUT.write_text(json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()} eval={len(segments)} clean={len(clean)}")


if __name__ == "__main__":
    main()
