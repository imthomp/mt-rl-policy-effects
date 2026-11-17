"""Freeze an en-xx GRPO protocol for a new target language (en-ja, en-ru; 2026-11-17).

Everything is inherited from the en-de protocol except the target language and prompt, as
for en-zh. Eval sources are the same 400 English segments (WMT24 en-xx tasks share sources);
no MQM file is read. MTME v2 wmt24 refA is used only by reference-based evaluation. Indicators
are those of the scale protocol (prospective fixes included).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENDE = ROOT / "results/grpo_ende_protocol.json"
ENDE_SHA256 = "a0bec8727d8bcb13f41074e51cc0e28086fa68c220bcf730c2d1e22052845779"
SCALE = ROOT / "results/grpo_scale_protocol.json"
SCALE_SHA256 = "f01a01915839f130b9ce70a0bb807b3cb1737bb29384a7627b78ef58ec072207"
LANGS = {"ja": "Japanese", "ru": "Russian"}


def main(code: str) -> None:
    lang = LANGS[code]
    out = ROOT / f"results/grpo_en{code}_protocol.json"
    if out.exists():
        raise FileExistsError(out)
    for p, d in ((ENDE, ENDE_SHA256), (SCALE, SCALE_SHA256)):
        if hashlib.sha256(p.read_bytes()).hexdigest() != d:
            raise ValueError(f"hash mismatch: {p}")
    ende, scale = json.loads(ENDE.read_text()), json.loads(SCALE.read_text())
    protocol = {k: ende[k] for k in ("policy_model", "train_prompts", "grpo", "rewards")}
    protocol.update({
        "status": f"frozen_before_any_en_{code}_audit_or_grpo_run",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "inherits": {"path": "results/grpo_ende_protocol.json", "sha256": ENDE_SHA256,
                     "unchanged": "policy model, English training prompts, GRPO hyperparameters, checkpoints, rewards, eval sources"},
        "direction": f"en-{code}",
        "target_lang": lang,
        "prompt": f"Translate the following English text into {lang}. Output only the translation.\n\nEnglish: {{source}}\n{lang}:",
        "eval_set": {k: ende["eval_set"][k] for k in ("segments", "sources", "count", "sources_sha256")}
                    | {"pool": "WMT24 English sources (same 400 as en-de/en-zh; no MQM file read)",
                       "references": f"MTME v2 wmt24 references/en-{code}.refA.txt (reference-based eval only)"},
        "hack_indicators": scale["indicators"],
        "seeds": [0, 1, 2],
    })
    out.write_text(json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"{out.name} sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main(sys.argv[1])
