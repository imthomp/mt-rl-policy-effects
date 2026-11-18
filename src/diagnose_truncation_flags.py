"""Exploratory, post hoc (2026-11-18): why safe truncation calls failed in the generalization study.
Counts outputs containing Han characters (code-switching into Chinese) at step 0 vs the final step for
the 7B en-ru runs, and lists every failed safe cell. Merges SSOT key truncation_flag_diagnostic."""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r"[一-鿿]")


def main() -> None:
    rows = {}
    for f in sorted(glob.glob(str(ROOT / "artifacts/grpo_gen/D_7b_en_ru_*/eval_outputs.jsonl"))):
        by = {}
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            by.setdefault(r["step"], []).append(r["output"])
        rows[Path(f).parent.name] = {"han_initial": sum(bool(HAN.search(x)) for x in by[min(by)]),
                                     "han_final": sum(bool(HAN.search(x)) for x in by[max(by)]), "n": len(by[max(by)])}
    gen = json.loads((ROOT / "results/grpo_generalization_result.json").read_text())
    failed = [c for c in gen["failures"] if c["prediction"] == "safe"]
    out = {"status": "exploratory_post_hoc", "en_ru_han_outputs": rows,
           "failed_safe_cells": len(failed), "failed_safe_types": sorted({c["type"] for c in failed})}
    ssot_p = ROOT / "results/paper_stats.json"
    ssot = json.loads(ssot_p.read_text())
    ssot["truncation_flag_diagnostic"] = out
    tmp = ssot_p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ssot, indent=2, sort_keys=True) + "\n")
    tmp.replace(ssot_p)
    print(out["failed_safe_cells"], out["failed_safe_types"])


if __name__ == "__main__":
    main()
