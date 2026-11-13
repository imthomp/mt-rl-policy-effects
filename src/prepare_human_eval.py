"""Prepare a blinded human evaluation of GRPO outputs (designed 2026-11-11).

Per language pair: 100 eval segments (seed 20261111; sources of 5-80 words), with
candidates from the initial policy and the final checkpoints of the runs below
(seed 0 unless noted). Identical outputs are merged, so each unique candidate is
scored once. Candidate order is shuffled per item. Annotators see only
artifacts/human_eval/<pair>/sheet.csv. The system key goes to key.json; its sha256 is
recorded in manifest.json so the key can be shown to be unchanged after annotation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/human_eval"
SEED, N = 20261111, 100
A = ROOT / "artifacts"
SYSTEMS = {
    "en_es": {"initial": (A / "grpo_v2/pilot/cometkiwi_23_xl", 0),
              "metricx_24_qe": (A / "grpo_v2/rule_test_en_es/metricx_24_qe_seed0", 500),
              "cometkiwi_22": (A / "grpo_v2/pilot/cometkiwi_22", 500),
              "cometkiwi_23_xl": (A / "grpo_v2/pilot/cometkiwi_23_xl", 500),
              "xcomet_xl_qe_guarded": (A / "grpo_v2/replication/xcomet_xl_qe_guarded_seed0", 500)},
    "en_de": {"initial": (A / "grpo_v2/rule_test_en_de/cometkiwi_22_seed0", 0),
              "metricx_24_qe": (A / "grpo_v2/rule_test_en_de/metricx_24_qe_seed0", 500),
              "cometkiwi_22": (A / "grpo_v2/rule_test_en_de/cometkiwi_22_seed0", 500),
              "cometkiwi_23_xl_seed2_unguarded": (A / "grpo_v2/rule_test_en_de/cometkiwi_23_xl_seed2", 500),
              "cometkiwi_23_xl_langid_guard": (A / "grpo_v2/ende_followup/cometkiwi_23_xl_langid_steps1000_seed0", 1000)},
    "en_zh": {"initial": (A / "grpo_v2/rule_test_en_zh/cometkiwi_22_seed0", 0),
              "metricx_24_qe": (A / "grpo_v2/rule_test_en_zh/metricx_24_qe_seed0", 500),
              "cometkiwi_22": (A / "grpo_v2/rule_test_en_zh/cometkiwi_22_seed0", 500),
              "cometkiwi_23_xl": (A / "grpo_v2/rule_test_en_zh/cometkiwi_23_xl_seed0", 500)},
}
PROTO = {"en_de": "ende", "en_es": "pilot", "en_zh": "enzh"}
TARGET = {"en_de": "German", "en_es": "Spanish", "en_zh": "Chinese"}


def outputs(run_dir: Path, step: int) -> dict[str, str]:
    out = {}
    for line in (run_dir / "eval_outputs.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["step"] == step:
            out[r["segment_key"]] = r["output"]
    if not out:
        raise ValueError(f"no step {step} in {run_dir}")
    return out


def main() -> None:
    from eval_grpo_pilot import eval_sources
    from train_grpo_pilot import load_protocol

    manifest = {"seed": SEED, "n_per_pair": N, "pairs": {}}
    for lp, systems in SYSTEMS.items():
        segs, sources = eval_sources(load_protocol(PROTO[lp]))
        src = dict(zip(segs, sources))
        outs = {name: outputs(d, step) for name, (d, step) in systems.items()}
        eligible = [s for s in segs if 5 <= len(src[s].split()) <= 80]
        rng = random.Random(f"{SEED}-{lp}")
        chosen = rng.sample(eligible, N)
        rows, key = [], {}
        for n, seg in enumerate(chosen, 1):
            uniq: dict[str, list[str]] = {}
            for name, o in outs.items():
                uniq.setdefault(o[seg].strip(), []).append(name)
            cands = list(uniq.items())
            rng.shuffle(cands)
            item = f"{lp}-{n:03d}"
            key[item] = {"segment_key": seg, "candidates": {}}
            for j, (text, names) in enumerate(cands):
                label = "ABCDEFG"[j]
                key[item]["candidates"][label] = names
                rows.append({"item": item, "candidate": label, "source": src[seg], "translation": text,
                             "score_0_100": "", "wrong_language": "", "untranslated": "",
                             "refusal_or_not_translation": "", "extra_text": "", "comment": ""})
        d = OUT / lp
        d.mkdir(parents=True, exist_ok=True)
        with (d / "sheet.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (d / "key.json").write_text(json.dumps(key, indent=2, sort_keys=True) + "\n")
        manifest["pairs"][lp] = {
            "target_lang": TARGET[lp], "systems": {k: [str(v[0].relative_to(ROOT)), v[1]] for k, v in systems.items()},
            "items": N, "rows": len(rows), "mean_unique_candidates": len(rows) / N,
            "sheet_sha256": hashlib.sha256((d / "sheet.csv").read_bytes()).hexdigest(),
            "key_sha256": hashlib.sha256((d / "key.json").read_bytes()).hexdigest()}
        print(lp, manifest["pairs"][lp]["rows"], "rows,", round(len(rows) / N, 2), "unique candidates/item")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    ssot = ROOT / "results/paper_stats.json"
    root = json.loads(ssot.read_text())
    root["human_eval_manifest"] = manifest
    tmp = ssot.with_suffix(".tmp")
    tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n")
    tmp.replace(ssot)


if __name__ == "__main__":
    main()
