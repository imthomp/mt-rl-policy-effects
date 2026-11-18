"""Prepare a blinded human evaluation of GRPO outputs (designed 2026-11-11; redesigned 2026-11-18).

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
SEED, N, N_ATTENTION = 20261118, 60, 10
A = ROOT / "artifacts"
S = A / "grpo_scale"
V2 = A / "grpo_v2"
# Redesigned 2026-11-18 around the paper's headline experiments (previous, never-sent sheets
# from the corrected 1.5B grid are archived in artifacts/human_eval/archive_2026-11-12/).
#   primary:   7B safe-called rewards (policy scale) and MetricX large/XL/XXL (reward scale)
#   secondary: corrected 1.5B en-es grid
# "attention" systems appear on N_ATTENTION fixed items only: their outputs are copies that the
# automatic indicators already flag, so they serve as attention checks, not as evidence.
SHEETS = {
    "policy7b_en_de": {"lp": "en_de", "role": "primary", "systems": {
        "initial_7b": (S / "A_7b_en_de_cometkiwi_22_seed0", 0),
        "cometkiwi_22": (S / "A_7b_en_de_cometkiwi_22_seed0", 500),
        "cometkiwi_23_xl": (S / "A_7b_en_de_cometkiwi_23_xl_seed0", 500),
        "metricx_24_qe": (S / "A_7b_en_de_metricx_24_qe_seed0", 500),
        "xcomet_xl_qe_no_copy": (S / "A_7b_en_de_xcomet_xl_qe_seed0", 500)}, "attention": {}},
    "policy7b_en_zh": {"lp": "en_zh", "role": "primary", "systems": {
        "initial_7b": (S / "A_7b_en_zh_cometkiwi_22_seed0", 0),
        "cometkiwi_22": (S / "A_7b_en_zh_cometkiwi_22_seed0", 500),
        "cometkiwi_23_xl": (S / "A_7b_en_zh_cometkiwi_23_xl_seed0", 500),
        "metricx_24_qe": (S / "A_7b_en_zh_metricx_24_qe_seed0", 500)},
        "attention": {"xcomet_xl_qe_copy": (S / "A_7b_en_zh_xcomet_xl_qe_seed0", 500)}},
    "rewardscale_en_de": {"lp": "en_de", "role": "primary", "systems": {
        "initial_1p5b": (S / "B_1p5b_en_de_metricx_24_xl_qe_seed0", 0),
        "metricx_24_large": (V2 / "rule_test_en_de/metricx_24_qe_seed0", 500),
        "metricx_24_xl": (S / "B_1p5b_en_de_metricx_24_xl_qe_seed0", 500),
        "metricx_24_xxl": (S / "B_1p5b_en_de_metricx_24_xxl_qe_seed0", 500)}, "attention": {}},
    "rewardscale_en_zh": {"lp": "en_zh", "role": "primary", "systems": {
        "initial_1p5b": (S / "B_1p5b_en_zh_metricx_24_xl_qe_seed0", 0),
        "metricx_24_large": (V2 / "rule_test_en_zh/metricx_24_qe_seed0", 500),
        "metricx_24_xl": (S / "B_1p5b_en_zh_metricx_24_xl_qe_seed0", 500),
        "metricx_24_xxl": (S / "B_1p5b_en_zh_metricx_24_xxl_qe_seed0", 500)}, "attention": {}},
    "grid1p5b_en_es": {"lp": "en_es", "role": "secondary", "systems": {
        "initial_1p5b": (V2 / "pilot/cometkiwi_23_xl", 0),
        "metricx_24_qe": (V2 / "rule_test_en_es/metricx_24_qe_seed0", 500),
        "cometkiwi_22": (V2 / "pilot/cometkiwi_22", 500),
        "cometkiwi_23_xl": (V2 / "pilot/cometkiwi_23_xl", 500),
        "xcomet_xl_qe_guarded": (V2 / "replication/xcomet_xl_qe_guarded_seed0", 500)}, "attention": {}},
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

    manifest = {"design": "2026-11-18 headline-aligned", "seed": SEED, "items_per_sheet": N,
                "attention_items_per_sheet": N_ATTENTION, "sheets": {}}
    for name, cfg in SHEETS.items():
        lp = cfg["lp"]
        segs, sources = eval_sources(load_protocol(PROTO[lp]))
        src = dict(zip(segs, sources))
        outs = {k: outputs(d, step) for k, (d, step) in cfg["systems"].items()}
        att = {k: outputs(d, step) for k, (d, step) in cfg["attention"].items()}
        for k, o in {**outs, **att}.items():
            if set(o) != set(segs):
                raise AssertionError(f"{name}/{k}: outputs do not cover the eval set one-to-one")
        eligible = [s for s in segs if 5 <= len(src[s].split()) <= 80]
        rng = random.Random(f"{SEED}-{name}")
        chosen = rng.sample(eligible, N)
        attention_items = set(rng.sample(range(N), N_ATTENTION)) if att else set()
        rows, key = [], {}
        for n, seg in enumerate(chosen):
            uniq: dict[str, list[str]] = {}
            for k, o in outs.items():
                uniq.setdefault(o[seg].strip(), []).append(k)
            if n in attention_items:
                for k, o in att.items():
                    uniq.setdefault(o[seg].strip(), []).append(k)
            cands = list(uniq.items())
            rng.shuffle(cands)
            item = f"{name}-{n + 1:03d}"
            key[item] = {"segment_key": seg, "attention_item": n in attention_items, "candidates": {}}
            for j, (text, names) in enumerate(cands):
                label = "ABCDEFGH"[j]
                key[item]["candidates"][label] = names
                rows.append({"item": item, "candidate": label, "source": src[seg], "translation": text,
                             "score_0_100": "", "wrong_language": "", "untranslated": "",
                             "refusal_or_not_translation": "", "extra_text": "", "comment": ""})
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        with (d / "sheet.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (d / "key.json").write_text(json.dumps(key, indent=2, sort_keys=True) + "\n")
        manifest["sheets"][name] = {
            "lp": lp, "target_lang": TARGET[lp], "role": cfg["role"],
            "systems": {k: [str(v[0].relative_to(ROOT)), v[1]] for k, v in cfg["systems"].items()},
            "attention_systems": {k: [str(v[0].relative_to(ROOT)), v[1]] for k, v in cfg["attention"].items()},
            "items": N, "rows": len(rows), "mean_unique_candidates": len(rows) / N,
            "sheet_sha256": hashlib.sha256((d / "sheet.csv").read_bytes()).hexdigest(),
            "key_sha256": hashlib.sha256((d / "key.json").read_bytes()).hexdigest()}
        print(f"{name:20s} {cfg['role']:9s} {len(rows)} rows, {len(rows) / N:.2f} unique candidates/item")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    ssot = ROOT / "results/paper_stats.json"
    root = json.loads(ssot.read_text())
    root["human_eval_manifest"] = manifest
    tmp = ssot.with_suffix(".tmp")
    tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n")
    tmp.replace(ssot)


if __name__ == "__main__":
    main()
