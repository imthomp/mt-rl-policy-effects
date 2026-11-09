"""Reference-based quality of every GRPO eval checkpoint (chrF and COMET-22 vs WMT24 refA).

Neither metric is a training reward: training rewards are QE-only (XCOMET-XL-QE,
CometKiwi-22, CometKiwi-23-XL, MetricX-24-QE). References come from
MT-Metrics-Eval v2 reference files, not from the MQM TSVs, so no MQM rating is
read (the en-de MQM outcomes stay sealed). Eval segments are aligned to MTME
lines by globalSegId and checked against the source text.

Scores are cached by (lang pair, source line, output) so reruns only score new
outputs. Writes artifacts/grpo_reference_eval.json, merges into paper_stats.
"""
from __future__ import annotations

import glob
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[1]
MTME = Path("/nobackup/archive/usr/it238/mtme/mt-metrics-eval-v2/wmt24")
CACHE = ROOT / "artifacts/grpo_reference_cache.jsonl"
OUTPUT = ROOT / "results/grpo_reference_eval.json"
SSOT = ROOT / "results/paper_stats.json"
GROUPS = {"grpo_pilot": "en_es", "grpo_replication": "en_es", "grpo_rule_tests/en_es": "en_es",
          "grpo_rule_tests/en_de": "en_de", "grpo_ende_followup": "en_de"}


def norm(t: str) -> str:
    return " ".join(unicodedata.normalize("NFC", t).split())


def mtme(lp: str):
    tag = lp.replace("_", "-")
    src = [l.rstrip("\n") for l in (MTME / f"sources/{tag}.txt").open(encoding="utf-8")]
    ref = [l.rstrip("\n") for l in (MTME / f"references/{tag}.refA.txt").open(encoding="utf-8")]
    assert len(src) == len(ref)
    return src, ref


def key(lp: str, line: int, output: str) -> str:
    return hashlib.sha256(f"{lp}\t{line}\t{output}".encode()).hexdigest()


def main() -> None:
    from eval_grpo_pilot import eval_sources
    from train_grpo_pilot import load_protocol

    data = {lp: mtme(lp) for lp in ("en_de", "en_es")}
    line_of = {}
    for lp, proto in (("en_de", "ende"), ("en_es", "pilot")):
        segs, sources = eval_sources(load_protocol(proto))
        src = data[lp][0]
        for seg, s in zip(segs, sources):
            i = int(seg.split("|")[2]) - 1
            if norm(src[i]) != norm(s):
                raise AssertionError(f"{lp} {seg}: MTME line {i} source mismatch")
            line_of[(lp, seg)] = i

    runs = {}
    for group, lp in GROUPS.items():
        for p in sorted(glob.glob(str(ROOT / "artifacts" / group / "*/eval_outputs.jsonl"))):
            runs[f"{group}/{Path(p).parent.name}"] = (lp, Path(p))

    cache = {}
    if CACHE.exists():
        for l in CACHE.open():
            r = json.loads(l)
            cache[r["k"]] = r["comet22"]
    todo = {}
    rows_by_run = {}
    for name, (lp, path) in runs.items():
        rows = [json.loads(l) for l in path.open()]
        rows_by_run[name] = rows
        for r in rows:
            i = line_of[(lp, r["segment_key"])]
            k = key(lp, i, r["output"])
            if k not in cache:
                todo[k] = {"src": data[lp][0][i], "mt": r["output"], "ref": data[lp][1][i]}
    print(f"{len(runs)} runs; {len(todo)} new (source, output) pairs to score; {len(cache)} cached", flush=True)
    if todo:
        from comet import load_from_checkpoint

        ckpt = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Unbabel--wmt22-comet-da"
                             / "snapshots/*/checkpoints/model.ckpt"))
        assert len(ckpt) == 1
        model = load_from_checkpoint(ckpt[0])
        keys = list(todo)
        scores = model.predict([todo[k] for k in keys], batch_size=64, gpus=1, progress_bar=False).scores
        with CACHE.open("a") as h:
            for k, s in zip(keys, scores):
                cache[k] = float(s)
                h.write(json.dumps({"k": k, "comet22": float(s)}) + "\n")

    chrf = CHRF()
    result = {"ci": "paired segment bootstrap (2000, seed 20261109) of final minus step-0 COMET-22",
              "metrics": "chrF (sacrebleu corpus, refA) and COMET-22 (Unbabel/wmt22-comet-da, mean, refA)",
              "reference_source": str(MTME), "runs": {}}
    for name, rows in rows_by_run.items():
        lp = runs[name][0]
        by = defaultdict(list)
        for r in rows:
            by[r["step"]].append(r)
        per = {}
        for step, rs in sorted(by.items()):
            idx = [line_of[(lp, r["segment_key"])] for r in rs]
            per[str(step)] = {
                "chrf": chrf.corpus_score([r["output"] for r in rs], [[data[lp][1][i] for i in idx]]).score,
                "comet22": float(np.mean([cache[key(lp, i, r["output"])] for r, i in zip(rs, idx)])),
                "n": len(rs),
            }
        first, last = min(by), max(by)
        a = {r["segment_key"]: cache[key(lp, line_of[(lp, r["segment_key"])], r["output"])] for r in by[first]}
        b = {r["segment_key"]: cache[key(lp, line_of[(lp, r["segment_key"])], r["output"])] for r in by[last]}
        d = np.array([b[k] - a[k] for k in sorted(a)])
        rng = np.random.default_rng(20261109)
        boot = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
        result["runs"][name] = {"lp": lp, "by_step": per,
                                "delta_comet22_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                                "delta_comet22": per[str(last)]["comet22"] - per[str(first)]["comet22"],
                                "delta_chrf": per[str(last)]["chrf"] - per[str(first)]["chrf"]}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    root = json.loads(SSOT.read_text(encoding="utf-8"))
    root["grpo_reference_eval"] = result
    tmp = SSOT.with_suffix(SSOT.suffix + ".tmp")
    tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(SSOT)
    for name, v in result["runs"].items():
        print(f"{name:60s} COMET22 {v['by_step']['0']['comet22']:.3f} -> {v['delta_comet22']:+.3f}   "
              f"[{v['delta_comet22_ci95'][0]:+.3f},{v['delta_comet22_ci95'][1]:+.3f}]   "
              f"chrF {v['by_step']['0']['chrf']:.1f} -> {v['delta_chrf']:+.1f}")


if __name__ == "__main__":
    main()
