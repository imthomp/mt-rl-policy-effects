"""Fair stripped comparison (exploratory, 2026-11-11): strip preambles at BOTH step 0 and the
final step (src/audit_preamble_exploratory.strip), then COMET-22 vs refA with a paired
segment bootstrap of final minus step 0. Runs: every run with >= 10 final-step preambles.
Uses the reference-eval score cache. Merges key "grpo_reference_eval_stripped" into the SSOT.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_preamble_exploratory import MIN_COUNT, preamble, strip
from eval_grpo_reference import CACHE, GROUPS, key, mtme

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/grpo_reference_eval_stripped.json"
SSOT = ROOT / "results/paper_stats.json"


def main() -> None:
    data = {lp: mtme(lp) for lp in ("en_de", "en_es", "en_zh")}
    cache = {}
    if CACHE.exists():
        for l in CACHE.open():
            r = json.loads(l)
            cache[r["k"]] = r["comet22"]
    runs = {}
    for group, lp in GROUPS.items():
        for p in sorted(glob.glob(str(ROOT / "artifacts" / group / "*/eval_outputs.jsonl"))):
            by = defaultdict(list)
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                by[r["step"]].append(r)
            if sum(preamble(r["output"]) is not None for r in by[max(by)]) >= MIN_COUNT:
                runs[f"{group}/{Path(p).parent.name}"] = (lp, by)
    todo = {}
    for lp, by in runs.values():
        for step in (min(by), max(by)):
            for r in by[step]:
                i = int(r["segment_key"].split("|")[2]) - 1
                for text in (r["output"], strip(r["output"])):
                    k = key(lp, i, text)
                    if k not in cache:
                        todo[k] = {"src": data[lp][0][i], "mt": text, "ref": data[lp][1][i]}
    print(f"{len(runs)} runs; {len(todo)} new pairs", flush=True)
    if todo:
        from comet import load_from_checkpoint
        ckpt = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Unbabel--wmt22-comet-da"
                             / "snapshots/*/checkpoints/model.ckpt"))
        model = load_from_checkpoint(ckpt[0])
        keys = list(todo)
        scores = model.predict([todo[k] for k in keys], batch_size=64, gpus=1, progress_bar=False).scores
        with CACHE.open("a") as h:
            for k, s in zip(keys, scores):
                cache[k] = float(s)
                h.write(json.dumps({"k": k, "comet22": float(s)}) + "\n")
    out = {"status": "exploratory_post_hoc",
           "method": "preambles stripped at step 0 and final; paired segment bootstrap (2000, seed 20261111)",
           "runs": {}}
    for name, (lp, by) in runs.items():
        res = {}
        for label, f in (("raw", lambda t: t), ("stripped", strip)):
            sc = {s: {r["segment_key"]: cache[key(lp, int(r["segment_key"].split("|")[2]) - 1, f(r["output"]))]
                      for r in by[s]} for s in (min(by), max(by))}
            a, b = sc[min(by)], sc[max(by)]
            d = np.array([b[k] - a[k] for k in sorted(a)])
            boot = d[np.random.default_rng(20261111).integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
            res[label] = {"initial": float(np.mean(list(a.values()))), "final": float(np.mean(list(b.values()))),
                          "delta": float(d.mean()),
                          "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}
        out["runs"][name] = {"lp": lp, **res}
        s = res["stripped"]
        print(f"{name:60s} raw {res['raw']['delta']:+.3f} | stripped {s['initial']:.3f}->{s['final']:.3f} "
              f"{s['delta']:+.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]")
    OUTPUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    root = json.loads(SSOT.read_text(encoding="utf-8"))
    root["grpo_reference_eval_stripped"] = out
    tmp = SSOT.with_suffix(SSOT.suffix + ".tmp")
    tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(SSOT)


if __name__ == "__main__":
    main()
