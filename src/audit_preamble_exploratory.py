"""EXPLORATORY, post hoc preamble audit (2026-11-10).

Some GRPO policies learn to prepend a short header ("Hier ist die Übersetzung ins
Deutsche:", "Translation:", "翻译如下：") before the translation. For every run this
counts preambles and trailing commentary at step 0 and at the final step. For runs
with >= MIN_COUNT preambles at the final step, it strips them and rescores the final
outputs with the run's own reward and with reference-based COMET-22, so we can see
how much of the reward gain the header carries.

Preamble: the first line has <= 8 whitespace words (<= 20 characters if it has no
spaces), ends in ':' or '：', and is followed by more text. Trailing commentary
(German): a final paragraph starting with Note/Hinweis/Anmerkung/Bitte beachten/
Beachten Sie/Allerdings. Only the preamble is stripped for rescoring.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_grpo_reference import GROUPS, mtme, norm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/preamble_audit_exploratory.json"
SSOT = ROOT / "results/paper_stats.json"
MIN_COUNT = 10
COMMENTARY = re.compile(r"^(Note|Hinweis|Anmerkung|Bitte beachten|Beachten Sie|Allerdings)\b")
PROTO = {"en_de": "ende", "en_es": "pilot", "en_zh": "enzh"}
LANG = {"en_de": "German", "en_es": "Spanish", "en_zh": "Chinese"}


def preamble(text: str) -> str | None:
    parts = text.strip().split("\n", 1)
    if len(parts) < 2 or not parts[1].strip():
        return None
    head = parts[0].strip()
    short = len(head.split()) <= 8 if " " in head else len(head) <= 20
    return head if short and head.endswith((":", "：")) else None


def strip(text: str) -> str:
    return text.strip().split("\n", 1)[1].strip() if preamble(text) else text


def commentary(text: str) -> bool:
    paras = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return len(paras) > 1 and bool(COMMENTARY.match(paras[-1]))


def main() -> None:
    from eval_grpo_pilot import eval_sources
    from train_grpo_pilot import load_protocol, make_scorer

    src_of, ref_of = {}, {}
    for lp, proto in PROTO.items():
        _, refs = mtme(lp)
        segs, sources = eval_sources(load_protocol(proto))
        for seg, s in zip(segs, sources):
            i = int(seg.split("|")[2]) - 1
            src_of[(lp, seg)], ref_of[(lp, seg)] = s, refs[i]

    runs = {}
    for group, lp in GROUPS.items():
        for p in sorted(glob.glob(str(ROOT / "artifacts" / group / "*/eval_outputs.jsonl"))):
            by = defaultdict(list)
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                by[r["step"]].append(r)
            runs[f"{group}/{Path(p).parent.name}"] = (lp, by)

    counts, heads = {}, {}
    for name, (lp, by) in runs.items():
        first, last = min(by), max(by)
        counts[name] = {"lp": lp, "final_step": last,
                        "preamble_initial": sum(preamble(r["output"]) is not None for r in by[first]),
                        "preamble_final": sum(preamble(r["output"]) is not None for r in by[last]),
                        "commentary_final": sum(commentary(r["output"]) for r in by[last]),
                        "n": len(by[last])}
        h = defaultdict(int)
        for r in by[last]:
            if (x := preamble(r["output"])):
                h[x] += 1
        heads[name] = dict(sorted(h.items(), key=lambda kv: -kv[1])[:3])

    targets = [n for n, c in counts.items() if c["preamble_final"] >= MIN_COUNT]
    print("runs with preambles:", {n: counts[n]["preamble_final"] for n in targets}, flush=True)
    rescored = {}
    by_reward = defaultdict(list)
    for n in targets:
        by_reward[n.split("/")[-1].split("_seed")[0].replace("_langid_steps1000", "").replace("_steps1000", "")
                   .replace("_guarded", "")].append(n)

    def score_sets(scorer, n):
        lp, by = runs[n]
        rows = by[max(by)]
        srcs = [src_of[(lp, r["segment_key"])] for r in rows]
        raw = np.asarray(scorer(srcs, [r["output"] for r in rows]))
        stp = np.asarray(scorer(srcs, [strip(r["output"]) for r in rows]))
        mask = np.array([preamble(r["output"]) is not None for r in rows])
        return raw, stp, mask

    for reward, names in by_reward.items():
        scorer = make_scorer(reward)
        for n in names:
            raw, stp, mask = score_sets(scorer, n)
            lp, by = runs[n]
            init = float(np.mean(scorer([src_of[(lp, r["segment_key"])] for r in by[min(by)]],
                                        [r["output"] for r in by[min(by)]])))
            rescored[n] = {"reward": reward, "own_initial": init, "own_final_raw": float(raw.mean()),
                           "own_final_stripped": float(stp.mean()),
                           "own_gain_raw": float(raw.mean()) - init,
                           "own_gain_stripped": float(stp.mean()) - init,
                           "affected_mean_raw": float(raw[mask].mean()),
                           "affected_mean_stripped": float(stp[mask].mean())}
            print(n, json.dumps({k: round(v, 3) for k, v in rescored[n].items() if k != "reward"}), flush=True)
        del scorer
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    from comet import load_from_checkpoint
    ckpt = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Unbabel--wmt22-comet-da"
                         / "snapshots/*/checkpoints/model.ckpt"))
    comet = load_from_checkpoint(ckpt[0])
    for n in targets:
        lp, by = runs[n]
        rows = by[max(by)]
        data = lambda f: [{"src": src_of[(lp, r["segment_key"])], "mt": f(r["output"]),
                           "ref": ref_of[(lp, r["segment_key"])]} for r in rows]
        raw = np.asarray(comet.predict(data(lambda o: o), batch_size=64, gpus=1, progress_bar=False).scores)
        stp = np.asarray(comet.predict(data(strip), batch_size=64, gpus=1, progress_bar=False).scores)
        rescored[n].update({"comet22_final_raw": float(raw.mean()), "comet22_final_stripped": float(stp.mean())})

    result = {"status": "exploratory_post_hoc", "min_count": MIN_COUNT, "counts": counts,
              "top_preambles": heads, "rescored": rescored}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    root = json.loads(SSOT.read_text(encoding="utf-8"))
    root["preamble_audit_exploratory"] = result
    tmp = SSOT.with_suffix(SSOT.suffix + ".tmp")
    tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(SSOT)


if __name__ == "__main__":
    main()
