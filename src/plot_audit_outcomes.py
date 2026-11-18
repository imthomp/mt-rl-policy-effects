"""Figure: audit preference rate p (x) vs observed indicator change (y), one point per audit task.

A task is (study, policy, reward, direction, failure type); y is the mean final-minus-initial
indicator change over its seeds. Outcome = "correct" (every scored seed correct), "incorrect"
(any scored seed incorrect), or "no call" (abstention). Encoded by colour AND marker shape.
Palette: dataviz reference slots 1-2 (validated, --pairs all); abstentions in recessive grey.
Sources: SSOT grpo_scale and grpo_generalization (if scored); studies scored with the fixed indicators only.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SSOT = ROOT / "results/paper_stats.json"
OUT = ROOT / "paper/grpo_audit_acl/fig_audit_outcomes.pdf"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#fcfcfb"
STYLE = {"correct": ("#2a78d6", "o", "Call correct"),
         "incorrect": ("#eb6834", "X", "Call incorrect"),
         "none": (MUTED, "o", "Abstention (no call)")}


def tasks_from_cells(cells, study, key_fn, p_key, delta_key, pred_key):
    groups = defaultdict(list)
    for c in cells:
        groups[key_fn(c)].append(c)
    pts = []
    for key, cs in groups.items():
        preds = {c[pred_key] for c in cs}
        if len(preds) != 1:
            raise AssertionError(f"{study} {key}: predictions differ across seeds")
        pred = preds.pop()
        scored = [c["correct"] for c in cs if c["correct"] is not None]
        outcome = "none" if pred == "none" else ("correct" if scored and all(scored) else "incorrect")
        pts.append({"study": study, "task": list(key), "p": cs[0][p_key], "prediction": pred,
                    "delta": sum(c[delta_key] for c in cs) / len(cs), "seeds": len(cs), "outcome": outcome})
    return pts


def main() -> None:
    ssot = json.loads(SSOT.read_text())
    scale = ssot["grpo_scale"]
    pts = tasks_from_cells(scale["cells"], "scale",
                           lambda c: (scale["runs"][c["run"]]["policy_model"], c["reward"], c["protocol"], c["type"]),
                           "audit_p", "indicator_delta", "prediction")
    # The corrected 1.5B en-zh grid is excluded: it was scored with the original indicator definitions,
    # whose Chinese truncation saturates and whose repetition check fires on copied English. Only
    # studies scored with the prospectively fixed definitions are plotted.
    if "grpo_generalization" in ssot:
        g = ssot["grpo_generalization"]
        pts += tasks_from_cells(g["cells"], "generalization",
                                lambda c: (c["policy_model"], c["reward"], c["protocol"], c["type"]),
                                "audit_p", "indicator_delta", "prediction")

    fig, ax = plt.subplots(figsize=(3.3, 2.5), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for x, lab in ((0.05, "safe"), (0.50, "hack")):
        ax.axvline(x, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.axhline(0.10, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
    for outcome in ("none", "correct", "incorrect"):
        color, marker, label = STYLE[outcome]
        sel = [p for p in pts if p["outcome"] == outcome]
        if not sel:
            continue
        ax.scatter([p["p"] for p in sel], [p["delta"] for p in sel], s=22 if outcome != "none" else 14,
                   marker=marker, c=color, edgecolors=SURFACE, linewidths=0.8, label=f"{label} ({len(sel)})",
                   zorder=3 if outcome != "none" else 2)
    # Direct labels only for hack-called tasks (the few points the text discusses).
    for p in pts:
        if p["prediction"] == "hack":
            pol = "7B" if "7B" in p["task"][0] else "1.5B"
            d = {"ende": "de", "enzh": "zh", "enja": "ja", "enru": "ru"}.get(p["task"][2], p["task"][2])
            rw = p["task"][1].replace("xcomet_xl_qe", "XCOMET-XL").replace("xcomet_xxl_qe", "XCOMET-XXL")
            below = p["delta"] < 0.5
            ax.annotate(f"{rw} {pol} {d}", (p["p"], p["delta"]), xytext=(-10, 6) if below else (-4, 0),
                        textcoords="offset points", ha="left" if below else "right", va="bottom" if below else "center",
                        fontsize=5.5, color=INK2)
    ax.set_xlabel("Audit preference rate $p_k$", fontsize=7, color=INK)
    ax.set_ylabel("Indicator change (final - initial)", fontsize=7, color=INK)
    ax.tick_params(labelsize=6, colors=INK2, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.set_xscale("function", functions=(lambda v: v ** 0.5, lambda v: v ** 2))
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.05, 0.2, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.05", "0.2", "0.5", "1"])
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    fig.savefig("/tmp/fig_audit_outcomes_preview.png", facecolor=SURFACE, dpi=200)  # layout check only
    counts = {o: sum(p["outcome"] == o for p in pts) for o in STYLE}
    by_study = {s: sum(p["study"] == s for p in pts) for s in sorted({p["study"] for p in pts})}
    ssot["audit_outcome_figure"] = {"points": pts, "counts": counts, "by_study": by_study,
                                    "generalization_included": "grpo_generalization" in ssot}
    tmp = SSOT.with_suffix(SSOT.suffix + ".tmp")
    tmp.write_text(json.dumps(ssot, indent=2, sort_keys=True) + "\n")
    tmp.replace(SSOT)
    print(OUT, counts, by_study)


if __name__ == "__main__":
    main()
