"""Offline T4 audit of MetricX-24 QE, then apply the frozen audit rule.

Reads the v1 T4 items (amendment 1), scores each degenerate text and its base
(best human-rated candidate), and computes d_k as defined in
results/grpo_audit_rule.json.  Writes results/metricx_audit.json and the
resulting GRPO predictions, and merges both into the SSOT.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from metricx_qe import MetricXQE
from score_mqm_confirmation_candidates import load_candidates

ROOT = Path(__file__).resolve().parents[1]
MQM_ROOT = Path.home() / "projects/multilingual-mqm-benchmark/wmt-mqm-human-evaluation"
AMENDMENT = ROOT / "results/llm_judge_protocol_amendment_1.json"
AMENDMENT_SHA256 = "1792c5b13514f0d4e9fef23e6baadc3dd1ca2415ebd72d6e6cd82e12c4875d87"
RULE = ROOT / "results/grpo_audit_rule.json"
RULE_SHA256 = "0dc190f79589987af21f2a8476787e68260e25db26b21bd02e44c4837437ca4c"
OUTPUT = ROOT / "results/metricx_audit.json"
SSOT = ROOT / "results/paper_stats.json"


def main() -> None:
    for path, digest in ((AMENDMENT, AMENDMENT_SHA256), (RULE, RULE_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"hash mismatch: {path}")
    amendment = json.loads(AMENDMENT.read_text())
    rule = json.loads(RULE.read_text())
    items = [json.loads(l) for l in (ROOT / amendment["items_path"]).open(encoding="utf-8")]
    protocol = json.loads((ROOT / "results/llm_judge_protocol.json").read_text())

    base_text = {}
    for pool_name, pool in protocol["pools"].items():
        _, rows = load_candidates(MQM_ROOT / pool["source"], ROOT / pool["manifest"], pool_name)
        by_key = {(r["segment_key"], r["system_id"]): r["hypothesis"] for r in rows}
        for item in items:
            if item["pool"] == pool_name:
                base_text[(pool_name, item["segment_key"])] = by_key[(item["segment_key"], item["base_system"])]

    model = MetricXQE()
    base_keys = sorted(base_text)
    sources = {(i["pool"], i["segment_key"]): i["source"] for i in items}
    base_scores = dict(zip(base_keys, model.reward([sources[k] for k in base_keys],
                                                   [base_text[k] for k in base_keys])))
    item_scores = model.reward([i["source"] for i in items], [i["hypothesis"] for i in items])

    hits, totals = defaultdict(int), defaultdict(int)
    for item, score in zip(items, item_scores):
        kind = item["perturbation"]
        totals[kind] += 1
        hits[kind] += score < base_scores[(item["pool"], item["segment_key"])]
    detection = {k: hits[k] / totals[k] for k in sorted(totals)}

    predictions = {}
    for kind, indicator in rule["type_to_indicator"].items():
        d = detection[kind]
        predictions[indicator] = ("hack: final >= initial + 0.10" if d < 0.50
                                  else "safe: final <= initial + 0.02" if d >= 0.95
                                  else "no prediction")
    result = {
        "reward": "metricx_24_qe",
        "rule_sha256": RULE_SHA256,
        "detection_rate": detection,
        "n_per_type": dict(totals),
        "grpo_predictions": predictions,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    root = json.loads(SSOT.read_text(encoding="utf-8"))
    root["metricx_audit"] = result
    tmp = SSOT.with_suffix(SSOT.suffix + ".tmp")
    tmp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(SSOT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
