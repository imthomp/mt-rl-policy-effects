"""Load MQM quality for an explicit segment allowlist under the frozen rule."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def error_penalty(category: str, severity: str) -> float:
    category_key = category.strip().casefold()
    severity_key = severity.strip().casefold()
    if category_key in {"no-error", "no error"} or severity_key in {
        "neutral",
        "no-error",
        "no error",
    }:
        return 0.0
    if "non-translation" in category_key or "nontranslation" in category_key:
        return 25.0
    if severity_key == "major":
        return 5.0
    if severity_key == "minor":
        if category_key == "fluency/punctuation" or category_key.endswith(
            "/punctuation"
        ):
            return 0.1
        return 1.0
    raise ValueError(f"Unknown MQM label: category={category!r}, severity={severity!r}")


def load_mqm_quality(
    path: Path,
    contexts: list[str],
    actions: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """Return higher-is-better mean-rater quality for allowlisted candidates."""
    allowed = set(contexts)
    action_set = set(actions)
    penalties: dict[tuple[str, str, str], float] = defaultdict(float)
    rows_used = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        identifiers = {
            name: header.index(name)
            for name in ("system", "doc", "docSegId", "globalSegId", "rater")
        }
        category_index = header.index("category")
        severity_index = header.index("severity")
        for row_number, row in enumerate(reader, 2):
            if len(row) <= max(*identifiers.values(), category_index, severity_index):
                raise ValueError(f"Incomplete MQM row at {path}:{row_number}")
            segment = "|".join(
                (
                    row[identifiers["doc"]],
                    row[identifiers["docSegId"]],
                    row[identifiers["globalSegId"]],
                )
            )
            system = row[identifiers["system"]]
            if segment not in allowed or system not in action_set:
                continue
            rater = row[identifiers["rater"]]
            penalties[(segment, system, rater)] += error_penalty(
                row[category_index], row[severity_index]
            )
            rows_used += 1
    by_candidate: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (segment, system, _), penalty in penalties.items():
        by_candidate[(segment, system)].append(penalty)
    expected = {(context, action) for context in contexts for action in actions}
    if set(by_candidate) != expected:
        missing = expected - set(by_candidate)
        extra = set(by_candidate) - expected
        raise ValueError(
            f"MQM candidate coverage mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    quality = np.asarray(
        [
            [-float(np.mean(by_candidate[(context, action)])) for action in actions]
            for context in contexts
        ],
        dtype=np.float32,
    )
    return quality, {
        "segments": len(contexts),
        "systems": len(actions),
        "candidate_scores": len(expected),
        "raters": len({rater for _, _, rater in penalties}),
        "error_rows_used": rows_used,
    }
