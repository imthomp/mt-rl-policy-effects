"""Build deterministic reward-training slice-exclusion conditions.

The targeted condition removes frozen flagged slices. Random controls remove
the same number of retained-clear slices and are accepted only when their label
volume is close to the targeted removal. A direction-matched sensitivity
control restricts candidates to retained-clear into-English slices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


GLOBAL_RANDOM_SEEDS = (48117, 90231, 177013)
DIRECTION_MATCHED_SEED = 611953
VOLUME_TOLERANCE = 0.05
MAX_ATTEMPTS = 200_000


def slice_key(row: dict[str, Any]) -> str:
    return f"{row['language_pair']}|{row['year']}"


def draw_matched_subset(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    target_rows: int,
    seed: int,
    forbidden_subsets: set[tuple[str, ...]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"Need {count} candidates, found {len(candidates)}")
    generator = random.Random(seed)
    forbidden_subsets = forbidden_subsets or set()
    best: tuple[float, list[dict[str, Any]], int] | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        chosen = generator.sample(candidates, count)
        identity = tuple(sorted(slice_key(row) for row in chosen))
        if identity in forbidden_subsets:
            continue
        removed_rows = sum(int(row["label_rows"]) for row in chosen)
        relative_difference = abs(removed_rows - target_rows) / target_rows
        if best is None or relative_difference < best[0]:
            best = (relative_difference, chosen, attempt)
        if relative_difference <= VOLUME_TOLERANCE:
            return sorted(chosen, key=slice_key), {
                "attempt": attempt,
                "removed_rows": removed_rows,
                "relative_volume_difference": relative_difference,
                "within_tolerance": True,
            }
    if best is None:
        raise RuntimeError("No candidate subset was drawn")
    relative_difference, chosen, attempt = best
    return sorted(chosen, key=slice_key), {
        "attempt": attempt,
        "removed_rows": sum(int(row["label_rows"]) for row in chosen),
        "relative_volume_difference": relative_difference,
        "within_tolerance": False,
    }


def build_conditions(ssot: dict[str, Any]) -> dict[str, Any]:
    inventory = ssot["reward_da_slice_inventory"]["rows"]
    decisions = {
        (row["language_pair"], int(row["year"])): row["decision"]
        for row in ssot["reward_da_slice_diagnostics"]["rows"]
    }
    for row in inventory:
        row["decision"] = decisions[(row["language_pair"], int(row["year"]))]

    flagged = [row for row in inventory if row["decision"] == "flagged"]
    clear = [row for row in inventory if row["decision"] == "retained_clear"]
    target_rows = sum(int(row["label_rows"]) for row in flagged)
    count = len(flagged)
    conditions: dict[str, Any] = {
        "all_data": {
            "excluded_slices": [],
            "removed_rows": 0,
            "role": "baseline",
        },
        "diagnostic_filtered": {
            "excluded_slices": [slice_key(row) for row in sorted(flagged, key=slice_key)],
            "removed_rows": target_rows,
            "role": "targeted_intervention",
        },
    }

    used: set[tuple[str, ...]] = set()
    for index, seed in enumerate(GLOBAL_RANDOM_SEEDS, start=1):
        chosen, matching = draw_matched_subset(
            clear,
            count=count,
            target_rows=target_rows,
            seed=seed,
            forbidden_subsets=used,
        )
        identity = tuple(slice_key(row) for row in chosen)
        used.add(identity)
        conditions[f"global_random_{index}"] = {
            "excluded_slices": list(identity),
            "seed": seed,
            "role": "size_matched_random_control",
            **matching,
        }

    direction_candidates = [row for row in clear if row["direction"] == "x_to_en"]
    chosen, matching = draw_matched_subset(
        direction_candidates,
        count=count,
        target_rows=target_rows,
        seed=DIRECTION_MATCHED_SEED,
    )
    conditions["direction_matched_earlier_year"] = {
        "excluded_slices": [slice_key(row) for row in chosen],
        "seed": DIRECTION_MATCHED_SEED,
        "role": "direction_matched_year_confound_sensitivity",
        **matching,
    }
    return {
        "slice_key_format": "language_pair|year",
        "target_excluded_slice_count": count,
        "target_removed_rows": target_rows,
        "volume_tolerance": VOLUME_TOLERANCE,
        "random_controls_use_only": "retained_clear slices",
        "inconclusive_slices_are_never_random_controls": True,
        "conditions": conditions,
        "known_limitation": (
            "No retained-clear into-English slices exist in the flagged later-year strata; "
            "the direction-matched sensitivity necessarily uses earlier years."
        ),
    }


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    existing = json.loads(path.read_text(encoding="utf-8"))
    existing["reward_training_conditions"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--ssot", type=Path, default=root / "results" / "paper_stats.json")
    parser.add_argument(
        "--output", type=Path, default=root / "data" / "reward_da" / "conditions.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ssot = json.loads(args.ssot.read_text(encoding="utf-8"))
    result = build_conditions(ssot)
    payload = {"conditions_sha256_basis": "canonical JSON below", **result}
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    result["manifest_path"] = str(args.output.resolve().relative_to(Path(__file__).resolve().parents[1]))
    result["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    merge_ssot(args.ssot, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
