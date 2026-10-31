"""Compute frozen D2 slice diagnostics from completed COMET/BLEURT scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


THRESHOLD = 0.50


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def align_scores(
    sample: list[dict[str, Any]],
    comet: list[dict[str, Any]],
    bleurt: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not (len(sample) == len(comet) == len(bleurt)):
        raise ValueError(
            f"Row-count mismatch: sample={len(sample)}, comet={len(comet)}, bleurt={len(bleurt)}"
        )
    aligned: list[dict[str, Any]] = []
    for position, (source, comet_row, bleurt_row) in enumerate(
        zip(sample, comet, bleurt, strict=True)
    ):
        expected = source["global_row_index"]
        if comet_row.get("global_row_index") != expected or bleurt_row.get("global_row_index") != expected:
            raise ValueError(f"Score alignment mismatch at position {position}")
        aligned.append(
            {
                "language_pair": source["language_pair"],
                "year": int(source["year"]),
                "da": float(source["da_score"]),
                "comet": float(comet_row["score"]),
                "bleurt": float(bleurt_row["score"]),
            }
        )
    return aligned


def d2_value(comet: np.ndarray, bleurt: np.ndarray, da: np.ndarray) -> float:
    metric_metric = float(spearmanr(comet, bleurt).statistic)
    metric_human = float(spearmanr(comet, da).statistic)
    if not np.isfinite(metric_metric) or not np.isfinite(metric_human):
        raise ValueError("Undefined slice correlation")
    return metric_metric - metric_human


def bootstrap_d2(
    comet: np.ndarray,
    bleurt: np.ndarray,
    da: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    if len(comet) < 2:
        raise ValueError("A slice needs at least two rows")
    estimate = d2_value(comet, bleurt, da)
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        indices = generator.integers(0, len(comet), size=len(comet))
        try:
            value = d2_value(comet[indices], bleurt[indices], da[indices])
        except ValueError:
            continue
        values.append(value)
    if len(values) < max(100, resamples // 2):
        raise ValueError(f"Too few finite bootstrap replicates: {len(values)}/{resamples}")
    lower, upper = np.percentile(values, [2.5, 97.5])
    return estimate, float(lower), float(upper)


def decision(lower: float, upper: float, threshold: float = THRESHOLD) -> str:
    if lower > threshold:
        return "flagged"
    if upper <= threshold:
        return "retained_clear"
    return "inconclusive_retain"


def verify_metadata(metadata_path: Path, sample_path: Path, score_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("complete"):
        raise ValueError(f"Metric output is incomplete: {metadata_path}")
    if metadata.get("input_sha256") != sha256_file(sample_path):
        raise ValueError(f"Metric input hash mismatch: {metadata_path}")
    if metadata.get("output_sha256") != sha256_file(score_path):
        raise ValueError(f"Metric output hash mismatch: {metadata_path}")
    return metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(existing, dict):
        raise ValueError(f"SSOT root must be a JSON object: {path}")
    existing["reward_da_slice_diagnostics"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--comet", type=Path, required=True)
    parser.add_argument("--bleurt", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261031)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output-csv", type=Path, default=root / "results" / "da_slice_diagnostics.csv")
    parser.add_argument("--ssot", type=Path, default=root / "results" / "paper_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap-resamples must be positive")
    comet_metadata = verify_metadata(
        args.comet.with_suffix(args.comet.suffix + ".meta.json"), args.sample, args.comet
    )
    bleurt_metadata = verify_metadata(
        args.bleurt.with_suffix(args.bleurt.suffix + ".meta.json"), args.sample, args.bleurt
    )
    aligned = align_scores(read_jsonl(args.sample), read_jsonl(args.comet), read_jsonl(args.bleurt))

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in aligned:
        grouped.setdefault((row["language_pair"], row["year"]), []).append(row)
    results: list[dict[str, Any]] = []
    for (language_pair, year), rows in sorted(grouped.items()):
        local_seed = int.from_bytes(
            hashlib.sha256(f"{args.seed}|{language_pair}|{year}".encode()).digest()[:8],
            byteorder="big",
        )
        estimate, lower, upper = bootstrap_d2(
            np.asarray([row["comet"] for row in rows]),
            np.asarray([row["bleurt"] for row in rows]),
            np.asarray([row["da"] for row in rows]),
            resamples=args.bootstrap_resamples,
            seed=local_seed,
        )
        results.append(
            {
                "language_pair": language_pair,
                "year": year,
                "rows": len(rows),
                "d2": estimate,
                "d2_ci_lower": lower,
                "d2_ci_upper": upper,
                "decision": decision(lower, upper),
            }
        )

    write_csv(args.output_csv, results)
    result = {
        "threshold": THRESHOLD,
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "rows": results,
        "slice_count": len(results),
        "flagged_slice_count": sum(row["decision"] == "flagged" for row in results),
        "inconclusive_slice_count": sum(
            row["decision"] == "inconclusive_retain" for row in results
        ),
        "comet_metadata": comet_metadata,
        "bleurt_metadata": bleurt_metadata,
        "sample_sha256": sha256_file(args.sample),
    }
    merge_ssot(args.ssot, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
