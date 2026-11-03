"""Freeze reward-training tasks for the safe-pressure confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CONDITION_ALIASES = {
    "all_available_DA": "all_data",
    "metric_annotation_disagreement_slices_removed": "diagnostic_filtered",
    "matched_random_slices_removed": "global_random_2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_ssot(path: Path, result: dict[str, Any]) -> None:
    root = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    root["safe_pressure_reward_plan"] = result
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "results/safe_pressure_protocol.json"
    manifest_path = root / "results/safe_pressure_manifest.json"
    indices_path = root / "data/reward_da/development_indices.json"
    conditions_path = root / "data/reward_da/conditions.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indices = json.loads(indices_path.read_text(encoding="utf-8"))
    if manifest["status"] != "fresh_confirmation_manifest_frozen_outcomes_sealed":
        raise ValueError("Fresh confirmation outcomes are not sealed")
    requested = protocol["reward_models"]["learned_conditions"]
    if set(requested) != set(CONDITION_ALIASES):
        raise ValueError("Condition aliases differ from the frozen protocol")
    if any(actual not in indices["conditions"] for actual in CONDITION_ALIASES.values()):
        raise ValueError("A selected condition is absent from the index manifest")
    tasks = []
    for seed in protocol["reward_models"]["independent_training_seeds"]:
        for protocol_condition in requested:
            actual = CONDITION_ALIASES[protocol_condition]
            tasks.append(
                {
                    "task_index": len(tasks),
                    "seed": seed,
                    "protocol_condition": protocol_condition,
                    "training_condition": actual,
                    "output_dir": f"artifacts/reward_models/safe_pressure/{seed}/models/{actual}",
                }
            )
    result = {
        "status": "safe_pressure_reward_training_plan_frozen",
        "protocol": str(protocol_path.relative_to(root)),
        "protocol_sha256": sha256(protocol_path),
        "confirmation_manifest": str(manifest_path.relative_to(root)),
        "confirmation_manifest_sha256": sha256(manifest_path),
        "indices": str(indices_path.relative_to(root)),
        "indices_sha256": sha256(indices_path),
        "condition_definitions": str(conditions_path.relative_to(root)),
        "condition_definitions_sha256": sha256(conditions_path),
        "condition_aliases": CONDITION_ALIASES,
        "training_arguments": {
            "steps": 500,
            "batch_size": 8,
            "gradient_accumulation": 4,
            "learning_rate": 2e-5,
            "warmup_steps": 50,
            "max_length": 256,
        },
        "tasks": tasks,
        "all_runs_reported": True,
        "human_outcomes_accessed": False,
    }
    output = root / "results/safe_pressure_reward_plan.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    merge_ssot(root / "results/paper_stats.json", result)
    print(json.dumps({"status": result["status"], "tasks": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()
