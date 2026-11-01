import json
import tempfile
import unittest
from pathlib import Path

from aggregate_reward_model_development import (
    REQUIRED_CONDITIONS,
    load_and_validate,
    merge_ssot,
)


def _metrics(condition: str) -> dict:
    return {
        "condition": condition,
        "git_worktree_dirty": False,
        "model_name": "model",
        "model_snapshot_hash": "snapshot",
        "indices_manifest_sha256": "indices",
        "dev_indices_sha256": "dev",
        "train_rows": 10,
        "dev_rows": 5,
        "steps": 2,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "learning_rate": 0.01,
        "warmup_steps": 0,
        "max_length": 8,
        "seed": 4,
        "executing_commit": "commit",
        "dev_mse": 1.0,
        "dev_pearson": 0.2,
        "dev_spearman": 0.3,
        "elapsed_seconds": 2.0,
        "peak_gpu_allocated_gib": 1.0,
    }


class AggregateRewardDevelopmentTests(unittest.TestCase):
    def test_validates_all_runs_and_saved_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition in REQUIRED_CONDITIONS:
                run = root / condition
                (run / "model").mkdir(parents=True)
                (run / "model" / "model.safetensors").touch()
                (run / "metrics.json").write_text(
                    json.dumps(_metrics(condition)), encoding="utf-8"
                )
            result = load_and_validate(root)
        self.assertEqual(set(result["conditions"]), set(REQUIRED_CONDITIONS))

    def test_rejects_noncomparable_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition in REQUIRED_CONDITIONS:
                run = root / condition
                (run / "model").mkdir(parents=True)
                (run / "model" / "model.safetensors").touch()
                metrics = _metrics(condition)
                if condition == "diagnostic_filtered":
                    metrics["seed"] = 99
                (run / "metrics.json").write_text(
                    json.dumps(metrics), encoding="utf-8"
                )
            with self.assertRaisesRegex(ValueError, "Non-comparable seed"):
                load_and_validate(root)

    def test_merge_preserves_existing_ssot_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper_stats.json"
            path.write_text(json.dumps({"existing": {"value": 1}}), encoding="utf-8")
            merge_ssot(path, {"status": "complete"})
            merged = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(merged["existing"], {"value": 1})
        self.assertEqual(merged["reward_model_development"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
