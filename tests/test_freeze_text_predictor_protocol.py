import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from freeze_text_predictor_protocol import build


class FreezeTextPredictorProtocolTests(unittest.TestCase):
    def test_rejects_budget_with_false_positive_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            manifest = {
                "status": "datasets_frozen_outcomes_sealed",
                "datasets": {
                    "x": {
                        "calibration_evaluation_split": {
                            "calibration_segment_ids": [str(i) for i in range(64)]
                        }
                    }
                },
            }
            development = {
                "frozen_confirmation_outcomes_accessed": False,
                "reward_instance_count": 1,
                "reward_seed_count": 1,
                "external_single_checkpoint_count": 0,
            }
            budget = {
                "frozen_confirmation_outcomes_accessed": False,
                "budgets": {
                    "64": {
                        "decision_agreement_with_full_development_predictor": {"low": 1.0},
                        "false_optimize_rate_on_development_failures": {"high": 0.1},
                        "optimize_rate_on_development_nonfailures": {"low": 1.0},
                    }
                },
            }
            for name, value in (
                ("predictor_confirmation_manifest.json", manifest),
                ("text_policy_predictor_development.json", development),
                ("text_policy_predictor_label_budget.json", budget),
            ):
                (root / "results" / name).write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                build(root)


if __name__ == "__main__":
    unittest.main()
