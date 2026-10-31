import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from audit_reward_holdouts import _has_valid_score, _production_candidates, _retained_content_lines, merge_ssot


class RewardHoldoutAuditTests(unittest.TestCase):
    def test_production_candidates_excludes_random_and_welcome_items(self):
        campaign = {
            "data": [
                [
                    {"item_id": "seg-a", "tgt": {"sys-1": "a", "sys-2": "b"}},
                    {"item_id": "seg-b", "tgt": {"sys-1": "c", "sys-2": "d"}},
                ]
            ],
            "data_random": [{"item_id": "random", "tgt": {"tutorial": "x"}}],
            "data_welcome": [{"item_id": "welcome", "tgt": {"tutorial": "x"}}],
        }
        self.assertEqual(
            _production_candidates(campaign),
            {"seg-a": {"sys-1", "sys-2"}, "seg-b": {"sys-1", "sys-2"}},
        )

    def test_merge_ssot_preserves_unrelated_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper_stats.json"
            path.write_text(json.dumps({"existing_analysis": {"value": 7}}), encoding="utf-8")
            merge_ssot(path, {"design_status": "coverage_only_outcomes_locked"})
            merged = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(merged["existing_analysis"], {"value": 7})
        self.assertEqual(
            merged["reward_holdout_feasibility"]["design_status"],
            "coverage_only_outcomes_locked",
        )

    def test_reset_discards_earlier_user_content(self):
        records = [
            {"user_id": "u", "item": [{"item_id": "old"}], "annotation": [{}]},
            {"user_id": "u", "annotation": "__RESET__"},
            {"user_id": "u", "item": [{"item_id": "new"}], "annotation": [{}]},
        ]
        retained = _retained_content_lines(records, {"old", "new"})
        self.assertEqual([row["item"][0]["item_id"] for row in retained], ["new"])

    def test_valid_score_check_rejects_missing_boolean_and_nonfinite_values(self):
        self.assertTrue(_has_valid_score({"score": 75}))
        self.assertFalse(_has_valid_score({"score": None}))
        self.assertFalse(_has_valid_score({"score": True}))
        self.assertFalse(_has_valid_score({"score": float("nan")}))
        self.assertFalse(_has_valid_score({"score": 101}))


if __name__ == "__main__":
    unittest.main()
