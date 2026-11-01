import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_confirmation_calibration import exact_best_of_n_human_change, read_reward


class ConfirmationCalibrationTests(unittest.TestCase):
    def test_best_of_n_detects_harmful_reward_selection(self):
        reward = np.asarray([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
        human = np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        self.assertLess(exact_best_of_n_human_change(reward, human, 2), 0)

    def test_reward_reader_requires_exact_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.jsonl"
            rows = [
                {"segment_key": "c", "system_id": "a", "predicted_reward": 1.0},
                {"segment_key": "c", "system_id": "b", "predicted_reward": 2.0},
            ]
            path.write_text(
                "\n".join(
                    [json.dumps({"rows": len(rows)})] + [json.dumps(row) for row in rows]
                )
                + "\n",
                encoding="utf-8",
            )
            np.testing.assert_allclose(read_reward(path, ["c"], ["a", "b"]), [[1, 2]])
            with self.assertRaises(ValueError):
                read_reward(path, ["c"], ["a", "b", "d"])


if __name__ == "__main__":
    unittest.main()
