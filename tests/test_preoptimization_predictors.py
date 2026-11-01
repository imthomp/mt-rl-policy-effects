import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_preoptimization_predictors import (
    first_order_alignment,
    read_reward_matrix,
    system_predictors,
)


class PreoptimizationPredictorTests(unittest.TestCase):
    def test_system_predictors_ignore_per_source_offsets(self):
        reward = np.asarray([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]])
        shifted = reward + np.asarray([[100.0], [-50.0]])
        self.assertEqual(system_predictors(reward), system_predictors(shifted))

    def test_first_order_alignment_sign(self):
        features = np.asarray([[1.0], [2.0]])
        reward = np.asarray([[0.0, 1.0], [0.0, 2.0]])
        aligned = first_order_alignment(reward, reward, features)
        opposed = first_order_alignment(reward, -reward, features)
        self.assertGreater(aligned["human_gradient_dot_reward_gradient"], 0)
        self.assertLess(opposed["human_gradient_dot_reward_gradient"], 0)
        self.assertAlmostEqual(aligned["human_reward_gradient_cosine"], 1.0)
        self.assertAlmostEqual(opposed["human_reward_gradient_cosine"], -1.0)

    def test_external_reward_requires_exact_environment(self):
        rows = [
            {
                "segment_key": "x",
                "system_id": action,
                "human_cesa": human,
                "predicted_reward": reward,
            }
            for action, human, reward in (("a", 1.0, 0.1), ("b", 2.0, 0.2))
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.jsonl"
            path.write_text(
                "\n".join(
                    [json.dumps({"record_type": "metadata", "rows": 2})]
                    + [json.dumps(row) for row in rows]
                )
                + "\n",
                encoding="utf-8",
            )
            observed = read_reward_matrix(
                path, ["x"], ["a", "b"], np.asarray([[1.0, 2.0]], dtype=np.float32)
            )
            np.testing.assert_allclose(observed, [[0.1, 0.2]])
            with self.assertRaises(ValueError):
                read_reward_matrix(
                    path,
                    ["x"],
                    ["a", "b"],
                    np.asarray([[1.0, 3.0]], dtype=np.float32),
                )


if __name__ == "__main__":
    unittest.main()
