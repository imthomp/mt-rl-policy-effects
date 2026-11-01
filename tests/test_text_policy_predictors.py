import unittest

import numpy as np

from analyze_text_policy_predictors import text_first_order_alignment, text_parameter_gradient


class TextPolicyPredictorTests(unittest.TestCase):
    def test_gradient_matches_manual_two_action_case(self):
        values = np.asarray([[0.0, 2.0]])
        embeddings = np.asarray([[[1.0, 0.0], [0.0, 1.0]]])
        np.testing.assert_allclose(text_parameter_gradient(values, embeddings), [-0.5, 0.5])

    def test_alignment_sign(self):
        values = np.asarray([[0.0, 2.0], [1.0, 0.0]])
        embeddings = np.asarray(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, -1.0]]]
        )
        aligned = text_first_order_alignment(values, values, embeddings)
        opposed = text_first_order_alignment(values, -values, embeddings)
        self.assertAlmostEqual(aligned["human_reward_gradient_cosine"], 1.0)
        self.assertAlmostEqual(opposed["human_reward_gradient_cosine"], -1.0)


if __name__ == "__main__":
    unittest.main()
