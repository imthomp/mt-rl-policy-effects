import unittest

import numpy as np

from analyze_predictor_label_budget import (
    cosine_with_sampled_human_gradient,
    one_segment_per_document_samples,
)
from analyze_preoptimization_predictors import parameter_gradient


class PredictorLabelBudgetTests(unittest.TestCase):
    def test_samples_distinct_documents_and_are_reproducible(self):
        contexts = ["news_###_doc1_###_0", "news_###_doc1_###_1", "news_###_doc2_###_0"]
        train = np.asarray([0, 1, 2])
        first = one_segment_per_document_samples(contexts, train, 2, 3, 7)
        second = one_segment_per_document_samples(contexts, train, 2, 3, 7)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)
            documents = {contexts[int(index)].rsplit("_###_", 1)[0] for index in left}
            self.assertEqual(len(documents), 2)

    def test_rejects_budget_above_document_count(self):
        with self.assertRaises(ValueError):
            one_segment_per_document_samples(["d_###_0"], np.asarray([0]), 2, 1, 7)

    def test_sampled_human_cosine_uses_supplied_reward_gradient(self):
        features = np.asarray([[1.0], [2.0]])
        human = np.asarray([[0.0, 1.0], [0.0, 2.0]])
        reward_gradient = parameter_gradient(human, features)
        self.assertAlmostEqual(
            cosine_with_sampled_human_gradient(reward_gradient, human, features), 1.0
        )
        self.assertAlmostEqual(
            cosine_with_sampled_human_gradient(-reward_gradient, human, features), -1.0
        )


if __name__ == "__main__":
    unittest.main()
