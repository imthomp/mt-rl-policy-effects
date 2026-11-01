import unittest

import numpy as np

from run_contextual_bandit import document_split, hash_features


class ContextualBanditTests(unittest.TestCase):
    def test_hash_features_are_deterministic_and_normalized(self):
        first = hash_features("これはテストです")
        second = hash_features("これはテストです")
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=6)

    def test_document_split_keeps_documents_disjoint(self):
        contexts = [
            "news_###_a_###_0", "news_###_a_###_1", "news_###_b_###_0",
            "social_###_c_###_0", "social_###_d_###_0",
        ]
        train, test, _ = document_split(contexts)
        train_docs = {contexts[index].rsplit("_###_", 1)[0] for index in train}
        test_docs = {contexts[index].rsplit("_###_", 1)[0] for index in test}
        self.assertFalse(train_docs & test_docs)


if __name__ == "__main__":
    unittest.main()
