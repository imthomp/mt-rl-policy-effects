import unittest

import numpy as np

from run_contextual_bandit import document_split, hash_features, machine_candidate_keys


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

    def test_human_postedited_output_is_not_a_bandit_action(self):
        keys = {
            ("source-a", "System A"),
            ("source-a", "Human (postediting)"),
            ("source-b", "System A"),
            ("source-b", "Human (postediting)"),
        }
        filtered = machine_candidate_keys(keys)
        self.assertEqual(filtered, {("source-a", "System A"), ("source-b", "System A")})

    def test_missing_frozen_exclusion_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Expected excluded actions are absent"):
            machine_candidate_keys({("source-a", "System A")})


if __name__ == "__main__":
    unittest.main()
