import unittest

import numpy as np

import run_text_policy_bandit as text_policy


class TextPolicyBanditTests(unittest.TestCase):
    def test_system_split_is_deterministic_disjoint_and_complete(self):
        actions = [f"system-{index}" for index in range(30)]
        first_train, first_test = text_policy.system_split(actions)
        second_train, second_test = text_policy.system_split(actions)
        self.assertEqual((first_train, first_test), (second_train, second_test))
        self.assertFalse(set(first_train) & set(first_test))
        self.assertEqual(sorted(first_train + first_test), list(range(len(actions))))
        self.assertTrue(first_train)
        self.assertTrue(first_test)

    def test_system_split_is_independent_of_input_order(self):
        actions = [f"system-{index}" for index in range(30)]
        train, test = text_policy.system_split(actions)
        train_names = {actions[index] for index in train}
        test_names = {actions[index] for index in test}
        reversed_actions = list(reversed(actions))
        reversed_train, reversed_test = text_policy.system_split(reversed_actions)
        self.assertEqual(train_names, {reversed_actions[index] for index in reversed_train})
        self.assertEqual(test_names, {reversed_actions[index] for index in reversed_test})

    def test_evaluation_baselines_trace_real_candidate_rows(self):
        reward = np.asarray([[0.1, 0.8], [0.7, 0.2]])
        human = np.asarray([[10.0, 20.0], [40.0, 30.0]])
        result = text_policy.evaluation_baselines(reward, human)
        self.assertEqual(result["uniform_human_cesa"], 25.0)
        self.assertEqual(result["proxy_best_human_cesa"], 30.0)
        self.assertEqual(result["human_best_ceiling_cesa"], 30.0)


if __name__ == "__main__":
    unittest.main()
