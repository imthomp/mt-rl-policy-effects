import unittest

from prepare_reward_training_indices import choose


class RewardTrainingIndexTests(unittest.TestCase):
    def test_choose_is_deterministic_and_bounded(self):
        first = choose(list(range(100)), 10, 17, "train")
        second = choose(list(range(100)), 10, 17, "train")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len(set(first)), 10)

    def test_namespace_changes_selection(self):
        self.assertNotEqual(
            choose(list(range(100)), 10, 17, "first"),
            choose(list(range(100)), 10, 17, "second"),
        )


if __name__ == "__main__":
    unittest.main()
