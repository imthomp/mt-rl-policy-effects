import unittest

from train_da_reward_model import format_input


class TrainDaRewardModelTests(unittest.TestCase):
    def test_format_input_preserves_field_order(self):
        self.assertEqual(
            format_input("source", "hypothesis", "reference", "</s>"),
            "source </s> hypothesis </s> reference",
        )


if __name__ == "__main__":
    unittest.main()
