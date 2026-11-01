import unittest

from analyze_wmt26_best_of_n import POOL_SIZES


class Wmt26BestOfNTests(unittest.TestCase):
    def test_frozen_pool_schedule_ends_with_full_pool(self):
        self.assertEqual(POOL_SIZES[-1], "full")
        self.assertEqual(POOL_SIZES[:5], (1, 2, 4, 8, 16))


if __name__ == "__main__":
    unittest.main()
