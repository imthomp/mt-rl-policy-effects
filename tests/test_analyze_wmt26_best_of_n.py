import unittest

from analyze_wmt26_best_of_n import POOL_SIZES
from wmt26_candidate_pool import machine_only_rows


class Wmt26BestOfNTests(unittest.TestCase):
    def test_frozen_pool_schedule_ends_with_full_pool(self):
        self.assertEqual(POOL_SIZES[-1], "full")
        self.assertEqual(POOL_SIZES[:5], (1, 2, 4, 8, 16))

    def test_human_postedit_is_excluded_from_selection_pool(self):
        rows = {
            "machine": {"system_id": "System A"},
            "human": {"system_id": "Human (postediting)"},
        }
        self.assertEqual(machine_only_rows(rows), {"machine": rows["machine"]})


if __name__ == "__main__":
    unittest.main()
