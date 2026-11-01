import unittest

from analyze_wmt25_best_of_n import candidate_order


class Wmt25BestOfNTests(unittest.TestCase):
    def test_candidate_order_is_deterministic_and_identity_based(self):
        rows = [
            {"system_id": "b"},
            {"system_id": "a"},
            {"system_id": "c"},
        ]
        first = [row["system_id"] for row in candidate_order(rows, "segment", 4)]
        second = [row["system_id"] for row in candidate_order(list(reversed(rows)), "segment", 4)]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
