import unittest

from build_reward_condition_manifests import draw_matched_subset


class RewardConditionManifestTests(unittest.TestCase):
    def test_draw_is_deterministic_and_respects_cardinality(self):
        candidates = [
            {"language_pair": f"x{i}-en", "year": 2019, "label_rows": 100 + i}
            for i in range(10)
        ]
        first, first_meta = draw_matched_subset(
            candidates, count=3, target_rows=306, seed=17
        )
        second, second_meta = draw_matched_subset(
            candidates, count=3, target_rows=306, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first_meta, second_meta)
        self.assertEqual(len(first), 3)
        self.assertTrue(first_meta["within_tolerance"])

    def test_forbidden_subset_is_not_reused(self):
        candidates = [
            {"language_pair": f"x{i}-en", "year": 2019, "label_rows": 100}
            for i in range(5)
        ]
        first, _ = draw_matched_subset(candidates, count=2, target_rows=200, seed=3)
        forbidden = {tuple(sorted(f"{row['language_pair']}|{row['year']}" for row in first))}
        second, _ = draw_matched_subset(
            candidates, count=2, target_rows=200, seed=3, forbidden_subsets=forbidden
        )
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
