import unittest

from analyze_bandit_seeds import average_ranks, spearman


class BanditSeedAnalysisTests(unittest.TestCase):
    def test_average_ranks_assigns_tied_midranks(self):
        self.assertEqual(average_ranks([4.0, 1.0, 4.0, 2.0]).tolist(), [2.5, 0.0, 2.5, 1.0])

    def test_spearman_detects_monotonic_order(self):
        self.assertAlmostEqual(spearman([1.0, 2.0, 3.0], [10.0, 20.0, 40.0]), 1.0)
        self.assertAlmostEqual(spearman([1.0, 2.0, 3.0], [40.0, 20.0, 10.0]), -1.0)


if __name__ == "__main__":
    unittest.main()
