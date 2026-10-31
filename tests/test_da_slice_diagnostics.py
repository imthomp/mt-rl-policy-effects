import unittest

from analyze_da_slice_diagnostics import align_scores, decision


class DaSliceDiagnosticTests(unittest.TestCase):
    def test_decision_rule(self):
        self.assertEqual(decision(0.51, 0.70), "flagged")
        self.assertEqual(decision(0.30, 0.50), "retained_clear")
        self.assertEqual(decision(0.30, 0.60), "inconclusive_retain")

    def test_alignment_rejects_reordered_metric_rows(self):
        sample = [
            {"global_row_index": 1, "language_pair": "en-de", "year": 2020, "da_score": 0.1}
        ]
        comet = [{"global_row_index": 2, "score": 0.2}]
        bleurt = [{"global_row_index": 1, "score": 0.3}]
        with self.assertRaises(ValueError):
            align_scores(sample, comet, bleurt)


if __name__ == "__main__":
    unittest.main()
