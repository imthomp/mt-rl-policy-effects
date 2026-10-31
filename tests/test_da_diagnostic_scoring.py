import json
import tempfile
import unittest
from pathlib import Path

from score_da_diagnostic_sample import completed_prefix


class DaDiagnosticScoringTests(unittest.TestCase):
    def test_completed_prefix_validates_global_indices(self):
        sample = [{"global_row_index": 4}, {"global_row_index": 9}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scores.jsonl"
            output.write_text(json.dumps({"global_row_index": 4, "score": 0.1}) + "\n")
            self.assertEqual(completed_prefix(output, sample), 1)

    def test_completed_prefix_rejects_misalignment(self):
        sample = [{"global_row_index": 4}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scores.jsonl"
            output.write_text(json.dumps({"global_row_index": 5, "score": 0.1}) + "\n")
            with self.assertRaises(ValueError):
                completed_prefix(output, sample)


if __name__ == "__main__":
    unittest.main()
