import tempfile
import unittest
from pathlib import Path

from mqm_human_scores import error_penalty, load_mqm_quality


class MQMHumanScoreTests(unittest.TestCase):
    def test_frozen_penalties(self):
        self.assertEqual(error_penalty("Accuracy/Mistranslation", "Major"), 5.0)
        self.assertEqual(error_penalty("Accuracy/Omission", "Critical"), 5.0)
        self.assertEqual(error_penalty("Accuracy/Mistranslation", "Minor"), 1.0)
        self.assertEqual(error_penalty("Fluency/Punctuation", "Minor"), 0.1)
        self.assertEqual(error_penalty("Non-translation", "Major"), 25.0)
        self.assertEqual(error_penalty("No-error", "No-error"), 0.0)

    def test_averages_rater_penalties_not_error_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mqm.tsv"
            path.write_text(
                "system\tdoc\tdocSegId\tglobalSegId\trater\tcategory\tseverity\n"
                "A\td1\t1\t1\tr1\tAccuracy/Mistranslation\tMajor\n"
                "A\td1\t1\t1\tr1\tFluency/Grammar\tMinor\n"
                "A\td1\t1\t1\tr2\tNo-error\tNo-error\n"
                "B\td1\t1\t1\tr1\tFluency/Punctuation\tMinor\n"
                "B\td1\t1\t1\tr2\tFluency/Punctuation\tMinor\n"
                "A\td2\t1\t2\tr1\tSHOULD_NOT_BE_READ\tSHOULD_NOT_BE_READ\n",
                encoding="utf-8",
            )
            quality, metadata = load_mqm_quality(path, ["d1|1|1"], ["A", "B"])
            self.assertAlmostEqual(float(quality[0, 0]), -3.0)
            self.assertAlmostEqual(float(quality[0, 1]), -0.1)
            self.assertEqual(metadata["candidate_scores"], 2)


if __name__ == "__main__":
    unittest.main()
