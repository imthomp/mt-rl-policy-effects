import tempfile
import unittest
from pathlib import Path

from freeze_predictor_confirmation_manifest import freeze_dataset


class PredictorConfirmationManifestTests(unittest.TestCase):
    def test_freeze_excludes_reference_and_ambiguous_segment(self):
        directory = Path(tempfile.mkdtemp())
        path = directory / "sample.tsv"
        path.write_text(
            "system\tdoc\tdocSegId\tglobalSegId\trater\tsource\ttarget\tcategory\tseverity\tmetadata\n"
            "SystemA\td1\t1\t1\tr1\ts1\ta\tX\tMajor\t{}\n"
            "SystemB\td1\t1\t1\tr1\ts1\tb\tX\tMajor\t{}\n"
            "refA\td1\t1\t1\tr1\ts1\tref\tX\tMajor\t{}\n"
            "SystemA\td1\t2\t2\tr1\ts2\tc\tX\tMajor\t{}\n"
            "SystemA\td1\t2\t2\tr2\ts2\td\tX\tMajor\t{}\n"
            "SystemB\td1\t2\t2\tr1\ts2\te\tX\tMajor\t{}\n"
            "refA\td1\t2\t2\tr1\ts2\tref2\tX\tMajor\t{}\n",
            encoding="utf-8",
        )
        result = freeze_dataset(path)
        self.assertEqual(result["excluded_non_machine_systems"], ["refA"])
        self.assertEqual(result["included_segment_ids"], ["d1|1|1"])
        self.assertEqual(result["excluded_segment_counts_by_reason"], {"candidate_text_variants": 1})
        self.assertEqual(result["outcome_fields_accessed"], [])

    def test_document_split_is_disjoint_and_outcome_blind(self):
        directory = Path(tempfile.mkdtemp())
        path = directory / "sample.tsv"
        lines = [
            "system\tdoc\tdocSegId\tglobalSegId\trater\tsource\ttarget\tcategory\tseverity\tmetadata"
        ]
        for document in ("d1", "d2", "d3"):
            for segment in ("1", "2"):
                for system in ("SystemA", "SystemB", "refA"):
                    lines.append(
                        f"{system}\t{document}\t{segment}\t{document}{segment}\tr1\ts{document}{segment}\t{system}{segment}\tX\tMajor\t{{}}"
                    )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = freeze_dataset(path, split_name="test", calibration_document_count=2)
        split = result["calibration_evaluation_split"]
        self.assertEqual(len(split["calibration_document_ids"]), 2)
        self.assertEqual(len(split["calibration_segment_ids"]), 2)
        self.assertEqual(len(split["optimization_segment_ids"]), 4)
        self.assertEqual(len(split["unlabeled_optimization_segment_ids"]), 2)
        self.assertEqual(len(split["evaluation_document_ids"]), 1)
        self.assertTrue(
            set(split["calibration_document_ids"]).isdisjoint(split["evaluation_document_ids"])
        )
        self.assertEqual(split["outcome_fields_accessed"], [])


if __name__ == "__main__":
    unittest.main()
