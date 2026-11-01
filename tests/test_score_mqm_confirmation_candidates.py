import json
import tempfile
import unittest
from pathlib import Path

from audit_reward_holdouts import sha256_file
from score_mqm_confirmation_candidates import load_candidates


class ScoreMQMConfirmationCandidatesTests(unittest.TestCase):
    def test_loader_uses_only_blind_fields_and_removes_highlights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mqm.tsv"
            source.write_text(
                "system\tdoc\tdocSegId\tglobalSegId\trater\tsource\ttarget\tcategory\tseverity\tmetadata\n"
                "SystemA\td1\t1\t1\tr1\tsrc\t<v>a</v>\tSHOULD_NOT_READ\tMajor\t{}\n"
                "SystemB\td1\t1\t1\tr1\tsrc\tb\tSHOULD_NOT_READ\tMajor\t{}\n"
                "refA\td1\t1\t1\tr1\tsrc\tref\tSHOULD_NOT_READ\tMajor\t{}\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "datasets": {
                            "test": {
                                "source_sha256": sha256_file(source),
                                "candidate_content_sha256": "content",
                                "included_segment_ids": ["d1|1|1"],
                                "machine_systems": ["SystemA", "SystemB"],
                                "excluded_non_machine_systems": ["refA"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            metadata, rows = load_candidates(source, manifest, "test")
            self.assertEqual(metadata["outcome_fields_accessed"], [])
            self.assertEqual([row["hypothesis"] for row in rows], ["a", "b"])
            self.assertEqual({row["reference"] for row in rows}, {"ref"})


if __name__ == "__main__":
    unittest.main()
