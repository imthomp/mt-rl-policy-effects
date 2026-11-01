import tempfile
import unittest
from pathlib import Path

from audit_predictor_confirmation_data import audit_tsv


HEADER = "system\tdoc\tdocSegId\tglobalSegId\trater\tsource\ttarget\tcategory\tseverity\tmetadata\n"


class PredictorConfirmationAuditTests(unittest.TestCase):
    def _write(self, outcome_suffixes: list[str]) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "sample.tsv"
        identifiers = [
            "SystemA\tdoc1\t1\t10\tr1\tsource one\ttarget a",
            "SystemB\tdoc1\t1\t10\tr2\tsource one\ttarget b",
            "ref-A\tdoc1\t1\t10\tr3\tsource one\treference",
            "SystemA\tdoc1\t2\t11\tr1\tsource two\ttarget c",
            "SystemB\tdoc1\t2\t11\tr2\tsource two\ttarget d",
            "ref-A\tdoc1\t2\t11\tr3\tsource two\treference two",
        ]
        path.write_text(
            HEADER + "".join(f"{prefix}\t{suffix}\n" for prefix, suffix in zip(identifiers, outcome_suffixes)),
            encoding="utf-8",
        )
        return path

    def test_outcome_columns_do_not_change_coverage_audit(self):
        first = audit_tsv(self._write(["A\tMajor\t{}"] * 6))
        second = audit_tsv(self._write(["anything\tunknown\tsecret"] * 6))
        for row in (first, second):
            row.pop("path")
            row.pop("sha256")
        self.assertEqual(first, second)
        self.assertEqual(first["outcome_fields_accessed"], [])

    def test_machine_pool_excludes_reference_and_is_rectangular(self):
        result = audit_tsv(self._write(["A\tMajor\t{}"] * 6))
        self.assertEqual(result["excluded_non_machine_systems"], ["ref-A"])
        self.assertEqual(result["machine_system_count"], 2)
        self.assertTrue(result["rectangular_machine_pool"])


if __name__ == "__main__":
    unittest.main()
