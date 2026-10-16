import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.reality_audit import build_report, collect_static_evidence


class RealityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_report_has_version_and_scope(self):
        self.assertEqual(self.report["audit_version"], "1.0")
        self.assertEqual(self.report["scope"], "conservative reality audit")

    def test_static_evidence_is_quantified(self):
        evidence = self.report["static_evidence"]
        self.assertGreater(evidence["core_lines"], 1000)
        self.assertGreater(evidence["classes"], 0)
        self.assertGreater(evidence["functions"], 0)
        self.assertGreater(evidence["embedded_test_markers"], 0)

    def test_embedded_marker_inventory_includes_fractional_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "constitutional_brain.py"
            source.write_text(
                'class Demo:\n    pass\n'
                'print("TEST 2 PASSED")\n'
                'print("TEST 2.5 PASSED")\n',
                encoding="utf-8",
            )
            with patch("tools.reality_audit.CORE", source):
                evidence = collect_static_evidence()

        self.assertEqual(evidence["embedded_test_markers"], 2)

    def test_claims_are_conservatively_classified(self):
        statuses = {item["status"] for item in self.report["claims"]}
        self.assertIn("unsubstantiated_by_software_tests", statuses)

    def test_documented_python_references_exist(self):
        paths = self.report["referenced_paths"]
        self.assertGreater(len(paths), 0)
        self.assertTrue(all(item["exists"] for item in paths))
        self.assertNotIn("NEXUS_CORE_v3.94.py", {item["path"] for item in paths})

    def test_known_risks_are_present(self):
        risks = " ".join(self.report["risks"])
        self.assertIn("Embedded demonstration tests", risks)
        self.assertIn("Optional dependency fallbacks", risks)


if __name__ == "__main__":
    unittest.main()
