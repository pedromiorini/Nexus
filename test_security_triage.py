import json
import tempfile
import unittest
from pathlib import Path

from tools.security_triage import build_report, classify, main


class SecurityTriageTests(unittest.TestCase):
    def test_known_findings_have_explicit_dispositions(self):
        self.assertEqual(classify({"test_id": "B608"}), "sql_construction_high_priority_review")
        self.assertEqual(classify({"test_id": "B101"}), "legacy_demo_assert_review")
        self.assertEqual(classify({"test_id": "B999"}), "manual_review")

    def test_report_preserves_findings_and_counts(self):
        report = build_report({
            "results": [
                {"test_id": "B608", "issue_severity": "MEDIUM", "filename": "core/x.py", "line_number": 4},
                {"test_id": "B101", "issue_severity": "LOW", "filename": "core/y.py", "line_number": 9},
            ]
        })
        self.assertIn("Total findings: **2**", report)
        self.assertIn("sql_construction_high_priority_review", report)
        self.assertIn("`core/x.py`", report)

    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bandit.json"
            output = Path(directory) / "triage.md"
            source.write_text(json.dumps({"results": []}), encoding="utf-8")
            self.assertEqual(main([str(source), str(output)]), 0)
            self.assertIn("Total findings: **0**", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
