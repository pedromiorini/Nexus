import json
import tempfile
import unittest
from pathlib import Path

from tools.security_triage import build_report, classify, main, merge_state_block, resolve_scope


class SecurityTriageTests(unittest.TestCase):
    def test_known_findings_have_explicit_dispositions(self):
        self.assertEqual(classify({"test_id": "B608"}), "sql_construction_high_priority_review")
        self.assertEqual(classify({"test_id": "B311"}), "simulation_only_random_review")
        self.assertEqual(classify({"test_id": "B101"}), "embedded_demo_assert_review")
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

    def test_findings_are_grouped_by_scope_and_stable_under_line_shifts(self):
        source = "class Memory:\n    def store(self):\n        return 1\n\nif __name__ == '__main__':\n    assert True\n    assert True\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "core").mkdir()
            (root / "core" / "m.py").write_text(source, encoding="utf-8")
            payload = {"results": [
                {"test_id": "B311", "issue_severity": "LOW", "filename": "core/m.py", "line_number": 3},
                {"test_id": "B101", "issue_severity": "LOW", "filename": "core/m.py", "line_number": 6},
                {"test_id": "B101", "issue_severity": "LOW", "filename": "core/m.py", "line_number": 7},
            ]}
            before = build_report(payload, root=root)
            self.assertIn("`Memory.store`", before)
            self.assertIn("`<__main__ block>` | 2 |", before)
            self.assertEqual(resolve_scope("core/missing.py", 1, root), "<unresolved>")
            # Inserir linhas no topo desloca todos os números de linha, mas não o relatório.
            (root / "core" / "m.py").write_text("# a\n# b\n" + source, encoding="utf-8")
            from tools.security_triage import _scope_spans
            _scope_spans.cache_clear()
            shifted = {"results": [dict(item, line_number=item["line_number"] + 2) for item in payload["results"]]}
            self.assertEqual(build_report(shifted, root=root), before)

    def test_cli_preserves_the_current_state_block_of_an_existing_report(self):
        block = "<!-- NEXUS-CURRENT-STATE\ncommit: HEAD\nci_run: CURRENT_RUN\ntests: 7\nbandit_low: 0\n-->"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bandit.json"
            output = Path(directory) / "triage.md"
            source.write_text(json.dumps({"results": []}), encoding="utf-8")
            output.write_text("# Bandit Security Triage\n\n" + block + "\n\nold body\n", encoding="utf-8")
            self.assertEqual(main([str(source), str(output)]), 0)
            text = output.read_text(encoding="utf-8")
            self.assertIn(block, text)
            self.assertNotIn("old body", text)
            self.assertEqual(merge_state_block("# Bandit Security Triage\n\nx\n", "sem bloco"), "# Bandit Security Triage\n\nx\n")

    def test_check_mode_detects_drift_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bandit.json"
            output = Path(directory) / "triage.md"
            source.write_text(json.dumps({"results": []}), encoding="utf-8")
            self.assertEqual(main([str(source), str(output)]), 0)
            self.assertEqual(main([str(source), str(output), "--check"]), 0)
            output.write_text(output.read_text(encoding="utf-8") + "\nedição manual\n", encoding="utf-8")
            before = output.read_text(encoding="utf-8")
            self.assertEqual(main([str(source), str(output), "--check"]), 1)
            self.assertEqual(output.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
