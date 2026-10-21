import ast
import json
import tempfile
import unittest
from pathlib import Path

from tools.security_triage import _is_main_guard, build_report, classify, main, merge_state_block, resolve_scope


class SecurityTriageTests(unittest.TestCase):
    def test_main_guard_requires_exact_single_equality(self):
        cases = (
            ("__name__ == '__main__'", True),
            ('(__name__ == "__main__")', True),
            ("__name__ != '__main__'", False),
            ("__name__ is '__main__'", False),
            ("__name__ is not '__main__'", False),
            ("__name__ in ('__main__',)", False),
            ("__name__ not in ('__main__',)", False),
            ("__name__ < '__main__'", False),
            ("__name__ <= '__main__'", False),
            ("__name__ > '__main__'", False),
            ("__name__ >= '__main__'", False),
            ("__name__ == 'other'", False),
            ("__name__ == None", False),
            ("__name__ == main_name", False),
            ("__name__ == '__main__' == other", False),
            ("'__main__' == __name__", False),
            ("other == '__main__'", False),
            ("__name__", False),
        )
        for condition, expected in cases:
            with self.subTest(condition=condition):
                node = ast.parse(f"if {condition}:\n    pass\n").body[0]
                self.assertEqual(_is_main_guard(node), expected)
        self.assertFalse(_is_main_guard(ast.parse("__name__ == '__main__'")))

    def test_scope_and_report_do_not_label_other_comparisons_as_main(self):
        conditions = (
            ("__name__ == '__main__'", "<__main__ block>"),
            ("__name__ != '__main__'", "<module>"),
            ("__name__ is '__main__'", "<module>"),
            ("__name__ in ('__main__',)", "<module>"),
            ("__name__ == 'other'", "<module>"),
            ("__name__ == '__main__' == other", "<module>"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (condition, expected) in enumerate(conditions):
                with self.subTest(condition=condition):
                    filename = f"case_{index}.py"
                    # Keep an unrelated scope so module lines resolve, not <unresolved>.
                    (root / filename).write_text(
                        f"def anchor():\n    return None\n\nif {condition}:\n    assert True\n",
                        encoding="utf-8",
                    )
                    payload = {"results": [
                        {"test_id": "B101", "issue_severity": "LOW",
                         "filename": filename, "line_number": line}
                        for line in (4, 5)
                    ]}
                    for line in (4, 5):
                        self.assertEqual(resolve_scope(filename, line, root), expected)
                    report = build_report(payload, root=root)
                    self.assertIn("Total findings: **2**", report)
                    self.assertIn(f"`{expected}` | 2 |", report)
                    if expected == "<module>":
                        self.assertNotIn("<__main__ block>", report)

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
