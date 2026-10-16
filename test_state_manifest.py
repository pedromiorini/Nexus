import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.state_manifest import build_manifest, main


class StateManifestTests(unittest.TestCase):
    def test_manifest_contains_current_commit_and_test_inventory(self):
        with patch.dict("os.environ", {"GITHUB_SHA": "abc123", "GITHUB_REF_NAME": "main"}, clear=False):
            manifest = build_manifest("passed", "python -m unittest discover", gates_status="success")
        self.assertEqual(manifest["repository"]["commit"], "abc123")
        self.assertEqual(manifest["repository"]["branch"], "main")
        self.assertEqual(manifest["verification"]["tests"]["result"], "passed")
        self.assertGreaterEqual(manifest["verification"]["tests"]["discovered_static"], 1)
        self.assertEqual(manifest["ci"]["gates_status"], "success")

    def test_cli_writes_json_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            with patch("tools.state_manifest.ROOT", Path.cwd()):
                self.assertEqual(main(["--output", str(output), "--test-result", "passed"]), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 2)
            self.assertIn("provenance", payload)
            self.assertIn("discovered_static", payload["verification"]["tests"])

    def test_manifest_preserves_executed_test_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test-summary.json").write_text(
                '{"executed": 11, "passed": 10, "failed": 1, "skipped": 0, "unexpected_successes": 1, "result": "failed"}',
                encoding="utf-8",
            )
            with patch("tools.state_manifest.ROOT", root), patch("tools.state_manifest._git", return_value="fixture"), patch("tools.state_manifest._document_commits", return_value={}):
                manifest = build_manifest("passed")
            tests = manifest["verification"]["tests"]
            self.assertEqual(tests["executed"], 11)
            self.assertEqual(tests["failed"], 1)
            self.assertEqual(tests["unexpected_successes"], 1)
            self.assertEqual(tests["result"], "failed")

    def test_manifest_rejects_inconsistent_test_summary(self):
        invalid_summaries = (
            '{"executed": 2, "passed": 2, "failed": 1, "skipped": 0, "unexpected_successes": 0, "result": "passed"}',
            '{"executed": true, "passed": 1, "failed": 0, "skipped": 0, "unexpected_successes": 0, "result": "passed"}',
            '{"executed": 1, "passed": 0, "failed": 0, "skipped": 0, "unexpected_successes": 0, "result": "passed"}',
            '{"executed": 1, "passed": 1, "failed": 0, "skipped": 0, "unexpected_successes": 1, "result": "passed"}',
        )
        for content in invalid_summaries:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "test-summary.json").write_text(content, encoding="utf-8")
                with patch("tools.state_manifest.ROOT", root), patch("tools.state_manifest._git", return_value="fixture"), patch("tools.state_manifest._document_commits", return_value={}):
                    self.assertEqual(build_manifest()["verification"]["tests"]["status"], "invalid_summary")


    def _manifest_with_files(self, files):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in files.items():
                (root / name).write_text(content, encoding="utf-8")
            with patch("tools.state_manifest.ROOT", root), patch("tools.state_manifest._git", return_value="fixture"), patch("tools.state_manifest._document_commits", return_value={}):
                return build_manifest("passed")

    def test_manifest_reports_integrated_checks_separately_from_unittest(self):
        summary = {
            "schema_version": 1,
            "scope": "embedded checks",
            "expected": ["1", "2", "3"],
            "passed": ["1", "2"],
            "skipped": ["3"],
            "missing": [],
            "unexpected": [],
            "conflicting": [],
            "result": "passed",
        }
        manifest = self._manifest_with_files({"integrated-summary.json": json.dumps(summary)})
        self.assertEqual(
            manifest["verification"]["integrated"],
            {
                "status": "generated", "scope": "embedded checks",
                "discovered_static": 3, "passed": 2, "skipped": 1,
                "failed": 0, "unexpected": 0, "result": "passed",
            },
        )

    def test_missing_or_invalid_integrated_summary_is_not_assumed(self):
        cases = (
            ({}, "not_generated"),
            ({"integrated-summary.json": "not json"}, "invalid_summary"),
            ({"integrated-summary.json": "{}"}, "invalid_summary"),
        )
        for files, status in cases:
            with self.subTest(status=status):
                manifest = self._manifest_with_files(files)
                self.assertEqual(manifest["verification"]["integrated"], {"status": status})

    def test_inconsistent_integrated_summary_is_invalid(self):
        summary = {
            "schema_version": 1,
            "scope": "embedded checks",
            "expected": ["1", "2"],
            "passed": ["1"],
            "skipped": [],
            "missing": [],
            "unexpected": [],
            "conflicting": [],
            "result": "passed",
        }
        manifest = self._manifest_with_files({"integrated-summary.json": json.dumps(summary)})
        self.assertEqual(manifest["verification"]["integrated"], {"status": "invalid_summary"})


if __name__ == "__main__":
    unittest.main()
