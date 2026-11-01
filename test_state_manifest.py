import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.state_manifest import _test_summary, build_manifest, main


class StateManifestTests(unittest.TestCase):
    def _check_test_summary(self, content, expected):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if content is not None:
                if isinstance(content, bytes):
                    (root / "test-summary.json").write_bytes(content)
                else:
                    (root / "test-summary.json").write_text(content, encoding="utf-8")
            with patch("tools.state_manifest.ROOT", root), patch("tools.state_manifest._git", return_value="fixture"), patch("tools.state_manifest._document_commits", return_value={}):
                self.assertEqual(_test_summary(), expected)
                manifest = build_manifest("passed", "fixture command")
            tests = manifest["verification"]["tests"]
            self.assertEqual(tests, {
                "scope": "unittest discover (test_*.py); embedded checks are under verification.integrated",
                "discovered_static": 0,
                **expected,
                "command": "fixture command",
                "result": expected.get("result", "passed"),
            })
            self.assertEqual(manifest["schema_version"], 2)

    def test_missing_test_summary_is_not_generated(self):
        self._check_test_summary(None, {"status": "not_generated"})

    def test_unreadable_test_summary_is_invalid(self):
        with patch.object(Path, "read_text", side_effect=PermissionError("fixture")):
            self._check_test_summary("{}", {"status": "invalid_summary"})

    def test_malformed_test_summary_is_invalid(self):
        for content in ("", "not json", '{"executed":', b'{\xff}'):
            with self.subTest(content=content):
                self._check_test_summary(content, {"status": "invalid_summary"})

    def test_non_json_constants_in_extra_fields_are_invalid(self):
        valid = dict(executed=1, passed=1, failed=0, skipped=0, unexpected_successes=0, result="passed")
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                content = json.dumps(valid)[:-1] + ', "extra": {"nested": [' + token + ']}}'
                self._check_test_summary(content, {"status": "invalid_summary"})

    def test_non_object_test_summary_is_invalid(self):
        for value in (42, 1.5, None, True, False, "executed", [], ["executed"]):
            with self.subTest(value=value):
                self._check_test_summary(json.dumps(value), {"status": "invalid_summary"})

    def test_test_summary_requires_all_fields(self):
        valid = dict(executed=1, passed=1, failed=0, skipped=0, unexpected_successes=0, result="passed")
        for missing in valid:
            with self.subTest(missing=missing):
                summary = {key: value for key, value in valid.items() if key != missing}
                self._check_test_summary(json.dumps(summary), {"status": "invalid_summary"})

    def test_test_summary_rejects_invalid_field_types_and_counts(self):
        valid = dict(executed=1, passed=1, failed=0, skipped=0, unexpected_successes=0, result="passed")
        changes = [
            {field: value}
            for field in ("executed", "passed", "failed", "skipped", "unexpected_successes")
            for value in (True, False, -1, 1.0, "1", None, [], {})
        ] + [
            {"result": value} for value in (None, True, 1, [], {}, "unknown", "failed")
        ] + [
            {"executed": 2}, {"unexpected_successes": 1},
            {"executed": 2, "failed": 1},
        ]
        for change in changes:
            with self.subTest(change=change):
                self._check_test_summary(json.dumps({**valid, **change}), {"status": "invalid_summary"})

    def test_valid_test_summary_is_generated(self):
        for summary in (
            dict(executed=0, passed=0, failed=0, skipped=0, unexpected_successes=0, result="passed"),
            dict(executed=3, passed=2, failed=0, skipped=1, unexpected_successes=0, result="passed"),
            dict(executed=3, passed=1, failed=1, skipped=1, unexpected_successes=1, result="failed"),
        ):
            with self.subTest(summary=summary):
                self._check_test_summary(json.dumps({**summary, "extra": "ignored"}), {"status": "generated", **summary})

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
