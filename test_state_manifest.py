import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.state_manifest import build_manifest, main


class StateManifestTests(unittest.TestCase):
    def test_manifest_contains_current_commit_and_test_inventory(self):
        with patch.dict("os.environ", {"GITHUB_SHA": "abc123", "GITHUB_REF_NAME": "main"}, clear=False):
            manifest = build_manifest("passed", "python -m unittest discover", "success")
        self.assertEqual(manifest["repository"]["commit"], "abc123")
        self.assertEqual(manifest["repository"]["branch"], "main")
        self.assertEqual(manifest["verification"]["tests"]["result"], "passed")
        self.assertGreaterEqual(manifest["verification"]["tests"]["discovered_static"], 1)
        self.assertEqual(manifest["ci"]["status"], "success")

    def test_cli_writes_json_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            with patch("tools.state_manifest.ROOT", Path.cwd()):
                self.assertEqual(main(["--output", str(output), "--test-result", "passed"]), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertIn("provenance", payload)
            self.assertIn("discovered_static", payload["verification"]["tests"])

    def test_manifest_preserves_executed_test_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test-summary.json").write_text(
                '{"executed": 11, "passed": 10, "failed": 1, "skipped": 0, "result": "failed"}',
                encoding="utf-8",
            )
            with patch("tools.state_manifest.ROOT", root), patch("tools.state_manifest._git", return_value="fixture"), patch("tools.state_manifest._document_commits", return_value={}):
                manifest = build_manifest("passed")
            tests = manifest["verification"]["tests"]
            self.assertEqual(tests["executed"], 11)
            self.assertEqual(tests["failed"], 1)
            self.assertEqual(tests["result"], "failed")


if __name__ == "__main__":
    unittest.main()
