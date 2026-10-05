import json
import tempfile
import unittest
from pathlib import Path

from tools.state_consistency_audit import DOCUMENTS, audit


class StateConsistencyAuditTests(unittest.TestCase):
    def _write_fixture(self, *, commit="abc123", tests=107, bandit_low=131, marker_overrides=None):
        directory = Path(tempfile.mkdtemp())
        manifest = {
            "schema_version": 1,
            "repository": {"commit": commit},
            "verification": {"tests": {"executed": tests}},
            "security": {"bandit": {"severity": {"LOW": bandit_low}}},
            "ci": {"run_id": "12345"},
        }
        manifest_path = directory / "STATE_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        overrides = marker_overrides or {}
        for name in DOCUMENTS:
            claims = {
                "commit": "HEAD",
                "ci_run": "CURRENT_RUN",
                "tests": str(tests),
                "bandit_low": str(bandit_low),
            }
            claims.update(overrides.get(name, {}))
            body = "\n".join(f"{key}: {value}" for key, value in claims.items())
            (directory / name).write_text(
                f"before\n<!-- NEXUS-CURRENT-STATE\n{body}\n-->\nafter\n",
                encoding="utf-8",
            )
        return directory, manifest_path

    def test_synchronized_documents_pass_and_history_is_ignored(self):
        directory, manifest = self._write_fixture()
        (directory / "NEXUS_KNOWLEDGE.md").write_text(
            (directory / "NEXUS_KNOWLEDGE.md").read_text(encoding="utf-8")
            + "\nHistorical checkpoint: 86 tests, commit old.\n",
            encoding="utf-8",
        )
        report = audit(manifest, directory)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["historical_text_ignored"])

    def test_divergent_test_count_is_a_conflict(self):
        directory, manifest = self._write_fixture(
            marker_overrides={"NEXUS_EVIDENCE.md": {"tests": "105"}}
        )
        report = audit(manifest, directory)
        self.assertEqual(report["status"], "fail")
        self.assertIn("tests", {item["field"] for item in report["conflicts"]})

    def test_divergent_commit_and_security_are_conflicts(self):
        directory, manifest = self._write_fixture(
            marker_overrides={
                "SECURITY_TRIAGE.md": {"commit": "oldsha", "bandit_low": "129"},
            }
        )
        report = audit(manifest, directory)
        self.assertEqual(report["status"], "fail")
        fields = {item["field"] for item in report["conflicts"]}
        self.assertEqual(fields, {"bandit_low", "commit"})

    def test_missing_marker_fails_without_inventing_state(self):
        directory, manifest = self._write_fixture()
        (directory / "CONTINUATION.md").write_text("historical only\n", encoding="utf-8")
        report = audit(manifest, directory)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("bloco" in error for error in report["errors"]))

    def test_invalid_manifest_fails_closed(self):
        directory, manifest = self._write_fixture()
        manifest.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            audit(manifest, directory)


if __name__ == "__main__":
    unittest.main()
