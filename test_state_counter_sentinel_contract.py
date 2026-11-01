"""Contrato: contador de testes derivado do manifesto (D1).

Os blocos NEXUS-CURRENT-STATE podem declarar ``tests: CURRENT_TESTS``. O valor é
resolvido para ``verification.tests.executed`` do manifesto gerado pelo CI, do mesmo
modo que ``commit: HEAD`` e ``ci_run: CURRENT_RUN``. Cada sentinela só vale no próprio
campo; números literais continuam exigindo igualdade exata.
"""
import json
import tempfile
import unittest
from pathlib import Path

from tools.state_consistency_audit import DOCUMENTS, ROOT, _parse_marker, audit

SYNCED = {"commit": "HEAD", "ci_run": "CURRENT_RUN", "tests": "CURRENT_TESTS", "bandit_low": "131"}


def _fixture(executed=307, claims=None, tests_overrides=None):
    directory = Path(tempfile.mkdtemp())
    tests = {"status": "generated", "executed": executed, "passed": executed, "failed": 0,
             "skipped": 0, "unexpected_successes": 0, "result": "passed"}
    tests.update(tests_overrides or {})
    manifest = {
        "schema_version": 2,
        "repository": {"commit": "abc123"},
        "verification": {"tests": tests, "integrated": {
            "status": "generated", "discovered_static": 3, "passed": 2,
            "skipped": 1, "failed": 0, "unexpected": 0, "result": "passed"}},
        "security": {"bandit": {"severity": {"LOW": 131}}},
        "ci": {"run_id": "999", "commit": "abc123", "gates_status": "success"},
    }
    path = directory / "STATE_MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    for name in DOCUMENTS:
        values = dict(SYNCED)
        values.update((claims or {}).get(name, {}))
        body = "\n".join(f"{k}: {v}" for k, v in values.items())
        (directory / name).write_text(f"x\n<!-- NEXUS-CURRENT-STATE\n{body}\n-->\n", encoding="utf-8")
    return path, directory


def _all(field, value):
    return {name: {field: value} for name in DOCUMENTS}


class CounterSentinelContract(unittest.TestCase):
    def test_sentinel_passes_for_any_executed_count(self):
        for executed in (0, 303, 307, 311):
            with self.subTest(executed=executed):
                report = audit(*_fixture(executed=executed))
                self.assertEqual(report["status"], "pass", report)

    def test_literal_equal_count_still_passes(self):
        report = audit(*_fixture(executed=303, claims=_all("tests", "303")))
        self.assertEqual(report["status"], "pass", report)

    def test_literal_stale_count_still_fails(self):
        report = audit(*_fixture(executed=307, claims={"CONTINUATION.md": {"tests": "303"}}))
        self.assertEqual(report["status"], "fail")
        self.assertIn(("CONTINUATION.md", "tests"),
                      {(c["document"], c["field"]) for c in report["conflicts"]})

    def test_sentinel_outside_its_field_fails(self):
        cases = [("tests", "HEAD"), ("tests", "CURRENT_RUN"), ("commit", "CURRENT_TESTS"),
                 ("ci_run", "CURRENT_TESTS"), ("bandit_low", "CURRENT_TESTS"),
                 ("tests", "current_tests"), ("commit", "CURRENT_RUN"), ("ci_run", "HEAD")]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                claims = _all("tests", "307")  # isola o efeito: só a sentinela fora do campo
                claims["NEXUS_EVIDENCE.md"] = {"tests": "307", field: value}
                report = audit(*_fixture(executed=307, claims=claims))
                self.assertEqual(report["status"], "fail", report)

    def test_sentinel_does_not_hide_failed_or_invalid_runs(self):
        for overrides in ({"result": "failed", "failed": 1, "passed": 306},
                          {"executed": True}, {"status": "missing"}):
            with self.subTest(overrides=overrides):
                report = audit(*_fixture(tests_overrides=overrides))
                self.assertEqual(report["status"], "fail", report)

    def test_repository_documents_use_derived_counter(self):
        for name in DOCUMENTS:
            with self.subTest(document=name):
                claims = _parse_marker((ROOT / name).read_text(encoding="utf-8"), name)
                self.assertEqual(claims["tests"], "CURRENT_TESTS")


if __name__ == "__main__":
    unittest.main()
