"""Contrato: linha de estado fora do bloco NEXUS-CURRENT-STATE falha o auditor.

O bloco interno pode concordar com o manifesto. Prosa histórica continua
ignorada. Na coluna 0, fora do bloco, ``commit:``, ``ci_run:``, ``tests:``,
``bandit_low:`` e um ``-->`` isolado são falha — o mesmo critério de
``tools/agente/replay/check_resolution.sh`` (L-002; forma do PR #54).

Teste dos três casos, escrito antes da correção:
(a) origem: ``tests:``, ``bandit_low:`` e ``-->`` soltos passam a falhar;
(b) sucesso anterior: a prosa histórica e os quatro documentos reais seguem em pass;
(c) semelhante incompatível: ``commit:`` e ``ci_run:`` soltos também falham.

Valores que parecem bool, NaN, infinito ou negativo não são normalizados: a
linha bruta entra no erro. Indentação, falta de espaço depois de ``:`` e
maiúscula ficam de fora, como no guarda de replay, e o teste trava esse limite.
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from hypothesis import assume, example, given, settings, strategies as st

from tools.state_consistency_audit import DOCUMENTS, ROOT, audit

SYNCED = {"commit": "HEAD", "ci_run": "CURRENT_RUN", "tests": "CURRENT_TESTS", "bandit_low": "131"}
# Critério copiado de tools/agente/replay/check_resolution.sh (coluna 0).
STRAY_RE = re.compile(r"^(?:commit|ci_run|tests|bandit_low): |^-->$")
OPEN_RE = re.compile(r"^<!--\s*NEXUS-CURRENT-STATE\b")

# (a) e (c), mais tipos inválidos escritos como texto (sem coerção).
STRAY_LINES = (
    "tests: CURRENT_TESTS",
    "bandit_low: 131",
    "-->",
    "commit: HEAD",
    "ci_run: CURRENT_RUN",
    "tests: True",
    "tests: False",
    "tests: nan",
    "tests: inf",
    "tests: -inf",
    "bandit_low: -1",
    "tests: 1.5",
    "commit: ",
)
# (b) e limites que o guarda de replay também não trata.
IGNORED_LINES = (
    "Historical checkpoint: 86 tests, commit old.",
    "see tests: 303 in the previous round",
    "  tests: 303",
    "tests:303",
    "Tests: 1",
    "--> ",
    "bandit_low:-1",
    "ci_run:CURRENT_RUN",
)


def _fixture(claims=None, suffixes=None, executed=307):
    directory = Path(tempfile.mkdtemp())
    manifest = {
        "schema_version": 2,
        "repository": {"commit": "abc123"},
        "verification": {
            "tests": {
                "status": "generated", "executed": executed, "passed": executed,
                "failed": 0, "skipped": 0, "unexpected_successes": 0, "result": "passed",
            },
            "integrated": {
                "status": "generated", "discovered_static": 3, "passed": 2,
                "skipped": 1, "failed": 0, "unexpected": 0, "result": "passed",
            },
        },
        "security": {"bandit": {"severity": {"LOW": 131}}},
        "ci": {"run_id": "999", "commit": "abc123", "gates_status": "success"},
    }
    path = directory / "STATE_MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    extras = suffixes or {}
    for name in DOCUMENTS:
        values = dict(SYNCED)
        values.update((claims or {}).get(name, {}))
        body = "\n".join(f"{key}: {value}" for key, value in values.items())
        extra = extras.get(name, "")
        (directory / name).write_text(
            f"before\n<!-- NEXUS-CURRENT-STATE\n{body}\n-->\nafter\n{extra}",
            encoding="utf-8",
        )
    return path, directory


def _with_line(name, line):
    return _fixture(suffixes={name: line + "\n"})


class StrayStateMarkerContract(unittest.TestCase):
    def test_stray_state_line_fails_and_keeps_the_raw_text(self):
        for line in STRAY_LINES:
            with self.subTest(line=line):
                report = audit(*_with_line("NEXUS_EVIDENCE.md", line))
                self.assertEqual(report["status"], "fail", report)
                self.assertEqual(report["conflicts"], [])
                self.assertTrue(report["historical_text_ignored"])
                matched = [
                    error for error in report["errors"]
                    if "fora do bloco" in error and line in error
                ]
                self.assertEqual(matched, [
                    f"NEXUS_EVIDENCE.md: linha de estado fora do bloco "
                    f"NEXUS-CURRENT-STATE: 9: {line}"
                ])

    def test_similar_fields_are_not_exempt(self):
        # (c) commit e ci_run soltos não podem passar só porque tests: foi o caso de origem.
        for line in ("commit: HEAD", "ci_run: CURRENT_RUN"):
            with self.subTest(line=line):
                report = audit(*_with_line("SECURITY_TRIAGE.md", line))
                self.assertEqual(report["status"], "fail")
                self.assertTrue(any(line in error for error in report["errors"]))

    def test_prose_history_and_replay_blind_spots_stay_accepted(self):
        for line in IGNORED_LINES:
            with self.subTest(line=line):
                report = audit(*_with_line("CONTINUATION.md", line))
                self.assertEqual(report["status"], "pass", report)
                self.assertEqual(report["errors"], [])
                self.assertEqual(report["conflicts"], [])

    def test_repository_documents_have_no_stray_state_lines(self):
        manifest, directory = _fixture()
        for name in DOCUMENTS:
            (directory / name).write_text(
                (ROOT / name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        report = audit(manifest, directory)
        self.assertEqual(report["status"], "pass", report)

    def test_several_stray_lines_are_all_reported(self):
        lines = ("tests: CURRENT_TESTS", "bandit_low: 131", "-->")
        report = audit(*_fixture(suffixes={"CONTINUATION.md": "\n".join(lines) + "\n"}))
        self.assertEqual(report["status"], "fail")
        blob = "\n".join(report["errors"])
        for line in lines:
            self.assertIn(line, blob)

    def test_stray_line_does_not_hide_another_documents_conflict(self):
        report = audit(*_fixture(
            claims={"NEXUS_KNOWLEDGE.md": {"tests": "1"}},
            suffixes={"NEXUS_EVIDENCE.md": "tests: True\n"},
        ))
        self.assertEqual(report["status"], "fail")
        self.assertIn(
            ("NEXUS_KNOWLEDGE.md", "tests"),
            {(item["document"], item["field"]) for item in report["conflicts"]},
        )
        self.assertTrue(any("tests: True" in error for error in report["errors"]))

    @settings(max_examples=40, deadline=None)
    @given(
        line=st.text(
            alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            min_size=0,
            max_size=48,
        )
    )
    @example("tests: CURRENT_TESTS")
    @example("commit: HEAD")
    @example("ci_run: CURRENT_RUN")
    @example("bandit_low: -1")
    @example("tests: True")
    @example("tests: nan")
    @example("-->")
    @example("Historical checkpoint: 86 tests, commit old.")
    @example("  tests: 303")
    @example("tests:303")
    @example("")
    def test_printable_line_fails_only_on_the_replay_pattern(self, line):
        assume(not OPEN_RE.match(line))
        report = audit(*_with_line("NEXUS_KNOWLEDGE.md", line))
        if STRAY_RE.match(line):
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any(line in error and "fora do bloco" in error for error in report["errors"]))
        else:
            self.assertEqual(report["status"], "pass", report)


if __name__ == "__main__":
    unittest.main()
