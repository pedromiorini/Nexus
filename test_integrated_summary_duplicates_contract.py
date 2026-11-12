"""Contrato: o manifesto lê o resumo que ``IntegratedSuiteTally`` grava.

Fenômeno em ``f2b439c`` (Python 3.12.3, sem patch). O tally publica
``passed`` e ``skipped`` já intersectados com ``expected``, e ``unexpected``
fica de fora desses dois conjuntos. ``duplicates`` lista ids terminais
repetidos e, sozinho, faz ``result`` ser ``failed``.

``tools.state_manifest._integrated_summary`` recalcula ``unexpected`` como
``(passed | skipped) - expected`` e ignora ``duplicates``.

Hipóteses, cada uma com o que a derruba:

- H1. Um resumo só com ``duplicates`` e ``result: failed`` já entra como
  suíte gerada que falhou. Derruba: o manifesto devolve ``invalid_summary``.
- H2. Um id em ``unexpected`` já entra como falha gerada. Derruba: o mesmo
  ``invalid_summary``.
- H3. ``result: passed`` com ``duplicates`` não vazio já é recusado. Derruba:
  o manifesto devolve ``generated`` e ``result: passed``.
- H4. Tipo inválido em ``duplicates`` (``true``, ``null``, ``str``, número,
  item ``bool``/número, id repetido, id ``-1``) já é recusado quando o resto
  do resumo passaria. Derruba: esses valores saem ``generated``.
- H5. Exigir a chave ``duplicates`` quebraria o resumo que já passa sem ela.
  Derruba: ausência da chave, lista vazia, skip e conflito continuam
  ``generated`` com o ``result`` do tally.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from core.integrated_suite_summary import IntegratedSuiteTally
from tools.state_manifest import build_manifest


def _tally(source: str, log: str) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "runner.py"
        path.write_text(source, encoding="utf-8")
        tally = IntegratedSuiteTally(StringIO())
        tally.write(log)
        return tally.build_summary(path)


def _manifest(summary: dict | None = None, raw: str | None = None) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        text = raw if raw is not None else json.dumps(summary)
        (root / "integrated-summary.json").write_text(text, encoding="utf-8")
        with patch("tools.state_manifest.ROOT", root), patch(
            "tools.state_manifest._git", return_value="fixture"
        ), patch("tools.state_manifest._document_commits", return_value={}):
            return build_manifest("passed")["verification"]["integrated"]


def _clean() -> dict:
    return {
        "schema_version": 1,
        "scope": "embedded checks",
        "expected": ["1"],
        "passed": ["1"],
        "skipped": [],
        "missing": [],
        "unexpected": [],
        "conflicting": [],
        "result": "passed",
    }


class IntegratedSummaryManifestContractTests(unittest.TestCase):
    def test_duplicate_only_tally_is_a_generated_failure(self):
        summary = _tally('"TEST 1 PASSED"\n', "TEST 1 PASSED\nTEST 1 PASSED\n")
        self.assertEqual(summary["duplicates"], ["1"])
        self.assertEqual(summary["unexpected"], [])
        self.assertEqual(summary["missing"], [])
        self.assertEqual(summary["conflicting"], [])
        self.assertEqual(summary["result"], "failed")

        got = _manifest(summary)

        self.assertEqual(got["status"], "generated")
        self.assertEqual(got["result"], "failed")
        self.assertEqual(got["discovered_static"], 1)
        self.assertEqual(got["passed"], 1)
        self.assertEqual(got["failed"], 0)
        self.assertEqual(got["unexpected"], 0)

    def test_unexpected_check_is_a_generated_failure(self):
        summary = _tally('"TEST 1 PASSED"\n', "TEST 1 PASSED\nTEST 2 PASSED\n")
        self.assertEqual(summary["unexpected"], ["2"])
        self.assertEqual(summary["passed"], ["1"])
        self.assertEqual(summary["result"], "failed")

        got = _manifest(summary)

        self.assertEqual(got["status"], "generated")
        self.assertEqual(got["result"], "failed")
        self.assertEqual(got["unexpected"], 1)
        self.assertEqual(got["discovered_static"], 1)
        self.assertEqual(got["passed"], 1)

    def test_duplicate_of_an_unexpected_id_is_a_generated_failure(self):
        summary = _tally(
            '"TEST 1 PASSED"\n"TEST 2 PASSED"\n',
            "TEST 1 PASSED\nTEST 1 PASSED\n"
            "TEST 10 PASSED\nTEST 10 PASSED\n"
            "TEST 0.5: feature DISABLED\nTEST 0.5: feature DISABLED\n",
        )
        self.assertEqual(summary["duplicates"], ["0.5", "1", "10"])
        self.assertEqual(summary["unexpected"], ["0.5", "10"])
        self.assertEqual(summary["missing"], ["2"])
        self.assertEqual(summary["result"], "failed")

        got = _manifest(summary)

        self.assertEqual(got["status"], "generated")
        self.assertEqual(got["result"], "failed")
        self.assertEqual(got["unexpected"], 2)
        self.assertEqual(got["failed"], 1)
        self.assertEqual(got["discovered_static"], 2)

    def test_passed_result_cannot_hide_duplicates(self):
        summary = _tally('"TEST 1 PASSED"\n', "TEST 1 PASSED\nTEST 1 PASSED\n")
        summary["result"] = "passed"

        self.assertEqual(_manifest(summary), {"status": "invalid_summary"})

    def test_invalid_duplicate_values_stay_invalid_on_a_passing_summary(self):
        invalid = (
            True,
            False,
            None,
            "1",
            1,
            1.0,
            {"1": 1},
            [True],
            [False],
            [1],
            [1.0],
            ["1", "1"],
            ["-1"],
            ["1.2.3"],
            [""],
            [" 1"],
            ["2"],
        )
        for value in invalid:
            with self.subTest(value=value):
                summary = _clean()
                summary["duplicates"] = value
                self.assertEqual(_manifest(summary), {"status": "invalid_summary"})

    def test_non_json_duplicate_constant_is_invalid(self):
        raw = json.dumps(_clean())[:-1] + ', "duplicates": NaN}'
        self.assertEqual(_manifest(raw=raw), {"status": "invalid_summary"})

    def test_summaries_the_tally_already_agrees_on_stay_generated(self):
        clean = _tally('"TEST 1 PASSED"\n', "TEST 1 PASSED\n")
        skipped = _tally(
            '"TEST 1 PASSED"\n"TEST 2 PASSED"\n',
            "TEST 1 PASSED\nTEST 2: feature not enabled\n",
        )
        conflict = _tally(
            '"TEST 1 PASSED"\n',
            "TEST 1 PASSED\nTEST 1: feature DISABLED\n",
        )
        self.assertEqual(conflict["duplicates"], ["1"])
        self.assertEqual(conflict["conflicting"], ["1"])

        for name, summary, result in (
            ("clean", clean, "passed"),
            ("skipped", skipped, "passed"),
            ("conflict", conflict, "failed"),
        ):
            with self.subTest(name=name):
                got = _manifest(summary)
                self.assertEqual(got["status"], "generated")
                self.assertEqual(got["result"], result)

        omitted = _clean()
        self.assertNotIn("duplicates", omitted)
        self.assertEqual(_manifest(omitted)["status"], "generated")
        self.assertEqual(_manifest(omitted)["result"], "passed")

        empty = _clean()
        empty["duplicates"] = []
        self.assertEqual(_manifest(empty)["result"], "passed")

        inconsistent = _clean()
        inconsistent["expected"] = ["1", "2"]
        inconsistent["missing"] = []
        self.assertEqual(_manifest(inconsistent), {"status": "invalid_summary"})

        orphan = _clean()
        orphan["duplicates"] = ["2"]
        orphan["result"] = "failed"
        self.assertEqual(_manifest(orphan), {"status": "invalid_summary"})


if __name__ == "__main__":
    unittest.main()
