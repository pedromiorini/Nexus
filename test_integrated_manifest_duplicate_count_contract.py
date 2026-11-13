"""Contrato: falha só por ids repetidos publica a contagem no manifesto.

Fenômeno em ``6c608f9`` (leitura de ``tools/state_manifest.py``).
``_integrated_summary`` aceita um resumo cujo único motivo de falha é
``duplicates`` não vazio, sem ``missing``, ``unexpected`` ou ``conflicting``,
e devolve ``status`` ``generated``, ``result`` ``failed``, ``failed`` 0 e
``unexpected`` 0. O dict de retorno não tem chave ``duplicates``.

Hipóteses, cada uma com o que a derruba:

- H1. A contagem já está no dict gerado. Derruba: ``duplicates`` ausente.
- H2. Id repetido fora de ``passed``, ``skipped`` e ``unexpected`` já fica
  ``invalid_summary``. Derruba: o manifesto devolve ``generated``.
- H3. ``result`` incoerente com as listas já fica ``invalid_summary``.
  Derruba: o manifesto devolve ``generated``.
- H4. Sem a chave, ou com lista vazia, a contagem publicada já é 0.
  Derruba: ``duplicates`` ausente no dict gerado.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.state_manifest import build_manifest


def _base() -> dict:
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


def _read(summary: dict) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "integrated-summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        with patch("tools.state_manifest.ROOT", root), patch(
            "tools.state_manifest._git", return_value="fixture"
        ), patch("tools.state_manifest._document_commits", return_value={}):
            return build_manifest("passed")["verification"]["integrated"]


class IntegratedDuplicateCountContractTests(unittest.TestCase):
    def test_duplicate_only_failure_records_the_count(self):
        cases = (
            ("passed id", {**_base(), "duplicates": ["1"], "result": "failed"}, 1, 1, 0),
            (
                "skipped id",
                {
                    **_base(),
                    "expected": ["1", "3"],
                    "skipped": ["3"],
                    "duplicates": ["3"],
                    "result": "failed",
                },
                1,
                2,
                1,
            ),
            (
                "two ids",
                {
                    **_base(),
                    "expected": ["1", "2"],
                    "passed": ["1", "2"],
                    "duplicates": ["1", "2"],
                    "result": "failed",
                },
                2,
                2,
                0,
            ),
        )
        for name, summary, count, discovered, skipped in cases:
            with self.subTest(name=name):
                got = _read(summary)
                self.assertEqual(got["status"], "generated")
                self.assertEqual(got["result"], "failed")
                self.assertEqual(got["failed"], 0)
                self.assertEqual(got["unexpected"], 0)
                self.assertEqual(got["passed"], discovered - skipped)
                self.assertEqual(got["skipped"], skipped)
                self.assertEqual(got["discovered_static"], discovered)
                self.assertEqual(got.get("duplicates"), count)
                self.assertEqual(
                    got,
                    {
                        "status": "generated",
                        "scope": "embedded checks",
                        "discovered_static": discovered,
                        "passed": discovered - skipped,
                        "skipped": skipped,
                        "failed": 0,
                        "unexpected": 0,
                        "duplicates": count,
                        "result": "failed",
                    },
                )

    def test_duplicate_outside_terminal_ids_stays_invalid(self):
        outside = (
            ["2"],
            ["9"],
            ["1", "2"],
        )
        for items in outside:
            with self.subTest(items=items):
                summary = {**_base(), "duplicates": items, "result": "failed"}
                self.assertEqual(_read(summary), {"status": "invalid_summary"})

        only_missing = _base()
        only_missing["expected"] = ["1", "2"]
        only_missing["missing"] = ["2"]
        only_missing["duplicates"] = ["2"]
        only_missing["result"] = "failed"
        self.assertEqual(_read(only_missing), {"status": "invalid_summary"})

    def test_inconsistent_result_stays_invalid(self):
        hidden = {**_base(), "duplicates": ["1"], "result": "passed"}
        self.assertEqual(_read(hidden), {"status": "invalid_summary"})

        false_failure = {**_base(), "duplicates": [], "result": "failed"}
        self.assertEqual(_read(false_failure), {"status": "invalid_summary"})

        no_cause = {**_base(), "result": "failed"}
        self.assertEqual(_read(no_cause), {"status": "invalid_summary"})

        wrong_token = {**_base(), "duplicates": ["1"], "result": "Failed"}
        self.assertEqual(_read(wrong_token), {"status": "invalid_summary"})

    def test_absent_duplicates_count_as_zero(self):
        absent = _base()
        self.assertNotIn("duplicates", absent)
        got = _read(absent)
        self.assertEqual(got["status"], "generated")
        self.assertEqual(got["result"], "passed")
        self.assertEqual(got["failed"], 0)
        self.assertEqual(got["unexpected"], 0)
        self.assertEqual(got.get("duplicates"), 0)

    def test_empty_duplicates_count_as_zero(self):
        empty = {**_base(), "duplicates": []}
        got_empty = _read(empty)
        self.assertEqual(got_empty["status"], "generated")
        self.assertEqual(got_empty["result"], "passed")
        self.assertEqual(got_empty["failed"], 0)
        self.assertEqual(got_empty["unexpected"], 0)
        self.assertEqual(got_empty.get("duplicates"), 0)
        self.assertEqual(
            got_empty,
            {
                "status": "generated",
                "scope": "embedded checks",
                "discovered_static": 1,
                "passed": 1,
                "skipped": 0,
                "failed": 0,
                "unexpected": 0,
                "duplicates": 0,
                "result": "passed",
            },
        )


if __name__ == "__main__":
    unittest.main()
