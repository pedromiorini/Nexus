"""Casos estruturais já cobertos pelo preflight de Axis e ausentes nos testes do patch.

Não altera a semântica aplicada em 1b2151c. Não exige rejeitar substituição idêntica:
uma agulha que ocorre uma vez continua válida, mesmo se o texto da substituição for igual.
"""
from __future__ import annotations

import unittest
from unittest import mock

from test_cognitivo_mutation_parallel_contract import TESTS, Project, valid_synthetic_catalog
from tools import cognitivo_mutation as cm
from tools.targeted_mutation import Mutation


def _good():
    return next(m for m in valid_synthetic_catalog() if m.name == "kills-f")


class StructuralPreflightEdges(unittest.TestCase):
    def _abort(self, mutations, *, create=True, argv=None):
        argv = ["--shard", "1/2"] if argv is None else argv
        with Project() as p:
            report = p.root / "report.json"
            prior = b"existing report must remain byte-for-byte"
            if create:
                report.write_bytes(prior)
            with mock.patch.object(cm, "run_catalog", return_value=[]) as run, mock.patch("sys.stderr"):
                code = cm.main(argv, mutations=mutations, root=p.root, report=report)
            if create:
                self.assertEqual(report.read_bytes(), prior)
            else:
                self.assertFalse(report.exists())
            run.assert_not_called()
            self.assertEqual(code, 2)
            self.assertTrue(cm.preflight_catalog(mutations, p.root))

    def test_empty_list_and_tuple_abort_and_keep_the_report(self):
        self._abort([])
        self._abort(())

    def test_missing_report_stays_missing(self):
        self._abort([], create=False)

    def test_duplicate_name_outside_the_selected_shard_aborts(self):
        self._abort([_good(), Mutation("kills-f", "mod.py", "return 1\n", "return 2\n", TESTS)])

    def test_paths_types_and_empty_needle_abort(self):
        good = _good()
        cases = [
            Mutation("abs", "/etc/passwd", "x", "y", TESTS),
            Mutation("up", "../outside.py", "x", "y", TESTS),
            Mutation("slash", "mod\\x.py", "x", "y", TESTS),
            Mutation("null", "mod.py\x00", "x", "y", TESTS),
            Mutation("empty-needle", "mod.py", "", "y", TESTS),
            Mutation("bad-tests", "mod.py", "return x + 1", "return 0", "test_mod.py"),
            object(),
        ]
        for item in cases:
            with self.subTest(item=getattr(item, "name", type(item).__name__)):
                self._abort([good, item])


if __name__ == "__main__":
    unittest.main()
