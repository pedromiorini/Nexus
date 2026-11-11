"""Contrato: ``line_number`` só nomeia escopo se for ``int`` exato.

Fenômeno em ``5776787``: ``resolve_scope`` faz ``int(line)``. ``bool`` é
subclasse de ``int``, então ``True`` vira a linha 1 e cai no escopo real.
``int(1.9)`` e ``int("1")`` também. ``False`` vira 0 e sai como ``<module>``.
``±inf`` levanta ``OverflowError`` e ``build_report`` não devolve o relatório.

Hipóteses:
- H1: ``True`` já fica ``<unresolved>``. Falsifica se o achado entra em
  ``anchor``.
- H2: float e str numérica já são recusados. Falsifica se ``1.0`` ou ``"1"``
  entram em ``anchor``.
- H3: valor não conversível já falha fechado, sem exceção. Falsifica se
  ``±inf`` levanta, ou se ``False`` / ``"1_0"`` viram ``<module>``.
- H4: ``int`` exato já resolve. Controle: as linhas 1 e 2 continuam
  ``anchor``; ``0``, ``-1`` e ``3`` continuam ``<module>``. Arquivo ausente
  continua ``<unresolved>``.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hypothesis import example, given, settings, strategies as st

from tools.security_triage import _scope_spans, build_report, resolve_scope


SOURCE = "def anchor():\n    return None\n"


class CitedLine(int):
    """Subclasse de int. Não é linha exata."""


class SecurityTriageLineContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.filename = "mod.py"
        (cls.root / cls.filename).write_text(SOURCE, encoding="utf-8")
        _scope_spans.cache_clear()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _scope(self, line: object, filename: str | None = None) -> str:
        return resolve_scope(filename or self.filename, line, self.root)

    def test_exact_int_inside_the_span_stays_named(self):
        self.assertEqual(self._scope(1), "anchor")
        self.assertEqual(self._scope(2), "anchor")

    def test_exact_int_outside_the_span_stays_module(self):
        for line in (0, -1, 3, 10**6):
            with self.subTest(line=line):
                self.assertEqual(self._scope(line), "<module>")

    def test_missing_or_unreadable_file_stays_unresolved(self):
        for line in (1, True, float("inf"), None):
            with self.subTest(line=line):
                self.assertEqual(self._scope(line, "ausente.py"), "<unresolved>")

    def test_non_exact_int_is_not_coerced_into_a_scope(self):
        samples = (
            True,
            False,
            1.0,
            1.5,
            2.9,
            -1.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            "1",
            "2",
            " 1",
            "1_0",
            "",
            "linha",
            None,
            [],
            {},
            (1,),
            CitedLine(1),
        )
        for line in samples:
            with self.subTest(line=line):
                self.assertEqual(self._scope(line), "<unresolved>")

    def test_invalid_line_does_not_hide_or_steal_a_valid_scope(self):
        payload = {"results": [
            {"test_id": "B101", "issue_severity": "LOW", "filename": self.filename,
             "line_number": 2, "code": "dentro"},
            {"test_id": "B101", "issue_severity": "LOW", "filename": self.filename,
             "line_number": True, "code": "bool"},
        ]}
        report = build_report(payload, root=self.root)
        self.assertIn("Total findings: **2**", report)
        self.assertIn("`anchor` | 1 |", report)
        self.assertIn("`<unresolved>` | 1 |", report)
        self.assertEqual(payload["results"][1]["line_number"], True)

    def test_infinity_does_not_abort_the_report(self):
        for line in (float("inf"), float("-inf")):
            with self.subTest(line=line):
                payload = {"results": [
                    {"test_id": "B311", "issue_severity": "LOW",
                     "filename": self.filename, "line_number": line},
                ]}
                report = build_report(payload, root=self.root)
                self.assertIn("Total findings: **1**", report)
                self.assertIn("`<unresolved>` | 1 |", report)
                self.assertNotIn("`anchor`", report)

    @settings(max_examples=40, deadline=None)
    @given(st.one_of(
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(max_size=12),
        st.none(),
        st.lists(st.integers(min_value=-5, max_value=5), max_size=3),
        st.tuples(st.integers(min_value=1, max_value=2)),
    ))
    @example(True)
    @example(False)
    @example(1.0)
    @example(float("nan"))
    @example(float("inf"))
    @example(float("-inf"))
    @example("1")
    @example("1_0")
    @example(None)
    def test_property_non_int_never_names_a_scope(self, line):
        self.assertEqual(self._scope(line), "<unresolved>")


if __name__ == "__main__":
    unittest.main()
