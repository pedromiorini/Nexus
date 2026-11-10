"""Contrato: ``line_number`` de B101 só isenta se for ``int`` exato.

Fenômeno em ``621ddb76``: ``audit`` compara a faixa do bloco ``__main__`` com
``int(line_number)``. ``bool`` é subclasse de ``int``, então ``True`` vira a
linha 1. ``int(1.5)`` e ``int("2")`` também caem numa linha real. ``NaN``,
infinito e ``None`` levantam no ``int()`` e o auditor não devolve resultado.

Hipóteses:
- H1: ``True`` já fica fora da faixa. Falsifica se o bloco nas linhas 1–2
  aceita o achado e ``ok`` fica verdadeiro.
- H2: float e str numérica já são recusados. Falsifica se ``1.0`` ou ``"2"``
  entram na faixa.
- H3: valor não conversível já falha fechado. Falsifica se ``NaN``, infinito
  ou ``None`` levantam em vez de marcar o achado como fora.
- H4: ``int`` exato dentro da faixa já é isento. Controle: a linha 2 continua
  aceita depois da correção. ``int`` fora da faixa continua fora.
"""
from __future__ import annotations

import math
import unittest

from hypothesis import example, given, settings, strategies as st

from tools.embedded_assert_audit import audit


SOURCE = "if __name__ == '__main__':\n    assert True\n"
FILENAME = "demo.py"


class CitedLine(int):
    """Subclasse de int. Não é linha exata."""


def _audit(line_number: object, *, include: bool = True) -> dict:
    item = {"test_id": "B101", "filename": FILENAME, "code": "assert True"}
    if include:
        item["line_number"] = line_number
    return audit({"results": [item]}, SOURCE, FILENAME)


def _same_line(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is float and math.isnan(actual) and math.isnan(expected):
        return True
    return actual == expected


class EmbeddedAssertLineContract(unittest.TestCase):
    def test_exact_int_inside_the_main_block_stays_exempt(self):
        result = _audit(2)
        self.assertTrue(result["ok"])
        self.assertEqual(result["main_block_ranges"], [(1, 2)])
        self.assertEqual(result["outside_main_block"], [])
        self.assertEqual(result["assert_lines_outside_main"], [])
        self.assertEqual(result["b101_count"], 1)

    def test_exact_int_outside_the_range_stays_outside(self):
        for line in (0, -1, 3, 10**6):
            with self.subTest(line=line):
                result = _audit(line)
                self.assertFalse(result["ok"])
                self.assertEqual(result["outside_main_block"], [{"line_number": line, "code": "assert True"}])
                self.assertEqual(result["assert_lines_outside_main"], [])

    def test_missing_line_number_is_not_invented_as_an_exempt_line(self):
        result = _audit(None, include=False)
        self.assertFalse(result["ok"])
        self.assertEqual(result["outside_main_block"], [{"line_number": None, "code": "assert True"}])

    def test_other_file_and_non_b101_do_not_become_findings(self):
        payload = {"results": [
            {"test_id": "B101", "filename": "other.py", "line_number": True},
            {"test_id": "B311", "filename": FILENAME, "line_number": True},
            {"test_id": "B101", "filename": FILENAME, "line_number": 2},
        ]}
        result = audit(payload, SOURCE, FILENAME)
        self.assertTrue(result["ok"])
        self.assertEqual(result["b101_count"], 2)
        self.assertEqual(result["outside_main_block"], [])

    def test_non_exact_int_is_not_coerced_into_the_main_block(self):
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
            " 2",
            "1_0",
            "",
            "linha",
            None,
            [],
            {},
            (2,),
            CitedLine(2),
        )
        for line in samples:
            with self.subTest(line=line):
                result = _audit(line)
                self.assertFalse(result["ok"])
                self.assertEqual(len(result["outside_main_block"]), 1)
                found = result["outside_main_block"][0]
                self.assertTrue(_same_line(found["line_number"], line))
                self.assertEqual(found["code"], "assert True")
                self.assertEqual(result["assert_lines_outside_main"], [])
                self.assertEqual(result["b101_count"], 1)

    def test_invalid_line_does_not_hide_a_valid_one_or_the_reverse(self):
        payload = {"results": [
            {"test_id": "B101", "filename": FILENAME, "line_number": 2, "code": "dentro"},
            {"test_id": "B101", "filename": FILENAME, "line_number": True, "code": "bool"},
        ]}
        result = audit(payload, SOURCE, FILENAME)
        self.assertFalse(result["ok"])
        self.assertEqual(result["outside_main_block"], [{"line_number": True, "code": "bool"}])
        self.assertEqual(result["b101_count"], 2)

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
    @example(1.0)
    @example(float("nan"))
    @example(float("inf"))
    @example("2")
    @example(None)
    def test_property_non_int_never_enters_the_exempt_range(self, line):
        result = _audit(line)
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["outside_main_block"]), 1)
        self.assertTrue(_same_line(result["outside_main_block"][0]["line_number"], line))


if __name__ == "__main__":
    unittest.main()
