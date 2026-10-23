"""Contrato CF8: diferença Jaccard de tokens — escrito antes do fix."""
from __future__ import annotations

import contextlib
import io
import re
import unittest

from core.cognitivo.counterfactual import RealCounterfactualReasoningEngine

J_PAIRS = [
    ("the cat sat on the mat", "the cat sat on a mat", 1 - 5 / 6),
    ("outcome without intervention", "outcome with apply patch", 1 - 1 / 6),
    ("alpha beta", "gamma delta", 1.0),
]


def _eng():
    with contextlib.redirect_stdout(io.StringIO()):
        return RealCounterfactualReasoningEngine()


def _jaccard_diff(a: str, b: str) -> float:
    if a == b:
        return 0.0
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta and not tb:
        return 0.0
    if not ta or not tb:
        return 1.0
    return 1.0 - (len(ta & tb) / len(ta | tb))


class CounterfactualGap4ContractTests(unittest.TestCase):
    def test_empty_and_identical(self):
        eng = _eng()
        self.assertEqual(eng._calculate_outcome_difference("", ""), 0.0)
        self.assertEqual(eng._calculate_outcome_difference("x", "x"), 0.0)

    def test_jaccard_pairs(self):
        eng = _eng()
        for a, b, expected in J_PAIRS:
            with self.subTest(a=a, b=b):
                got = eng._calculate_outcome_difference(a, b)
                self.assertAlmostEqual(got, expected, places=6)
                self.assertAlmostEqual(got, _jaccard_diff(a, b), places=6)

    def test_default_outcomes_share_outcome_token(self):
        eng = _eng()
        d = eng._calculate_outcome_difference(
            "outcome_without_intervention", "outcome_with_apply_patch"
        )
        self.assertLess(d, 1.0)
        self.assertAlmostEqual(d, 1 - 1 / 6, places=6)


if __name__ == "__main__":
    unittest.main()
