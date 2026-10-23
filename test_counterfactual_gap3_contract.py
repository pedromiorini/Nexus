"""Contrato CF7: API explícita (situation, intervention) — escrito antes do fix."""
from __future__ import annotations

import contextlib
import inspect
import io
import unittest

from core.cognitivo.counterfactual import RealCounterfactualReasoningEngine

E_PAIRS = [
    ("during the outage", "we had patched"),
    ("on host db-1", "isolate the host"),
    ("system vulnerable", "apply patch"),
    ("alarm firing", "rollback"),
]


def _eng():
    with contextlib.redirect_stdout(io.StringIO()):
        return RealCounterfactualReasoningEngine()


class CounterfactualGap3ContractTests(unittest.TestCase):
    def test_explore_what_if_accepts_explicit_kwargs(self):
        sig = inspect.signature(RealCounterfactualReasoningEngine.explore_what_if)
        params = sig.parameters
        self.assertIn("situation", params)
        self.assertIn("intervention", params)

    def test_explicit_api_preserves_situation(self):
        eng = _eng()
        for sit, inter in E_PAIRS:
            with self.subTest(sit=sit, inter=inter):
                out = eng.explore_what_if(situation=sit, intervention=inter)
                sc = out["scenario"]
                self.assertEqual(sc.original_situation, sit)
                self.assertEqual(sc.intervention, inter)

    def test_string_api_still_parses_what_if(self):
        eng = _eng()
        sit, inter = eng._parse_what_if_question("What if we isolate the host?")
        self.assertEqual(sit, "current_situation")
        self.assertEqual(inter, "we isolate the host")
        out = eng.explore_what_if("What if we isolate the host?")
        self.assertEqual(out["scenario"].intervention, "we isolate the host")


if __name__ == "__main__":
    unittest.main()
