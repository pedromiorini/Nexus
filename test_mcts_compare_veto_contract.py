"""Contrato da fronteira de vetoes em compare_allowed_futures.

A API auxiliar deve aplicar a mesma validação de Mapping[str, str] já esperada
pelo construtor de RealMCTSPlanner: entradas inválidas são rejeitadas, nomes
vazios após normalização não são ignorados, e mapas válidos preservam o motivo.
"""
from __future__ import annotations

import unittest

from core.cognitivo.mcts_planner import compare_allowed_futures


class _WorldModel:
    def compare_futures(self, sequences):
        return {
            "results": [{"sequence": list(sequence)} for sequence in sequences],
            "best_future_index": 0 if sequences else None,
            "best_future": list(sequences[0]) if sequences else None,
        }


class CompareAllowedFuturesVetoContractTests(unittest.TestCase):
    def test_non_string_action_name_is_rejected(self):
        with self.assertRaises(TypeError):
            compare_allowed_futures(_WorldModel(), [["KEEP"]], vetoes={1: "reason"})

    def test_non_string_reason_is_rejected(self):
        for reason in (None, 2):
            with self.subTest(reason=reason):
                with self.assertRaises(TypeError):
                    compare_allowed_futures(_WorldModel(), [["BLOCK"]], vetoes={"BLOCK": reason})

    def test_empty_or_whitespace_only_action_name_is_rejected(self):
        for name in ("", "   ", "\t\n"):
            with self.subTest(name=repr(name)):
                with self.assertRaises(ValueError):
                    compare_allowed_futures(_WorldModel(), [["KEEP"]], vetoes={name: "reason"})

    def test_valid_veto_preserves_its_reason(self):
        out = compare_allowed_futures(
            _WorldModel(), [["BLOCK"], ["KEEP"]], vetoes={"BLOCK": "review before use"}
        )
        self.assertEqual(out["allowed_indices"], [1])
        self.assertEqual(out["vetoed_futures"], [0])
        self.assertEqual(out["vetoed_actions"], [{"action": "BLOCK", "reason": "review before use"}])


if __name__ == "__main__":
    unittest.main()
