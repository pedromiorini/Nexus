"""Contrato do avaliador funcional C (causal + world model) pré-registrado em 285851b.

Fixa o conjunto L (30 frases: 18 positivas em 6 formas, 12 negativas), os 6 conjuntos do C3 com
ótimo único, e que o avaliador mede o código extraído sem alterá-lo.
"""

import unittest

from tools import cognitivo_eval as ev


class CausalEvalContractTests(unittest.TestCase):
    def test_frozen_sets_match_preregistration(self):
        self.assertEqual(len(ev.CAUSAL_L), 30)
        self.assertEqual(sum(lab is not None for _, lab in ev.CAUSAL_L), 18)
        self.assertEqual(len(ev.CAUSAL_C3_SETS), 6)
        self.assertEqual(len(ev.CAUSAL_ACTION_VOCAB), 10)
        self.assertEqual(len(ev.CAUSAL_H), 24)
        self.assertEqual(sum(lab is not None for _, lab in ev.CAUSAL_H), 14)
        self.assertFalse({t for t, _ in ev.CAUSAL_H} & {t for t, _ in ev.CAUSAL_L})

    def test_c3_sets_have_unique_valid_optimum(self):
        for seqs, best in ev.CAUSAL_C3_SETS:
            self.assertTrue(0 <= best < len(seqs))

    def test_evaluator_reports_all_criteria(self):
        ids = [c["id"] for c in ev.eval_causal()]
        self.assertEqual(ids, ["C1a", "C1b", "C2a", "C2b", "C3", "C4", "H-F1", "H-FP"])

    def test_cli_accepts_causal_capability(self):
        self.assertIn("causal", ev.CAPABILITIES)
        self.assertIn(ev.main(["--capability", "causal", "--criteria", "v1"]), (ev.EXIT_OK, ev.EXIT_FAIL))


if __name__ == "__main__":
    unittest.main()
