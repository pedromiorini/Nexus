"""Contrato R2: compare_futures ranqueia por (sucesso, utilidade ordinal do estado final).

(a) origem: quando todos os futuros falham, o menos ruim vence (vulnerable > compromised);
(b) não regressão: futuro com sucesso continua vencendo; chaves de retorno mantidas;
(c) inválido: lista vazia continua levantando ValueError; empate → menor índice.
"""

import unittest

from core.cognitivo.causal_world_model import STATUS_UTILITY, RealWorldModelSimulator


class FuturesRankingContractTests(unittest.TestCase):
    def setUp(self):
        self.wm = RealWorldModelSimulator()

    def best(self, seqs):
        return self.wm.compare_futures(seqs)["best_future_index"]

    def test_utility_order(self):
        u = STATUS_UTILITY
        self.assertGreater(u["secure"], u["stable"])
        self.assertGreater(u["stable"], u["vulnerable"])
        self.assertGreater(u["vulnerable"], u["compromised"])

    def test_least_bad_future_wins_when_all_fail(self):
        self.assertEqual(self.best([["IGNORE_ALERT"], ["MONITOR"]]), 1)
        self.assertEqual(self.best([["IGNORE_ALERT", "MONITOR"], ["MONITOR", "MONITOR"], ["IGNORE_ALERT"]]), 1)

    def test_success_still_dominates_and_secure_beats_stable(self):
        self.assertEqual(self.best([["MONITOR"], ["APPLY_PATCH"], ["IGNORE_ALERT"]]), 1)
        self.assertEqual(self.best([["ROLLBACK"], ["APPLY_PATCH"]]), 1)

    def test_ties_keep_lowest_index_and_keys(self):
        out = self.wm.compare_futures([["MONITOR"], ["ESCALATE"]])
        self.assertEqual(out["best_future_index"], 0)
        self.assertEqual({"results", "best_future_index", "best_future"} <= set(out), True)
        self.assertIs(out["best_future"], out["results"][0])

    def test_unknown_status_ranks_lowest_and_empty_raises(self):
        self.wm.current_state.state["system_status"] = "weird"
        self.assertEqual(self.best([["MONITOR"], ["IGNORE_ALERT"]]), 1)
        with self.assertRaises(ValueError):
            self.wm.compare_futures([])


if __name__ == "__main__":
    unittest.main()
