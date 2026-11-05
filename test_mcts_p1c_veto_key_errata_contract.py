"""Errata E-P1c-VK (#147): compare_allowed_futures usa o contrato veto_key/veto_lookup da main (V3, V4)."""
from __future__ import annotations

import unittest

from core.cognitivo.mcts_planner import DEFAULT_VETOES, compare_allowed_futures


class _WorldModel:
    def compare_futures(self, sequences):
        return {"results": [list(s) for s in sequences],
                "best_future_index": 0 if sequences else None,
                "best_future": list(sequences[0]) if sequences else None}


class V3CanonicalVariantsTests(unittest.TestCase):
    def test_main_canonical_variants_are_vetoed(self):
        for name in ("shutdown.service", "Shutdown-Service()", "shutdown\u200b_service"):  # E-P1c-VK3
            with self.subTest(name=repr(name)):
                out = compare_allowed_futures(_WorldModel(), [[name], ["MONITOR"]])
                self.assertEqual(out["allowed_indices"], [1])
                self.assertEqual(out["vetoed_futures"], [0])
                self.assertEqual(out["best_future_index"], 1)
                self.assertEqual(out["vetoed_actions"],
                                 [{"action": name, "reason": DEFAULT_VETOES["SHUTDOWN_SERVICE"]}])


class V4ExactNamePrecedenceTests(unittest.TestCase):
    def test_exact_name_wins_then_first_canonical(self):
        vetoes = {"my action": "x", "MY_ACTION": "y"}
        out = compare_allowed_futures(_WorldModel(), [["MY_ACTION"], ["my-action"], ["MONITOR"]], vetoes=vetoes)
        self.assertEqual(out["allowed_indices"], [2])
        self.assertEqual(out["vetoed_actions"], [{"action": "MY_ACTION", "reason": "y"},
                                                 {"action": "my-action", "reason": "x"}])


if __name__ == "__main__":
    unittest.main()
