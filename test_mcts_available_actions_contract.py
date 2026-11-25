"""Contrato de fronteira de ``available_actions`` em ``RealMCTSPlanner.plan``.

Critérios registrados antes da correção:
- Recipientes textuais (str/bytes/bytearray) são incompatíveis com a lista de
  ações e devem levantar TypeError antes de incrementar contadores do plano.
- Itens que não sejam str não são ações válidas e são ignorados; entradas
  inválidas não podem virar ações no resultado nem fazer o planejador falhar.
- Uma ação textual válida continua disponível quando misturada a itens inválidos.
"""

import random
import unittest

from core.cognitivo.mcts_planner import RealMCTSPlanner


class MCTSAvailableActionsContractTests(unittest.TestCase):
    def test_non_string_items_are_ignored_and_valid_string_is_preserved(self):
        planner = RealMCTSPlanner(max_depth=1)
        random.seed(1)

        result = planner.plan(
            "secure the system",
            available_actions=["ROLLBACK", {"invalid": "unhashable"}],
            max_iterations=3,
        )

        self.assertEqual(result.best_action_sequence, ["ROLLBACK"])

    def test_non_string_only_items_do_not_become_actions(self):
        planner = RealMCTSPlanner(max_depth=1)
        random.seed(1)

        result = planner.plan(
            "secure the system",
            available_actions=[None],
            max_iterations=3,
        )

        self.assertEqual(result.best_action_sequence, [])

    def test_invalid_containers_are_rejected_before_plan_counters_change(self):
        for actions in ("MONITOR", b"MONITOR", bytearray(b"MONITOR"), 42):
            with self.subTest(actions=type(actions).__name__):
                planner = RealMCTSPlanner(max_depth=1)
                with self.assertRaisesRegex(TypeError, "available_actions"):
                    planner.plan("secure the system", available_actions=actions, max_iterations=3)
                self.assertEqual(planner.total_plans, 0)
                self.assertEqual(planner.total_iterations, 0)


if __name__ == "__main__":
    unittest.main()
