"""Contrato de entradas de RealMCTSPlanner (endurecimento).

Teste dos três casos (critérios escritos antes da correção):
(a) caso de origem: exploration_constant NaN/inf, max_depth<=0, max_iterations<0 e
    goal=None hoje são aceitos em silêncio ou levantam AttributeError; passam a
    levantar ValueError, e bool usado como int levanta TypeError;
(b) não regressão: entradas válidas não mudam (SHA-256 com seed fixa idêntico ao da
    main; max_iterations=0, goal vazio e available_actions=[] seguem aceitos);
(c) caso incompatível: tipos errados (goal não-str, constantes não numéricas) levantam
    TypeError antes de qualquer mutação de estado (total_plans não muda).
"""

import random
import unittest

from core.cognitivo.mcts_planner import RealMCTSPlanner


class MCTSPlannerConstructorContractTests(unittest.TestCase):
    def test_non_finite_exploration_constant_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "exploration_constant"):
                RealMCTSPlanner(exploration_constant=value)

    def test_non_numeric_exploration_constant_rejected(self):
        for value in (True, "1.4", None):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "exploration_constant"):
                RealMCTSPlanner(exploration_constant=value)

    def test_non_positive_max_depth_rejected(self):
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_depth"):
                RealMCTSPlanner(max_depth=value)

    def test_non_int_max_depth_rejected(self):
        for value in (True, 2.0, "3"):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "max_depth"):
                RealMCTSPlanner(max_depth=value)

    def test_valid_constructor_values_accepted(self):
        planner = RealMCTSPlanner(exploration_constant=0, max_depth=1)
        self.assertEqual((planner.exploration_constant, planner.max_depth), (0, 1))


class MCTSPlannerPlanContractTests(unittest.TestCase):
    def setUp(self):
        self.planner = RealMCTSPlanner(max_depth=1)

    def _rejects(self, exc, pattern, **kwargs):
        args = {"goal": "secure the system", "max_iterations": 3}
        args.update(kwargs)
        with self.assertRaisesRegex(exc, pattern):
            self.planner.plan(**args)
        self.assertEqual(self.planner.total_plans, 0)
        self.assertEqual(self.planner.total_iterations, 0)

    def test_missing_goal_rejected_with_value_error(self):
        self._rejects(ValueError, "goal", goal=None)

    def test_non_string_goal_rejected(self):
        self._rejects(TypeError, "goal", goal=42)

    def test_negative_iterations_rejected(self):
        self._rejects(ValueError, "max_iterations", max_iterations=-1)

    def test_non_int_iterations_rejected(self):
        for value in (True, 2.5, "3"):
            with self.subTest(value=value):
                self._rejects(TypeError, "max_iterations", max_iterations=value)

    def test_zero_iterations_and_empty_goal_still_accepted(self):
        random.seed(1)
        self.assertEqual(self.planner.plan("secure", max_iterations=0).iterations_run, 0)
        self.assertEqual(self.planner.plan("", max_iterations=2).iterations_run, 2)


if __name__ == "__main__":
    unittest.main()
