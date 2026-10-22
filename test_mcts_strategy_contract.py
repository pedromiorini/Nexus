"""Contrato da estratégia opt-in do RealMCTSPlanner (M6: NMCS nível 2, Cazenave 2009)."""
import inspect
import random
import unittest

from core.cognitivo.mcts_planner import RealMCTSPlanner


class _Tree(RealMCTSPlanner):
    """Árvore 3x3 determinística; folha 'C,C,C' é o único ótimo (1.0)."""

    def __init__(self, **kw):
        super().__init__(max_depth=3, **kw)
        self.leaves = 0

    def _get_initial_state(self, goal):
        return {"steps_taken": []}

    def _get_available_actions(self, state, depth=0):
        return [] if len(state["steps_taken"]) >= 3 else ["A", "B", "C"]

    def _apply_action(self, state, action):
        return {"steps_taken": state["steps_taken"] + [action]}

    def _evaluate_state(self, state, goal):
        steps = state["steps_taken"]
        if len(steps) < 3:
            return 0.0
        self.leaves += 1
        return 1.0 if steps == ["C", "C", "C"] else 0.1 * steps.count("C")


class _Additive(RealMCTSPlanner):
    """Árvore 3^8 aditiva (valor = fração de 'C'); seguir a melhor sequência leva ao ótimo."""

    def __init__(self, **kw):
        super().__init__(max_depth=8, **kw)

    def _get_initial_state(self, goal):
        return {"steps_taken": []}

    def _get_available_actions(self, state, depth=0):
        return [] if len(state["steps_taken"]) >= 8 else ["A", "B", "C"]

    def _apply_action(self, state, action):
        return {"steps_taken": state["steps_taken"] + [action]}

    def _evaluate_state(self, state, goal):
        steps = state["steps_taken"]
        return steps.count("C") / 8 if len(steps) == 8 else 0.0


class StrategyContract(unittest.TestCase):
    def test_nmcs_follows_best_child(self):
        hits = 0
        for seed in range(20):
            random.seed(seed)
            hits += _Additive(strategy="nmcs").plan("g", max_iterations=300).expected_reward == 1.0
        self.assertGreaterEqual(hits, 2)  # busca uniforme: 300/6561 por execução


    def test_default_is_uct_and_keyword_only(self):
        self.assertEqual(RealMCTSPlanner().strategy, "uct")
        param = inspect.signature(RealMCTSPlanner).parameters["strategy"]
        self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(param.default, "uct")

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            RealMCTSPlanner(strategy="nrpa")
        with self.assertRaises(TypeError):
            RealMCTSPlanner(strategy=1)

    def test_nmcs_respects_rollout_budget(self):
        for budget in (1, 7, 50):
            random.seed(0)
            p = _Tree(strategy="nmcs")
            p.plan("g", max_iterations=budget)
            self.assertLessEqual(p.leaves, budget)
            self.assertGreaterEqual(p.leaves, 1)

    def test_nmcs_finds_optimum_with_enough_budget(self):
        random.seed(0)
        r = _Tree(strategy="nmcs").plan("g", max_iterations=60)
        self.assertEqual(r.best_action_sequence, ["C", "C", "C"])
        self.assertEqual(r.expected_reward, 1.0)

    def test_nmcs_default_domain_plan_is_valid(self):
        random.seed(0)
        r = RealMCTSPlanner(max_depth=3, strategy="nmcs").plan(
            "secure the system", available_actions=["APPLY_PATCH", "MONITOR"], max_iterations=100)
        self.assertTrue(r.best_action_sequence)
        self.assertTrue(set(r.best_action_sequence) <= {"APPLY_PATCH", "MONITOR"})

    def test_uct_default_unchanged_by_explicit_strategy(self):
        random.seed(3)
        a = RealMCTSPlanner(max_depth=3).plan("secure the system", max_iterations=200).best_action_sequence
        random.seed(3)
        b = RealMCTSPlanner(max_depth=3, strategy="uct").plan("secure the system",
                                                              max_iterations=200).best_action_sequence
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
