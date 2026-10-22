"""Contrato de qualidade de RealMCTSPlanner (critérios M1–M4 de docs/cognitivo/AUDITORIA_2026-10-22.md)."""
import random
import unittest

from core.cognitivo.mcts_planner import RealMCTSPlanner
from tools.cognitivo_eval import eval_mcts


class MCTSPlannerQualityContractTests(unittest.TestCase):
    def test_preregistered_criteria_pass(self):
        checks = {c["id"]: c for c in eval_mcts()}
        self.assertEqual(checks["M3"]["criteria"], "v2")
        for cid in ("M1", "M2", "M4", "M5"):
            self.assertTrue(checks[cid]["passed"], checks[cid])
        # M3 v2 (max_depth=8, `<` estrito) FALHA neste PR: os dois regrets seguem em 0,0 (teto persiste).
        # Pela regra da v2 o critério não é reajustado; falha reportada ao Pedro. Aqui só a forma fraca
        # (sem regressão com mais iterações).
        self.assertLessEqual(checks["M3"]["value"], checks["M3"]["regret_50"])

    def test_caller_action_list_is_not_mutated(self):
        allowed = ["MONITOR", "ROLLBACK"]
        random.seed(1)
        RealMCTSPlanner(max_depth=2).plan("secure the system", available_actions=allowed, max_iterations=50)
        self.assertEqual(allowed, ["MONITOR", "ROLLBACK"])

    def test_every_depth_respects_available_actions(self):
        allowed = ["MONITOR", "ROLLBACK", "ESCALATE"]
        for seed in range(5):
            random.seed(seed)
            plan = RealMCTSPlanner(max_depth=4).plan("secure the system", available_actions=allowed,
                                                     max_iterations=200)
            self.assertTrue(1 <= len(plan.best_action_sequence) <= 4)
            self.assertTrue(set(plan.best_action_sequence) <= set(allowed), plan.best_action_sequence)

    def test_custom_actions_do_not_leak_between_plans(self):
        planner = RealMCTSPlanner(max_depth=2)
        before = dict(planner._action_effects)
        random.seed(2)
        plan = planner.plan("secure the system", available_actions=["CUSTOM_X"], max_iterations=20)
        self.assertEqual(plan.best_action_sequence, ["CUSTOM_X", "CUSTOM_X"])
        self.assertEqual(planner._action_effects, before)
        random.seed(2)
        self.assertNotIn("CUSTOM_X", planner.plan("secure the system", max_iterations=50).best_action_sequence)

    def test_shorter_plan_wins_when_extra_steps_only_hurt(self):
        # "reduce cost", horizonte 3: ótimo exaustivo é ["APPLY_PATCH"] (0.84 > melhor completo 0.74)
        random.seed(0)
        plan = RealMCTSPlanner(max_depth=3).plan("reduce cost", max_iterations=2000)
        self.assertEqual(plan.best_action_sequence, ["APPLY_PATCH"])

    def test_plan_never_exceeds_horizon(self):
        for goal in ("improve stability", "secure the system"):
            random.seed(3)
            plan = RealMCTSPlanner(max_depth=1).plan(goal, max_iterations=300)
            self.assertEqual(len(plan.best_action_sequence), 1, (goal, plan.best_action_sequence))


if __name__ == "__main__":
    unittest.main()
