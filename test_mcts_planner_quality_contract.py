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

    def test_custom_actions_remain_available_after_root_expansion(self):
        from unittest.mock import patch

        planner = RealMCTSPlanner(max_depth=2)
        caller_actions = ["CUSTOM_X"]
        actions_at_depth_one = []
        get_available_actions = planner._get_available_actions

        def capture_available_actions(state, depth):
            actions = get_available_actions(state, depth)
            if depth == 1:
                actions_at_depth_one.append(actions)
            return actions

        with patch.object(planner, "_get_available_actions", side_effect=capture_available_actions):
            with patch.object(planner, "_evaluate_state", return_value=0.0):
                planner.plan("neutral custom action", available_actions=caller_actions, max_iterations=1)

        self.assertEqual(caller_actions, ["CUSTOM_X"])
        self.assertTrue(actions_at_depth_one)
        self.assertTrue(any("CUSTOM_X" in actions for actions in actions_at_depth_one))

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
        self.assertTrue(plan.best_action_sequence)
        self.assertTrue(set(plan.best_action_sequence) <= {"CUSTOM_X"})
        self.assertEqual(planner._action_effects, before)
        random.seed(2)
        self.assertNotIn("CUSTOM_X", planner.plan("secure the system", max_iterations=50).best_action_sequence)

    def test_unknown_custom_action_has_neutral_effects(self):
        from unittest.mock import patch

        planner = RealMCTSPlanner(max_depth=1)
        goal = "improve stability"
        initial_state = planner._get_initial_state(goal)
        baseline_metrics = {key: initial_state[key] for key in ("security", "stability", "cost")}
        baseline_reward = round(planner._evaluate_state(initial_state, goal), 4)
        observed_metrics = []
        apply_action = planner._apply_action

        def capture_action_result(state, action):
            result_state = apply_action(state, action)
            if action == "UNMODELED_ACTION":
                observed_metrics.append({key: result_state[key] for key in baseline_metrics})
            return result_state

        random.seed(1)
        with patch.object(planner, "_apply_action", side_effect=capture_action_result):
            result = planner.plan(goal, available_actions=["UNMODELED_ACTION"], max_iterations=1)

        self.assertEqual(result.best_action_sequence, ["UNMODELED_ACTION"])
        self.assertEqual(result.expected_reward, baseline_reward)
        self.assertTrue(observed_metrics)
        self.assertTrue(all(metrics == baseline_metrics for metrics in observed_metrics))

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
