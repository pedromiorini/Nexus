"""Contrato MW1–MW9 de docs/cognitivo/AUDITORIA_2026-10-22.md: o MCTS usa os efeitos reais do modelo de mundo."""
import copy
import itertools
import random
import time
import unittest

from core.cognitivo.action_effects import ACTION_EFFECTS
from core.cognitivo.causal_world_model import RealWorldModelSimulator
from core.cognitivo.mcts_planner import RealMCTSPlanner

SEEDS = [20261101 + k for k in range(20)]
GOAL = "secure the system"
# Referência independente (não chama o módulo para calcular o esperado).
UTILITY = {"secure": 3, "stable": 2, "isolated": 1, "offline": 1, "vulnerable": 1, "compromised": 0}
INITIAL = {"security_patch_applied": False, "alert_level": "high", "system_status": "vulnerable"}
BASE_WM = round(UTILITY["vulnerable"] / 3, 4)


def ref_reward(sequence):
    state = dict(INITIAL)
    for action in sequence:
        state.update(ACTION_EFFECTS.get(action, {}).get("world", {}))
    return UTILITY[state["system_status"]] / 3


def _plan(seed, goal=GOAL, iterations=300, depth=3, actions=None, model=True):
    random.seed(seed)
    wm = RealWorldModelSimulator() if model else None
    planner = RealMCTSPlanner(world_model=wm, max_depth=depth)
    return planner.plan(goal, available_actions=actions, max_iterations=iterations)


def _strip(result):
    return {k: v for k, v in result.__dict__.items() if k != "planning_time"}


class MCTSWorldEffectsContractTests(unittest.TestCase):
    def test_mw1_reward_is_not_stuck(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                r = _plan(seed)
                self.assertGreater(r.expected_reward, 0.3334)
                self.assertGreaterEqual(len({a["expected_reward"] for a in r.alternative_plans}), 2)

    def test_mw2_plan_follows_real_effects(self):
        actions = list(ACTION_EFFECTS)
        best = max(ref_reward(seq) for n in (1, 2, 3) for seq in itertools.product(actions, repeat=n))
        self.assertEqual(best, 1.0)
        for seed in SEEDS:
            with self.subTest(seed=seed):
                self.assertEqual(ref_reward(_plan(seed).best_action_sequence), best)

    def test_mw3_shortest_plan_wins(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                self.assertIn(_plan(seed).best_action_sequence, (["APPLY_PATCH"], ["DEPLOY_UPDATE"]))

    def test_mw4_harmful_action_and_subsets(self):
        for seed in SEEDS:
            with self.subTest(seed=seed, case="patch"):
                r = _plan(seed, actions=["IGNORE_ALERT", "MONITOR", "APPLY_PATCH"])
                self.assertEqual(r.best_action_sequence, ["APPLY_PATCH"])
            with self.subTest(seed=seed, case="monitor"):
                r = _plan(seed, actions=["IGNORE_ALERT", "MONITOR"])
                self.assertEqual(r.best_action_sequence, ["MONITOR"])

    def test_mw5_unmodeled_action_does_not_inflate_success(self):
        for seed in SEEDS:
            action = "ZZ_%d" % random.Random(seed).randrange(10 ** 6)
            with self.subTest(seed=seed, model=True):
                r = _plan(seed, actions=[action])
                self.assertEqual(r.best_action_sequence, [action])
                self.assertEqual(r.expected_reward, BASE_WM)
                self.assertEqual(r.success_probability, 0.0)
            with self.subTest(seed=seed, model=False):
                random.seed(seed)
                planner = RealMCTSPlanner(max_depth=3)
                goal = "improve stability"
                base = round(planner._evaluate_state(planner._get_initial_state(goal), goal), 4)
                r = planner.plan(goal, available_actions=[action], max_iterations=300)
                self.assertEqual(r.best_action_sequence, [action])
                self.assertEqual(r.expected_reward, base)
                self.assertEqual(r.success_probability, 0.0)
            with self.subTest(seed=seed, case="mixed"):
                r = _plan(seed, actions=["APPLY_PATCH", action])
                self.assertEqual(r.best_action_sequence, ["APPLY_PATCH"])
                self.assertAlmostEqual(r.success_probability, min(1.0, 1.1 * r.expected_reward), delta=0.0002)
            with self.subTest(seed=seed, case="known_without_model"):
                r = _plan(seed, model=False)
                self.assertAlmostEqual(r.success_probability, min(1.0, 1.1 * r.expected_reward), delta=0.0002)

    def test_mw6_integrated_brain_planner(self):
        from core.constitutional_brain import CompleteNexusBrain
        brain = CompleteNexusBrain(enable_router=True)
        for seed in SEEDS[:5]:
            with self.subTest(seed=seed):
                random.seed(seed)
                r = brain.mcts_planner.plan(GOAL, max_iterations=200)
                self.assertNotEqual(r.expected_reward, 0.5)
                sim = brain.world_model.simulate(list(r.best_action_sequence))
                self.assertIs(sim.success, True)

    def test_mw7_no_side_effects_and_no_cache_leak(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                wm = RealWorldModelSimulator()
                before = copy.deepcopy(wm.current_state.state)
                planner = RealMCTSPlanner(world_model=wm, max_depth=3)
                effects = copy.deepcopy(planner._action_effects)
                random.seed(seed)
                first = _strip(planner.plan(GOAL, max_iterations=200))
                self.assertEqual(wm.current_state.state, before)
                self.assertEqual(planner._action_effects, effects)
                planner2 = RealMCTSPlanner(world_model=RealWorldModelSimulator(), max_depth=3)
                random.seed(seed)
                self.assertEqual(_strip(planner2.plan(GOAL, max_iterations=200)), first)
                wm.current_state.state["system_status"] = "compromised"
                r = planner.plan(GOAL, available_actions=["MONITOR"], max_iterations=50)
                self.assertEqual(r.expected_reward, 0.0)

    def test_mw9_model_without_simulate_keeps_the_numeric_path(self):
        # Acrescentado após o RED (guiado pelo mutante mw_world_forced_without_simulate): dublê só com current_state.
        class Stub:
            current_state = type("S", (), {"state": {"system_status": "vulnerable"}})()

        for seed in SEEDS[:5]:
            with self.subTest(seed=seed):
                random.seed(seed)
                got = RealMCTSPlanner(world_model=Stub(), max_depth=3).plan(GOAL, max_iterations=100)
                self.assertEqual(got.expected_reward, 0.5)
                self.assertEqual(got.success_probability, 0.55)

    def test_mw8_planning_time_p95(self):
        times = []
        for seed in SEEDS:
            random.seed(seed)
            planner = RealMCTSPlanner(world_model=RealWorldModelSimulator(), max_depth=5)
            t0 = time.perf_counter()
            planner.plan(GOAL, max_iterations=200)
            times.append(time.perf_counter() - t0)
        times.sort()
        self.assertLessEqual(times[int(0.95 * len(times)) - 1], 0.10, times)


if __name__ == "__main__":
    unittest.main()
