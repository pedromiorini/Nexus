"""Contrato ER1–ER9 de docs/cognitivo/AUDITORIA_2026-10-22.md: expected_reward e success_probability
refletem a sequência devolvida pelo MCTS no modo mundo."""
import dataclasses
import random
import time
import unittest

from core.cognitivo.action_effects import ACTION_EFFECTS
from core.cognitivo.causal_world_model import RealWorldModelSimulator
from core.cognitivo.mcts_planner import PlanResult, RealMCTSPlanner

SEEDS = [20261102 + k for k in range(20)]
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


def iterations_for(seed):
    return (30, 300)[seed % 2]


def subset_for(seed):
    rng = random.Random(seed)
    pool = list(ACTION_EFFECTS) + ["ZZ_%d" % rng.randrange(10 ** 6)]
    return rng.sample(pool, rng.randint(1, 4))


def _plan(seed, actions=None, model=True, depth=3, iterations=None, planner=None):
    random.seed(seed)
    if planner is None:
        planner = RealMCTSPlanner(world_model=RealWorldModelSimulator() if model else None, max_depth=depth)
    return planner.plan(GOAL, available_actions=actions, max_iterations=iterations or iterations_for(seed))


def expected_success(result):
    """Errata E1: derivado do valor NÃO arredondado (como o código e o MW5)."""
    seq = result.best_action_sequence
    if any(a.startswith("ZZ_") for a in seq):
        return 0.0
    return min(1.0, 1.1 * (ref_reward(seq) if seq else 0.0))


class MCTSExpectedRewardContractTests(unittest.TestCase):
    def _check_coherent(self, result):
        seq = result.best_action_sequence
        want = round(ref_reward(seq), 4) if seq else 0.0
        self.assertEqual(result.expected_reward, want, seq)
        self.assertAlmostEqual(result.success_probability, expected_success(result), delta=0.0002, msg=seq)

    def test_er1_reward_and_success_reflect_the_returned_sequence(self):
        for seed in SEEDS:
            with self.subTest(seed=seed, case="default"):
                self._check_coherent(_plan(seed))
            with self.subTest(seed=seed, case="subset", actions=subset_for(seed)):
                self._check_coherent(_plan(seed, actions=subset_for(seed)))

    def test_er2_top_value_does_not_come_from_the_best_child(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                r = _plan(seed)
                self.assertIn(r.best_action_sequence, (["APPLY_PATCH"], ["DEPLOY_UPDATE"]))
                self.assertEqual(r.expected_reward, 1.0)
                self.assertEqual(r.success_probability, 1.0)

    def test_er3_empty_missing_and_unknown_inputs(self):
        for seed in SEEDS:
            with self.subTest(seed=seed, case="empty"):
                r = _plan(seed, actions=[])
                self.assertEqual((r.best_action_sequence, r.expected_reward, r.success_probability), ([], 0.0, 0.0))
            action = "ZZ_%d" % random.Random(seed).randrange(10 ** 6)
            with self.subTest(seed=seed, case="unknown"):
                r = _plan(seed, actions=[action])
                self.assertEqual(r.best_action_sequence, [action])
                self.assertEqual((r.expected_reward, r.success_probability), (BASE_WM, 0.0))
            with self.subTest(seed=seed, case="compromised"):
                wm = RealWorldModelSimulator()
                wm.current_state.state["system_status"] = "compromised"
                r = _plan(seed, actions=["MONITOR"], planner=RealMCTSPlanner(world_model=wm, max_depth=3))
                self.assertEqual((r.best_action_sequence, r.expected_reward, r.success_probability), ([], 0.0, 0.0))
        planner = RealMCTSPlanner(world_model=RealWorldModelSimulator(), max_depth=3)
        with self.assertRaisesRegex(ValueError, "goal"):
            planner.plan(None)
        random.seed(SEEDS[0])
        self._check_coherent(planner.plan("", max_iterations=30))

    def test_er4_no_leak_between_plans(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                wm = RealWorldModelSimulator()
                before = dict(wm.current_state.state)
                planner = RealMCTSPlanner(world_model=wm, max_depth=3)
                _plan(seed, planner=planner)
                self.assertEqual(planner._world_rewards, {})
                self.assertEqual(wm.current_state.state, before)
                wm.current_state.state["system_status"] = "compromised"
                r = _plan(seed, actions=["MONITOR"], planner=planner)
                self.assertEqual((r.best_action_sequence, r.expected_reward), ([], 0.0))

    def test_er5_structure_is_intact(self):
        fields = {"goal", "best_action_sequence", "expected_reward", "iterations_run", "tree_nodes_created",
                  "alternative_plans", "planning_time", "success_probability"}
        self.assertEqual({f.name for f in dataclasses.fields(PlanResult)}, fields)
        for seed in SEEDS:
            actions = subset_for(seed)
            with self.subTest(seed=seed):
                r = _plan(seed, actions=actions)
                self.assertLessEqual(len(r.alternative_plans), 3)
                for alt in r.alternative_plans:
                    self.assertEqual(set(alt), {"first_action", "expected_reward", "visits"})
                    self.assertIn(alt["first_action"], actions)

    def test_er6_without_model_keeps_the_best_child_semantics(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                r = _plan(seed, model=False)
                self.assertTrue(r.alternative_plans)
                self.assertEqual(r.expected_reward, r.alternative_plans[0]["expected_reward"])

    def test_er7_integrated_brain_planner(self):
        from core.constitutional_brain import CompleteNexusBrain
        brain = CompleteNexusBrain(enable_router=True)
        for seed in SEEDS[:5]:
            with self.subTest(seed=seed):
                random.seed(seed)
                r = brain.mcts_planner.plan(GOAL, max_iterations=200)
                self.assertTrue(r.best_action_sequence)
                self.assertEqual(r.expected_reward, round(ref_reward(r.best_action_sequence), 4))

    def test_er8_planning_time_p95(self):
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
