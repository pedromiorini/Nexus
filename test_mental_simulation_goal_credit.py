"""Contrato do crédito parcial ciente do objetivo (RealMentalSimulationSystem).

Critérios MS3-v2 e MS7 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit 82308d0) antes do teste RED e antes da mudança de comportamento.
"""

import contextlib
import io
import random
import unittest

from core.constitutional_brain import MentalScenario, RealMentalSimulationSystem

SEEDS = tuple(range(20))
ROLLOUTS = 100


def _system():
    with contextlib.redirect_stdout(io.StringIO()):
        return RealMentalSimulationSystem()


def _simulate(state, actions, goal):
    return _system().simulate_scenario(MentalScenario("c", state, actions, [], goal))


def _picks(state, actions, goal):
    out = []
    for seed in SEEDS:
        system = _system()
        random.seed(seed)  # nosec B311 - seed fixa pré-registrada
        out.append(system.rollout_planning(dict(state), list(actions), goal, ROLLOUTS))
    return out


class GoalAwarePartialCreditTableTests(unittest.TestCase):
    """MS7: valores exatos pré-registrados para objetivos só de posição, sem sucesso."""

    def test_position_only_goal_credits_progress_toward_ten(self):
        for position, quality in ((0, 0.30), (5, 0.50), (9, 0.66), (-3, 0.30)):
            with self.subTest(position=position):
                result = _simulate({"position": position}, [], "increase position")
                self.assertFalse(result.success)
                self.assertAlmostEqual(result.outcome_quality, quality, places=9)
                self.assertAlmostEqual(result.predicted_utility, quality * 0.5, places=9)

    def test_resources_and_structures_are_not_progress_toward_a_position_goal(self):
        result = _simulate({"position": 0, "resources": 3, "structures": 2}, [], "increase position")
        self.assertAlmostEqual(result.outcome_quality, 0.30, places=9)

    def test_invalid_positions_without_actions_give_no_credit_and_no_error(self):
        for position in (None, "a", float("nan")):
            with self.subTest(position=position):
                result = _simulate({"position": position}, [], "increase position")
                self.assertAlmostEqual(result.outcome_quality, 0.30, places=9)

    def test_a_wait_step_keeps_the_progress_of_the_position(self):
        self.assertAlmostEqual(_simulate({"position": 5}, ["wait"], "increase position").outcome_quality, 0.50, places=9)

    def test_other_goals_keep_the_legacy_credit_exactly(self):
        mixed = _simulate({"resources": 1, "position": 0}, [], "increase position and gather resources")
        self.assertAlmostEqual(mixed.outcome_quality, 0.50, places=9)
        survive = _simulate({"structures": 1}, [], "survive")
        self.assertAlmostEqual(survive.outcome_quality, 0.50, places=9)
        for goal in (None, 3):
            with self.subTest(goal=goal):
                self.assertAlmostEqual(_simulate({"resources": 1}, [], goal).outcome_quality, 0.50, places=9)

    def test_success_is_unchanged(self):
        result = _simulate({"position": 9}, ["move"], "increase position")
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.outcome_quality, 0.90, places=9)


class GoalAwareRolloutContractTests(unittest.TestCase):
    """MS3-v2: `increase position` passa a escolher `move`; controles não mudam."""

    def test_increase_position_chooses_move(self):
        picks = _picks({"position": 0}, ["move", "wait", "acquire"], "increase position")
        self.assertGreaterEqual(picks.count("move") / len(picks), 0.80, picks)

    def test_increase_position_chooses_move_when_it_is_not_the_first_action(self):
        picks = _picks({"position": 3}, ["wait", "acquire", "move"], "increase position")
        self.assertGreaterEqual(picks.count("move") / len(picks), 0.80, picks)

    def test_controls_keep_the_base_choice(self):
        for goal in ("survive", "increase position and gather resources"):
            with self.subTest(goal=goal):
                picks = _picks({"position": 0}, ["move", "wait", "acquire"], goal)
                self.assertEqual(picks, ["acquire"] * len(SEEDS))

    def test_gather_resources_still_chooses_acquire(self):
        picks = _picks({"resources": 3}, ["acquire resource", "wait", "build structure"], "gather resources")
        self.assertGreaterEqual(picks.count("acquire resource") / len(picks), 0.90, picks)


class GoalAwareHarnessTests(unittest.TestCase):
    def test_every_pre_registered_criterion_passes(self):
        from tools.cognitivo_eval import eval_mental_simulation
        checks = {item["id"]: item for item in eval_mental_simulation()}
        self.assertEqual(set(checks), {"MS1", "MS2", "MS3", "MS4", "MS5", "MS6", "MS7"})
        for cid, item in checks.items():
            with self.subTest(cid=cid):
                self.assertTrue(item["passed"], item)


if __name__ == "__main__":
    unittest.main()
