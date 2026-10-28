"""Contrato do mental simulation: entradas inválidas, harness MS e extração strangler.

Critérios MS1–MS6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit cf8f9f2) antes do move e antes do eval. O harness é
tools/cognitivo_eval.py --capability mental_simulation.
"""

import contextlib
import io
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from core.constitutional_brain import (
    MentalScenario,
    RealMentalSimulationSystem,
)

ROOT = Path(__file__).resolve().parent
_NAMES = ("MentalScenario", "MentalSimulationResult", "RealMentalSimulationSystem")
_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.mental_simulation as mod
system = mod.RealMentalSimulationSystem()
result = system.simulate_scenario(mod.MentalScenario("s", {"position": 9}, ["move"], [], "position"))
assert result.success and result.final_state == {"position": 10} and result.predicted_utility == 0.9
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    import core.cognitivo.mental_simulation as mental_simulation
    return mental_simulation


def _quiet(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealMentalSimulationSystem(*args, **kwargs)


def _scenario(state, actions, goal="survive", sid="s"):
    return MentalScenario(sid, state, actions, [], goal)


class _Boom:
    def __getattr__(self, name):
        raise AssertionError("called " + name)


class MentalSimulationInvalidInputContractTests(unittest.TestCase):
    def test_non_string_action_and_wrong_state_types_are_not_coerced(self):
        for action in (1, None, ["move"]):
            with self.subTest(action=action), self.assertRaises(AttributeError):
                _quiet().simulate_scenario(_scenario({}, [action]))
        with self.assertRaises(AttributeError):
            _quiet().simulate_scenario(_scenario(None, []))
        with self.assertRaises(AttributeError):
            _quiet().simulate_scenario(None)
        with self.assertRaises(TypeError):
            _quiet().simulate_scenario(_scenario({}, None))
        for position in (None, "a"):
            with self.subTest(position=position), self.assertRaises(TypeError):
                _quiet().simulate_scenario(_scenario({"position": position}, ["move"]))

    def test_goal_is_read_only_after_an_action(self):
        self.assertFalse(_quiet().simulate_scenario(_scenario({"position": 3}, [], None)).success)
        for goal in (None, 3):
            with self.subTest(goal=goal), self.assertRaises(AttributeError):
                _quiet().simulate_scenario(_scenario({}, ["wait"], goal))

    def test_rollout_rejects_empty_actions_and_zero_rollouts(self):
        with self.assertRaises(ValueError):
            _quiet().rollout_planning({}, [], "goal", 3)
        with self.assertRaises(ZeroDivisionError):
            _quiet().rollout_planning({}, ["move"], "goal", 0)
        with self.assertRaises(TypeError):
            _quiet().rollout_planning({}, None, "goal", 1)
        with self.assertRaises(AttributeError):
            _quiet().rollout_planning(None, ["move"], "goal", 1)
        self.assertEqual(_quiet().rollout_planning({}, ["move"], "goal", -1), "move")

    def test_rollout_cache_and_counter_grow_by_actions_times_rollouts(self):
        system = _quiet()
        random.seed(3)
        system.rollout_planning({"position": 0}, ["move", "wait"], "survive", 5)
        stats = system.get_statistics()
        self.assertEqual(stats["total_simulations"], 10)
        self.assertEqual(stats["cached_simulations"], 10)
        self.assertEqual(stats["successful_predictions"], 0)
        self.assertEqual(stats["simulation_accuracy"], 0.0)

    def test_counterfactual_replaces_only_the_first_action_and_keeps_the_cache(self):
        system = _quiet()
        actual = _scenario({"position": 8}, ["wait", "move", "move"], "position", "real")
        original = system.simulate_scenario(actual)
        result = system.counterfactual_simulation(actual, "move")
        self.assertEqual(result.scenario_id, "counterfactual_real")
        self.assertEqual((result.success, result.steps_taken, result.final_state), (True, 2, {"position": 10}))
        self.assertIs(system.simulation_cache["real"], original)
        self.assertEqual(actual.actions, ["wait", "move", "move"])
        empty = system.counterfactual_simulation(_scenario({"position": 0}, [], "survive", "e"), "move")
        self.assertEqual((empty.steps_taken, empty.final_state), (1, {"position": 1}))

    def test_compare_prefers_the_second_scenario_on_a_tie(self):
        system = _quiet()
        tie = system.compare_scenarios(_scenario({}, ["wait"]), _scenario({}, ["wait"], sid="t"))
        self.assertEqual((tie["preferred_scenario"], tie["utility_difference"]), (2, 0.0))
        self.assertEqual(system.get_statistics()["total_simulations"], 2)

    def test_collaborators_are_stored_and_never_read(self):
        system = _quiet(planner=_Boom(), reasoning=_Boom(), causal=_Boom())
        system.simulate_scenario(_scenario({}, ["move"]))
        system.rollout_planning({}, ["move"], "survive", 2)
        hooked = type("W", (), {"predict_next_state": lambda self, state, action: {"position": 99}})()
        self.assertEqual(
            _quiet(world_model=hooked).simulate_scenario(_scenario({}, ["wait"], "position")).final_state,
            {"position": 99},
        )
        falsy = type("F", (), {"__len__": lambda self: 0,
                               "predict_next_state": lambda self, state, action: {"position": 99}})()
        self.assertEqual(
            _quiet(world_model=falsy).simulate_scenario(_scenario({"position": 1}, ["move"])).final_state,
            {"position": 2},
        )
        with self.assertRaises(TypeError):
            RealMentalSimulationSystem(1, 2, 3, 4, 5)

    def test_substring_match_and_bool_position_are_current_behavior(self):
        self.assertEqual(_quiet().simulate_scenario(_scenario({}, ["remove"])).final_state, {"position": 1})
        self.assertEqual(_quiet().simulate_scenario(_scenario({}, ["rebuild"])).final_state, {"structures": 1})
        self.assertEqual(_quiet().simulate_scenario(_scenario({"position": True}, ["move"])).final_state["position"], 2)


class MentalSimulationExtractionContractTests(unittest.TestCase):
    def test_monolith_and_package_reexport_the_same_objects(self):
        mental_simulation = _module()
        import core.cognitivo as package
        import core.constitutional_brain as brain
        for name in _NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(mental_simulation, name))
                self.assertIs(getattr(package, name), getattr(mental_simulation, name))
                self.assertEqual(getattr(mental_simulation, name).__module__, "core.cognitivo.mental_simulation")

    def test_module_imports_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", _BLOCKER], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_pre_registered_checks_match_the_frozen_fingerprint(self):
        _module()
        from tools.cognitivo_eval import MS_BEHAVIOR_SHA, MS_SOURCE_SHA, eval_mental_simulation
        checks = {item["id"]: item for item in eval_mental_simulation()}
        self.assertEqual(set(checks), {"MS1", "MS2", "MS3", "MS4", "MS5", "MS6"})
        for cid in ("MS1", "MS2", "MS4", "MS5", "MS6"):
            with self.subTest(cid=cid):
                self.assertTrue(checks[cid]["passed"], checks[cid])
        # MS3 mede o propósito do rollout; o resultado é registrado, não forçado a passar aqui.
        self.assertEqual(set(checks["MS3"]["value"]), {"reach", "increase", "gather"})
        for name, item in checks["MS3"]["value"].items():
            with self.subTest(case=name):
                self.assertTrue(0.0 <= item["rate"] <= 1.0)
        self.assertEqual(checks["MS6"]["value"]["sha_module"], MS_BEHAVIOR_SHA)
        self.assertEqual(checks["MS6"]["value"]["sha_brain"], MS_BEHAVIOR_SHA)
        self.assertEqual(checks["MS6"]["value"]["source_sha"], MS_SOURCE_SHA)
        self.assertEqual(checks["MS6"]["value"]["clock"], "none")
        self.assertTrue(checks["MS6"]["value"]["identity"])


class MentalSimulationHarnessNegativeContractTests(unittest.TestCase):
    """O gate falha fechado: comportamento ou fonte adulterados reprovam o harness."""

    def _run(self, argv):
        from tools import cognitivo_eval
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            code = cognitivo_eval.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_tampered_behavior_changes_the_fingerprint(self):
        _module()
        import core.constitutional_brain as brain
        from tools import cognitivo_eval

        class Tampered(brain.RealMentalSimulationSystem):
            def _check_goal_achieved(self, state, goal):
                return False

        ns = {"RealMentalSimulationSystem": Tampered, "MentalScenario": brain.MentalScenario,
              "MentalSimulationResult": brain.MentalSimulationResult}
        tampered = cognitivo_eval._ms_sha(cognitivo_eval.mental_simulation_fingerprint(ns))
        self.assertNotEqual(tampered, cognitivo_eval.MS_BEHAVIOR_SHA)

    def test_wrong_golden_fails_ms6_and_exits_1(self):
        _module()
        from tools import cognitivo_eval
        with mock.patch.object(cognitivo_eval, "MS_BEHAVIOR_SHA", "0" * 64):
            checks = {item["id"]: item for item in cognitivo_eval.eval_mental_simulation()}
            code, out, _ = self._run(["--capability", "mental_simulation"])
        self.assertFalse(checks["MS6"]["passed"])
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL)
        self.assertIn('"MS6"', out.split('"failed"')[1])
        with mock.patch.object(cognitivo_eval, "MS_SOURCE_SHA", "0" * 64):
            self.assertFalse({i["id"]: i for i in cognitivo_eval.eval_mental_simulation()}["MS6"]["passed"])

    def test_cli_rejects_unknown_spellings_with_exit_2(self):
        for value in ("mental-simulation", "Mental_Simulation", "mentalsimulation", "mental_simulation2"):
            with self.subTest(value=value):
                code, _, err = self._run(["--capability", value])
                self.assertEqual(code, 2)
                self.assertIn("desconhecida", err)

    def test_cli_only_ms3_may_fail_and_exit_code_matches(self):
        _module()
        from tools import cognitivo_eval
        code, out, _ = self._run(["--capability", "mental_simulation"])
        import json
        failed = json.loads(out)["failed"]
        self.assertLessEqual(set(failed), {"MS3"})
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL if failed else cognitivo_eval.EXIT_OK)

    def test_fingerprint_restores_the_global_random_state(self):
        _module()
        import core.constitutional_brain as brain
        from tools import cognitivo_eval
        random.seed(11)
        before = random.getstate()
        cognitivo_eval.mental_simulation_fingerprint(brain)
        self.assertEqual(random.getstate(), before)

    def test_source_text_requires_the_marker(self):
        from tools import cognitivo_eval
        with self.assertRaises(ValueError):
            cognitivo_eval.mental_simulation_source_text("nothing here")
        with self.assertRaises(ValueError):
            cognitivo_eval.mental_simulation_source_text("x\n" + cognitivo_eval._MS_MARKER + "\nno end")


if __name__ == "__main__":
    unittest.main()
