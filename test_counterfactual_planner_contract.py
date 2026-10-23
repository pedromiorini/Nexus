"""Contrato CF3b: planner injetado é chamado no what-if.

Critérios pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit a4f591e) antes deste teste e antes da correção.
No base fe57c82, self.planner só é guardado em __init__: as asserções
de chamada falham.
"""
from __future__ import annotations

import contextlib
import io
import math
import unittest
from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st

from core.cognitivo.causal_world_model import RealCausalReasoning
from core.cognitivo import counterfactual as cf_mod
from core.cognitivo.counterfactual import (
    CounterfactualScenario,
    RealCounterfactualReasoningEngine,
)
from core.cognitivo.mcts_planner import RealMCTSPlanner

PROBE = 8


def _engine(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealCounterfactualReasoningEngine(**kwargs)


class _Stub:
    def __init__(self, sequence=None, as_dict=False):
        self.calls = []
        self.sequence = ["APPLY_PATCH"] if sequence is None else sequence
        self.as_dict = as_dict

    def plan(self, goal, available_actions=None, max_iterations=200):
        self.calls.append((goal, available_actions, max_iterations))
        if self.as_dict:
            return {"best_action_sequence": self.sequence}
        return SimpleNamespace(best_action_sequence=self.sequence)


class _RecordingMCTS(RealMCTSPlanner):
    def __init__(self):
        super().__init__(max_depth=1)
        self.calls = []
        self.last = None

    def plan(self, goal, available_actions=None, max_iterations=200):
        copied = list(available_actions) if available_actions is not None else None
        self.calls.append((goal, copied, max_iterations))
        self.last = super().plan(
            goal, available_actions=available_actions, max_iterations=max_iterations)
        return self.last


class CounterfactualPlannerContractTests(unittest.TestCase):
    def test_probe_budget_is_the_preregistered_constant(self):
        self.assertTrue(hasattr(cf_mod, "PLANNER_PROBE_ITERATIONS"))
        self.assertEqual(cf_mod.PLANNER_PROBE_ITERATIONS, PROBE)

    def test_generate_calls_plan_and_appends_string_actions(self):
        stub = _Stub(sequence=["APPLY_PATCH", 1, True, None])
        eng = _engine(planner=stub)
        sc = eng.generate_counterfactual_scenario("system vulnerable", "isolate")
        self.assertEqual(stub.calls, [("system vulnerable", ["isolate"], PROBE)])
        self.assertEqual(sc.original_situation, "system vulnerable")
        self.assertEqual(sc.intervention, "isolate")
        self.assertEqual(sc.causal_chain[-1], "planner: APPLY_PATCH")
        self.assertNotIn("planner: 1", sc.causal_chain)
        self.assertNotIn("planner: True", sc.causal_chain)
        self.assertEqual(sc.actual_outcome, "outcome_without_intervention")
        self.assertEqual(sc.counterfactual_outcome, "outcome_with_isolate")

    def test_dict_sequence_and_explicit_what_if(self):
        stub = _Stub(sequence=["ROLLBACK"], as_dict=True)
        eng = _engine(planner=stub)
        out = eng.explore_what_if(situation="during the outage", intervention="we had patched")
        self.assertEqual(
            stub.calls, [("during the outage", ["we had patched"], PROBE)])
        sc = out["scenario"]
        self.assertEqual(sc.original_situation, "during the outage")
        self.assertEqual(sc.intervention, "we had patched")
        self.assertIn("planner: ROLLBACK", sc.causal_chain)

    def test_string_what_if_uses_parsed_pair(self):
        stub = _Stub(sequence=[])
        eng = _engine(planner=stub)
        out = eng.explore_what_if("What if we isolate the host?")
        self.assertEqual(
            stub.calls, [("current_situation", ["we isolate the host"], PROBE)])
        self.assertEqual(out["scenario"].intervention, "we isolate the host")
        self.assertEqual(len(out["scenario"].causal_chain), 5)
        self.assertFalse(any(step.startswith("planner:") for step in out["scenario"].causal_chain))

    def test_string_sequence_is_not_iterated_as_characters(self):
        stub = _Stub(sequence="APPLY")
        eng = _engine(planner=stub)
        sc = eng.generate_counterfactual_scenario("host", "isolate")
        self.assertEqual(len(stub.calls), 1)
        self.assertFalse(any(str(step).startswith("planner:") for step in sc.causal_chain))

    def test_missing_sequence_does_not_extend_chain(self):
        class _NoSequence:
            def __init__(self):
                self.calls = 0

            def plan(self, goal, available_actions=None, max_iterations=200):
                self.calls += 1
                return None

        planner = _NoSequence()
        eng = _engine(planner=planner)
        sc = eng.generate_counterfactual_scenario("host", "isolate")
        self.assertEqual(planner.calls, 1)
        self.assertEqual(len(sc.causal_chain), 5)
        self.assertFalse(any(str(step).startswith("planner:") for step in sc.causal_chain))

    def test_absent_or_noncallable_plan_keeps_heuristic(self):
        eng = _engine()
        sc = eng.generate_counterfactual_scenario("system vulnerable", "apply patch")
        self.assertEqual(sc.actual_outcome, "outcome_without_intervention")
        self.assertEqual(sc.counterfactual_outcome, "outcome_with_apply_patch")
        self.assertEqual(len(sc.causal_chain), 5)
        self.assertEqual(sc.original_situation, "system vulnerable")
        self.assertEqual(sc.intervention, "apply patch")

        class _Bad:
            def __init__(self, plan):
                self.plan = plan

        for plan in (True, -1, math.nan, math.inf, -math.inf):
            with self.subTest(plan=plan):
                bad = _engine(planner=_Bad(plan))
                got = bad.generate_counterfactual_scenario("system vulnerable", "isolate")
                self.assertEqual(len(got.causal_chain), 5)
                self.assertTrue(all(not str(step).startswith("planner:") for step in got.causal_chain))

        missing = _engine(planner=object())
        got = missing.generate_counterfactual_scenario("alarm", "patch")
        self.assertEqual(len(got.causal_chain), 5)

    def test_empty_strings_still_call_plan(self):
        stub = _Stub(sequence=[])
        called = _engine(planner=stub)
        called.generate_counterfactual_scenario("", "")
        self.assertEqual(stub.calls, [("", [""], PROBE)])
        silent = _engine()
        sc = silent.generate_counterfactual_scenario("", "")
        self.assertEqual(len(sc.causal_chain), 5)
        self.assertFalse(any(str(step).startswith("planner:") for step in sc.causal_chain))

    def test_other_methods_do_not_call_plan(self):
        stub = _Stub()
        eng = _engine(planner=stub)
        eng._predict_outcome("sit", "act")
        eng.model_causal_intervention("alert_level", "high", "low")
        ready = CounterfactualScenario(
            scenario_id="s",
            original_situation="sit",
            intervention="act",
            counterfactual_outcome="a",
            actual_outcome="b",
            causal_chain=["1"],
            plausibility_score=0.5,
        )
        eng.compare_outcomes(ready)
        self.assertEqual(stub.calls, [])

    def test_world_model_and_causal_protocols_remain(self):
        class _WM:
            def predict_outcome(self, situation, intervention):
                return f"wm:{situation}:{intervention}"

        stub = _Stub(sequence=["PATCH"])
        eng = _engine(world_model=_WM(), causal_reasoning=RealCausalReasoning(), planner=stub)
        sc = eng.generate_counterfactual_scenario("alarm", "patch")
        self.assertEqual(sc.actual_outcome, "wm:alarm:None")
        self.assertEqual(sc.counterfactual_outcome, "wm:alarm:patch")
        self.assertEqual(sc.causal_chain[0], "alarm causes patch")
        self.assertEqual(sc.causal_chain[-1], "planner: PATCH")
        self.assertEqual(stub.calls, [("alarm", ["patch"], PROBE)])

    def test_real_mcts_planner_is_called_on_the_documented_path(self):
        planner = _RecordingMCTS()
        eng = _engine(planner=planner)
        sc = eng.explore_what_if(situation="system vulnerable", intervention="APPLY_PATCH")
        self.assertEqual(planner.calls, [("system vulnerable", ["APPLY_PATCH"], PROBE)])
        self.assertEqual(planner.total_plans, 1)
        self.assertEqual(sc["scenario"].original_situation, "system vulnerable")
        self.assertEqual(sc["scenario"].intervention, "APPLY_PATCH")
        for action in planner.last.best_action_sequence:
            if isinstance(action, str):
                self.assertIn(f"planner: {action}", sc["scenario"].causal_chain)

    def test_invalid_goal_propagates_from_real_planner(self):
        planner = RealMCTSPlanner(max_depth=1)
        eng = _engine(planner=planner)
        with self.assertRaises(ValueError):
            eng.generate_counterfactual_scenario(None, "isolate")
        with self.assertRaises(TypeError):
            eng.generate_counterfactual_scenario(True, "isolate")
        self.assertEqual(planner.total_plans, 0)

    @given(st.text(max_size=24), st.text(max_size=24))
    @settings(max_examples=20, deadline=None, database=None, derandomize=True)
    def test_plan_receives_caller_strings(self, situation, intervention):
        stub = _Stub(sequence=["X"])
        eng = _engine(planner=stub)
        sc = eng.generate_counterfactual_scenario(situation, intervention)
        self.assertEqual(stub.calls, [(situation, [intervention], PROBE)])
        self.assertEqual(sc.original_situation, situation)
        self.assertEqual(sc.intervention, intervention)
        self.assertIn("planner: X", sc.causal_chain)


if __name__ == "__main__":
    unittest.main()
