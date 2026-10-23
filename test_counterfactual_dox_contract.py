"""Contrato CF9: intervenção do(X) no contrafactual.

Critérios pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit c187c12) antes deste teste e antes da correção.
Na base 629541f, simulate/do/_do não são consultados: o outcome cai em
outcome_with_* e a cadeia com aresta SCM continua só o template lexical.
O oráculo de mapeamento abaixo copia o protocolo do pré-registro; não chama
o helper do módulo.
"""
from __future__ import annotations

import contextlib
import io
import math
import re
import unittest
from types import SimpleNamespace

from hypothesis import example, given, settings
from hypothesis import strategies as st

from core.cognitivo.action_effects import ACTION_EFFECTS
from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator
from core.cognitivo.counterfactual import RealCounterfactualReasoningEngine

PROBE = 8
APPLY_STATE = (
    "state:alert_level=low,security_patch_applied=True,system_status=secure"
)
DEPLOY_STATE = (
    "state:alert_level=high,security_patch_applied=False,system_status=secure"
)
SPY_STATE = "state:marker=from-model,system_status=secure"
DEPLOY_CHAIN = [
    "do(DEPLOY_FIX): system_status=secure",
    "host exposed causes DEPLOY_FIX",
]


def _engine(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealCounterfactualReasoningEngine(**kwargs)


def _normalize(text: str) -> str:
    collapsed = re.sub(r"[\s\-]+", "_", text.strip())
    return re.sub(r"_+", "_", collapsed).strip("_").upper()


def spec_action(intervention, adjacency_keys):
    """Oráculo do pré-registro CF9. None = não mapeia."""
    if not isinstance(intervention, str):
        return None
    raw = intervention.strip()
    if not raw:
        return None
    normalized = _normalize(intervention)
    keys = [str(key) for key in adjacency_keys]
    if normalized in ACTION_EFFECTS:
        return normalized
    if raw in keys:
        return raw
    if normalized in keys:
        return normalized
    folded = [key for key in keys if _normalize(key) == normalized]
    if len(folded) == 1:
        return folded[0]
    if len(folded) > 1:
        return None
    tokens = set(re.findall(r"[a-z0-9]+", intervention.lower()))
    if not tokens:
        return None
    matches = [
        name
        for name in ACTION_EFFECTS
        if tokens == set(re.findall(r"[a-z0-9]+", name.lower()))
    ]
    if len(matches) == 1:
        return matches[0]
    return None


class SpySim:
    def __init__(self, state=None, exc=None):
        self.calls = []
        self.exc = exc
        self.current_state = SimpleNamespace(state={"system_status": "vulnerable"})
        self._state = {"marker": "from-model", "system_status": "secure"} if state is None else state

    def simulate(self, actions):
        self.calls.append(list(actions))
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(
            final_state=SimpleNamespace(state=dict(self._state)),
            warnings=[],
        )


class RecordingWorld(RealWorldModelSimulator):
    def __init__(self, causal_reasoning=None):
        super().__init__(causal_reasoning=causal_reasoning)
        self.calls = []

    def simulate(self, actions):
        self.calls.append(list(actions))
        return super().simulate(actions)


class _Stub:
    def __init__(self, sequence):
        self.calls = []
        self.sequence = sequence

    def plan(self, goal, available_actions=None, max_iterations=200):
        self.calls.append((goal, available_actions, max_iterations))
        return SimpleNamespace(best_action_sequence=self.sequence)


def _with_edge(sentence="DEPLOY_FIX causes system_status secure"):
    with contextlib.redirect_stdout(io.StringIO()):
        causal = RealCausalReasoning()
    causal.extract_causal_relations(sentence)
    return causal


class CounterfactualDoContractTests(unittest.TestCase):
    def test_spy_simulate_outcome_is_model_state_not_prefix(self):
        spy = SpySim()
        eng = _engine(world_model=spy)
        self.assertEqual(eng._predict_outcome("sit", None), "outcome_without_intervention")
        self.assertEqual(spy.calls, [])
        direct = eng._predict_outcome("system vulnerable", "APPLY_PATCH")
        scenario = eng.generate_counterfactual_scenario("system vulnerable", "APPLY_PATCH")
        self.assertEqual(direct, SPY_STATE)
        self.assertEqual(scenario.counterfactual_outcome, SPY_STATE)
        self.assertFalse(scenario.counterfactual_outcome.startswith("outcome_with_"))
        self.assertEqual(scenario.actual_outcome, "outcome_without_intervention")
        self.assertEqual(scenario.intervention, "APPLY_PATCH")
        self.assertEqual(spy.calls, [["APPLY_PATCH"], ["APPLY_PATCH"]])

    def test_real_simulator_maps_apply_patch_token_orders(self):
        for intervention in ("apply patch", "patch apply", "apply_patch", "APPLY_PATCH"):
            with self.subTest(intervention=intervention):
                causal = _with_edge("noop causes nothing relevant")
                world = RecordingWorld(causal_reasoning=causal)
                eng = _engine(world_model=world, causal_reasoning=causal)
                scenario = eng.generate_counterfactual_scenario("system vulnerable", intervention)
                self.assertEqual(world.calls, [["APPLY_PATCH"]])
                self.assertEqual(scenario.counterfactual_outcome, APPLY_STATE)
                self.assertEqual(scenario.actual_outcome, "outcome_without_intervention")
                self.assertEqual(scenario.intervention, intervention)
                self.assertTrue(all(not step.startswith("do(") for step in scenario.causal_chain))

    def test_deploy_fix_outcome_comes_from_c4_do(self):
        causal = _with_edge()
        world = RecordingWorld(causal_reasoning=causal)
        eng = _engine(world_model=world, causal_reasoning=causal)
        scenario = eng.generate_counterfactual_scenario("host exposed", "DEPLOY_FIX")
        self.assertEqual(world.calls, [["DEPLOY_FIX"]])
        self.assertEqual(scenario.counterfactual_outcome, DEPLOY_STATE)
        self.assertEqual(scenario.causal_chain, DEPLOY_CHAIN)
        self.assertNotEqual(scenario.causal_chain, ["host exposed causes DEPLOY_FIX"])

    def test_chain_uses_prior_edge_without_world_model(self):
        causal = _with_edge()
        eng = _engine(causal_reasoning=causal)
        self.assertEqual(eng._build_causal_chain("host exposed", "DEPLOY_FIX"), DEPLOY_CHAIN)
        scenario = eng.generate_counterfactual_scenario("host exposed", "DEPLOY_FIX")
        self.assertEqual(scenario.causal_chain, DEPLOY_CHAIN)
        self.assertEqual(scenario.counterfactual_outcome, "outcome_with_DEPLOY_FIX")

    def test_lowercase_graph_key_is_the_simulate_action(self):
        causal = _with_edge("deploy_fix causes system_status secure")
        world = RecordingWorld(causal_reasoning=causal)
        eng = _engine(world_model=world)
        scenario = eng.generate_counterfactual_scenario("host exposed", "DEPLOY_FIX")
        self.assertEqual(world.calls, [["deploy_fix"]])
        self.assertEqual(scenario.causal_chain, ["do(deploy_fix): system_status=secure"])
        self.assertIn("system_status=secure", scenario.counterfactual_outcome)
        self.assertFalse(scenario.counterfactual_outcome.startswith("outcome_with_"))

    def test_world_graph_and_separate_extractor_both_appear(self):
        edged = _with_edge()
        other = _with_edge("unrelated causes something else")
        world = RecordingWorld(causal_reasoning=edged)
        eng = _engine(world_model=world, causal_reasoning=other)
        scenario = eng.generate_counterfactual_scenario("host exposed", "DEPLOY_FIX")
        self.assertEqual(scenario.causal_chain, DEPLOY_CHAIN)
        self.assertEqual(world.calls, [["DEPLOY_FIX"]])

    def test_explore_what_if_preserves_intervention(self):
        spy = SpySim()
        eng = _engine(world_model=spy)
        out = eng.explore_what_if(situation="system vulnerable", intervention="APPLY_PATCH")
        self.assertEqual(spy.calls, [["APPLY_PATCH"]])
        self.assertEqual(out["scenario"].intervention, "APPLY_PATCH")
        self.assertEqual(out["scenario"].original_situation, "system vulnerable")
        self.assertEqual(out["scenario"].counterfactual_outcome, SPY_STATE)

    def test_public_do_and_private_do_hooks(self):
        class PublicDo:
            def __init__(self):
                self.calls = []
                self.private = []
                self.current_state = SimpleNamespace(
                    state={"alert_level": "high", "system_status": "vulnerable"}
                )

            def do(self, action):
                self.calls.append(action)
                return {"system_status": "secure"}

            def _do(self, action):
                self.private.append(action)
                return {"system_status": "compromised"}

        hook = PublicDo()
        eng = _engine(world_model=hook)
        scenario = eng.generate_counterfactual_scenario("sit", "APPLY_PATCH")
        self.assertEqual(hook.calls, ["APPLY_PATCH"])
        self.assertEqual(hook.private, [])
        self.assertEqual(scenario.counterfactual_outcome, "state:alert_level=high,system_status=secure")

        class PrivateDo:
            def __init__(self):
                self.calls = []
                self.current_state = SimpleNamespace(state={"system_status": "vulnerable"})

            def _do(self, action):
                self.calls.append(action)
                return {"system_status": "secure"}

        private = PrivateDo()
        eng = _engine(world_model=private)
        scenario = eng.generate_counterfactual_scenario("sit", "APPLY_PATCH")
        self.assertEqual(private.calls, ["APPLY_PATCH"])
        self.assertEqual(scenario.counterfactual_outcome, "state:system_status=secure")

    def test_empty_do_effects_fall_back_after_the_call(self):
        class EmptyDo:
            def __init__(self):
                self.calls = []

            def _do(self, action):
                self.calls.append(action)
                return {}

        hook = EmptyDo()
        eng = _engine(world_model=hook)
        scenario = eng.generate_counterfactual_scenario("sit", "APPLY_PATCH")
        self.assertEqual(hook.calls, ["APPLY_PATCH"])
        self.assertEqual(scenario.counterfactual_outcome, "outcome_with_APPLY_PATCH")

    def test_planner_stays_after_chain_and_formula_holds(self):
        stub = _Stub(["MONITOR"])
        spy = SpySim()
        eng = _engine(world_model=spy, planner=stub)
        scenario = eng.generate_counterfactual_scenario("system vulnerable", "apply patch")
        self.assertEqual(stub.calls, [("system vulnerable", ["apply patch"], PROBE)])
        self.assertEqual(spy.calls, [["APPLY_PATCH"]])
        self.assertEqual(scenario.intervention, "apply patch")
        self.assertEqual(scenario.causal_chain[-1], "planner: MONITOR")
        self.assertAlmostEqual(
            scenario.plausibility_score,
            max(0.1, 0.7 - 0.05 * len(scenario.causal_chain)),
        )

    def test_predict_outcome_preempts_simulate(self):
        class Both:
            def __init__(self):
                self.calls = []

            def predict_outcome(self, situation, intervention):
                return f"wm:{situation}:{intervention}"

            def simulate(self, actions):
                self.calls.append(list(actions))
                return SimpleNamespace(final_state=SimpleNamespace(state={"x": "y"}))

        both = Both()
        eng = _engine(world_model=both)
        self.assertEqual(eng._predict_outcome("sit", "act"), "wm:sit:act")
        scenario = eng.generate_counterfactual_scenario("sit", "APPLY_PATCH")
        self.assertEqual(both.calls, [])
        self.assertEqual(scenario.counterfactual_outcome, "wm:sit:APPLY_PATCH")

    def test_unmapped_empty_and_false_do_not_call_simulate(self):
        spy = SpySim()
        eng = _engine(world_model=spy)
        expected = {
            "patch": "outcome_with_patch",
            "hello": "outcome_with_hello",
            "isolate the system": "outcome_with_isolate_the_system",
            "   ": "outcome_with____",
        }
        for text, outcome in expected.items():
            with self.subTest(text=text):
                spy.calls.clear()
                scenario = eng.generate_counterfactual_scenario("sit", text)
                self.assertEqual(spy.calls, [])
                self.assertEqual(scenario.counterfactual_outcome, outcome)
        spy.calls.clear()
        blank = eng.generate_counterfactual_scenario("sit", "")
        self.assertEqual(spy.calls, [])
        self.assertEqual(blank.counterfactual_outcome, "outcome_without_intervention")
        absent = eng.generate_counterfactual_scenario("sit", False)
        self.assertEqual(spy.calls, [])
        self.assertEqual(absent.counterfactual_outcome, "outcome_without_intervention")

    def test_non_str_raises_and_bool_hook_is_not_called(self):
        spy = SpySim()
        eng = _engine(world_model=spy)
        for bad in (True, -1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                spy.calls.clear()
                before = eng.total_scenarios_generated
                with self.assertRaises(AttributeError):
                    eng.generate_counterfactual_scenario("sit", bad)
                self.assertEqual(spy.calls, [])
                self.assertEqual(eng.total_scenarios_generated, before)

        class BoolHook:
            simulate = True

        heuristic = _engine(world_model=BoolHook())
        scenario = heuristic.generate_counterfactual_scenario("sit", "APPLY_PATCH")
        self.assertEqual(scenario.counterfactual_outcome, "outcome_with_APPLY_PATCH")

    def test_simulate_error_propagates_and_skips_store(self):
        spy = SpySim(exc=ValueError("boom"))
        eng = _engine(world_model=spy)
        with self.assertRaises(ValueError):
            eng.generate_counterfactual_scenario("sit", "APPLY_PATCH")
        self.assertEqual(eng.total_scenarios_generated, 0)
        self.assertEqual(spy.calls, [["APPLY_PATCH"]])
        self.assertEqual(eng.scenarios, [])

    def test_other_methods_do_not_simulate(self):
        spy = SpySim()
        eng = _engine(world_model=spy)
        eng.model_causal_intervention("status", "vulnerable", "secure")
        scenario = eng.generate_counterfactual_scenario("sit", "hello")
        eng.compare_outcomes(scenario)
        self.assertEqual(spy.calls, [])

    def test_no_model_keeps_heuristic_and_cf2(self):
        eng = _engine()
        scenario = eng.generate_counterfactual_scenario("alarm", "patch")
        self.assertEqual(scenario.counterfactual_outcome, "outcome_with_patch")
        self.assertEqual(scenario.actual_outcome, "outcome_without_intervention")
        self.assertEqual(len(scenario.causal_chain), 5)
        self.assertTrue(all(not step.startswith("do(") for step in scenario.causal_chain))
        causal = _with_edge("unrelated causes something else")
        wired = _engine(causal_reasoning=causal)
        self.assertEqual(wired._build_causal_chain("alarm", "patch"), ["alarm causes patch"])

    def test_ambiguous_graph_keys_do_not_map(self):
        class Ambiguous:
            causal_graph = {
                "adjacency": {
                    "deploy_fix": {"system_status secure"},
                    "Deploy-Fix": {"system_status secure"},
                }
            }

        spy = SpySim()
        eng = _engine(world_model=spy, causal_reasoning=Ambiguous())
        scenario = eng.generate_counterfactual_scenario("sit", "DEPLOY_FIX")
        self.assertEqual(spy.calls, [])
        self.assertEqual(scenario.counterfactual_outcome, "outcome_with_DEPLOY_FIX")
        self.assertTrue(all(not step.startswith("do(") for step in scenario.causal_chain))

    def test_plausibility_formula_not_retuned(self):
        eng = _engine()
        score = eng._calculate_plausibility("x", ["a", "b"])
        self.assertTrue(math.isclose(score, 0.6))

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz _-", min_size=0, max_size=20))
    @example("APPLY_PATCH")
    @example("apply patch")
    @example("patch apply")
    @example("patch")
    @example("")
    @example("hello")
    @settings(max_examples=40, deadline=None)
    def test_mapping_matches_preregistered_oracle(self, text):
        spy = SpySim()
        eng = _engine(world_model=spy)
        action = spec_action(text, ())
        scenario = eng.generate_counterfactual_scenario("sit", text)
        if action is None:
            self.assertEqual(spy.calls, [])
        else:
            self.assertEqual(spy.calls, [[action]])
            self.assertEqual(scenario.counterfactual_outcome, SPY_STATE)
            self.assertEqual(scenario.intervention, text)


if __name__ == "__main__":
    unittest.main()
