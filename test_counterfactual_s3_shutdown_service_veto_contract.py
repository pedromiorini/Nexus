"""Contrato S3: o fallback não devolve nome de DEFAULT_VETOES.

Critérios S3-C1..C3 e S3-N1..N7 do pré-registro
docs/cognitivo/S3_SHUTDOWN_SERVICE_MESMA_VETO_KEY_PRE_REGISTRO_2026-11-26.md
(SHA-256 17565093a0e960ba543cd68d8f0b54abae2278998dcdb2348061f656b5680741),
alternativa 1. Escrito antes da correção de _map_intervention.

S3-N8..N13 são o diff e os gates, não asserts deste arquivo: o pré-registro
os verifica contra a base, não com um teste novo.
"""
from __future__ import annotations

import contextlib
import io
import unittest

from hypothesis import example, given, settings
from hypothesis import strategies as st

from core.cognitivo.action_effects import ACTION_EFFECTS, veto_key
from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator
from core.cognitivo.counterfactual import RealCounterfactualReasoningEngine, _normalize_action_key
from core.cognitivo.mcts_planner import DEFAULT_VETOES, compare_allowed_futures

SITUATION = "system vulnerable"
OFFLINE = "state:alert_level=low,security_patch_applied=False,system_status=offline"
SECURE = "state:alert_level=low,security_patch_applied=True,system_status=secure"
VERBOSE = (
    "state:alert_level=high,logging_level=verbose,"
    "security_patch_applied=False,system_status=vulnerable"
)
OBS = {
    "system_status": "compromised",
    "alert_level": "critical",
    "security_patch_applied": False,
    "logging_level": "normal",
}
C1_CASES = (
    ("shutdown.service", "outcome_with_shutdown.service"),
    ("shutdown_service()", "outcome_with_shutdown_service()"),
    ("shutdown/service", "outcome_with_shutdown/service"),
    ("shutdown:service", "outcome_with_shutdown:service"),
    ("Shutdown, Service", "outcome_with_Shutdown,_Service"),
    ("ignore.alert", "outcome_with_ignore.alert"),
)
N1_CASES = (
    ("service shutdown", "outcome_with_service_shutdown"),
    ("Service, Shutdown!", "outcome_with_Service,_Shutdown!"),
    ("service_shutdown()", "outcome_with_service_shutdown()"),
    ("patch apply", "outcome_with_patch_apply"),
    ("alert ignore", "outcome_with_alert_ignore"),
    ("\ufb01INCREASE_LOGGING", "outcome_with_\ufb01INCREASE_LOGGING"),
    ("alert.ignore", "outcome_with_alert.ignore"),
)
S3_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-./:,!()\u200b"
SHUTDOWN_REASON = DEFAULT_VETOES["SHUTDOWN_SERVICE"]


class RecordingWorld(RealWorldModelSimulator):
    def __init__(self, causal_reasoning=None):
        super().__init__(causal_reasoning=causal_reasoning)
        self.calls = []

    def simulate(self, actions):
        self.calls.append(list(actions))
        return super().simulate(actions)


def _engine(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealCounterfactualReasoningEngine(**kwargs)


def _causal():
    with contextlib.redirect_stdout(io.StringIO()):
        causal = RealCausalReasoning()
    causal.extract_causal_relations("noop causes nothing relevant")
    return causal


def _wired(intervention):
    causal = _causal()
    world = RecordingWorld(causal_reasoning=causal)
    engine = _engine(world_model=world, causal_reasoning=causal)
    scenario = engine.generate_counterfactual_scenario(SITUATION, intervention)
    return world, scenario


def _is_step1(text):
    return isinstance(text, str) and _normalize_action_key(text) in ACTION_EFFECTS


def _assert_unmapped(test, text, outcome):
    world, scenario = _wired(text)
    test.assertEqual(world.calls, [])
    test.assertEqual(scenario.counterfactual_outcome, outcome)
    test.assertEqual(scenario.actual_outcome, "outcome_without_intervention")
    test.assertEqual(scenario.intervention, text)
    test.assertEqual(scenario.causal_chain, [f"{SITUATION} causes {text}"])


class _CompareWorld:
    def compare_futures(self, sequences):
        return {
            "results": [list(sequence) for sequence in sequences],
            "best_future_index": 0 if sequences else None,
            "best_future": list(sequences[0]) if sequences else None,
        }


class S3ShutdownServiceVetoContractTests(unittest.TestCase):
    def test_s3_c1_same_veto_key_default_veto_does_not_simulate(self):
        for text, outcome in C1_CASES:
            with self.subTest(intervention=text):
                _assert_unmapped(self, text, outcome)

    @given(st.text(alphabet=S3_ALPHABET, min_size=0, max_size=24))
    @example("shutdown.service")
    @example("shutdown_service()")
    @example("shutdown/service")
    @example("shutdown:service")
    @example("Shutdown, Service")
    @example("ignore.alert")
    @example("apply.patch")
    @example("increase.logging")
    @example("hello")
    @example("patch apply")
    @example("shutdown service")
    @example("SHUTDOWN_SERVICE")
    @example("A\u200bPPLY_PATCH")
    @settings(max_examples=40, deadline=None)
    def test_s3_c2_fallback_simulation_is_not_a_default_veto(self, text):
        if _is_step1(text):
            return
        world, _scenario = _wired(text)
        if not world.calls:
            return
        self.assertEqual(len(world.calls), 1)
        self.assertEqual(len(world.calls[0]), 1)
        self.assertNotIn(world.calls[0][0], DEFAULT_VETOES)

    def test_s3_c3_rank_of_the_gap_stays(self):
        engine = _engine(world_model=RecordingWorld())
        for spelling in (
            "shutdown.service",
            "shutdown_service()",
            "ignore.alert",
            "shutdown service",
            "apply.patch",
        ):
            with self.subTest(spelling=spelling):
                out = engine.rank_counterfactuals(OBS, ["IGNORE_ALERT"], [[spelling]])
                self.assertEqual(out["ranking"], [])
                self.assertEqual(out["vetoed"], [])
                self.assertEqual(out["unmodeled"], [[spelling]])
                self.assertIsNone(out["best"])
        out = engine.rank_counterfactuals(
            OBS, ["IGNORE_ALERT"], [["SHUTDOWN_SERVICE"], ["ISOLATE_SYSTEM"]]
        )
        self.assertEqual(out["best"]["actions"], ["ISOLATE_SYSTEM"])
        self.assertEqual(out["unmodeled"], [])
        self.assertIn(["SHUTDOWN_SERVICE"], [item["actions"] for item in out["ranking"]])
        self.assertEqual(
            out["vetoed"],
            [{
                "actions": ["SHUTDOWN_SERVICE"],
                "vetoes": [{
                    "action": "SHUTDOWN_SERVICE",
                    "reason": SHUTDOWN_REASON,
                }],
            }],
        )

    def test_s3_n1_s2_c1_spellings_stay_unmapped(self):
        for text, outcome in N1_CASES:
            with self.subTest(intervention=text):
                _assert_unmapped(self, text, outcome)

    def test_s3_n2_step1_names_still_map(self):
        expected = (
            ("apply patch", "APPLY_PATCH", SECURE),
            ("apply_patch", "APPLY_PATCH", SECURE),
            ("APPLY_PATCH", "APPLY_PATCH", SECURE),
            ("SHUTDOWN_SERVICE", "SHUTDOWN_SERVICE", OFFLINE),
            ("shutdown service", "SHUTDOWN_SERVICE", OFFLINE),
        )
        for text, action, outcome in expected:
            with self.subTest(intervention=text):
                world, scenario = _wired(text)
                self.assertEqual(world.calls, [[action]])
                self.assertEqual(scenario.counterfactual_outcome, outcome)
                self.assertEqual(scenario.intervention, text)

    def test_s3_n3_unidentified_strings_stay(self):
        for text, outcome in (
            ("hello", "outcome_with_hello"),
            ("patch", "outcome_with_patch"),
            ("isolate the system", "outcome_with_isolate_the_system"),
        ):
            with self.subTest(intervention=text):
                _assert_unmapped(self, text, outcome)
        for text in ("", False):
            with self.subTest(intervention=text):
                world, scenario = _wired(text)
                self.assertEqual(world.calls, [])
                self.assertEqual(scenario.counterfactual_outcome, "outcome_without_intervention")
        spaces = "   "
        world, scenario = _wired(spaces)
        self.assertEqual(world.calls, [])
        self.assertEqual(scenario.counterfactual_outcome, "outcome_with____")
        self.assertEqual(
            scenario.causal_chain,
            [
                f"1. {SITUATION}",
                f"2. Intervention: {spaces}",
                "3. Immediate effect",
                "4. Secondary effect",
                "5. Final outcome",
            ],
        )

    def test_s3_n4_simulated_action_keeps_same_veto_key_or_is_absent(self):
        samples = [text for text, _outcome in C1_CASES]
        samples.extend((
            "apply.patch",
            "increase.logging",
            "hello",
            "patch apply",
            "shutdown service",
            "SHUTDOWN_SERVICE",
            "A\u200bPPLY_PATCH",
        ))
        for text in samples:
            with self.subTest(intervention=text):
                world, _scenario = _wired(text)
                if not world.calls:
                    continue
                self.assertEqual(len(world.calls), 1)
                self.assertEqual(len(world.calls[0]), 1)
                self.assertEqual(veto_key(text), veto_key(world.calls[0][0]))
        world, _scenario = _wired("A\u200bPPLY_PATCH")
        self.assertEqual(world.calls, [])

    def test_s3_n5_s2_c6_rank_stays(self):
        engine = _engine(world_model=RecordingWorld())
        for spelling in ("service shutdown", "patch apply", "shutdown service"):
            with self.subTest(spelling=spelling):
                out = engine.rank_counterfactuals(OBS, ["IGNORE_ALERT"], [[spelling]])
                self.assertEqual(out["ranking"], [])
                self.assertEqual(out["vetoed"], [])
                self.assertEqual(out["unmodeled"], [[spelling]])
                self.assertIsNone(out["best"])
        out = engine.rank_counterfactuals(
            OBS, ["IGNORE_ALERT"], [["SHUTDOWN_SERVICE"], ["ISOLATE_SYSTEM"]]
        )
        self.assertEqual(out["best"]["actions"], ["ISOLATE_SYSTEM"])
        self.assertEqual(out["unmodeled"], [])
        self.assertIn(["SHUTDOWN_SERVICE"], [item["actions"] for item in out["ranking"]])
        self.assertEqual(out["vetoed"][0]["vetoes"][0]["reason"], SHUTDOWN_REASON)

    def test_s3_n6_non_veto_fallback_still_maps(self):
        expected = (
            ("apply.patch", "APPLY_PATCH", SECURE),
            ("increase.logging", "INCREASE_LOGGING", VERBOSE),
        )
        for text, action, outcome in expected:
            with self.subTest(intervention=text):
                world, scenario = _wired(text)
                self.assertEqual(world.calls, [[action]])
                self.assertEqual(scenario.counterfactual_outcome, outcome)
                self.assertEqual(scenario.intervention, text)
                self.assertEqual(scenario.causal_chain, [f"{SITUATION} causes {text}"])

    def test_s3_n7_compare_allowed_futures_still_vetoes_shutdown_service(self):
        out = compare_allowed_futures(_CompareWorld(), [["shutdown.service"], ["MONITOR"]])
        self.assertEqual(out["allowed_indices"], [1])
        self.assertEqual(out["vetoed_futures"], [0])
        self.assertEqual(out["best_future_index"], 1)
        self.assertEqual(
            out["vetoed_actions"],
            [{"action": "shutdown.service", "reason": SHUTDOWN_REASON}],
        )


if __name__ == "__main__":
    unittest.main()
