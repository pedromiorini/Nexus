"""Contrato S2: o fallback de tokens só devolve nome com a mesma veto_key.

Critérios S2-C1..C6 do pré-registro
docs/cognitivo/S2_FALLBACK_TOKENS_VETO_KEY_PRE_REGISTRO_2026-11-26.md
(SHA-256 5f3e666464c8c9dafb1086533cbd805829f19042754a5b1c8cacb48d9bc58f99),
alternativa 1. Escrito antes da correção de _map_intervention.
"""
from __future__ import annotations

import contextlib
import io
import unittest

from hypothesis import example, given, settings
from hypothesis import strategies as st

from core.cognitivo.action_effects import veto_key
from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator
from core.cognitivo.counterfactual import RealCounterfactualReasoningEngine
from core.cognitivo.mcts_planner import DEFAULT_VETOES

SITUATION = "system vulnerable"
OFFLINE = "state:alert_level=low,security_patch_applied=False,system_status=offline"
SECURE = "state:alert_level=low,security_patch_applied=True,system_status=secure"
OBS = {
    "system_status": "compromised",
    "alert_level": "critical",
    "security_patch_applied": False,
    "logging_level": "normal",
}
C1_CASES = (
    ("service shutdown", "outcome_with_service_shutdown"),
    ("Service, Shutdown!", "outcome_with_Service,_Shutdown!"),
    ("service_shutdown()", "outcome_with_service_shutdown()"),
    ("patch apply", "outcome_with_patch_apply"),
    ("alert ignore", "outcome_with_alert_ignore"),
    ("\ufb01INCREASE_LOGGING", "outcome_with_\ufb01INCREASE_LOGGING"),
)
S2_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.!()\u200b"


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


class S2FallbackVetoKeyContractTests(unittest.TestCase):
    def test_s2_c1_different_veto_key_does_not_map(self):
        for text, outcome in C1_CASES:
            with self.subTest(intervention=text):
                world, scenario = _wired(text)
                self.assertEqual(world.calls, [])
                self.assertEqual(scenario.counterfactual_outcome, outcome)
                self.assertEqual(scenario.actual_outcome, "outcome_without_intervention")
                self.assertEqual(scenario.intervention, text)
                self.assertEqual(scenario.causal_chain, [f"{SITUATION} causes {text}"])

    def test_s2_c2_same_veto_key_fallback_still_maps(self):
        # Errata: docs/cognitivo/S3_ERRATA_S2C2_2026-11-26.md
        # O nome do método permanece o do contrato S2. As duas grafias deixam de mapear.
        for text, outcome in (
            ("shutdown.service", "outcome_with_shutdown.service"),
            ("shutdown_service()", "outcome_with_shutdown_service()"),
        ):
            with self.subTest(intervention=text):
                world, scenario = _wired(text)
                self.assertEqual(world.calls, [])
                self.assertEqual(scenario.counterfactual_outcome, outcome)
                self.assertEqual(scenario.intervention, text)
                self.assertEqual(scenario.causal_chain, [f"{SITUATION} causes {text}"])

    def test_s2_c3_step1_names_still_map(self):
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

    def test_s2_c4_unidentified_strings_stay(self):
        for text, outcome in (
            ("hello", "outcome_with_hello"),
            ("patch", "outcome_with_patch"),
            ("isolate the system", "outcome_with_isolate_the_system"),
        ):
            with self.subTest(intervention=text):
                world, scenario = _wired(text)
                self.assertEqual(world.calls, [])
                self.assertEqual(scenario.counterfactual_outcome, outcome)
                self.assertEqual(scenario.intervention, text)
                self.assertEqual(scenario.causal_chain, [f"{SITUATION} causes {text}"])
        for text in ("", False):
            with self.subTest(intervention=text):
                world, scenario = _wired(text)
                self.assertEqual(world.calls, [])
                self.assertEqual(scenario.counterfactual_outcome, "outcome_without_intervention")
        world, scenario = _wired("   ")
        self.assertEqual(world.calls, [])
        self.assertEqual(scenario.counterfactual_outcome, "outcome_with_" + "___")

    @given(st.text(alphabet=S2_ALPHABET, min_size=0, max_size=24))
    @example("service shutdown")
    @example("Service, Shutdown!")
    @example("service_shutdown()")
    @example("patch apply")
    @example("alert ignore")
    @example("\ufb01INCREASE_LOGGING")
    @example("shutdown.service")
    @example("shutdown_service()")
    @example("apply patch")
    @example("apply_patch")
    @example("APPLY_PATCH")
    @example("SHUTDOWN_SERVICE")
    @example("shutdown service")
    @example("hello")
    @example("patch")
    @example("A\u200bPPLY_PATCH")
    @settings(max_examples=40, deadline=None)
    def test_s2_c5_simulated_action_is_none_or_same_veto_key(self, text):
        world, _scenario = _wired(text)
        if not world.calls:
            return
        self.assertEqual(len(world.calls), 1)
        self.assertEqual(len(world.calls[0]), 1)
        action = world.calls[0][0]
        self.assertEqual(veto_key(text), veto_key(action))

    def test_s2_c6_rank_noncanonical_stays_unmodeled_and_exact_stays_vetoed(self):
        engine = _engine(world_model=RecordingWorld())
        for spelling in ("service shutdown", "shutdown.service", "patch apply", "shutdown service"):
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
                    "reason": DEFAULT_VETOES["SHUTDOWN_SERVICE"],
                }],
            }],
        )


if __name__ == "__main__":
    unittest.main()
