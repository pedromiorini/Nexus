"""Contrato SC3R/SC3N (errata E2 do SC3, pré-registro em docs/cognitivo/AUDITORIA_2026-10-22.md,
seção «SCM do(X) — errata E2»): ação declarada fora de ACTION_EFFECTS gera aviso, sem rejeitar.

Referência independente: estado inicial e tabela congelados abaixo (não calcula o esperado chamando o
módulo). Seeds fixas: random.Random(20261101 + k), k = 0..19.
"""
from __future__ import annotations

import random
import string
import unittest

from core.cognitivo.action_effects import ACTION_EFFECTS
from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator

INITIAL = {"security_patch_applied": False, "alert_level": "high", "system_status": "vulnerable"}
APPLY_PATCH = {"security_patch_applied": True, "alert_level": "low", "system_status": "secure"}
PAIRS = (
    ("system_status", "secure"),
    ("system_status", "stable"),
    ("system_status", "offline"),
    ("alert_level", "low"),
    ("logging_level", "verbose"),
)
SEEDS = range(20)


def _warning(action):
    return f"declared intervention for action outside ACTION_EFFECTS: {action!r}"


def _name(rng):
    n = rng.randint(6, 12)
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def _fresh():
    cr = RealCausalReasoning()
    return cr, RealWorldModelSimulator(causal_reasoning=cr)


class SC3RDeclaredOutsideTableWarnsTests(unittest.TestCase):
    def test_warns_and_still_applies(self):
        for k in SEEDS:
            rng = random.Random(20261101 + k)
            action = _name(rng)
            key, value = rng.choice(PAIRS)
            with self.subTest(k=k, action=action, pair=(key, value)):
                self.assertNotIn(action, ACTION_EFFECTS)
                cr, world = _fresh()
                self.assertEqual(cr.declare_intervention(action, {key: value}), [])
                self.assertEqual(world._do(action), {key: value})
                result = world.simulate([action])
                expected = dict(INITIAL)
                expected[key] = value
                self.assertEqual(result.warnings, [_warning(action)])
                self.assertEqual(result.claims, [])
                self.assertEqual(result.final_state.state, expected)
                self.assertEqual(result.success, expected["system_status"] in ("secure", "stable"))
                self.assertEqual(result.verified_status, expected["system_status"])
                self.assertFalse(any("unknown action" in w for w in result.warnings))

    def test_one_warning_per_occurrence(self):
        for k in SEEDS:
            rng = random.Random(20261101 + k)
            action = _name(rng)
            with self.subTest(k=k):
                cr, world = _fresh()
                cr.declare_intervention(action, {"system_status": "stable"})
                self.assertEqual(world.simulate([action, action]).warnings, [_warning(action)] * 2)
                mixed = world.simulate(["APPLY_PATCH", action])
                self.assertEqual(mixed.warnings, [_warning(action)])
                self.assertEqual(mixed.final_state.state["system_status"], "stable")
                self.assertEqual(world.simulate(["APPLY_PATCH"]).warnings, [])
                self.assertEqual(world.simulate(["APPLY_PATCH"]).final_state.state,
                                 dict(INITIAL, **APPLY_PATCH))


class SC3NNoFalsePositiveTests(unittest.TestCase):
    """SC3N: entrada vazia/ausente/inválida não gera o aviso novo."""

    def test_empty_or_invalid_effects(self):
        for k in SEEDS:
            rng = random.Random(20261101 + k)
            action = _name(rng)
            with self.subTest(k=k):
                cr, world = _fresh()
                self.assertEqual(cr.declare_intervention(action, {}), [])
                self.assertEqual(world.simulate([action]).warnings, [f"unknown action: {action!r}"])
                cr2, world2 = _fresh()
                self.assertEqual(len(cr2.declare_intervention(action, {"system_status": "pwned"})), 1)
                self.assertEqual(world2.simulate([action]).warnings, [f"unknown action: {action!r}"])

    def test_empty_sequence_and_blank_actions(self):
        _, world = _fresh()
        self.assertEqual(world.simulate([]).warnings, [])
        self.assertEqual(world.simulate([""]).warnings, ["unknown action: ''"])
        self.assertEqual(world.simulate([None]).warnings, ["unknown action: None"])

    def test_without_causal_reasoning(self):
        world = RealWorldModelSimulator()
        self.assertEqual(world.simulate(["FOO"]).warnings, ["unknown action: 'FOO'"])

    def test_text_claim_only_has_no_declared_warning(self):
        cr, world = _fresh()
        cr.extract_causal_relations("FOO causes system_status secure")
        result = world.simulate(["FOO"])
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("unverified causal claim", result.warnings[0])

    def test_table_action_with_declaration_does_not_warn(self):
        cr, world = _fresh()
        cr.declare_intervention("APPLY_PATCH", {"alert_level": "medium"})
        result = world.simulate(["APPLY_PATCH"])
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.final_state.state["alert_level"], "medium")

    def test_invalid_arguments_keep_their_codes(self):
        cr = RealCausalReasoning()
        with self.assertRaises(ValueError):
            cr.declare_intervention("", {"system_status": "secure"})
        with self.assertRaises(TypeError):
            cr.declare_intervention("A", [])


if __name__ == "__main__":
    unittest.main()
