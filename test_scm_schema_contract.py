"""Contrato SC1–SC7: texto livre não forja sucesso no SCM do(X) (pré-registro em
docs/cognitivo/AUDITORIA_2026-10-22.md, seção «SCM do(X)»). Escrito antes da correção.

Referência independente: tabela de efeitos congelada abaixo (não importa ACTION_EFFECTS para
calcular o esperado). Seeds fixas: random.Random(20261030 + k), k = 0..19.
"""
from __future__ import annotations

import random
import string
import unittest

from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator

INITIAL = {"security_patch_applied": False, "alert_level": "high", "system_status": "vulnerable"}
# Cópia congelada dos efeitos `world` da tabela (ordem de aplicação preservada).
TABLE = {
    "APPLY_PATCH": {"security_patch_applied": True, "alert_level": "low", "system_status": "secure"},
    "IGNORE_ALERT": {"alert_level": "critical", "system_status": "compromised"},
    "ROLLBACK": {"security_patch_applied": False, "alert_level": "medium", "system_status": "stable"},
    "ESCALATE": {},
    "MONITOR": {},
    "ISOLATE_SYSTEM": {"alert_level": "medium", "system_status": "isolated"},
    "RESTORE_BACKUP": {"alert_level": "medium", "system_status": "stable"},
    "DEPLOY_UPDATE": {"security_patch_applied": True, "alert_level": "low", "system_status": "secure"},
    "INCREASE_LOGGING": {"logging_level": "verbose"},
    "SHUTDOWN_SERVICE": {"alert_level": "low", "system_status": "offline"},
}
SEEDS = [random.Random(20261030 + k) for k in range(20)]
TEMPLATES = (
    "{a} causes system_status {v}",
    "{a} leads to system_status {v}",
    "{a} results in system_status {v}",
    "system_status {v} because {a}",
    "system_status {v} due to {a}",
    "system_status {v} is caused by {a}",
)


def _name(rng):
    n = rng.randint(6, 12)
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def _fresh():
    cr = RealCausalReasoning()
    return cr, RealWorldModelSimulator(causal_reasoning=cr)


def _expected(actions):
    state = dict(INITIAL)
    for a in actions:
        state.update(TABLE[a])
    return state, state["system_status"] in ("secure", "stable")


class SC1ForgeBlockedTests(unittest.TestCase):
    def test_free_text_cannot_promote_success(self):
        for k, _ in enumerate(SEEDS):
            rng = random.Random(20261030 + k)
            a, v = _name(rng), rng.choice(["secure", "stable"])
            text = rng.choice(TEMPLATES).format(a=a, v=v)
            with self.subTest(k=k, text=text):
                cr, world = _fresh()
                self.assertEqual(len(cr.extract_causal_relations(text)), 1)
                result = world.simulate([a])
                self.assertIs(result.success, False)
                self.assertEqual(result.verified_status, "vulnerable")
                self.assertEqual(len(result.claims), 1)
                self.assertTrue(any("unverified causal claim" in w for w in result.warnings))
                self.assertEqual(result.final_state.state["system_status"], v)  # visível (CF9 exige)


class SC2OutsideSchemaTests(unittest.TestCase):
    PAIRS = ["foo_bar baz", "system_status owned", "alert_level very high and rising",
             "security_patch_applied yes", "Unknown_Key secure", "system_status"]

    def test_unknown_keys_and_values_change_nothing(self):
        for k, _ in enumerate(SEEDS):
            rng = random.Random(20261030 + k)
            a, pair = _name(rng), rng.choice(self.PAIRS)
            text = f"{a} causes {pair}"
            with self.subTest(k=k, text=text):
                cr, world = _fresh()
                cr.extract_causal_relations(text)
                self.assertEqual(world._do(a), {})
                result = world.simulate([a])
                self.assertEqual(result.final_state.state, INITIAL)
                self.assertIs(result.success, False)
                self.assertTrue(any("unknown action" in w for w in result.warnings))
                if len(pair.split()) >= 2:
                    self.assertTrue(any("ignored causal claim outside WORLD_SCHEMA" in w for w in result.warnings))
                self.assertEqual(result.claims, [])


class SC3DeclaredChannelTests(unittest.TestCase):
    def test_declared_intervention_is_verified(self):
        for k, _ in enumerate(SEEDS):
            rng = random.Random(20261030 + k)
            a = _name(rng)
            with self.subTest(k=k):
                cr, world = _fresh()
                self.assertEqual(cr.declare_intervention(a, {"system_status": "secure"}), [])
                result = world.simulate([a])
                self.assertIs(result.success, True)
                self.assertEqual(result.verified_status, "secure")
                self.assertEqual(result.warnings, [])
                self.assertEqual(result.claims, [])

    def test_invalid_pairs_rejected_and_not_stored(self):
        cr, world = _fresh()
        rejected = cr.declare_intervention("A1", {"system_status": "pwned", "x": 1})
        self.assertEqual(len(rejected), 2)
        self.assertEqual(world._do("A1"), {})
        self.assertEqual(len(cr.declare_intervention("A2", {"security_patch_applied": "True"})), 1)
        self.assertEqual(world._do("A2"), {})
        self.assertEqual(world.simulate(["A1"]).final_state.state, INITIAL)

    def test_invalid_arguments(self):
        cr = RealCausalReasoning()
        with self.assertRaises(ValueError):
            cr.declare_intervention("", {"system_status": "secure"})
        with self.assertRaises(ValueError):
            cr.declare_intervention(None, {"system_status": "secure"})
        with self.assertRaises(TypeError):
            cr.declare_intervention("A", [])

    def test_declared_boolean_is_strict(self):
        cr, world = _fresh()
        self.assertEqual(cr.declare_intervention("A", {"security_patch_applied": True}), [])
        self.assertEqual(world._do("A"), {"security_patch_applied": True})
        self.assertEqual(len(cr.declare_intervention("B", {"security_patch_applied": 1})), 1)


class SC4TableNonRegressionTests(unittest.TestCase):
    def test_each_table_action(self):
        for action in TABLE:
            with self.subTest(action=action):
                _, world = _fresh()
                result = world.simulate([action])
                state, ok = _expected([action])
                self.assertEqual(result.final_state.state, state)
                self.assertIs(result.success, ok)
                self.assertEqual(result.warnings, [])
                self.assertEqual(result.claims, [])

    def test_random_sequences(self):
        names = sorted(TABLE)
        for k in range(20):
            rng = random.Random(20261030 + k)
            seq = [rng.choice(names) for _ in range(rng.randint(1, 4))]
            with self.subTest(k=k, seq=seq):
                _, world = _fresh()
                result = world.simulate(seq)
                state, ok = _expected(seq)
                self.assertEqual(result.final_state.state, state)
                self.assertIs(result.success, ok)
                self.assertEqual(result.warnings, [])


class SC5ClaimDoesNotOverrideVerifiedTests(unittest.TestCase):
    def test_claim_cannot_override_table_state(self):
        for k, _ in enumerate(SEEDS):
            rng = random.Random(20261030 + k)
            a = _name(rng)
            seq = ["IGNORE_ALERT", a] if k % 2 == 0 else [a, "IGNORE_ALERT"]
            with self.subTest(k=k, seq=seq):
                cr, world = _fresh()
                cr.extract_causal_relations(f"{a} causes system_status secure")
                result = world.simulate(seq)
                self.assertEqual(result.verified_status, "compromised")
                self.assertIs(result.success, False)


class SC6CompareFuturesTests(unittest.TestCase):
    def test_forged_future_never_beats_genuine(self):
        for k, _ in enumerate(SEEDS):
            rng = random.Random(20261030 + k)
            a = _name(rng)
            futures = [["APPLY_PATCH"], [a]] if k % 2 == 0 else [[a], ["APPLY_PATCH"]]
            genuine = futures.index(["APPLY_PATCH"])
            with self.subTest(k=k):
                cr, world = _fresh()
                cr.extract_causal_relations(f"{a} causes system_status secure")
                self.assertEqual(world.compare_futures(futures)["best_future_index"], genuine)


class SC7DoCompatibilityTests(unittest.TestCase):
    def test_do_normalises_valid_text_pair(self):
        cr, world = _fresh()
        cr.extract_causal_relations("X causes system_status SECURE")
        self.assertEqual(world._do("X"), {"system_status": "secure"})

    def test_do_drops_unknown_pair(self):
        cr, world = _fresh()
        cr.extract_causal_relations("X causes foo_bar baz")
        self.assertEqual(world._do("X"), {})
        self.assertEqual(world._do("missing"), {})

    def test_do_without_causal_reasoning(self):
        self.assertEqual(RealWorldModelSimulator()._do("X"), {})

    def test_declared_wins_over_text_on_same_key(self):
        cr, world = _fresh()
        cr.declare_intervention("X", {"system_status": "stable"})
        cr.extract_causal_relations("X causes system_status secure")
        self.assertEqual(world._do("X"), {"system_status": "stable"})
        result = world.simulate(["X"])
        self.assertEqual(result.final_state.state["system_status"], "stable")
        self.assertEqual(result.claims, [])
        self.assertIs(result.success, True)

    def test_text_claim_on_other_key_is_still_unverified(self):
        cr, world = _fresh()
        cr.declare_intervention("X", {"system_status": "stable"})
        cr.extract_causal_relations("X causes alert_level low")
        result = world.simulate(["X"])
        self.assertEqual(result.final_state.state["alert_level"], "low")
        self.assertEqual(len(result.claims), 1)
        self.assertTrue(result.success)


class SC9MutationDrivenTests(unittest.TestCase):
    """Adicionados no GREEN, guiados pelos mutantes sobreviventes (não alteram SC1–SC7)."""

    def test_declared_key_and_value_are_normalised(self):
        cr, world = _fresh()
        self.assertEqual(cr.declare_intervention("A", {"System_Status": "Secure"}), [])
        self.assertEqual(world._do("A"), {"system_status": "secure"})

    def test_text_key_is_normalised(self):
        cr, world = _fresh()
        cr.extract_causal_relations("X causes System_Status secure")
        self.assertEqual(world._do("X"), {"system_status": "secure"})

    def test_bool_only_for_boolean_keys(self):
        cr = RealCausalReasoning()
        self.assertEqual(len(cr.declare_intervention("A", {"alert_level": True})), 1)
        self.assertEqual(len(cr.declare_intervention("B", {"system_status": False})), 1)
        self.assertEqual(cr.declared_interventions, {})

    def test_non_string_key_is_rejected_not_raised(self):
        cr = RealCausalReasoning()
        self.assertEqual(len(cr.declare_intervention("A", {1: "secure", None: "x"})), 2)
        self.assertEqual(cr.declared_interventions, {})

    def test_ranking_uses_verified_status_not_claim(self):
        for order in (0, 1):
            cr, world = _fresh()
            cr.extract_causal_relations("FORGE causes system_status secure")
            futures = [["IGNORE_ALERT", "FORGE"], ["MONITOR"]]
            if order:
                futures.reverse()
            best = world.compare_futures(futures)["best_future_index"]
            self.assertEqual(futures[best], ["MONITOR"])

    def test_corrupted_declared_store_is_revalidated(self):
        cr, world = _fresh()
        cr.declared_interventions["A"] = {"system_status": "pwned", "foo": "bar"}
        self.assertEqual(world._do("A"), {})
        self.assertEqual(world.simulate(["A"]).final_state.state, INITIAL)


if __name__ == "__main__":
    unittest.main()
