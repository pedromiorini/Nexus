"""Contrato SCM-CF: contrafactual por abdução-ação-predição (Pearl, 2009, cap. 7).

Critérios pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md (seção «SCM contrafactual»)
em commit anterior a este teste e ao código. O oráculo abaixo não importa nada do módulo:
enumera os 96 estados iniciais possíveis do WORLD_SCHEMA, filtra os que reproduzem a
observação sob a sequência factual e lê, entre eles, o que a sequência contrafactual
determina. Seeds: ``random.Random(20261301 + k)``.
"""
from __future__ import annotations

import contextlib
import copy
import io
import itertools
import random
import unittest

from core.cognitivo.action_effects import ACTION_EFFECTS
from core.cognitivo.causal_world_model import (
    STATUS_UTILITY,
    WORLD_SCHEMA,
    RealCausalReasoning,
    RealWorldModelSimulator,
)
from core.cognitivo.counterfactual import RealCounterfactualReasoningEngine

SEED = 20261301
DECLARED = ("DECL_A", "DECL_B", "DECL_C")
KEYS = sorted(WORLD_SCHEMA)


def scm():
    from core.cognitivo import scm_counterfactual

    return scm_counterfactual


def _engine(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealCounterfactualReasoningEngine(**kwargs)


def _instance(k):
    """Instância aleatória: mundo com declaradas, F, C, estado inicial e observação."""
    rng = random.Random(SEED + k)
    causal = RealCausalReasoning()
    effects = {name: dict(entry["world"]) for name, entry in ACTION_EFFECTS.items()}
    for name in DECLARED:
        chosen = rng.sample(KEYS, rng.randint(1, 3))
        declared = {key: rng.choice(WORLD_SCHEMA[key]) for key in chosen}
        assert causal.declare_intervention(name, declared) == []
        effects[name] = declared
    world = RealWorldModelSimulator(causal)
    pool = sorted(effects)
    factual = [rng.choice(pool) for _ in range(rng.randint(0, 3))]
    counter = [rng.choice(pool) for _ in range(rng.randint(0, 3))]
    initial = {key: rng.choice(WORLD_SCHEMA[key]) for key in KEYS}
    final = apply(initial, factual, effects)
    keep = KEYS if rng.random() < 0.5 else rng.sample(KEYS, rng.randint(1, 3))
    observed = {key: final[key] for key in keep}
    if rng.random() < 0.25:  # corrompe uma chave: pode virar inconsistente
        key = rng.choice(sorted(observed))
        observed[key] = rng.choice([v for v in WORLD_SCHEMA[key] if v != observed[key]])
    return world, effects, factual, counter, observed


def apply(state, actions, effects):
    out = dict(state)
    for action in actions:
        out.update(effects[action])
    return out


def all_worlds():
    domains = [WORLD_SCHEMA[key] for key in KEYS]
    for values in itertools.product(*domains):
        yield dict(zip(KEYS, values))


def oracle(observed, factual, counter, effects):
    """(consistente, estado identificado, não identificadas, U identificado) por enumeração."""
    worlds = [
        w for w in all_worlds()
        if all(apply(w, factual, effects)[key] == value for key, value in observed.items())
    ]
    if not worlds:
        return False, {}, KEYS, {}
    finals = [apply(w, counter, effects) for w in worlds]
    state, unidentified, backdrop = {}, [], {}
    for key in KEYS:
        values = {f[key] for f in finals}
        if len(values) == 1:
            state[key] = next(iter(values))
        else:
            unidentified.append(key)
        initial_values = {w[key] for w in worlds}
        if len(initial_values) == 1:
            backdrop[key] = next(iter(initial_values))
    return True, state, unidentified, backdrop


class AbductionExactnessTests(unittest.TestCase):
    def test_a1_matches_enumeration_oracle(self):
        inconsistent = with_unidentified = 0
        for k in range(200):
            world, effects, factual, counter, observed = _instance(k)
            with self.subTest(seed=SEED + k):
                result = scm().counterfactual_outcome(observed, factual, counter, world)
                ok, state, unidentified, backdrop = oracle(observed, factual, counter, effects)
                self.assertEqual(result.consistent, ok)
                self.assertEqual(result.state, state)
                self.assertEqual(list(result.unidentified), unidentified)
                self.assertEqual(result.backdrop, backdrop if ok else {})
                inconsistent += not ok
                with_unidentified += bool(ok and unidentified)
        # a amostra precisa exercitar os dois ramos (limites pré-registrados)
        self.assertGreaterEqual(inconsistent, 10)
        self.assertGreaterEqual(with_unidentified, 60)

    def test_a2_same_action_returns_the_observation(self):
        """Axioma de consistência de Pearl: do(factual) reproduz toda chave observada."""
        for k in range(100):
            world, effects, factual, _counter, observed = _instance(k)
            ok, *_ = oracle(observed, factual, factual, effects)
            if not ok:
                continue
            with self.subTest(seed=SEED + k):
                result = scm().counterfactual_outcome(observed, factual, factual, world)
                for key, value in observed.items():
                    self.assertEqual(result.state[key], value)
                self.assertTrue(set(result.unidentified).isdisjoint(observed))

    def test_a3_inputs_and_model_are_not_mutated(self):
        for k in range(30):
            world, _effects, factual, counter, observed = _instance(k)
            before = copy.deepcopy((observed, factual, counter, world.current_state,
                                    world.causal_reasoning.declared_interventions,
                                    dict(world.causal_reasoning.causal_graph["adjacency"])))
            scm().counterfactual_outcome(observed, factual, counter, world)
            after = (observed, factual, counter, world.current_state,
                     world.causal_reasoning.declared_interventions,
                     dict(world.causal_reasoning.causal_graph["adjacency"]))
            with self.subTest(seed=SEED + k):
                self.assertEqual(before, copy.deepcopy(after))

    def test_a4_free_text_is_not_a_mechanism(self):
        causal = RealCausalReasoning()
        world = RealWorldModelSimulator(causal)
        base = scm().counterfactual_outcome(
            {"system_status": "compromised", "alert_level": "critical"}, ["IGNORE_ALERT"], ["MONITOR"], world)
        causal.causal_graph["adjacency"]["MONITOR"].add("system_status secure")
        causal.causal_graph["adjacency"]["IGNORE_ALERT"].add("logging_level verbose")
        again = scm().counterfactual_outcome(
            {"system_status": "compromised", "alert_level": "critical"}, ["IGNORE_ALERT"], ["MONITOR"], world)
        self.assertEqual(base, again)
        self.assertNotIn("logging_level", again.state)
        causal.causal_graph["adjacency"]["GHOST"] = {"system_status secure"}
        with self.assertRaises(scm().ScmInputError) as ctx:
            scm().counterfactual_outcome({"system_status": "stable"}, [], ["GHOST"], world)
        self.assertEqual(ctx.exception.code, "SCM_UNMODELED")
        self.assertEqual(ctx.exception.actions, ("GHOST",))

    def test_a4b_declared_overrides_table_for_the_same_action(self):
        causal = RealCausalReasoning()
        causal.declare_intervention("APPLY_PATCH", {"alert_level": "medium"})
        world = RealWorldModelSimulator(causal)
        result = scm().counterfactual_outcome({"logging_level": "normal"}, [], ["APPLY_PATCH"], world)
        self.assertEqual(result.state, {"alert_level": "medium", "logging_level": "normal",
                                        "security_patch_applied": True, "system_status": "secure"})
        plain = scm().counterfactual_outcome({"logging_level": "normal"}, [], ["APPLY_PATCH"])
        self.assertEqual(plain.state["alert_level"], "low")

    def test_a5_simulator_initial_state_is_not_the_background(self):
        world, _effects, factual, counter, observed = _instance(3)
        expected = scm().counterfactual_outcome(observed, factual, counter, world)
        world.current_state.state.update({"system_status": "offline", "alert_level": "critical",
                                          "logging_level": "verbose"})
        self.assertEqual(scm().counterfactual_outcome(observed, factual, counter, world), expected)

    def test_a6_status_and_utility(self):
        for k in range(100):
            world, _effects, factual, counter, observed = _instance(k)
            result = scm().counterfactual_outcome(observed, factual, counter, world)
            with self.subTest(seed=SEED + k):
                status = result.state.get("system_status")
                self.assertEqual(result.verified_status, status)
                self.assertEqual(result.utility, STATUS_UTILITY[status] if status else None)

    def test_a7_inconsistent_observation_withholds_prediction(self):
        result = scm().counterfactual_outcome(
            {"system_status": "compromised"}, ["APPLY_PATCH"], ["MONITOR"])
        self.assertFalse(result.consistent)
        self.assertEqual(result.conflicts, {"system_status": ("compromised", "secure")})
        self.assertEqual((result.state, result.verified_status, result.utility), ({}, None, None))
        self.assertEqual(list(result.unidentified), KEYS)

    def test_a8_observation_is_canonicalized(self):
        result = scm().counterfactual_outcome({" System_Status ": "SECURE"}, [], [])
        self.assertEqual(result.state, {"system_status": "secure"})


class RankAlternativesTests(unittest.TestCase):
    OBS = {"system_status": "compromised", "alert_level": "critical", "security_patch_applied": False}
    ALTS = [["SHUTDOWN_SERVICE"], ["MONITOR"], ["INVENTED"], ["ISOLATE_SYSTEM"], ["APPLY_PATCH"],
            ["ROLLBACK"], ["MONITOR", "APPLY_PATCH"]]

    def test_b1_concrete_scenario(self):
        out = scm().rank_alternatives(self.OBS, ["IGNORE_ALERT"], self.ALTS)
        self.assertTrue(out["consistent"])
        self.assertEqual(out["factual_utility"], 0)
        self.assertEqual([r["actions"] for r in out["ranking"]],
                         [["APPLY_PATCH"], ["MONITOR", "APPLY_PATCH"], ["ROLLBACK"],
                          ["ISOLATE_SYSTEM"], ["SHUTDOWN_SERVICE"]])
        self.assertEqual([r["utility"] for r in out["ranking"]], [3, 3, 2, 1, 1])
        self.assertEqual([r["gain"] for r in out["ranking"]], [3, 3, 2, 1, 1])
        self.assertEqual(out["ranking"][0]["verified_status"], "secure")
        self.assertEqual(out["best"], out["ranking"][0])
        self.assertEqual(out["unidentified"], [["MONITOR"]])
        self.assertEqual(out["unmodeled"], [["INVENTED"]])

    def test_b2_order_of_alternatives_does_not_matter(self):
        for k in range(30):
            rng = random.Random(SEED + 500 + k)
            shuffled = [list(a) for a in self.ALTS]
            rng.shuffle(shuffled)
            with self.subTest(seed=SEED + 500 + k):
                self.assertEqual(scm().rank_alternatives(self.OBS, ["IGNORE_ALERT"], shuffled),
                                 scm().rank_alternatives(self.OBS, ["IGNORE_ALERT"], self.ALTS))

    def test_b3_ranking_agrees_with_enumeration_oracle(self):
        for k in range(100):
            world, effects, factual, _counter, observed = _instance(k)
            rng = random.Random(SEED + 900 + k)
            pool = sorted(effects)
            alts = [[rng.choice(pool) for _ in range(rng.randint(0, 2))] for _ in range(4)]
            ok, *_ = oracle(observed, factual, factual, effects)
            out = scm().rank_alternatives(observed, factual, alts, world)
            with self.subTest(seed=SEED + k):
                self.assertEqual(out["consistent"], ok)
                if not ok:
                    self.assertEqual((out["ranking"], out["best"], out["unidentified"]), ([], None, []))
                    continue
                expected, expected_unid = [], []
                for alt in alts:
                    _, state, _, _ = oracle(observed, factual, alt, effects)
                    status = state.get("system_status")
                    if status is None:
                        expected_unid.append(alt)
                    else:
                        expected.append((-STATUS_UTILITY[status], len(alt), tuple(alt)))
                self.assertEqual([(-r["utility"], len(r["actions"]), tuple(r["actions"]))
                                  for r in out["ranking"]], sorted(expected))
                self.assertEqual(out["unidentified"], sorted(expected_unid, key=lambda s: (len(s), tuple(s))))

    def test_b4_gain_is_none_without_observed_status(self):
        out = scm().rank_alternatives({"alert_level": "high"}, [], [["APPLY_PATCH"]])
        self.assertIsNone(out["factual_utility"])
        self.assertIsNone(out["ranking"][0]["gain"])
        self.assertEqual(out["ranking"][0]["utility"], 3)

    def test_b5_inconsistent_observation(self):
        out = scm().rank_alternatives({"system_status": "compromised"}, ["APPLY_PATCH"], [["MONITOR"]])
        self.assertFalse(out["consistent"])
        self.assertEqual(out["conflicts"], {"system_status": ("compromised", "secure")})
        self.assertEqual((out["ranking"], out["best"]), ([], None))

    def test_b6_nothing_identified_gives_no_best(self):
        out = scm().rank_alternatives({"alert_level": "critical"}, ["IGNORE_ALERT"], [["MONITOR"]])
        self.assertEqual((out["ranking"], out["best"], out["unidentified"]), ([], None, [["MONITOR"]]))


class NegativeInputTests(unittest.TestCase):
    def _code(self, call, *args, **kwargs):
        with self.assertRaises(scm().ScmInputError) as ctx:
            call(*args, **kwargs)
        return ctx.exception.code

    def test_n1_missing_or_empty_observation(self):
        for bad in (None, {}):
            with self.subTest(bad=bad):
                self.assertEqual(self._code(scm().counterfactual_outcome, bad, [], []), "SCM_NO_OBSERVATION")
                self.assertEqual(self._code(scm().rank_alternatives, bad, [], [["MONITOR"]]), "SCM_NO_OBSERVATION")

    def test_n2_observation_outside_schema(self):
        for bad in ({"colour": "red"}, {"system_status": "melting"}, {"system_status": True},
                    {"security_patch_applied": "true"}, {"security_patch_applied": 1}, {3: "x"}):
            with self.subTest(bad=bad):
                self.assertEqual(self._code(scm().counterfactual_outcome, bad, [], []), "SCM_BAD_OBSERVATION")
        self.assertEqual(self._code(scm().counterfactual_outcome,
                                    {"system_status": "secure", " SYSTEM_STATUS": "stable"}, [], []),
                         "SCM_BAD_OBSERVATION")

    def test_n3_missing_actions(self):
        obs = {"system_status": "secure"}
        self.assertEqual(self._code(scm().counterfactual_outcome, obs, None, []), "SCM_NO_ACTIONS")
        self.assertEqual(self._code(scm().counterfactual_outcome, obs, [], None), "SCM_NO_ACTIONS")
        self.assertEqual(self._code(scm().rank_alternatives, obs, None, [["MONITOR"]]), "SCM_NO_ACTIONS")

    def test_n4_blank_action(self):
        obs = {"system_status": "secure"}
        for bad in ("", "   "):
            with self.subTest(bad=bad):
                self.assertEqual(self._code(scm().counterfactual_outcome, obs, [bad], []), "SCM_BAD_ACTION")
                self.assertEqual(self._code(scm().counterfactual_outcome, obs, [], [bad]), "SCM_BAD_ACTION")

    def test_n5_wrong_types(self):
        obs = {"system_status": "secure"}
        for args in (([1], []), ([], [None]), ("APPLY_PATCH", []), ([], "MONITOR"), ([], b"MONITOR")):
            with self.subTest(args=args), self.assertRaises(TypeError):
                scm().counterfactual_outcome(obs, *args)
        with self.assertRaises(TypeError):
            scm().counterfactual_outcome(["system_status"], [], [])
        with self.assertRaises(TypeError):
            scm().rank_alternatives(obs, [], "MONITOR")
        with self.assertRaises(TypeError):
            scm().rank_alternatives(obs, [], [["MONITOR"], "APPLY_PATCH"])

    def test_n6_missing_or_empty_alternatives(self):
        obs = {"system_status": "secure"}
        for bad in (None, [], ()):
            with self.subTest(bad=bad):
                self.assertEqual(self._code(scm().rank_alternatives, obs, [], bad), "SCM_NO_ALTERNATIVES")

    def test_n7_unmodeled_factual_is_an_input_error(self):
        obs = {"system_status": "secure"}
        self.assertEqual(self._code(scm().counterfactual_outcome, obs, ["INVENTED"], ["MONITOR"]), "SCM_UNMODELED")
        self.assertEqual(self._code(scm().counterfactual_outcome, obs, [], ["INVENTED", "MONITOR"]), "SCM_UNMODELED")
        self.assertEqual(self._code(scm().rank_alternatives, obs, ["INVENTED"], [["MONITOR"]]), "SCM_UNMODELED")

    def test_n8_action_names_are_exact(self):
        obs = {"system_status": "secure"}
        self.assertEqual(self._code(scm().counterfactual_outcome, obs, [], ["apply_patch"]), "SCM_UNMODELED")


class EngineIntegrationTests(unittest.TestCase):
    def test_e1_engine_delegates_to_its_world_model(self):
        causal = RealCausalReasoning()
        causal.declare_intervention("HARDEN_HOST", {"system_status": "secure"})
        world = RealWorldModelSimulator(causal)
        engine = _engine(causal_reasoning=causal, world_model=world)
        obs = {"system_status": "compromised", "alert_level": "critical"}
        got = engine.counterfactual_query(obs, ["IGNORE_ALERT"], ["HARDEN_HOST"])
        self.assertEqual(got, scm().counterfactual_outcome(obs, ["IGNORE_ALERT"], ["HARDEN_HOST"], world))
        self.assertEqual(got.state["system_status"], "secure")
        ranked = engine.rank_counterfactuals(obs, ["IGNORE_ALERT"], [["HARDEN_HOST"], ["ROLLBACK"]])
        self.assertEqual(ranked["best"]["actions"], ["HARDEN_HOST"])

    def test_e2_engine_without_world_model_uses_the_table_only(self):
        engine = _engine()
        got = engine.counterfactual_query({"system_status": "compromised"}, ["IGNORE_ALERT"], ["APPLY_PATCH"])
        self.assertEqual(got.state["system_status"], "secure")
        with self.assertRaises(scm().ScmInputError) as ctx:
            engine.counterfactual_query({"system_status": "stable"}, [], ["HARDEN_HOST"])
        self.assertEqual(ctx.exception.code, "SCM_UNMODELED")

    def test_e3_query_leaves_no_trace_in_the_engine(self):
        engine = _engine(world_model=RealWorldModelSimulator(RealCausalReasoning()))
        before = (engine.get_statistics(), list(engine.scenarios), list(engine.insights),
                  dict(engine.unverified_claims))
        engine.counterfactual_query({"system_status": "stable"}, [], ["ROLLBACK"])
        engine.rank_counterfactuals({"system_status": "stable"}, [], [["ROLLBACK"]])
        self.assertEqual(before, (engine.get_statistics(), list(engine.scenarios), list(engine.insights),
                                  dict(engine.unverified_claims)))

    def test_e4_negative_inputs_propagate(self):
        engine = _engine()
        with self.assertRaises(scm().ScmInputError) as ctx:
            engine.counterfactual_query({}, [], [])
        self.assertEqual(ctx.exception.code, "SCM_NO_OBSERVATION")
        with self.assertRaises(scm().ScmInputError) as ctx:
            engine.rank_counterfactuals({"system_status": "stable"}, [], [])
        self.assertEqual(ctx.exception.code, "SCM_NO_ALTERNATIVES")


if __name__ == "__main__":
    unittest.main()
