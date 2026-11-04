"""Contrato da revisão da Pearl no PR #154 (B1, S1, S2, S3 e documentação).

Reaproveita o oráculo por enumeração de ``test_scm_counterfactual_contract`` (não importa
o módulo sob teste). Seeds: ``random.Random(20261401 + k)``.
"""
from __future__ import annotations

import random
import unittest

from core.cognitivo.causal_world_model import WORLD_SCHEMA, RealCausalReasoning, RealWorldModelSimulator
from core.cognitivo.mcts_planner import DEFAULT_VETOES
from test_scm_counterfactual_contract import KEYS, SEED, _engine, _instance, all_worlds, apply, oracle, scm

SEED2 = SEED + 100


def _world():
    return RealWorldModelSimulator(RealCausalReasoning())


def _effects(world):
    from core.cognitivo.action_effects import ACTION_EFFECTS

    return {name: dict(entry["world"]) for name, entry in ACTION_EFFECTS.items()}


def _veto(action):
    return {"action": action, "reason": DEFAULT_VETOES[action]}


class ConflictBeforeUnmodeledTests(unittest.TestCase):
    """B1: a evidência inconsistente com a factual vence a ação contrafactual sem efeito verificado."""

    OBS = {"system_status": "compromised"}
    CONFLICT = {"system_status": ("compromised", "secure")}

    def test_b1a_rank_with_only_unmodeled_alternatives_reports_the_conflict(self):
        out = scm().rank_alternatives(self.OBS, ["APPLY_PATCH"], [["apply_patch"], ["FOO"]], _world())
        self.assertFalse(out["consistent"])
        self.assertEqual(out["conflicts"], self.CONFLICT)
        self.assertEqual((out["ranking"], out["best"], out["tied_best"], out["vetoed"],
                          out["unidentified"], out["unmodeled"]), ([], None, [], [], [], []))

    def test_b1b_outcome_conflict_wins_over_unmodeled(self):
        for counter in (["apply_patch"], ["FOO"], ["MONITOR"], []):
            with self.subTest(counter=counter):
                got = scm().counterfactual_outcome(self.OBS, ["APPLY_PATCH"], counter, _world())
                self.assertFalse(got.consistent)
                self.assertEqual(got.conflicts, self.CONFLICT)
                self.assertEqual((got.state, got.possible, got.utility), ({}, {}, None))

    def test_b1c_conflict_does_not_depend_on_the_counterfactual_sequence(self):
        checked = 0
        for k in range(100):
            world, effects, factual, counter, observed = _instance(k)
            if oracle(observed, factual, counter, effects)[0]:
                continue
            checked += 1
            with self.subTest(seed=SEED2 + k):
                base = scm().counterfactual_outcome(observed, factual, [], world)
                masked = scm().counterfactual_outcome(observed, factual, counter + ["GHOST"], world)
                self.assertFalse(masked.consistent)
                self.assertEqual(masked, base)
                out = scm().rank_alternatives(observed, factual, [["GHOST"], counter], world)
                self.assertEqual((out["consistent"], out["conflicts"]), (False, base.conflicts))
        self.assertGreaterEqual(checked, 5)

    def test_b1d_consistent_evidence_still_rejects_unmodeled_counterfactual(self):
        with self.assertRaises(scm().ScmInputError) as ctx:
            scm().counterfactual_outcome({"system_status": "secure"}, ["APPLY_PATCH"], ["apply_patch"], _world())
        self.assertEqual((ctx.exception.code, ctx.exception.actions), ("SCM_UNMODELED", ("apply_patch",)))

    def test_b1e_input_validation_precedes_the_conflict_check(self):
        for bad, code in (([" "], "SCM_BAD_ACTION"), (None, "SCM_NO_ACTIONS")):
            with self.subTest(bad=bad):
                with self.assertRaises(scm().ScmInputError) as ctx:
                    scm().counterfactual_outcome(self.OBS, ["APPLY_PATCH"], bad, _world())
                self.assertEqual(ctx.exception.code, code)
        with self.assertRaises(TypeError):
            scm().counterfactual_outcome(self.OBS, ["APPLY_PATCH"], "MONITOR", _world())


class VetoTests(unittest.TestCase):
    """S1: veto padrão marca e rebaixa; não exclui."""

    OBS = {"system_status": "compromised"}

    def _rank(self, alternatives, **kwargs):
        return scm().rank_alternatives(self.OBS, ["MONITOR"], alternatives, _world(), **kwargs)

    def test_s1a_vetoed_alternative_is_marked_kept_and_skipped_by_best(self):
        out = self._rank([["SHUTDOWN_SERVICE"], ["MONITOR"]])
        self.assertEqual([r["actions"] for r in out["ranking"]], [["SHUTDOWN_SERVICE"], ["MONITOR"]])
        self.assertEqual([r["utility"] for r in out["ranking"]], [1, 0])
        self.assertEqual(out["ranking"][0]["vetoed"], [_veto("SHUTDOWN_SERVICE")])
        self.assertEqual(out["ranking"][1]["vetoed"], [])
        self.assertEqual(out["vetoed"], [{"actions": ["SHUTDOWN_SERVICE"], "vetoes": [_veto("SHUTDOWN_SERVICE")]}])
        self.assertEqual(out["best"]["actions"], ["MONITOR"])
        self.assertIs(out["best"], out["ranking"][1])

    def test_s1b_empty_mapping_turns_the_veto_off(self):
        out = self._rank([["SHUTDOWN_SERVICE"], ["MONITOR"]], vetoes={})
        self.assertEqual(out["best"]["actions"], ["SHUTDOWN_SERVICE"])
        self.assertEqual(out["vetoed"], [])
        self.assertEqual(out["ranking"][0]["vetoed"], [])

    def test_s1c_custom_mapping_replaces_the_default(self):
        out = self._rank([["SHUTDOWN_SERVICE"], ["MONITOR"]], vetoes={"MONITOR": "motivo próprio"})
        self.assertEqual(out["best"]["actions"], ["SHUTDOWN_SERVICE"])
        self.assertEqual(out["vetoed"], [{"actions": ["MONITOR"], "vetoes": [{"action": "MONITOR", "reason": "motivo próprio"}]}])

    def test_s1d_all_vetoed_leaves_no_best_and_no_tie(self):
        out = self._rank([["SHUTDOWN_SERVICE"], ["IGNORE_ALERT"]])
        self.assertEqual(len(out["ranking"]), 2)
        self.assertEqual((out["best"], out["tied_best"]), (None, []))
        self.assertEqual(len(out["vetoed"]), 2)

    def test_s1e_vetoes_must_be_a_mapping(self):
        for bad in (["SHUTDOWN_SERVICE"], "SHUTDOWN_SERVICE", 3):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self._rank([["MONITOR"]], vetoes=bad)
        self.assertEqual(self._rank([["MONITOR"]], vetoes=None)["best"]["actions"], ["MONITOR"])

    def test_s1f_any_action_of_a_sequence_vetoes_it_with_reasons_in_order_without_duplicates(self):
        out = self._rank([["IGNORE_ALERT", "MONITOR", "SHUTDOWN_SERVICE", "IGNORE_ALERT"], ["MONITOR"]])
        item = next(r for r in out["ranking"] if len(r["actions"]) == 4)
        self.assertEqual(item["vetoed"], [_veto("IGNORE_ALERT"), _veto("SHUTDOWN_SERVICE")])
        self.assertEqual(out["best"]["actions"], ["MONITOR"])

    def test_s1g_veto_never_reorders_the_ranking(self):
        for k in range(100):
            world, effects, factual, _counter, observed = _instance(k)
            rng = random.Random(SEED2 + 300 + k)
            pool = sorted(effects) + ["SHUTDOWN_SERVICE", "IGNORE_ALERT"]
            alts = [[rng.choice(pool) for _ in range(rng.randint(0, 2))] for _ in range(5)]
            ok = oracle(observed, factual, factual, effects)[0]
            if not ok:
                continue
            with self.subTest(seed=SEED2 + 300 + k):
                on = scm().rank_alternatives(observed, factual, alts, world)
                off = scm().rank_alternatives(observed, factual, alts, world, vetoes={})
                self.assertEqual([r["actions"] for r in on["ranking"]], [r["actions"] for r in off["ranking"]])
                allowed = [r for r in on["ranking"] if not r["vetoed"]]
                self.assertEqual(on["best"], allowed[0] if allowed else None)
                for r in on["ranking"]:
                    names = [a for a in dict.fromkeys(r["actions"]) if a in DEFAULT_VETOES]
                    self.assertEqual([v["action"] for v in r["vetoed"]], names)

    def test_s1h_engine_applies_the_default_veto_and_forwards_vetoes(self):
        engine = _engine()
        obs = {"system_status": "compromised", "alert_level": "critical"}
        out = engine.rank_counterfactuals(obs, ["MONITOR"], [["SHUTDOWN_SERVICE"], ["MONITOR"]])
        self.assertEqual(out["best"]["actions"], ["MONITOR"])
        self.assertEqual(len(out["vetoed"]), 1)
        off = engine.rank_counterfactuals(obs, ["MONITOR"], [["SHUTDOWN_SERVICE"], ["MONITOR"]], vetoes={})
        self.assertEqual(off["best"]["actions"], ["SHUTDOWN_SERVICE"])


class TiedBestTests(unittest.TestCase):
    """S2: empates no topo e estado/não identificadas por alternativa."""

    OBS = {"system_status": "vulnerable"}

    def test_s2a_three_way_tie_is_exposed_with_the_vetoed_one_flagged(self):
        out = scm().rank_alternatives(self.OBS, ["MONITOR"],
                                      [["SHUTDOWN_SERVICE"], ["ISOLATE_SYSTEM"], ["MONITOR"]], _world())
        self.assertEqual([r["utility"] for r in out["ranking"]], [1, 1, 1])
        self.assertEqual([r["verified_status"] for r in out["tied_best"]], ["isolated", "vulnerable", "offline"])
        self.assertEqual([r["actions"] for r in out["tied_best"]],
                         [["ISOLATE_SYSTEM"], ["MONITOR"], ["SHUTDOWN_SERVICE"]])
        self.assertEqual(out["best"]["actions"], ["ISOLATE_SYSTEM"])
        self.assertEqual([bool(r["vetoed"]) for r in out["tied_best"]], [False, False, True])

    def test_s2b_unique_top_has_a_single_tied_best_and_lower_utilities_are_left_out(self):
        out = scm().rank_alternatives({"system_status": "compromised"}, ["MONITOR"],
                                      [["APPLY_PATCH"], ["ROLLBACK"], ["MONITOR"]], _world())
        self.assertEqual(out["tied_best"], [out["best"]])
        self.assertEqual(out["best"]["actions"], ["APPLY_PATCH"])

    def test_s2c_tied_best_ignores_a_higher_utility_vetoed_alternative(self):
        out = scm().rank_alternatives({"system_status": "compromised"}, ["MONITOR"],
                                      [["SHUTDOWN_SERVICE"], ["MONITOR"]], _world())
        self.assertEqual([r["actions"] for r in out["tied_best"]], [["MONITOR"]])

    def test_s2d_items_carry_state_and_unidentified_equal_to_the_oracle(self):
        for k in range(100):
            world, effects, factual, _counter, observed = _instance(k)
            rng = random.Random(SEED2 + 600 + k)
            pool = sorted(effects)
            alts = [[rng.choice(pool) for _ in range(rng.randint(0, 2))] for _ in range(4)]
            if not oracle(observed, factual, factual, effects)[0]:
                continue
            out = scm().rank_alternatives(observed, factual, alts, world)
            with self.subTest(seed=SEED2 + 600 + k):
                for item in out["ranking"]:
                    _, state, unid, _ = oracle(observed, factual, item["actions"], effects)
                    self.assertEqual(item["state"], state)
                    self.assertEqual(item["unidentified"], unid)
                if out["best"] is None:
                    self.assertEqual(out["tied_best"], [])
                else:
                    top = out["best"]["utility"]
                    self.assertEqual([r["actions"] for r in out["tied_best"]],
                                     [r["actions"] for r in out["ranking"] if r["utility"] == top])

    def test_s2e_inconsistent_evidence_has_no_tie(self):
        out = scm().rank_alternatives({"system_status": "compromised"}, ["APPLY_PATCH"], [["MONITOR"]], _world())
        self.assertEqual((out["tied_best"], out["vetoed"]), ([], []))


class PossibleValuesTests(unittest.TestCase):
    """S3: domínio possível de cada chave não identificada."""

    def test_s3a_overwritten_status_has_the_whole_domain(self):
        got = scm().counterfactual_outcome({"system_status": "secure"}, ["APPLY_PATCH"], ["MONITOR"], _world())
        self.assertTrue(got.consistent)
        self.assertIn("system_status", got.unidentified)
        self.assertEqual(got.possible["system_status"], WORLD_SCHEMA["system_status"])
        self.assertEqual(len(got.possible["system_status"]), 6)

    def test_s3b_possible_equals_the_values_the_oracle_leaves_open(self):
        for k in range(200):
            world, effects, factual, counter, observed = _instance(k)
            got = scm().counterfactual_outcome(observed, factual, counter, world)
            worlds = [w for w in all_worlds()
                      if all(apply(w, factual, effects)[key] == v for key, v in observed.items())]
            with self.subTest(seed=SEED + k):
                if not worlds:
                    self.assertEqual(got.possible, {})
                    continue
                finals = [apply(w, counter, effects) for w in worlds]
                self.assertEqual(set(got.possible), set(got.unidentified))
                for key, domain in got.possible.items():
                    self.assertEqual(set(domain), {f[key] for f in finals})
                    self.assertEqual(domain, WORLD_SCHEMA[key])
                for key in got.state:
                    self.assertNotIn(key, got.possible)

    def test_s3c_inconsistent_has_no_possible(self):
        got = scm().counterfactual_outcome({"system_status": "compromised"}, ["APPLY_PATCH"], [], _world())
        self.assertEqual(got.possible, {})
        self.assertEqual(set(got.unidentified), set(KEYS))


class DocumentationTests(unittest.TestCase):
    def test_d1_module_states_its_limits(self):
        doc = scm().__doc__
        for needle in ("sem arestas", "do(ação)", "nomes de ação exatos", "veto"):
            with self.subTest(needle=needle):
                self.assertIn(needle, doc)


if __name__ == "__main__":
    unittest.main()
