"""Contrato P1b de docs/cognitivo/AUDITORIA_2026-10-22.md, seção «MCTS P1b — veto padrão de ações perigosas e três
pontos da Pearl»: (V) veto padrão de IGNORE_ALERT/SHUTDOWN_SERVICE com o motivo em ``PlanResult.vetoed_actions``;
(S) ``_is_unmodeled`` é um predicado estrutural (fora da tabela e não declarada), texto não decide;
(T) desempate determinístico lexicográfico, e ação modelada antes de declarada, com a mesma recompensa.

Referências escritas no teste, sem calcular o esperado chamando o módulo. Seeds: 20261201 + k.
"""
from __future__ import annotations

import copy
import dataclasses
import random
import unittest
from types import SimpleNamespace

from core.cognitivo.action_effects import ACTION_EFFECTS
from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator
from core.cognitivo import mcts_planner as mp
from core.cognitivo.mcts_planner import MCTSNode, PlanResult, RealMCTSPlanner

SEEDS20 = [20261201 + k for k in range(20)]
SEEDS40 = [20261201 + k for k in range(40)]
GOAL = "secure the system"
VETOED = ("IGNORE_ALERT", "SHUTDOWN_SERVICE")


def _wm(causal=None):
    return RealWorldModelSimulator(causal_reasoning=causal)


def _plan(seed, actions=None, model=True, causal=None, iterations=300, depth=3, **kwargs):
    random.seed(seed)
    planner = RealMCTSPlanner(world_model=_wm(causal) if model else None, max_depth=depth, **kwargs)
    return planner.plan(GOAL, available_actions=actions, max_iterations=iterations)


def _causal_with(*texts):
    causal = RealCausalReasoning()
    for text in texts:
        causal.extract_causal_relations(text)
    return causal


def _strip(result):
    data = dataclasses.asdict(result)
    data.pop("planning_time")
    return data


class V1DefaultVetoTests(unittest.TestCase):
    def test_all_candidates_vetoed_gives_empty_plan(self):
        for model in (True, False):
            for seed in SEEDS20:
                with self.subTest(model=model, seed=seed):
                    r = _plan(seed, actions=list(VETOED), model=model)
                    self.assertEqual(r.best_action_sequence, [])
                    self.assertEqual(r.expected_reward, 0.0)

    def test_vetoed_action_loses_to_a_safe_one(self):
        cases = [(["IGNORE_ALERT", "MONITOR"], ["MONITOR"]), (["SHUTDOWN_SERVICE", "APPLY_PATCH"], ["APPLY_PATCH"])]
        for model in (True, False):
            for seed in SEEDS20:
                for actions, expected in cases:
                    with self.subTest(model=model, seed=seed, actions=actions):
                        self.assertEqual(_plan(seed, actions=actions, model=model).best_action_sequence, expected)

    def test_default_action_space_never_returns_a_vetoed_action(self):
        for model in (True, False):
            for seed in SEEDS20:
                with self.subTest(model=model, seed=seed):
                    r = _plan(seed, model=model)
                    self.assertFalse(set(r.best_action_sequence) & set(VETOED))
                    self.assertFalse({a["first_action"] for a in r.alternative_plans} & set(VETOED))


class V2ReasonInResultTests(unittest.TestCase):
    def test_default_vetoes_are_exported_with_reasons(self):
        self.assertEqual(set(mp.DEFAULT_VETOES), set(VETOED))
        for reason in mp.DEFAULT_VETOES.values():
            self.assertIsInstance(reason, str)
            self.assertTrue(reason.strip())

    def test_reason_is_returned_in_order_without_duplicates(self):
        for seed in SEEDS20:
            with self.subTest(seed=seed):
                r = _plan(seed, actions=["SHUTDOWN_SERVICE", "MONITOR", "IGNORE_ALERT", "SHUTDOWN_SERVICE"])
                self.assertEqual(r.vetoed_actions, [
                    {"action": "SHUTDOWN_SERVICE", "reason": mp.DEFAULT_VETOES["SHUTDOWN_SERVICE"]},
                    {"action": "IGNORE_ALERT", "reason": mp.DEFAULT_VETOES["IGNORE_ALERT"]}])
                full = _plan(seed)
                self.assertEqual([v["action"] for v in full.vetoed_actions], ["IGNORE_ALERT", "SHUTDOWN_SERVICE"])

    def test_nothing_vetoed_gives_empty_list_and_no_leak_between_plans(self):
        planner = RealMCTSPlanner(world_model=_wm(), max_depth=3)
        random.seed(SEEDS20[0])
        first = planner.plan(GOAL, available_actions=["IGNORE_ALERT", "MONITOR"], max_iterations=50)
        second = planner.plan(GOAL, available_actions=["MONITOR", "APPLY_PATCH"], max_iterations=50)
        self.assertEqual([v["action"] for v in first.vetoed_actions], ["IGNORE_ALERT"])
        self.assertEqual(second.vetoed_actions, [])
        self.assertEqual(_plan(SEEDS20[1], actions=["MONITOR"]).vetoed_actions, [])

    def test_reason_is_an_instance_attribute_not_a_dataclass_field(self):
        names = {f.name for f in dataclasses.fields(PlanResult)}
        self.assertNotIn("vetoed_actions", names)
        r = _plan(SEEDS20[0], actions=["IGNORE_ALERT", "MONITOR"])
        self.assertNotIn("vetoed_actions", dataclasses.asdict(r))
        self.assertTrue(r.vetoed_actions)


class V3SwitchTests(unittest.TestCase):
    def test_empty_mapping_turns_the_veto_off(self):
        for model in (True, False):
            for seed in SEEDS20:
                with self.subTest(model=model, seed=seed):
                    r = _plan(seed, actions=["SHUTDOWN_SERVICE"], model=model, vetoes={})
                    self.assertEqual(r.best_action_sequence, ["SHUTDOWN_SERVICE"])
                    self.assertEqual(r.vetoed_actions, [])

    def test_custom_mapping_replaces_the_default(self):
        for seed in SEEDS20:
            with self.subTest(seed=seed):
                r = _plan(seed, actions=["MONITOR", "IGNORE_ALERT"], vetoes={"MONITOR": "x"})
                self.assertEqual(r.vetoed_actions, [{"action": "MONITOR", "reason": "x"}])
                self.assertNotIn("MONITOR", r.best_action_sequence)

    def test_mapping_is_copied_and_defaults_are_not_mutated(self):
        mine = {"MONITOR": "x"}
        planner = RealMCTSPlanner(world_model=_wm(), vetoes=mine)
        mine["ESCALATE"] = "y"
        self.assertEqual(planner.vetoes, {"MONITOR": "x"})
        before = copy.deepcopy(mp.DEFAULT_VETOES)
        _plan(SEEDS20[0], actions=["IGNORE_ALERT", "MONITOR"])
        self.assertEqual(mp.DEFAULT_VETOES, before)

    def test_non_mapping_is_rejected(self):
        for bad in (["IGNORE_ALERT"], "IGNORE_ALERT", 3):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                RealMCTSPlanner(vetoes=bad)


class V4NoSideEffectTests(unittest.TestCase):
    def test_veto_does_not_touch_model_or_tables(self):
        table = copy.deepcopy(ACTION_EFFECTS)
        for seed in SEEDS20:
            with self.subTest(seed=seed):
                random.seed(seed)
                wm = _wm()
                before = copy.deepcopy(wm.current_state.state)
                planner = RealMCTSPlanner(world_model=wm, max_depth=3)
                actions_before = copy.deepcopy(planner._action_effects)
                planner.plan(GOAL, available_actions=["IGNORE_ALERT", "MONITOR"], max_iterations=100)
                self.assertEqual(wm.current_state.state, before)
                self.assertEqual(planner._action_effects, actions_before)
                self.assertEqual(ACTION_EFFECTS, table)


class S1StructuralPredicateTests(unittest.TestCase):
    def test_is_modeled_is_structural(self):
        causal = RealCausalReasoning()
        causal.declare_intervention("FOO_BAR", {"system_status": "secure"})
        causal.extract_causal_relations("APPLY_PATCH causes system_status compromised")
        causal.extract_causal_relations("ZZ_QQ causes system_status secure")
        wm = _wm(causal)
        self.assertTrue(wm.is_modeled("APPLY_PATCH"))
        self.assertTrue(wm.is_modeled("FOO_BAR"))
        self.assertFalse(wm.is_modeled("ZZ_QQ"))
        self.assertFalse(wm.is_modeled("NEVER_SEEN"))
        self.assertFalse(_wm().is_modeled("ZZ_QQ"))

    TEXTS = ["APPLY_PATCH causes system_status compromised", "MONITOR causes system_status secure",
             "ROLLBACK leads to system_status secure", "DEPLOY_UPDATE results in alert_level critical",
             "APPLY_PATCH triggers system_status vulnerable", "ESCALATE causes system_status secure"]
    ACTION_SETS = [["APPLY_PATCH", "MONITOR"], ["MONITOR", "ESCALATE"], ["ROLLBACK", "MONITOR"],
                   ["APPLY_PATCH", "DEPLOY_UPDATE", "ROLLBACK"]]

    def test_text_about_modeled_actions_does_not_change_the_plan(self):
        """IJ1b (critério novo): plano inteiro igual, incluindo success_probability."""
        for text in self.TEXTS:
            for actions in self.ACTION_SETS:
                for seed in SEEDS20[:5]:
                    with self.subTest(text=text, actions=actions, seed=seed):
                        plain = _plan(seed, actions=actions)
                        with_text = _plan(seed, actions=actions, causal=_causal_with(text))
                        self.assertEqual(_strip(with_text), _strip(plain))
                        self.assertEqual(with_text.success_probability, plain.success_probability)

    def test_text_about_an_invented_action_keeps_zero(self):
        for seed in SEEDS20[:5]:
            with self.subTest(seed=seed):
                plain = _plan(seed, actions=["ZZ_QQ"])
                with_text = _plan(seed, actions=["ZZ_QQ"], causal=_causal_with("ZZ_QQ causes system_status secure"))
                self.assertEqual(plain.success_probability, 0.0)
                self.assertEqual(with_text.success_probability, 0.0)

    def test_double_without_is_modeled_keeps_the_warning_criterion(self):
        def double(warnings):
            return SimpleNamespace(
                current_state=SimpleNamespace(state={"system_status": "vulnerable"}),
                simulate=lambda steps: SimpleNamespace(
                    verified_status="vulnerable", warnings=list(warnings), success=False))
        self.assertTrue(RealMCTSPlanner(world_model=double(["unknown action: 'X'"]))._is_unmodeled("X"))
        self.assertFalse(RealMCTSPlanner(world_model=double([]))._is_unmodeled("X"))
        self.assertTrue(RealMCTSPlanner()._is_unmodeled("ZZ_QQ"))
        self.assertFalse(RealMCTSPlanner()._is_unmodeled("MONITOR"))


class T1DeterministicTieBreakTests(unittest.TestCase):
    def test_forty_seeds_give_one_sequence_and_sorted_alternatives(self):
        found = set()
        for seed in SEEDS40:
            r = _plan(seed)
            found.add(tuple(r.best_action_sequence))
            keys = [(-a["expected_reward"], a["first_action"]) for a in r.alternative_plans]
            with self.subTest(seed=seed):
                self.assertEqual(keys, sorted(keys))
        self.assertEqual(found, {("APPLY_PATCH",)})

    def test_record_found_is_independent_of_arrival_order(self):
        def best(order):
            planner = RealMCTSPlanner(world_model=_wm())
            for steps in order:
                planner._record_found(1.0, {"steps_taken": list(steps)})
            return planner._best_found[1]
        a, b = ["APPLY_PATCH"], ["DEPLOY_UPDATE"]
        self.assertEqual(best([a, b]), a)
        self.assertEqual(best([b, a]), a)
        two = ["MONITOR", "APPLY_PATCH"]
        self.assertEqual(best([two, a]), a)
        self.assertEqual(best([a, two]), a)
        self.assertEqual(best([["Z_DECL"], two]), two)
        self.assertEqual(best([two, ["Z_DECL"]]), two)

    def test_record_found_without_model_keeps_the_first_on_a_tie(self):
        planner = RealMCTSPlanner()
        planner._record_found(0.8, {"steps_taken": ["DEPLOY_UPDATE"]})
        planner._record_found(0.8, {"steps_taken": ["APPLY_PATCH"]})
        self.assertEqual(planner._best_found[1], ["DEPLOY_UPDATE"])

    def test_alternatives_with_equal_reward_are_ordered_by_name(self):
        planner = RealMCTSPlanner(world_model=_wm())
        names = ["MONITOR", "ESCALATE", "APPLY_PATCH", "ROLLBACK"]
        outputs = []
        for order in (names, list(reversed(names)), sorted(names)):
            root = MCTSNode(state={}, action_taken=None, parent=None)
            for name in order:
                root.children.append(MCTSNode(state={}, action_taken=name, parent=root, visits=2, total_reward=1.0))
            outputs.append(planner._extract_alternatives(root, top_n=4))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])
        self.assertEqual([a["first_action"] for a in outputs[0]], sorted(names))


class T2ModeledBeforeDeclaredTests(unittest.TestCase):
    def test_declared_secure_action_does_not_beat_apply_patch(self):
        causal = RealCausalReasoning()
        causal.declare_intervention("FOO_BAR", {"system_status": "secure"})
        causal.declare_intervention("AAA_DECL", {"system_status": "secure"})
        for seed in SEEDS40:
            with self.subTest(seed=seed):
                r = _plan(seed, actions=["FOO_BAR", "AAA_DECL", "APPLY_PATCH", "MONITOR"], causal=causal)
                self.assertEqual(r.best_action_sequence, ["APPLY_PATCH"])


class T3MixedSequenceHypothesisTests(unittest.TestCase):
    def test_unmodeled_prefix_keeps_the_verified_status_but_not_success(self):
        sim = _wm().simulate(["ZZ_X", "APPLY_PATCH"])
        self.assertEqual(sim.verified_status, "secure")
        self.assertIs(sim.success, False)
        planner = RealMCTSPlanner(world_model=_wm())
        self.assertEqual(planner._evaluate_world({"steps_taken": ["ZZ_X", "APPLY_PATCH"]}), 1.0)

    def test_planner_never_returns_the_mixed_sequence(self):
        for seed in SEEDS40:
            with self.subTest(seed=seed):
                r = _plan(seed, actions=["ZZ_X", "APPLY_PATCH"])
                self.assertEqual(r.best_action_sequence, ["APPLY_PATCH"])
                self.assertEqual(r.success_probability, 1.0)


if __name__ == "__main__":
    unittest.main()
