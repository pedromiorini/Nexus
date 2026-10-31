"""Contrato P1c de docs/cognitivo/AUDITORIA_2026-10-22.md, seção «MCTS P1c — veto no cérebro central, sucesso de ação
declarada e contratos do veto»: (B) ``compare_allowed_futures`` e os dois pontos ``compare_futures`` do cérebro nunca
indexam sequência vetada; (D) ação declarada fora da tabela não chega a ``success_probability == 1.0``; (N) o veto
casa pelo nome normalizado, valida o mapa e documenta o que ``replace`` perde e o que ``alternative_plans`` garante.

Referências escritas no teste. Seeds: 20261301 + k (B), 20261302 + k (D), 20261303 + k (N).
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import pickle
import random
import unittest

from core.cognitivo import mcts_planner as mp
from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator
from core.cognitivo.mcts_planner import DEFAULT_VETOES, MCTSNode, PlanResult, RealMCTSPlanner

SEEDS_B = [20261301 + k for k in range(40)]
SEEDS_D = [20261302 + k for k in range(40)]
SEEDS_N = [20261303 + k for k in range(20)]
GOAL = "secure the system"
POOL = ["IGNORE_ALERT", "SHUTDOWN_SERVICE", "MONITOR", "APPLY_PATCH", "ROLLBACK", "ZZ_Q", "ignore alert", "Shutdown-Service"]
VETOED_KEYS = {"IGNORE_ALERT", "SHUTDOWN_SERVICE"}


def _key(action):
    return action.strip().replace(" ", "_").replace("-", "_").replace("__", "_").upper()


def _is_vetoed(action):
    return _key(action) in VETOED_KEYS


def _wm(causal=None):
    return RealWorldModelSimulator(causal_reasoning=causal)


def _compare(sequences, **kwargs):
    return mp.compare_allowed_futures(_wm(), sequences, **kwargs)


def _declared():
    causal = RealCausalReasoning()
    causal.declare_intervention("AAA_DECL", {"system_status": "secure"})
    return causal


def _plan(seed, actions=None, model=True, causal=None, iterations=300, depth=3, planner=None, **kwargs):
    random.seed(seed)
    planner = planner or RealMCTSPlanner(world_model=_wm(causal) if model else None, max_depth=depth, **kwargs)
    return planner.plan(GOAL, available_actions=actions, max_iterations=iterations)


class B1AllowedFuturesTests(unittest.TestCase):
    def test_best_index_is_never_a_vetoed_sequence(self):
        for seed in SEEDS_B:
            rng = random.Random(seed)
            sequences = [[rng.choice(POOL) for _ in range(rng.randint(1, 3))] for _ in range(rng.randint(2, 5))]
            allowed = [i for i, seq in enumerate(sequences) if not any(_is_vetoed(a) for a in seq)]
            with self.subTest(seed=seed, sequences=sequences):
                out = _compare(sequences)
                if not allowed:
                    self.assertIsNone(out["best_future_index"])
                    self.assertIsNone(out["best_future"])
                    self.assertEqual(out["results"], [])
                else:
                    ref = _wm().compare_futures([sequences[i] for i in allowed])
                    self.assertEqual(out["best_future_index"], allowed[ref["best_future_index"]])
                    self.assertNotIn(out["best_future_index"], out["vetoed_futures"])
                self.assertEqual(out["allowed_indices"], allowed)
                self.assertEqual(out["vetoed_futures"], [i for i in range(len(sequences)) if i not in allowed])
                self.assertEqual(len(out["results"]), len(allowed))

    def test_fixed_cases_and_vetoed_actions(self):
        cases = [
            ([["SHUTDOWN_SERVICE"], ["IGNORE_ALERT"]], None, ["SHUTDOWN_SERVICE", "IGNORE_ALERT"]),
            ([["SHUTDOWN_SERVICE", "APPLY_PATCH"], ["MONITOR"]], 1, ["SHUTDOWN_SERVICE"]),
            ([["MONITOR"], ["IGNORE_ALERT", "APPLY_PATCH"]], 0, ["IGNORE_ALERT"]),
            ([["ignore alert"], ["ZZ_Q"]], 1, ["ignore alert"]),
            ([["IGNORE_ALERT"], ["IGNORE_ALERT", "MONITOR"], ["MONITOR"]], 2, ["IGNORE_ALERT"]),
        ]
        for sequences, best, names in cases:
            with self.subTest(sequences=sequences):
                out = _compare(sequences)
                self.assertEqual(out["best_future_index"], best)
                self.assertEqual([v["action"] for v in out["vetoed_actions"]], names)
                for item in out["vetoed_actions"]:
                    self.assertEqual(item["reason"], DEFAULT_VETOES[_key(item["action"])])

    def test_empty_input(self):
        out = _compare([])
        self.assertEqual((out["results"], out["best_future_index"], out["best_future"], out["vetoed_actions"]),
                         ([], None, None, []))

    def test_without_veto_equals_the_model_comparison(self):
        pool = ["MONITOR", "APPLY_PATCH", "ROLLBACK", "ZZ_Q", "DEPLOY_UPDATE"]
        for seed in SEEDS_B:
            rng = random.Random(seed)
            sequences = [[rng.choice(pool) for _ in range(rng.randint(1, 3))] for _ in range(rng.randint(1, 4))]
            with self.subTest(seed=seed):
                model = _wm()  # o mesmo modelo nas duas chamadas (o carimbo de tempo do estado inicial é por instância)
                out = mp.compare_allowed_futures(model, sequences)
                ref = model.compare_futures(sequences)
                self.assertEqual(out["best_future_index"], ref["best_future_index"])
                self.assertEqual(out["best_future"], ref["best_future"])
                self.assertEqual(out["results"], ref["results"])
                self.assertEqual(out["vetoed_actions"], [])
                self.assertEqual(out["vetoed_futures"], [])

    def test_custom_and_disabled_vetoes_and_no_mutation(self):
        sequences = [["SHUTDOWN_SERVICE"], ["MONITOR"]]
        self.assertEqual(_compare(sequences, vetoes={})["best_future_index"], 0)
        self.assertEqual(_compare(sequences, vetoes={})["vetoed_actions"], [])
        out = _compare([["APPLY_PATCH"], ["MONITOR"]], vetoes={"APPLY_PATCH": "x"})
        self.assertEqual(out["best_future_index"], 1)
        self.assertEqual(out["vetoed_actions"], [{"action": "APPLY_PATCH", "reason": "x"}])
        out = _compare([["IGNORE_ALERT"], ["MONITOR"]], vetoes={"MONITOR": "custom"})
        self.assertEqual(out["best_future_index"], 0)
        self.assertEqual(out["vetoed_actions"], [{"action": "MONITOR", "reason": "custom"}])
        with self.assertRaises(TypeError):
            _compare(sequences, vetoes=["IGNORE_ALERT"])
        before_defaults, before_input = dict(DEFAULT_VETOES), copy.deepcopy(sequences)
        _compare(sequences)
        self.assertEqual(DEFAULT_VETOES, before_defaults)
        self.assertEqual(sequences, before_input)

    def test_compare_rejects_invalid_veto_entries(self):
        cases = [
            ({"MONITOR": "custom", 1: "invalid key"}, TypeError),
            ({"IGNORE_ALERT": 1}, TypeError),
            ({"  ": "empty normalized key"}, ValueError),
        ]
        for vetoes, error in cases:
            with self.subTest(vetoes=vetoes), self.assertRaises(error):
                _compare([["IGNORE_ALERT"]], vetoes=vetoes)

    def test_compare_rejects_malformed_future_element(self):
        for sequences in (["IGNORE_ALERT"], [["MONITOR", None]]):
            with self.subTest(sequences=sequences), self.assertRaises(TypeError):
                _compare(sequences)

        class RecordingModel:
            called = False

            def compare_futures(self, sequences):
                self.called = True
                return {"results": []}

        model = RecordingModel()
        with self.assertRaises(TypeError):
            mp.compare_allowed_futures(model, [["MONITOR"], "IGNORE_ALERT"])
        self.assertFalse(model.called)

    def test_exported(self):
        self.assertIn("compare_allowed_futures", mp.__all__)
        self.assertEqual([f.name for f in dataclasses.fields(PlanResult)],
                         ["goal", "best_action_sequence", "expected_reward", "iterations_run", "tree_nodes_created",
                          "alternative_plans", "planning_time", "success_probability"])


class BrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.constitutional_brain import CompleteNexusBrain
        with contextlib.redirect_stdout(io.StringIO()):
            cls.brain = CompleteNexusBrain(enable_router=True)

    def _think(self, actions, goal="secure the system", seed=SEEDS_B[0]):
        random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            return self.brain.think("What should we do?", possible_actions=actions, planning_goal=goal,
                                    mcts_iterations=60, use_router=False)

    def test_b5_think_never_indexes_a_vetoed_sequence(self):
        for seed in SEEDS_B[:5]:
            with self.subTest(seed=seed, case="all vetoed"):
                r = self._think([["SHUTDOWN_SERVICE"], ["IGNORE_ALERT"]], seed=seed)
                self.assertIsNone(r["best_future"])
                self.assertEqual({v["action"] for v in r["vetoed_actions"]}, VETOED_KEYS)
            with self.subTest(seed=seed, case="one vetoed"):
                r = self._think([["IGNORE_ALERT"], ["MONITOR"]], seed=seed)
                self.assertEqual(r["best_future"], 1)
                self.assertEqual([v["action"] for v in r["vetoed_actions"]], ["IGNORE_ALERT"])
            with self.subTest(seed=seed, case="mixed sequence"):
                r = self._think([["SHUTDOWN_SERVICE", "APPLY_PATCH"], ["MONITOR"]], seed=seed)
                self.assertEqual(r["best_future"], 1)
            with self.subTest(seed=seed, case="none vetoed"):
                r = self._think([["MONITOR"], ["APPLY_PATCH"]], seed=seed)
                self.assertEqual(r["vetoed_actions"], [])
                self.assertIn(r["best_future"], (0, 1))

    def test_b5_vetoed_actions_from_planner_and_from_later_steps(self):
        r = self._think([["SHUTDOWN_SERVICE"]])  # planner e comparação veem o veto; o resultado o deduplica
        self.assertEqual([v["action"] for v in r["vetoed_actions"]], ["SHUTDOWN_SERVICE"])
        r = self._think([["MONITOR", "IGNORE_ALERT"], ["MONITOR"]])  # só a comparação vê o passo vetado
        self.assertEqual(r["best_future"], 1)
        self.assertEqual([v["action"] for v in r["vetoed_actions"]], ["IGNORE_ALERT"])

    def test_b5_default_vetoes_propagate_without_possible_actions(self):
        r = self._think(None)
        self.assertEqual(
            {(v["action"], v["reason"]) for v in r["vetoed_actions"]},
            set(DEFAULT_VETOES.items()),
        )

    def test_b5_think_compares_a_single_candidate_sequence(self):
        r = self._think([["MONITOR", "IGNORE_ALERT"]])
        self.assertIn("world_comparison", r)
        self.assertIsNone(r["best_future"])
        self.assertEqual(r["world_comparison"]["vetoed_futures"], [0])
        self.assertEqual([v["action"] for v in r["vetoed_actions"]], ["IGNORE_ALERT"])

    def test_b5_disabled_vetoes_in_the_brain_planner(self):
        saved = self.brain.mcts_planner.vetoes
        self.brain.mcts_planner.vetoes = {}
        try:
            r = self._think([["SHUTDOWN_SERVICE"], ["IGNORE_ALERT"]])
        finally:
            self.brain.mcts_planner.vetoes = saved
        self.assertEqual(r["best_future"], 0)
        self.assertEqual(r["vetoed_actions"], [])

    def test_b6_without_planning_goal_there_is_no_comparison(self):
        r = self._think([["SHUTDOWN_SERVICE"], ["IGNORE_ALERT"]], goal=None)
        self.assertNotIn("best_future", r)
        self.assertNotIn("world_comparison", r)

    def test_b7_router_world_model_site(self):
        from core.constitutional_brain import ModuleType
        router = self.brain.router
        cases = [([["SHUTDOWN_SERVICE"], ["IGNORE_ALERT"]], None), ([["SHUTDOWN_SERVICE", "APPLY_PATCH"], ["MONITOR"]], 1),
                 ([["MONITOR"], ["IGNORE_ALERT", "APPLY_PATCH"]], 0), ([["ignore alert"], ["ZZ_Q"]], 1)]
        for actions, best in cases:
            with self.subTest(actions=actions):
                out = router._invoke_module(ModuleType.WORLD_MODEL, self.brain.world_model, "p",
                                            {"possible_actions": actions})
                self.assertEqual(out["best_future_index"], best)
                self.assertTrue(out["vetoed_actions"])
        out = router._invoke_module(ModuleType.WORLD_MODEL, self.brain.world_model, "p", {})
        self.assertEqual(out["vetoed_actions"], [])
        self.assertEqual(out["best_future_index"], 0)

    def test_b7_router_uses_the_planner_vetoes(self):
        from core.constitutional_brain import ModuleType
        planner = self.brain.router.modules[ModuleType.MCTS_PLANNER]
        saved = planner.vetoes
        planner.vetoes = {"monitor": "só neste teste"}
        try:
            out = self.brain.router._invoke_module(ModuleType.WORLD_MODEL, self.brain.world_model, "p",
                                                   {"possible_actions": [["MONITOR"], ["IGNORE_ALERT"]]})
        finally:
            planner.vetoes = saved
        self.assertEqual(out["best_future_index"], 1)
        self.assertEqual(out["vetoed_actions"], [{"action": "MONITOR", "reason": "só neste teste"}])


class D1DeclaredSuccessTests(unittest.TestCase):
    def test_declared_secure_action_does_not_reach_full_success(self):
        for seed in SEEDS_D:
            for actions in (["AAA_DECL", "ROLLBACK"], ["AAA_DECL"]):
                with self.subTest(seed=seed, actions=actions):
                    r = _plan(seed, actions=actions, causal=_declared())
                    self.assertEqual(r.best_action_sequence, ["AAA_DECL"])
                    self.assertEqual(r.expected_reward, 1.0)
                    self.assertEqual(r.success_probability, 0.9)
                    self.assertLess(r.success_probability, 1.0)

    def test_table_actions_keep_full_success(self):
        for seed in SEEDS_D:
            with self.subTest(seed=seed):
                r = _plan(seed, actions=["APPLY_PATCH", "MONITOR"], causal=_declared())
                self.assertEqual(r.best_action_sequence, ["APPLY_PATCH"])
                self.assertEqual(r.success_probability, 1.0)

    def test_unmodeled_still_zero_and_beats_the_cap(self):
        for seed in SEEDS_D[:20]:
            with self.subTest(seed=seed):
                self.assertEqual(_plan(seed, actions=["ZZ_X"], causal=_declared()).success_probability, 0.0)
        planner = RealMCTSPlanner(world_model=_wm(_declared()), max_depth=3)
        planner._best_found = (9.0, ["AAA_DECL", "ZZ_unmodeled"])
        r = planner._run_plan(GOAL, planner._get_initial_state(GOAL), ["AAA_DECL"], 5, 0.0)
        self.assertEqual(r.success_probability, 0.0)

    def test_declared_action_after_a_table_action_is_capped(self):
        planner = RealMCTSPlanner(world_model=_wm(_declared()), max_depth=3)
        planner._best_found = (9.0, ["APPLY_PATCH", "AAA_DECL"])
        r = planner._run_plan(GOAL, planner._get_initial_state(GOAL), ["AAA_DECL"], 5, 0.0)
        self.assertEqual(r.expected_reward, 1.0)
        self.assertEqual(r.success_probability, 0.9)

    def test_cap_only_lowers(self):
        for seed in SEEDS_D:
            with self.subTest(seed=seed):
                r = _plan(seed, actions=["AAA_DECL", "ROLLBACK", "MONITOR"], causal=_declared())
                self.assertLessEqual(r.success_probability, min(1.0, 1.1 * r.expected_reward))
                self.assertLessEqual(r.success_probability, 0.9)

    def test_models_without_declared_knowledge_are_unchanged(self):
        class Double:
            current_state = type("S", (), {"state": {"system_status": "vulnerable"}})()

            @staticmethod
            def simulate(steps):
                return type("R", (), {"verified_status": "secure", "warnings": [], "success": True})()

        for seed in SEEDS_D[:10]:
            with self.subTest(seed=seed):
                r = _plan(seed, actions=["MONITOR"], model=False)
                self.assertAlmostEqual(r.success_probability, min(1.0, 1.1 * r.expected_reward), delta=0.0002)
                random.seed(seed)
                r = RealMCTSPlanner(world_model=Double(), max_depth=3).plan(GOAL, available_actions=["MONITOR"],
                                                                            max_iterations=50)
                self.assertEqual(r.success_probability, 1.0)
                random.seed(seed)  # fora da tabela e sem is_modeled: decide o aviso do simulador, sem teto
                r = RealMCTSPlanner(world_model=Double(), max_depth=3).plan(GOAL, available_actions=["CUSTOM_X"],
                                                                            max_iterations=50)
                self.assertEqual(r.best_action_sequence, ["CUSTOM_X"])
                self.assertEqual(r.success_probability, 1.0)


class N1NormalizedVetoTests(unittest.TestCase):
    SPELLINGS = ["ignore alert", "Ignore-Alert", " ignore_alert ", "IGNORE__ALERT"]

    def test_alternative_spellings_are_vetoed(self):
        for model in (True, False):
            for seed in SEEDS_N:
                with self.subTest(model=model, seed=seed):
                    r = _plan(seed, actions=self.SPELLINGS + ["MONITOR"], model=model)
                    self.assertEqual(r.best_action_sequence and set(r.best_action_sequence), {"MONITOR"})
                    self.assertEqual([v["action"] for v in r.vetoed_actions], self.SPELLINGS)
                    for item in r.vetoed_actions:
                        self.assertEqual(item["reason"], DEFAULT_VETOES["IGNORE_ALERT"])
                    for alt in r.alternative_plans:
                        self.assertNotIn(alt["first_action"], self.SPELLINGS)

    def test_custom_keys_are_normalized_and_original_keys_kept(self):
        planner = RealMCTSPlanner(vetoes={"my action": "x", "MY_ACTION": "y"})
        self.assertEqual(list(planner.vetoes), ["my action", "MY_ACTION"])
        for seed in SEEDS_N[:5]:
            for name in ("MY_ACTION", "my-action"):
                with self.subTest(seed=seed, name=name):
                    r = _plan(seed, actions=[name, "MONITOR"], planner=RealMCTSPlanner(vetoes={"my action": "x", "MY_ACTION": "y"}))
                    self.assertEqual(r.vetoed_actions, [{"action": name, "reason": "x"}])
                    self.assertEqual(set(r.best_action_sequence), {"MONITOR"})

    def test_invalid_maps(self):
        for bad in ({1: 2}, {"A": 2}, {1: "x"}, {None: None}):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                RealMCTSPlanner(vetoes=bad)
        for bad in ({"  ": "x"}, {"--": "x"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RealMCTSPlanner(vetoes=bad)
        self.assertEqual(RealMCTSPlanner(vetoes={}).vetoes, {})
        self.assertEqual(RealMCTSPlanner(vetoes=None).vetoes, dict(DEFAULT_VETOES))


class N3CopyAndDocsTests(unittest.TestCase):
    def test_replace_loses_vetoed_actions_but_copies_keep_them(self):
        for seed in SEEDS_N:
            with self.subTest(seed=seed):
                r = _plan(seed, actions=["IGNORE_ALERT", "MONITOR"])
                want = [{"action": "IGNORE_ALERT", "reason": DEFAULT_VETOES["IGNORE_ALERT"]}]
                self.assertEqual(r.vetoed_actions, want)
                self.assertEqual(dataclasses.replace(r).vetoed_actions, ())
                self.assertEqual(copy.copy(r).vetoed_actions, want)
                self.assertEqual(copy.deepcopy(r).vetoed_actions, want)
                self.assertEqual(pickle.loads(pickle.dumps(r)).vetoed_actions, want)
                self.assertNotIn("vetoed_actions", dataclasses.asdict(r))

    def test_alternatives_are_deterministic_only_on_ties(self):
        names = ["Z_ACT", "A_ACT", "M_ACT"]
        for k, seed in enumerate(SEEDS_N[:6]):
            order = list(names)
            random.Random(seed).shuffle(order)
            planner = RealMCTSPlanner(max_depth=3)
            root = MCTSNode(state={}, action_taken=None, parent=None)
            for name in order:
                root.children.append(MCTSNode(state={}, action_taken=name, parent=root, visits=2, total_reward=1.0))
            with self.subTest(k=k, case="tie"):
                self.assertEqual([a["first_action"] for a in planner._extract_alternatives(root, top_n=3)], sorted(names))
            root = MCTSNode(state={}, action_taken=None, parent=None)
            for rank, name in enumerate(order):
                root.children.append(MCTSNode(state={}, action_taken=name, parent=root, visits=2, total_reward=1.0 + rank))
            with self.subTest(k=k, case="no tie"):
                got = [a["first_action"] for a in planner._extract_alternatives(root, top_n=3)]
                self.assertEqual(got, list(reversed(order)))

    def test_docs_state_the_limits(self):
        self.assertIn("replace", PlanResult.__doc__)
        self.assertIn("vetoed_actions", PlanResult.__doc__)
        self.assertIn("só no empate", RealMCTSPlanner._extract_alternatives.__doc__)


if __name__ == "__main__":
    unittest.main()
