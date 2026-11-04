"""DC1–DC10: ciclo de decisão do roteador (propor → rejeitar → selecionar → aplicar; impasse).

Pré-registrado em docs/cognitivo/AUDITORIA_2026-10-22.md (seção «Ciclo de decisão do roteador»).
Seeds random.Random(20261401 + k), k = 0..19. O oráculo abaixo é independente do código de
produção (fecho transitivo por Floyd–Warshall; o módulo usa outro algoritmo).
"""
import ast
import dataclasses
import importlib
import random
import re
import unittest
from pathlib import Path

SEEDS = range(20)
POOL = ["APPLY_PATCH", "MONITOR", "ISOLATE_HOST", "RESTART_SERVICE", "ROLLBACK", "SCAN", "IGNORE_ALERT",
        "SHUTDOWN_SERVICE"]
ROOT = Path(__file__).resolve().parent


def dc():
    return importlib.import_module("core.cognitivo.decision_cycle")


def norm(text):
    return re.sub(r"_+", "_", re.sub(r"[\s\-]+", "_", text.strip())).strip("_").upper()


def spell(op, rng):
    return rng.choice([op, op.lower(), op.replace("_", " ").title(), " " + op.replace("_", "-") + "  "])


def oracle(proposals, rejects, better):
    """(selected, kind, candidates, rejected) esperados; kind None = seleção."""
    first = {}
    for text in proposals:
        first.setdefault(norm(text), text)
    if not first:
        return None, "NO_CHANGE", (), {}
    rej = {}
    for key, reason in rejects.items():
        rej.setdefault(norm(key), reason)
    rejected = {first[k]: rej[k] for k in first if k in rej}
    remaining = [k for k in first if k not in rej]
    if not remaining:
        return None, "CONSTRAINT_FAILURE", tuple(first[k] for k in first), rejected
    edges = {(norm(a), norm(b)) for a, b in better if norm(a) in remaining and norm(b) in remaining}
    reach = {(a, b): (a, b) in edges for a in remaining for b in remaining}
    for m in remaining:
        for a in remaining:
            for b in remaining:
                if reach[(a, m)] and reach[(m, b)]:
                    reach[(a, b)] = True
    if any(reach[(a, a)] for a in remaining):
        return None, "CONFLICT", tuple(first[k] for k in remaining), rejected
    free = [k for k in remaining if not any((a, k) in edges for a in remaining)]
    if len(free) == 1:
        return first[free[0]], None, (first[free[0]],), rejected
    return None, "TIE", tuple(first[k] for k in free), rejected


def build_case(k):
    rng = random.Random(20261401 + k)
    ops = rng.sample(POOL, rng.randint(2, 6))
    proposals = [spell(o, rng) for o in ops]
    if rng.random() < 0.5:
        proposals.append(spell(rng.choice(ops), rng))  # repetição com outra grafia
    rng.shuffle(proposals)
    rejects = {spell(o, rng): f"motivo-{rng.randint(0, 99)}" for o in rng.sample(POOL, rng.randint(0, 3))}
    order = rng.sample(POOL, len(POOL))
    better = []
    for _ in range(rng.randint(0, 8)):
        x, y = rng.sample(order, 2)
        if order.index(x) > order.index(y):
            x, y = y, x
        better.append((spell(x, rng), spell(y, rng)))
    if k % 2:
        better.append((spell(order[-1], rng), spell(order[0], rng)))  # tende a fechar ciclo
    return rng, proposals, rejects, better


class Rec:
    def __init__(self, result="ok", exc=None):
        self.calls = []
        self.result = result
        self.exc = exc

    def __call__(self, op, state):
        self.calls.append((op, state))
        if self.exc:
            raise self.exc
        return self.result


class DC1Oracle(unittest.TestCase):
    def test_select_operator_matches_oracle(self):
        m = dc()
        kinds = set()
        for k in SEEDS:
            with self.subTest(k=k):
                _, proposals, rejects, better = build_case(k)
                sel, kind, cands, rejected = oracle(proposals, rejects, better)
                decision = m.select_operator(proposals, rejects=rejects, better=better, use_default_rejects=False)
                kinds.add(kind)
                self.assertEqual(decision.selected, sel)
                self.assertEqual(decision.impasse.kind if decision.impasse else None, kind)
                self.assertEqual(decision.candidates, cands)
                self.assertEqual(decision.rejected, rejected)
        self.assertGreaterEqual(len(kinds), 3, kinds)  # a semente cobre vários desfechos

    def test_run_cycle_matches_oracle_and_apply_once(self):
        m = dc()
        for k in SEEDS:
            with self.subTest(k=k):
                _, proposals, rejects, better = build_case(k)
                sel, kind, cands, rejected = oracle(proposals, rejects, better)
                half = len(proposals) // 2
                state = {"k": k}
                rec = Rec(result=("r", k))
                decision = m.run_decision_cycle(
                    state, [lambda s: proposals[:half], lambda s: tuple(proposals[half:])],
                    apply=rec, rejects=rejects, better=better, use_default_rejects=False)
                self.assertEqual(decision.selected, sel)
                self.assertEqual(decision.impasse.kind if decision.impasse else None, kind)
                self.assertEqual(decision.candidates, cands)
                self.assertEqual(decision.rejected, rejected)
                if sel is None:
                    self.assertEqual(rec.calls, [])
                    self.assertFalse(decision.applied)
                    self.assertNotIn("apply", decision.phases)
                else:
                    self.assertEqual(len(rec.calls), 1)
                    self.assertEqual(rec.calls[0][0], sel)
                    self.assertIs(rec.calls[0][1], state)
                    self.assertTrue(decision.applied)
                    self.assertEqual(decision.result, ("r", k))


class DC2NoChange(unittest.TestCase):
    def check(self, decision, rec):
        self.assertIsNone(decision.selected)
        self.assertEqual(decision.impasse.kind, "NO_CHANGE")
        self.assertEqual(decision.impasse.code, "DC_IMPASSE_NO_CHANGE")
        self.assertEqual(decision.candidates, ())
        self.assertEqual(decision.subgoal.goal, "resolve:NO_CHANGE")
        self.assertEqual(decision.subgoal.kind, "NO_CHANGE")
        self.assertFalse(decision.applied)
        self.assertEqual(rec.calls, [])

    def test_empty_inputs_are_an_impasse_not_an_error(self):
        m = dc()
        for k in SEEDS:
            with self.subTest(k=k):
                rec = Rec()
                variants = [
                    m.select_operator([]), m.select_operator(()),
                    m.run_decision_cycle({}, [], apply=rec), m.run_decision_cycle({}, (), apply=rec),
                    m.run_decision_cycle({}, [lambda s: [], lambda s: ()], apply=rec),
                ]
                for decision in variants:
                    self.check(decision, rec)

    def test_none_is_an_input_error_with_a_different_code(self):
        m = dc()
        for call in (lambda: m.select_operator(None), lambda: m.run_decision_cycle({}, None)):
            with self.assertRaises(m.DecisionInputError) as ctx:
                call()
            self.assertEqual(ctx.exception.code, "DC_NO_INPUT")
            self.assertIsInstance(ctx.exception, ValueError)


class DC3Tie(unittest.TestCase):
    def test_tie_keeps_proposal_order_and_selects_nothing(self):
        m = dc()
        for k in SEEDS:
            with self.subTest(k=k):
                rng = random.Random(20261401 + k)
                ops = rng.sample(POOL[:6], rng.randint(2, 4))
                proposals = [spell(o, rng) for o in ops]
                decision = m.select_operator(proposals, rejects={})
                self.assertIsNone(decision.selected)
                self.assertEqual(decision.impasse.kind, "TIE")
                self.assertEqual(decision.impasse.code, "DC_IMPASSE_TIE")
                self.assertEqual(decision.candidates, tuple(proposals))
                self.assertEqual(decision.subgoal.goal, "resolve:TIE")
                self.assertEqual(decision.subgoal.candidates, tuple(proposals))

    def test_duplicate_or_alias_spelling_is_not_a_tie(self):
        m = dc()
        decision = m.select_operator(["a", "A", " a "], rejects={})
        self.assertEqual(decision.selected, "a")
        self.assertIsNone(decision.impasse)
        self.assertIsNone(decision.subgoal)
        self.assertEqual(decision.candidates, ("a",))
        decision = m.select_operator(["Apply-Patch", "APPLY_PATCH", "apply patch"], rejects={})
        self.assertEqual(decision.selected, "Apply-Patch")

    def test_better_resolves_a_tie(self):
        m = dc()
        decision = m.select_operator(["MONITOR", "SCAN", "ROLLBACK"], rejects={},
                                     better=[("scan", "monitor"), ("SCAN", "ROLLBACK")])
        self.assertEqual(decision.selected, "SCAN")


class DC4Conflict(unittest.TestCase):
    def test_cycles_fail_closed(self):
        m = dc()
        cases = [
            (["A1", "B1"], [("a1", "b1"), ("B1", "A1")]),
            (["A1", "B1", "C1"], [("A1", "B1"), ("B1", "C1"), ("C1", "A1")]),
            (["A1", "B1", "C1", "D1"], [("A1", "B1"), ("B1", "A1")]),  # D1 solto não vence
            (["A1", "B1", "C1", "D1"], [("A1", "B1"), ("B1", "C1"), ("C1", "A1"), ("D1", "A1")]),
        ]
        for proposals, better in cases:
            with self.subTest(proposals=proposals, better=better):
                decision = m.select_operator(proposals, rejects={}, better=better)
                self.assertIsNone(decision.selected)
                self.assertEqual(decision.impasse.kind, "CONFLICT")
                self.assertEqual(decision.impasse.code, "DC_IMPASSE_CONFLICT")
                self.assertEqual(decision.candidates, tuple(proposals))
                self.assertEqual(decision.subgoal.goal, "resolve:CONFLICT")

    def test_cycle_through_a_rejected_operator_is_ignored(self):
        m = dc()
        decision = m.select_operator(["A1", "B1", "C1"], rejects={"c1": "não"},
                                     better=[("A1", "B1"), ("B1", "C1"), ("C1", "A1")])
        self.assertEqual(decision.selected, "A1")
        self.assertEqual(decision.rejected, {"C1": "não"})

    def test_preference_about_absent_operator_is_ignored(self):
        m = dc()
        decision = m.select_operator(["A1", "B1"], rejects={}, better=[("ZZ", "A1"), ("B1", "YY")])
        self.assertIsNone(decision.selected)
        self.assertEqual(decision.impasse.kind, "TIE")


class DC5Constraint(unittest.TestCase):
    def test_all_rejected(self):
        m = dc()
        for k in SEEDS:
            with self.subTest(k=k):
                rng = random.Random(20261401 + k)
                ops = rng.sample(POOL, rng.randint(1, 4))
                proposals = [spell(o, rng) for o in ops]
                rejects = {o: f"r{k}-{o}" for o in ops}
                decision = m.select_operator(proposals, rejects=rejects, use_default_rejects=False)
                self.assertIsNone(decision.selected)
                self.assertEqual(decision.impasse.kind, "CONSTRAINT_FAILURE")
                self.assertEqual(decision.impasse.code, "DC_IMPASSE_CONSTRAINT")
                self.assertEqual(decision.candidates, tuple(proposals))
                self.assertEqual(decision.rejected, {p: rejects[norm(p)] for p in proposals})
                self.assertEqual(decision.subgoal.goal, "resolve:CONSTRAINT_FAILURE")

    def test_default_vetoes(self):
        m = dc()
        decision = m.select_operator(["shutdown_service", "Ignore alert"])
        self.assertEqual(decision.impasse.kind, "CONSTRAINT_FAILURE")
        self.assertEqual(set(decision.rejected), {"shutdown_service", "Ignore alert"})
        self.assertTrue(all(decision.rejected.values()))
        decision = m.select_operator(["SHUTDOWN_SERVICE", "MONITOR"])
        self.assertEqual(decision.selected, "MONITOR")
        self.assertIn("SHUTDOWN_SERVICE", decision.rejected)

    def test_explicit_empty_rejects_no_longer_disables_the_default(self):
        # Errata E-DC5: o veto padrão é aditivo; desligar exige use_default_rejects=False.
        m = dc()
        self.assertIsNone(m.select_operator(["SHUTDOWN_SERVICE"], rejects={}).selected)
        self.assertEqual(
            m.select_operator(["SHUTDOWN_SERVICE"], rejects={}, use_default_rejects=False).selected,
            "SHUTDOWN_SERVICE")

    def test_better_does_not_rehabilitate_a_vetoed_operator(self):
        m = dc()
        decision = m.select_operator(["SHUTDOWN_SERVICE", "MONITOR"], better=[("SHUTDOWN_SERVICE", "MONITOR")])
        self.assertEqual(decision.selected, "MONITOR")

    def test_default_vetoes_are_frozen_at_import(self):
        m = dc()
        from core.cognitivo import mcts_planner
        saved = dict(mcts_planner.DEFAULT_VETOES)
        try:
            mcts_planner.DEFAULT_VETOES.clear()
            decision = m.select_operator(["SHUTDOWN_SERVICE", "MONITOR"])
            self.assertEqual(decision.selected, "MONITOR")
        finally:
            mcts_planner.DEFAULT_VETOES.update(saved)

    def test_duplicate_reject_keys_first_wins(self):
        m = dc()
        decision = m.select_operator(["A1"], rejects={"a1": "primeiro", "A 1": "x", "A1": "segundo"})
        self.assertEqual(decision.rejected, {"A1": "primeiro"})


class DC6InvalidInput(unittest.TestCase):
    def assertCode(self, code, call):
        with self.assertRaises(dc().DecisionInputError) as ctx:
            call()
        self.assertEqual(ctx.exception.code, code)
        self.assertTrue(str(ctx.exception).startswith(code))

    def test_bad_proposals_container(self):
        m = dc()
        for bad in ("abc", {"a"}, {"a": 1}, 5, True, b"ab", iter(["a"]), range(2)):
            with self.subTest(bad=bad):
                self.assertCode("DC_BAD_INPUT", lambda: m.select_operator(bad))

    def test_bad_proposal_elements(self):
        m = dc()
        for bad in (1, True, None, "", "  ", "-", "_-_", b"a", ["a"]):
            with self.subTest(bad=bad):
                self.assertCode("DC_BAD_PROPOSAL", lambda: m.select_operator(["MONITOR", bad]))

    def test_bad_better(self):
        m = dc()
        for bad in ([("a",)], [("a", "b", "c")], [("a", 1)], [(1, "a")], [("a", "A")], [("a", " a-")],
                    "ab", {"a": "b"}, 5, [["a"]], [None], [("a", True)], [("", "b")]):
            with self.subTest(bad=bad):
                self.assertCode("DC_BAD_PREFERENCE", lambda: m.select_operator(["a", "b"], better=bad))

    def test_bad_rejects(self):
        m = dc()
        for bad in ([("a", "b")], "ab", 5, {1: "x"}, {"a": 1}, {"a": None}, {True: "x"}):
            with self.subTest(bad=bad):
                self.assertCode("DC_BAD_REJECTS", lambda: m.select_operator(["a"], rejects=bad))

    def test_bad_proposers_and_apply(self):
        m = dc()
        for bad in ("abc", {"a": 1}, 5, lambda s: [], iter([lambda s: []]), [lambda s: [], 5], [None]):
            with self.subTest(bad=bad):
                self.assertCode("DC_BAD_INPUT", lambda: m.run_decision_cycle({}, bad))
        for bad in (5, "x", True, [1]):
            with self.subTest(apply=bad):
                calls = []
                self.assertCode("DC_BAD_INPUT", lambda: m.run_decision_cycle(
                    {}, [lambda s: calls.append(1) or ["a"]], apply=bad))
                self.assertEqual(calls, [])

    def test_validation_happens_before_any_effect(self):
        m = dc()
        calls = []
        rec = Rec()
        self.assertCode("DC_BAD_REJECTS", lambda: m.run_decision_cycle(
            {}, [lambda s: calls.append(1) or ["a"]], apply=rec, rejects=5))
        self.assertCode("DC_BAD_PREFERENCE", lambda: m.run_decision_cycle(
            {}, [lambda s: calls.append(1) or ["a"]], apply=rec, better="ab"))
        self.assertEqual(calls, [])
        self.assertEqual(rec.calls, [])

    def test_hostile_str_subclass_is_never_called(self):
        m = dc()

        class Hostile(str):
            def strip(self, *args):
                raise RuntimeError("strip")

            def __str__(self):
                raise RuntimeError("str")

            def upper(self):
                raise RuntimeError("upper")

        decision = m.select_operator([Hostile("go")], rejects={Hostile("nope"): Hostile("why")},
                                     better=[(Hostile("go"), Hostile("other"))])
        self.assertEqual(decision.selected, "go")
        self.assertIs(type(decision.selected), str)


class DC7ProposerFailure(unittest.TestCase):
    def check(self, decision, rec, index):
        self.assertIsNone(decision.selected)
        self.assertEqual(decision.impasse.kind, "PROPOSER_FAILURE")
        self.assertEqual(decision.impasse.code, "DC_IMPASSE_PROPOSER")
        self.assertEqual(decision.subgoal.goal, "resolve:PROPOSER_FAILURE")
        self.assertTrue(decision.error.startswith(f"DC_PROPOSER_FAILED: proposer[{index}] "), decision.error)
        self.assertNotIn("segredo-xyz", decision.error)
        self.assertEqual(rec.calls, [])
        self.assertFalse(decision.applied)
        self.assertEqual(decision.candidates, ())

    def test_failures_stop_the_cycle(self):
        m = dc()

        def boom(state):
            raise RuntimeError("segredo-xyz")

        for k, bad in enumerate([boom, lambda s: "abc", lambda s: None, lambda s: {"a"}, lambda s: [1],
                                 lambda s: ["MONITOR", ""], lambda s: [["a"]]]):
            with self.subTest(k=k):
                rec = Rec()
                later = []
                decision = m.run_decision_cycle(
                    {}, [lambda s: [], bad, lambda s: later.append(1) or ["MONITOR"]], apply=rec)
                self.check(decision, rec, 1)
                self.assertEqual(later, [])

    def test_failure_is_not_masked_by_other_proposers(self):
        m = dc()
        rec = Rec()

        def boom(state):
            raise ValueError("segredo-xyz")

        decision = m.run_decision_cycle({}, [boom, lambda s: ["MONITOR"]], apply=rec)
        self.check(decision, rec, 0)
        self.assertEqual(decision.error, "DC_PROPOSER_FAILED: proposer[0] ValueError")

    def test_base_exceptions_propagate(self):
        m = dc()

        def stop(state):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            m.run_decision_cycle({}, [stop])


class DC8Apply(unittest.TestCase):
    def test_success(self):
        m = dc()
        rec = Rec(result={"ok": 1})
        decision = m.run_decision_cycle("estado", [lambda s: ["Monitor"]], apply=rec, rejects={})
        self.assertTrue(decision.applied)
        self.assertEqual(decision.result, {"ok": 1})
        self.assertIsNone(decision.error)
        self.assertEqual(decision.phases, ("propose", "select", "apply"))
        self.assertEqual(rec.calls, [("Monitor", "estado")])

    def test_apply_exception(self):
        m = dc()
        rec = Rec(exc=RuntimeError("segredo-xyz"))
        decision = m.run_decision_cycle({}, [lambda s: ["Monitor"]], apply=rec, rejects={})
        self.assertFalse(decision.applied)
        self.assertEqual(decision.selected, "Monitor")
        self.assertIsNone(decision.result)
        self.assertEqual(decision.error, "DC_APPLY_FAILED: RuntimeError")
        self.assertEqual(decision.phases, ("propose", "select", "apply"))
        self.assertEqual(len(rec.calls), 1)

    def test_apply_none_only_decides(self):
        m = dc()
        decision = m.run_decision_cycle({}, [lambda s: ["Monitor"]], rejects={})
        self.assertEqual(decision.selected, "Monitor")
        self.assertFalse(decision.applied)
        self.assertIsNone(decision.error)
        self.assertEqual(decision.phases, ("propose", "select"))

    def test_impasse_phases_and_no_apply(self):
        m = dc()
        rec = Rec()
        decision = m.run_decision_cycle({}, [lambda s: ["A1", "B1"]], apply=rec, rejects={})
        self.assertEqual(decision.phases, ("propose", "select"))
        self.assertEqual(rec.calls, [])
        self.assertIsNone(decision.error)


class DC9Invariants(unittest.TestCase):
    def test_inputs_are_not_mutated_and_result_is_deterministic(self):
        m = dc()
        import copy
        for k in SEEDS:
            with self.subTest(k=k):
                _, proposals, rejects, better = build_case(k)
                snap = copy.deepcopy((proposals, rejects, better))
                first = m.select_operator(proposals, rejects=rejects, better=better)
                second = m.select_operator(proposals, rejects=rejects, better=better)
                self.assertEqual((proposals, rejects, better), snap)
                self.assertEqual(first, second)

    def test_permutation_invariance(self):
        m = dc()
        for k in SEEDS:
            with self.subTest(k=k):
                rng, proposals, rejects, better = build_case(k)
                base = m.select_operator(proposals, rejects=rejects, better=better)
                shuffled = list(proposals)
                rng.shuffle(shuffled)
                other = m.select_operator(shuffled, rejects=rejects, better=better)
                self.assertEqual(base.impasse.kind if base.impasse else None,
                                 other.impasse.kind if other.impasse else None)
                self.assertEqual({norm(c) for c in base.candidates}, {norm(c) for c in other.candidates})
                if base.selected is not None:
                    self.assertEqual(norm(base.selected), norm(other.selected))

    def test_decision_is_frozen(self):
        m = dc()
        decision = m.select_operator(["MONITOR"], rejects={})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            decision.selected = "X"
        impasse = m.select_operator([], rejects={})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            impasse.impasse.kind = "X"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            impasse.subgoal.goal = "X"

    def test_no_impasse_means_no_subgoal(self):
        m = dc()
        decision = m.select_operator(["MONITOR"], rejects={})
        self.assertIsNone(decision.impasse)
        self.assertIsNone(decision.subgoal)
        self.assertEqual(decision.phases, ("propose", "select"))
        self.assertEqual(decision.selected, "MONITOR")

    def test_state_is_passed_by_reference(self):
        m = dc()
        seen = []
        state = {"x": 1}
        m.run_decision_cycle(state, [lambda s: seen.append(s) or ["MONITOR"]], rejects={})
        self.assertIs(seen[0], state)

    def test_proposer_returning_a_tuple_or_list(self):
        m = dc()
        self.assertEqual(m.run_decision_cycle({}, [lambda s: ("MONITOR",)], rejects={}).selected, "MONITOR")
        self.assertEqual(m.run_decision_cycle({}, [lambda s: ["MONITOR"]], rejects={}).selected, "MONITOR")


class DC10Isolation(unittest.TestCase):
    def test_module_does_not_import_the_monolith(self):
        m = dc()
        tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertFalse([name for name in imported if "constitutional_brain" in name], imported)

    def test_public_surface(self):
        m = dc()
        self.assertEqual(sorted(m.__all__), sorted([
            "Decision", "DecisionInputError", "Impasse", "Subgoal", "run_decision_cycle", "select_operator"]))
        for name in m.__all__:
            self.assertTrue(hasattr(m, name), name)


if __name__ == "__main__":
    unittest.main()
