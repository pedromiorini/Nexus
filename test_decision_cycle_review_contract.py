"""DC11–DC15: revisão do Laird na PR #156 (B1, B2, vocabulário fechado, rejected imutável, docs).

Pré-registrado em docs/cognitivo/AUDITORIA_2026-10-22.md (seção «Errata E-DC5 e revisão do Laird»).
Seeds random.Random(20261401 + k), k = 0..19. Oráculo independente do código (fecho transitivo).
Os testes `test_laird_red*` portam os REDs da revisão (/workspace/probes_pr156/test_red_pr156.py).
"""
import importlib
import random
import unittest

from core.cognitivo import mcts_planner
from test_decision_cycle_contract import POOL, SEEDS, Rec, build_case, norm, spell

VARIANTS = ["SHUTDOWN_SERVICE", "shutdown service", "Shutdown-Service", " shutdown_service ", "SHUTDOWN__SERVICE",
            "shutdown\u00a0service", "shutdown_service\u200b", "shutdown\u200b_service",
            "ＳＨＵＴＤＯＷＮ_ＳＥＲＶＩＣＥ", "shutdown.service", "shutdown/service", "shutdown_service()",
            "shutdown\u2011service", "shutdown\u2013service", "shutdown_service\x00", "shutdownservice",
            "shutdown_services", "SHUTDOWN_SERVICE;", "ignore_alert\ufeff"]
NOISE = ["\u200b", ".", "()", "s", "\x00", "\u2011"]


def dc():
    return importlib.import_module("core.cognitivo.decision_cycle")


def default_pairs():
    return [(norm(k), v) for k, v in mcts_planner.DEFAULT_VETOES.items()]


def oracle(proposals, reject_pairs, better, allowed=None):
    """(selected, kind, candidates, rejected); rejected usa 'VOCAB' como motivo de vocabulário."""
    first = {}
    for text in proposals:
        first.setdefault(norm(text), text)
    if not first:
        return None, "NO_CHANGE", (), {}
    rej = {}
    for key, reason in reject_pairs:
        rej.setdefault(norm(key), reason)
    blocked = {}
    for key in first:
        if key in rej:
            blocked[key] = rej[key]
        elif allowed is not None and key not in {norm(a) for a in allowed}:
            blocked[key] = "VOCAB"
    rejected = {first[k]: blocked[k] for k in first if k in blocked}
    remaining = [k for k in first if k not in blocked]
    if not remaining:
        return None, "CONSTRAINT_FAILURE", tuple(first.values()), rejected
    edges = {(norm(a), norm(b)) for a, b in better if norm(a) in remaining and norm(b) in remaining}
    reach = {(a, b): (a, b) in edges for a in remaining for b in remaining}
    for mid in remaining:
        for a in remaining:
            for b in remaining:
                if reach[(a, mid)] and reach[(mid, b)]:
                    reach[(a, b)] = True
    if any(reach[(a, a)] for a in remaining):
        return None, "CONFLICT", tuple(first[k] for k in remaining), rejected
    free = [k for k in remaining if not any((a, k) in edges for a in remaining)]
    if len(free) == 1:
        return first[free[0]], None, (first[free[0]],), rejected
    return None, "TIE", tuple(first[k] for k in free), rejected


def view(rejected):
    return {k: ("VOCAB" if "vocabul" in v else v) for k, v in rejected.items()}


class DC11AdditiveRejects(unittest.TestCase):
    def test_laird_red1_default_veto_survives_non_empty_rejects(self):
        m = dc()
        d = m.select_operator(["SHUTDOWN_SERVICE"], rejects={"OUTRO": "motivo"})
        self.assertIsNone(d.selected)
        self.assertEqual(d.impasse.kind, "CONSTRAINT_FAILURE")

    def test_none_and_empty_rejects_are_equivalent(self):
        m = dc()
        a = m.select_operator(["SHUTDOWN_SERVICE", "MONITOR"], rejects=None)
        b = m.select_operator(["SHUTDOWN_SERVICE", "MONITOR"], rejects={})
        self.assertEqual(a, b)
        self.assertEqual(a.selected, "MONITOR")

    def test_seeded_oracle_with_additive_default(self):
        m = dc()
        for k in SEEDS:
            with self.subTest(k=k):
                _, proposals, rejects, better = build_case(k)
                pairs = default_pairs() + [(key, reason) for key, reason in rejects.items()]
                sel, kind, cands, rejected = oracle(proposals, pairs, better)
                d = m.select_operator(proposals, rejects=rejects, better=better)
                self.assertEqual(d.selected, sel)
                self.assertEqual(d.impasse.kind if d.impasse else None, kind)
                self.assertEqual(d.candidates, cands)
                self.assertEqual(dict(d.rejected), rejected)

    def test_use_default_rejects_false_disables_only_the_default(self):
        m = dc()
        d = m.select_operator(["SHUTDOWN_SERVICE"], rejects={"OUTRO": "x"}, use_default_rejects=False)
        self.assertEqual(d.selected, "SHUTDOWN_SERVICE")
        d = m.select_operator(["SHUTDOWN_SERVICE", "OUTRO"], rejects={"outro": "x"}, use_default_rejects=False)
        self.assertEqual(d.selected, "SHUTDOWN_SERVICE")
        self.assertEqual(dict(d.rejected), {"OUTRO": "x"})
        self.assertEqual(m.select_operator(["SHUTDOWN_SERVICE"], use_default_rejects=False).selected,
                         "SHUTDOWN_SERVICE")

    def test_use_default_rejects_must_be_bool(self):
        m = dc()
        for bad in (0, 1, None, "no", [], 1.0):
            with self.subTest(bad=bad):
                with self.assertRaises(m.DecisionInputError) as ctx:
                    m.select_operator(["a"], use_default_rejects=bad)
                self.assertEqual(ctx.exception.code, "DC_BAD_INPUT")
                with self.assertRaises(m.DecisionInputError) as ctx:
                    m.run_decision_cycle({}, [lambda s: ["a"]], use_default_rejects=bad)
                self.assertEqual(ctx.exception.code, "DC_BAD_INPUT")

    def test_default_reason_wins_on_key_collision(self):
        m = dc()
        d = m.select_operator(["SHUTDOWN_SERVICE"], rejects={"shutdown service": "meu motivo"})
        self.assertEqual(d.rejected["SHUTDOWN_SERVICE"], mcts_planner.DEFAULT_VETOES["SHUTDOWN_SERVICE"])

    def test_run_cycle_honours_flag_and_additive_default(self):
        m = dc()
        rec = Rec()
        d = m.run_decision_cycle({}, [lambda s: ["SHUTDOWN_SERVICE"]], apply=rec, rejects={"OUTRO": "x"})
        self.assertEqual(d.impasse.kind, "CONSTRAINT_FAILURE")
        self.assertEqual(rec.calls, [])
        d = m.run_decision_cycle({}, [lambda s: ["SHUTDOWN_SERVICE"]], apply=rec, rejects={"OUTRO": "x"},
                                 use_default_rejects=False)
        self.assertTrue(d.applied)
        self.assertEqual(len(rec.calls), 1)

    def test_default_without_rejects_is_unchanged(self):
        m = dc()
        self.assertEqual(m.select_operator(["SHUTDOWN_SERVICE", "MONITOR"]).selected, "MONITOR")
        self.assertEqual(m.select_operator(["Ignore alert"]).impasse.kind, "CONSTRAINT_FAILURE")

    def test_invalid_rejects_still_raise_with_flag(self):
        m = dc()
        with self.assertRaises(m.DecisionInputError) as ctx:
            m.select_operator(["a"], rejects=5, use_default_rejects=False)
        self.assertEqual(ctx.exception.code, "DC_BAD_REJECTS")


class DC12Allowed(unittest.TestCase):
    def test_laird_red3_operator_outside_vocabulary_is_rejected(self):
        m = dc()
        d = m.select_operator(["shutdown_service\u200b"], allowed=["MONITOR", "SHUTDOWN_SERVICE"])
        self.assertIsNone(d.selected)
        self.assertEqual(m.select_operator(["MONITOR"], allowed=["MONITOR", "SHUTDOWN_SERVICE"]).selected,
                         "MONITOR")

    def test_s3a_variants_fail_closed_with_closed_vocabulary(self):
        m = dc()
        allowed = ["MONITOR", "SHUTDOWN_SERVICE"]
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                d = m.select_operator([variant], allowed=allowed, use_default_rejects=False)
                if norm(variant) in allowed:
                    self.assertEqual(d.selected, variant)
                else:
                    self.assertIsNone(d.selected)
                    self.assertEqual(d.impasse.kind, "CONSTRAINT_FAILURE")
                    self.assertIn("vocabul", d.rejected[variant])
                d = m.select_operator([variant], allowed=allowed)  # com o veto padrão: nenhuma variante passa
                self.assertIsNone(d.selected)

    def test_none_means_no_restriction(self):
        m = dc()
        self.assertEqual(m.select_operator(["QUALQUER_COISA"], allowed=None).selected, "QUALQUER_COISA")

    def test_seeded_oracle_with_vocabulary(self):
        m = dc()
        for k in SEEDS:
            with self.subTest(k=k):
                rng, proposals, rejects, better = build_case(k)
                allowed = [spell(o, rng) for o in rng.sample(POOL, rng.randint(2, 6))]
                proposals = proposals + [rng.choice(POOL) + rng.choice(NOISE) for _ in range(rng.randint(0, 3))]
                rng.shuffle(proposals)
                sel, kind, cands, rejected = oracle(proposals, list(rejects.items()), better, allowed)
                d = m.select_operator(proposals, rejects=rejects, better=better, allowed=allowed,
                                      use_default_rejects=False)
                self.assertEqual(d.selected, sel)
                self.assertEqual(d.impasse.kind if d.impasse else None, kind)
                self.assertEqual(d.candidates, cands)
                self.assertEqual(view(d.rejected), rejected)

    def test_vocabulary_never_rehabilitates_a_vetoed_operator(self):
        m = dc()
        d = m.select_operator(["SHUTDOWN_SERVICE"], allowed=["SHUTDOWN_SERVICE"])
        self.assertIsNone(d.selected)
        self.assertEqual(d.rejected["SHUTDOWN_SERVICE"], mcts_planner.DEFAULT_VETOES["SHUTDOWN_SERVICE"])

    def test_reject_reason_beats_vocabulary_reason(self):
        m = dc()
        d = m.select_operator(["X1"], rejects={"x1": "motivo-x"}, allowed=["Y1"])
        self.assertEqual(dict(d.rejected), {"X1": "motivo-x"})

    def test_preferences_over_out_of_vocabulary_operators_are_ignored(self):
        m = dc()
        d = m.select_operator(["A1", "B1"], allowed=["A1"], better=[("B1", "A1"), ("A1", "B1")],
                              use_default_rejects=False)
        self.assertEqual(d.selected, "A1")

    def test_all_outside_vocabulary_is_constraint_failure_with_candidates(self):
        m = dc()
        d = m.select_operator(["A1", "b 1"], allowed=["Z1"], use_default_rejects=False)
        self.assertEqual(d.impasse.kind, "CONSTRAINT_FAILURE")
        self.assertEqual(d.candidates, ("A1", "b 1"))

    def test_invalid_allowed_has_its_own_code(self):
        m = dc()
        for bad in ([], (), "ab", {"a"}, {"a": 1}, 5, True, [1], [""], ["-"], [None], [True], iter(["a"]),
                    ["a", b"b"]):
            with self.subTest(bad=bad):
                with self.assertRaises(m.DecisionInputError) as ctx:
                    m.select_operator(["a"], allowed=bad)
                self.assertEqual(ctx.exception.code, "DC_BAD_ALLOWED")
                self.assertTrue(str(ctx.exception).startswith("DC_BAD_ALLOWED"))

    def test_run_cycle_validates_allowed_before_any_effect(self):
        m = dc()
        calls = []
        rec = Rec()
        with self.assertRaises(m.DecisionInputError) as ctx:
            m.run_decision_cycle({}, [lambda s: calls.append(1) or ["a"]], apply=rec, allowed=[])
        self.assertEqual(ctx.exception.code, "DC_BAD_ALLOWED")
        self.assertEqual((calls, rec.calls), ([], []))

    def test_run_cycle_applies_vocabulary(self):
        m = dc()
        rec = Rec()
        d = m.run_decision_cycle({}, [lambda s: ["MONITOR\u200b"]], apply=rec, allowed=["MONITOR"])
        self.assertIsNone(d.selected)
        self.assertEqual(rec.calls, [])
        d = m.run_decision_cycle({}, [lambda s: ["monitor"]], apply=rec, allowed=["MONITOR"])
        self.assertTrue(d.applied)

    def test_hostile_str_in_allowed_is_never_called(self):
        m = dc()

        class Hostile(str):
            def strip(self, *args):
                raise RuntimeError("strip")

            def __str__(self):
                raise RuntimeError("str")

        self.assertEqual(m.select_operator(["go"], allowed=[Hostile("go")]).selected, "go")


class DC13OperatorNoChange(unittest.TestCase):
    def test_laird_red2_apply_failure_is_an_operator_no_change_impasse(self):
        m = dc()

        def boom(op, st):
            raise RuntimeError("x")

        d = m.run_decision_cycle({}, [lambda s: ["MONITOR"]], apply=boom)
        self.assertIs(d.applied, False)
        self.assertEqual(d.selected, "MONITOR")
        self.assertEqual(d.impasse.kind, "OPERATOR_NO_CHANGE")
        self.assertIsNotNone(d.subgoal)

    def test_apply_failure_contract_over_seeds(self):
        m = dc()
        excs = [RuntimeError, ValueError, KeyError, OSError]
        for k in SEEDS:
            with self.subTest(k=k):
                rng = random.Random(20261401 + k)
                op = rng.choice(POOL[:6])
                proposals = [spell(op, rng) for _ in range(rng.randint(1, 3))] + [rng.choice(["SHUTDOWN_SERVICE"])]
                rng.shuffle(proposals)
                exc = rng.choice(excs)
                rec = Rec(exc=exc("segredo-xyz"))
                state = {"k": k}
                d = m.run_decision_cycle(state, [lambda s: proposals], apply=rec)
                self.assertEqual(norm(d.selected), op)
                self.assertEqual(d.impasse.kind, "OPERATOR_NO_CHANGE")
                self.assertEqual(d.impasse.code, "DC_IMPASSE_OPERATOR_NO_CHANGE")
                self.assertEqual(d.impasse.candidates, (d.selected,))
                self.assertEqual(d.subgoal.kind, "OPERATOR_NO_CHANGE")
                self.assertEqual(d.subgoal.goal, "resolve:OPERATOR_NO_CHANGE")
                self.assertEqual(d.subgoal.candidates, (d.selected,))
                self.assertEqual(d.candidates, (d.selected,))
                self.assertFalse(d.applied)
                self.assertIsNone(d.result)
                self.assertEqual(d.error, f"DC_APPLY_FAILED: {exc.__name__}")
                self.assertNotIn("segredo-xyz", d.error)
                self.assertEqual(d.phases, ("propose", "select", "apply"))
                self.assertEqual(len(rec.calls), 1)
                self.assertIs(rec.calls[0][1], state)
                self.assertEqual(list(d.rejected), ["SHUTDOWN_SERVICE"])

    def test_success_and_no_apply_have_no_impasse(self):
        m = dc()
        d = m.run_decision_cycle({}, [lambda s: ["MONITOR"]], apply=Rec())
        self.assertTrue(d.applied)
        self.assertIsNone(d.impasse)
        self.assertIsNone(d.subgoal)
        d = m.run_decision_cycle({}, [lambda s: ["MONITOR"]])
        self.assertIsNone(d.impasse)
        self.assertIsNone(d.subgoal)
        self.assertIsNone(d.error)

    def test_other_impasses_still_select_nothing_and_never_apply(self):
        m = dc()
        for proposals, kwargs, kind in [
            ([], {}, "NO_CHANGE"), (["A1", "B1"], {}, "TIE"),
            (["A1", "B1"], {"better": [("A1", "B1"), ("B1", "A1")]}, "CONFLICT"),
            (["SHUTDOWN_SERVICE"], {}, "CONSTRAINT_FAILURE"),
        ]:
            with self.subTest(kind=kind):
                rec = Rec()
                d = m.run_decision_cycle({}, [lambda s: proposals], apply=rec, **kwargs)
                self.assertEqual(d.impasse.kind, kind)
                self.assertIsNone(d.selected)
                self.assertEqual(rec.calls, [])
                self.assertEqual(d.phases, ("propose", "select"))

    def test_base_exception_in_apply_propagates(self):
        m = dc()
        with self.assertRaises(KeyboardInterrupt):
            m.run_decision_cycle({}, [lambda s: ["MONITOR"]], apply=Rec(exc=KeyboardInterrupt()))


class DC14RejectedImmutable(unittest.TestCase):
    def test_rejected_is_a_read_only_mapping(self):
        from collections.abc import Mapping
        m = dc()
        cases = [m.select_operator(["SHUTDOWN_SERVICE", "MONITOR"]), m.select_operator(["MONITOR"]),
                 m.select_operator([]), m.run_decision_cycle({}, [lambda s: ["A1", "B1"]])]
        for d in cases:
            with self.subTest(selected=d.selected, rejected=dict(d.rejected)):
                self.assertIsInstance(d.rejected, Mapping)
                before = dict(d.rejected)
                for mutate in (lambda: d.rejected.clear(), lambda: d.rejected.__setitem__("X", "y"),
                               lambda: d.rejected.pop("SHUTDOWN_SERVICE", None), lambda: d.rejected.update(X="y")):
                    with self.assertRaises((TypeError, AttributeError)):
                        mutate()
                self.assertEqual(dict(d.rejected), before)

    def test_rejected_does_not_alias_the_callers_rejects(self):
        m = dc()
        rejects = {"X1": "motivo"}
        d = m.select_operator(["X1", "Y1"], rejects=rejects)
        rejects["Y1"] = "depois"
        rejects["X1"] = "mudou"
        self.assertEqual(dict(d.rejected), {"X1": "motivo"})

    def test_equality_with_plain_dict_still_works(self):
        m = dc()
        d = m.select_operator(["SHUTDOWN_SERVICE", "MONITOR"])
        self.assertEqual(d.rejected, {"SHUTDOWN_SERVICE": mcts_planner.DEFAULT_VETOES["SHUTDOWN_SERVICE"]})
        self.assertEqual(d, m.select_operator(["SHUTDOWN_SERVICE", "MONITOR"]))


class DC15Docs(unittest.TestCase):
    def test_module_docstring_documents_the_review_points(self):
        doc = dc().__doc__
        for token in ("Soar", "mais estrito", "state no-change", "DecisionInputError", "bloqueio", "rejectors",
                      "required", "OPERATOR_NO_CHANGE", "use_default_rejects", "allowed", "aditivo"):
            with self.subTest(token=token):
                self.assertIn(token, doc)

    def test_select_operator_docstring_mentions_new_parameters(self):
        for fn in (dc().select_operator, dc().run_decision_cycle):
            with self.subTest(fn=fn.__name__):
                for token in ("allowed", "use_default_rejects"):
                    self.assertIn(token, fn.__doc__)


if __name__ == "__main__":
    unittest.main()
