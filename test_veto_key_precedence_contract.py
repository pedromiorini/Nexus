"""Guardas de precedência e colisão do veto canônico (mutantes de tools/cognitivo_mutation.py).

Complementa test_red_pr156_veto_names.py (C1–C7): fixa a regra «nome exato vence; na colisão de chave canônica
vence o primeiro» nos quatro pontos de consulta e o fail-closed de um veto cujo motivo é ``None`` (planejador).
"""
import contextlib
import io
import unittest

from core.cognitivo import decision_cycle as DC
from core.cognitivo import mcts_planner as MP
from core.cognitivo import scm_counterfactual as S
from core.cognitivo.action_effects import veto_index, veto_key, veto_lookup
from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator

OBS = {"system_status": "compromised", "alert_level": "critical", "security_patch_applied": False,
       "logging_level": "normal"}


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def world():
    return RealWorldModelSimulator(RealCausalReasoning())


class VetoLookupTests(unittest.TestCase):
    def test_exact_name_wins_over_canonical_collision(self):
        vetoes = {"foo": "primeiro", "FOO": "exato"}
        self.assertEqual(veto_index(vetoes), {"FOO": "primeiro"})  # na colisão canônica, o primeiro vence
        self.assertEqual(veto_lookup("FOO", vetoes), (True, "exato"))  # nome exato tem precedência
        self.assertEqual(veto_lookup("foo", vetoes), (True, "primeiro"))
        self.assertEqual(veto_lookup("F O O", vetoes), (False, None))  # outra identidade (FOO != F_O_O)
        self.assertEqual(veto_lookup("f-o-o", vetoes), (False, None))
        self.assertEqual(veto_lookup("Foo()", vetoes), (True, "primeiro"))

    def test_key_that_does_not_canonicalize_stays_exact_only(self):
        vetoes = {"não ascii": "x", "SEM CHAVE ÷": "y", "ok": "z"}
        self.assertEqual(veto_index(vetoes), {"OK": "z"})
        self.assertEqual(veto_lookup("não ascii", vetoes), (True, "x"))  # exato ainda vale
        self.assertEqual(veto_lookup("NÃO ASCII", vetoes), (False, None))
        self.assertEqual(veto_lookup(None, vetoes), (False, None))
        self.assertEqual(veto_lookup(5, vetoes), (False, None))
        self.assertEqual(veto_lookup("", vetoes), (False, None))

    def test_none_reason_is_still_a_veto(self):
        self.assertEqual(veto_lookup("a", {"a": None}), (True, None))
        self.assertEqual(veto_lookup("A ", {"a": None}), (True, None))

    def test_precomputed_index_is_used_and_equals_the_recomputed_one(self):
        vetoes = {"x y": "r"}
        self.assertEqual(veto_lookup("X_Y", vetoes, {"X_Y": "do índice"}), (True, "do índice"))
        self.assertEqual(veto_lookup("X_Y", vetoes, {}), (False, None))
        self.assertEqual(veto_lookup("X_Y", vetoes), (True, "r"))

    def test_veto_key_is_not_fooled_by_internal_underscores_or_edges(self):
        self.assertEqual(veto_key("__a__b__"), "A_B")
        self.assertEqual(veto_key("a - b"), "A_B")
        self.assertIsNone(veto_key("___"))
        self.assertIsNone(veto_key("()"))
        self.assertIsNone(veto_key(";"))
        self.assertEqual(veto_key("a();"), "A")
        self.assertEqual(veto_key("a(;)"), None)
        self.assertEqual(veto_key("a( _ )"), "A")
        self.assertEqual(veto_key("a?!;"), "A")
        self.assertEqual(veto_key("a()()"), None)


class VetoKeyEdgeTests(unittest.TestCase):
    """Bordas do canonicalizador que os mutantes de ``veto_key`` exigem fixadas."""

    def test_each_separator_alone_becomes_underscore(self):
        for sep in [" ", "-", ".", ":", "/", "\\", ",", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014",
                    "\u2015", "\u2212", "\u00b7", "\t", "\n", "\u00a0"]:
            with self.subTest(sep=sep):
                self.assertEqual(veto_key("shutdown" + sep + "service"), "SHUTDOWN_SERVICE")
                self.assertEqual(veto_key("shutdown" + sep * 3 + "service"), "SHUTDOWN_SERVICE")

    def test_non_space_control_characters_are_removed_not_turned_into_separators(self):
        for char in ["\x00", "\x07", "\x1b", "\x7f", "\x9f"]:
            with self.subTest(char=repr(char)):
                self.assertEqual(veto_key("SHUTDOWN_SERVICE" + char), "SHUTDOWN_SERVICE")
                self.assertEqual(veto_key("SHUTDOWN" + char + "_SERVICE"), "SHUTDOWN_SERVICE")

    def test_call_syntax_after_trailing_or_inner_space(self):
        for text in ["shutdown_service() ", "shutdown_service();  ", "shutdown_service ()",
                     "  shutdown_service()!?", "shutdown_service(\t)"]:
            with self.subTest(text=text):
                self.assertEqual(veto_key(text), "SHUTDOWN_SERVICE")

    def test_ascii_is_required_before_upper(self):
        for char in ["\u0131", "\u00df", "\u0406", "\u0405"]:  # ı, ß, І, Ѕ (ı.upper() seria «I», ß.upper() «SS»)
            with self.subTest(char=char):
                self.assertIsNone(veto_key("shutdown_serv" + char + "ce"))
                self.assertIsNone(veto_key(char))
        self.assertEqual(veto_key("\u212a"), "K")  # NFKC leva o kelvin a ASCII antes da checagem
        self.assertEqual(veto_key("shutdown_\u017fervice"), "SHUTDOWN_SERVICE")  # idem ſ (NFKC: «s»)

    def test_call_syntax_with_space_between_parens_and_mark_does_not_canonicalize(self):
        # Limite documentado: «() ;» (espaço entre ``)`` e ``;``) não canonicaliza; o registro o recusa.
        self.assertIsNone(veto_key("shutdown_service ( ) ;"))


class DeclareInterventionMessageTests(unittest.TestCase):
    def test_blank_and_non_canonical_names_are_refused_with_distinct_messages(self):
        from core.cognitivo.causal_world_model import RealCausalReasoning
        model = RealCausalReasoning()
        for blank in ("", "   ", "\t\n"):
            with self.subTest(blank=blank), self.assertRaisesRegex(ValueError, "não vazia"):
                model.declare_intervention(blank, {"alert_level": "low"})
        with self.assertRaisesRegex(ValueError, "canonicalizar"):
            model.declare_intervention("shutdown service (dry run)", {"alert_level": "low"})


class PlannerVetoTests(unittest.TestCase):
    def plan(self, vetoes, actions):
        planner = MP.RealMCTSPlanner(vetoes=vetoes)
        return quiet(planner.plan, "goal", available_actions=actions, max_iterations=4)

    def test_variant_duplicates_are_listed_once_each_spelling(self):
        result = self.plan({"SHUTDOWN_SERVICE": "r"}, ["shutdown service", "shutdown service", "SHUTDOWN_SERVICE",
                                                       "SHUTDOWN_SERVICE", "MONITOR"])
        self.assertEqual([v["action"] for v in result.vetoed_actions], ["shutdown service", "SHUTDOWN_SERVICE"])

    def test_exact_name_reason_wins_over_the_canonical_one(self):
        result = self.plan({"foo": "primeiro", "FOO": "exato"}, ["FOO", "foo", "Foo()"])
        self.assertEqual(result.vetoed_actions, [{"action": "FOO", "reason": "exato"},
                                                 {"action": "foo", "reason": "primeiro"},
                                                 {"action": "Foo()", "reason": "primeiro"}])

    def test_veto_with_none_reason_still_vetoes_variants(self):
        result = self.plan({"X": None}, ["x ", "MONITOR"])
        self.assertEqual(result.vetoed_actions, [{"action": "x ", "reason": None}])

    def test_non_str_candidates_are_ignored_not_vetoed(self):
        result = self.plan({"X": "r"}, [None, 5, "MONITOR"])
        self.assertEqual(list(result.vetoed_actions), [])

    def test_vetoes_mutated_after_construction_are_seen(self):
        planner = MP.RealMCTSPlanner(vetoes={})
        planner.vetoes["alvo"] = "depois"
        result = quiet(planner.plan, "goal", available_actions=["ALVO ", "MONITOR"], max_iterations=4)
        self.assertEqual(result.vetoed_actions, [{"action": "ALVO ", "reason": "depois"}])


class ScmVetoTests(unittest.TestCase):
    def rank(self, vetoes, sequences):
        return S.rank_alternatives(OBS, ["MONITOR"], sequences, world(), vetoes=vetoes)

    def test_variant_in_a_sequence_uses_the_canonical_reason_and_exact_wins(self):
        model = world()
        self.assertEqual(model.causal_reasoning.declare_intervention("monitor", {"alert_level": "low"}), [])
        out = S.rank_alternatives(OBS, ["MONITOR"], [["monitor"], ["MONITOR"]], model,
                                  vetoes={"MONITOR": "canônico", "monitor": "outro"})
        by_seq = {tuple(item["actions"]): item["vetoed"] for item in out["ranking"]}
        self.assertEqual(by_seq[("monitor",)], [{"action": "monitor", "reason": "outro"}])
        self.assertEqual(by_seq[("MONITOR",)], [{"action": "MONITOR", "reason": "canônico"}])

    def test_declared_variant_without_exact_entry_gets_the_canonical_reason(self):
        model = world()
        self.assertEqual(model.causal_reasoning.declare_intervention("monitor", {"alert_level": "low"}), [])
        out = S.rank_alternatives(OBS, ["MONITOR"], [["monitor"]], model, vetoes={"MONITOR": "canônico"})
        self.assertEqual(out["ranking"][0]["vetoed"], [{"action": "monitor", "reason": "canônico"}])

    def test_unmodeled_variant_is_not_ranked_even_when_its_canonical_form_is_vetoed(self):
        out = self.rank({"MONITOR": "r"}, [["monitor"]])
        self.assertEqual(out["ranking"], [])
        self.assertEqual(out["unmodeled"], [["monitor"]])

    def test_duplicate_spelling_in_a_sequence_is_listed_once(self):
        out = self.rank({"MONITOR": "r"}, [["MONITOR", "MONITOR", "MONITOR"]])
        self.assertEqual(out["ranking"][0]["vetoed"], [{"action": "MONITOR", "reason": "r"}])

    def test_empty_vetoes_turn_the_canonical_veto_off(self):
        out = self.rank({}, [["MONITOR"]])
        self.assertEqual(out["ranking"][0]["vetoed"], [])


class DecisionCycleVetoTests(unittest.TestCase):
    def test_reject_with_call_syntax_blocks_the_plain_proposal(self):
        decision = DC.select_operator(["deploy", "ok"], rejects={"deploy()": "motivo"}, use_default_rejects=False)
        self.assertEqual(decision.selected, "ok")
        self.assertEqual(dict(decision.rejected), {"deploy": "motivo"})

    def test_exact_reject_beats_canonical_extra_on_collision(self):
        rejects = {"x y()": "r0", "x_y": "r1"}
        decision = DC.select_operator(["x_y"], rejects=rejects, use_default_rejects=False)
        self.assertEqual(dict(decision.rejected), {"x_y": "r1"})

    def test_first_reject_wins_on_canonical_collision(self):
        rejects = {"a b()": "primeiro", "a_b();": "segundo"}
        decision = DC.select_operator(["A-B"], rejects=rejects, use_default_rejects=False)
        self.assertEqual(dict(decision.rejected), {"A-B": "primeiro"})

    def test_default_reason_wins_over_a_user_reason_for_the_same_canonical_key(self):
        decision = DC.select_operator(["shutdown service()"], rejects={"SHUTDOWN SERVICE": "usuário"})
        self.assertEqual(dict(decision.rejected), {"shutdown service()": DC.DEFAULT_VETOES["SHUTDOWN_SERVICE"]})

    def test_exact_user_reject_of_the_variant_beats_the_canonical_default(self):
        decision = DC.select_operator(["shutdown service()"], rejects={"SHUTDOWN_SERVICE()": "usuário"})
        self.assertEqual(dict(decision.rejected), {"shutdown service()": "usuário"})

    def test_veto_reason_beats_the_vocabulary_reason(self):
        decision = DC.select_operator(["ignore alert"], allowed=["MONITOR"])
        self.assertEqual(dict(decision.rejected), {"ignore alert": DC.DEFAULT_VETOES["IGNORE_ALERT"]})
        decision = DC.select_operator(["other"], allowed=["MONITOR"])
        self.assertEqual(dict(decision.rejected), {"other": "operador fora do vocabulário permitido"})

    def test_use_default_rejects_false_turns_the_canonical_default_off(self):
        decision = DC.select_operator(["shutdown service"], use_default_rejects=False)
        self.assertEqual(decision.selected, "shutdown service")

    def test_proposal_that_does_not_canonicalize_is_not_vetoed_by_it(self):
        decision = DC.select_operator(["não ascii"], rejects={"ok": "r"}, use_default_rejects=False)
        self.assertEqual(decision.selected, "não ascii")

    def test_reject_that_does_not_canonicalize_still_blocks_its_exact_key(self):
        decision = DC.select_operator(["não ascii", "ok"], rejects={"não ascii": "r"}, use_default_rejects=False)
        self.assertEqual(decision.selected, "ok")

    def test_default_canonical_map_is_built_from_default_vetoes_and_frozen(self):
        from unittest import mock
        with mock.patch.object(DC, "DEFAULT_VETOES", {"Deploy  Now()": "motivo-padrao", "não ascii": "x"}):
            frozen = DC._freeze_default_canon()
            exact = DC._freeze_default_vetoes()
        self.assertEqual(dict(frozen), {"DEPLOY_NOW": "motivo-padrao"})
        with self.assertRaises(TypeError):
            frozen["OUTRO"] = "x"
        with mock.patch.object(DC, "_FROZEN_DEFAULT_VETOES", exact), \
                mock.patch.object(DC, "_FROZEN_DEFAULT_CANON", frozen):
            decision = DC.select_operator(["deploy now", "ok"])
        self.assertEqual(decision.selected, "ok")
        self.assertEqual(dict(decision.rejected), {"deploy now": "motivo-padrao"})

    def test_default_added_after_import_is_not_seen_by_exact_or_canonical_maps(self):
        from unittest import mock
        with mock.patch.dict(DC.DEFAULT_VETOES, {"NOVO_OPERADOR": "motivo-novo"}):
            for name in ("NOVO_OPERADOR", "novo operador"):
                with self.subTest(name=name):
                    decision = DC.select_operator([name])
                    self.assertEqual(decision.selected, name)
                    self.assertEqual(dict(decision.rejected), {})

    def test_import_time_canonical_default_is_frozen(self):
        saved = dict(DC.DEFAULT_VETOES)
        try:
            DC.DEFAULT_VETOES.clear()
            decision = DC.select_operator(["shutdown service"])
            self.assertIsNone(decision.selected)
        finally:
            DC.DEFAULT_VETOES.update(saved)


if __name__ == "__main__":
    unittest.main()
