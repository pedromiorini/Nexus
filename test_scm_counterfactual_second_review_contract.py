"""Contrato da 2ª revisão da Pearl no PR #154 (sugestões S1, S2 e S3).

H1: ``vetoes`` validado e congelado na entrada, ``DEFAULT_VETOES`` congelado no import;
H2: limitação conhecida do alias declarado (o veto é por nome, não por efeito);
H3: a decisão é ``best``, não ``ranking[0]``.
"""
from __future__ import annotations

import unittest
from collections.abc import Mapping
from types import MappingProxyType
from unittest import mock

from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator
from core.cognitivo.mcts_planner import DEFAULT_VETOES
from test_scm_counterfactual_contract import _engine, scm

OBS = {"system_status": "compromised", "alert_level": "critical"}
ALT = [["SHUTDOWN_SERVICE"], ["MONITOR"]]


def _world(declared=None):
    causal = RealCausalReasoning()
    for name, effects in (declared or {}).items():
        assert causal.declare_intervention(name, effects) == []
    return RealWorldModelSimulator(causal)


def _rank(**kwargs):
    return scm().rank_alternatives(OBS, ["MONITOR"], ALT, _world(), **kwargs)


class BadStr:
    def __str__(self):
        raise RuntimeError("boom")


class ReadOnce(Mapping):
    """Mapa que falha se a mesma chave for lida duas vezes (a entrada é lida uma única vez)."""

    def __init__(self, data):
        self.data, self.seen = dict(data), set()

    def __getitem__(self, key):
        if key in self.seen:
            raise AssertionError("chave lida duas vezes: %r" % (key,))
        self.seen.add(key)
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class VetoValidationTests(unittest.TestCase):
    """H1: validar e congelar ``vetoes`` na entrada."""

    def test_h1a_non_str_key_or_value_is_a_type_error_before_ranking(self):
        bad = ({1: "x"}, {"SHUTDOWN_SERVICE": 1}, {"SHUTDOWN_SERVICE": None},
               {None: "x"}, {"SHUTDOWN_SERVICE": ["x"]}, {b"SHUTDOWN_SERVICE": "x"})
        for vetoes in bad:
            with self.subTest(vetoes=vetoes), mock.patch.object(
                    scm(), "counterfactual_outcome", side_effect=AssertionError("rankeou antes de validar")):
                with self.assertRaises(TypeError):
                    _rank(vetoes=vetoes)

    def test_h1b_value_with_a_raising_str_is_a_type_error_not_runtime_error(self):
        with self.assertRaises(TypeError):
            _rank(vetoes={"SHUTDOWN_SERVICE": BadStr()})

    def test_h1c_vetoes_are_validated_before_the_alternatives(self):
        with self.assertRaises(TypeError):
            scm().rank_alternatives(OBS, ["MONITOR"], [], _world(), vetoes={1: "x"})

    def test_h1d_valid_mappings_are_still_accepted(self):
        for vetoes in ({}, {"SHUTDOWN_SERVICE": "motivo"}, MappingProxyType({"SHUTDOWN_SERVICE": "motivo"})):
            with self.subTest(vetoes=vetoes):
                out = _rank(vetoes=vetoes)
                self.assertEqual(out["best"]["actions"], ["MONITOR"] if vetoes else ["SHUTDOWN_SERVICE"])
        out = _rank(vetoes={"SHUTDOWN_SERVICE": "motivo"})
        self.assertEqual(out["ranking"][0]["vetoed"], [{"action": "SHUTDOWN_SERVICE", "reason": "motivo"}])

    def test_h1e_the_input_mapping_is_read_once_and_the_reasons_are_exact_str(self):
        mapping = ReadOnce({"SHUTDOWN_SERVICE": "motivo", "MONITOR": "outro"})
        out = scm().rank_alternatives(OBS, ["MONITOR"], ALT + [["SHUTDOWN_SERVICE", "MONITOR"]], _world(),
                                      vetoes=mapping)
        self.assertEqual(mapping.seen, {"SHUTDOWN_SERVICE", "MONITOR"})
        for item in out["ranking"]:
            for veto in item["vetoed"]:
                self.assertIs(type(veto["reason"]), str)
                self.assertIs(type(veto["action"]), str)

    def test_h1f_mutating_the_mapping_after_the_call_does_not_change_the_result(self):
        vetoes = {"SHUTDOWN_SERVICE": "motivo A"}
        out = _rank(vetoes=vetoes)
        vetoes["SHUTDOWN_SERVICE"] = "motivo B"
        vetoes.clear()
        self.assertEqual([v["reason"] for item in out["ranking"] for v in item["vetoed"]], ["motivo A"])

    def test_h1g_engine_rejects_invalid_vetoes_too(self):
        with self.assertRaises(TypeError):
            _engine().rank_counterfactuals(OBS, ["MONITOR"], ALT, vetoes={1: "x"})


class FrozenDefaultTests(unittest.TestCase):
    """H1: ``DEFAULT_VETOES`` é copiado/congelado no import; mexer nele depois não desliga o veto."""

    def setUp(self):
        saved = dict(DEFAULT_VETOES)
        self.expected = {name: saved[name] for name in saved}

        def restore():
            DEFAULT_VETOES.clear()
            DEFAULT_VETOES.update(saved)

        self.addCleanup(restore)

    def test_h1h_clearing_the_default_mapping_does_not_turn_the_veto_off(self):
        before = _rank()
        self.assertEqual(before["best"]["actions"], ["MONITOR"])
        DEFAULT_VETOES.clear()
        after = _rank()
        self.assertEqual(after["best"]["actions"], ["MONITOR"])
        self.assertEqual(after["vetoed"], before["vetoed"])
        self.assertEqual(_engine().rank_counterfactuals(OBS, ["MONITOR"], ALT)["vetoed"], before["vetoed"])

    def test_h1i_adding_to_the_default_mapping_does_not_add_vetoes(self):
        DEFAULT_VETOES["MONITOR"] = "intruso"
        out = _rank()
        self.assertEqual(out["best"]["actions"], ["MONITOR"])
        self.assertEqual([v["action"] for item in out["ranking"] for v in item["vetoed"]], ["SHUTDOWN_SERVICE"])

    def test_h1j_default_reasons_are_those_of_the_planner_table_at_import(self):
        out = _rank()
        reasons = {v["action"]: v["reason"] for item in out["ranking"] for v in item["vetoed"]}
        self.assertEqual(reasons, {"SHUTDOWN_SERVICE": self.expected["SHUTDOWN_SERVICE"]})


class AliasLimitationTests(unittest.TestCase):
    """H2: o veto é por nome exato, não por efeito (limitação conhecida, fixada aqui)."""

    ALIAS = {"shutdown_service": {"system_status": "offline"}}

    def test_h2a_declared_alias_with_the_same_effect_is_not_vetoed_by_default(self):
        out = scm().rank_alternatives(OBS, ["MONITOR"], [["shutdown_service"], ["MONITOR"]],
                                      _world(self.ALIAS))
        alias = next(item for item in out["ranking"] if item["actions"] == ["shutdown_service"])
        self.assertEqual(alias["verified_status"], "offline")
        self.assertEqual(alias["vetoed"], [])
        self.assertEqual(out["vetoed"], [])
        self.assertEqual(out["best"]["actions"], ["shutdown_service"])

    def test_h2b_the_alias_is_vetoed_when_its_name_is_in_the_mapping(self):
        out = scm().rank_alternatives(OBS, ["MONITOR"], [["shutdown_service"], ["MONITOR"]], _world(self.ALIAS),
                                      vetoes={"shutdown_service": "alias"})
        self.assertEqual(out["best"]["actions"], ["MONITOR"])

    def test_h2c_the_module_docstring_states_the_limitation(self):
        doc = scm().__doc__
        for needle in ("alias", "declare_intervention", "veto por efeito", "Limitação conhecida"):
            with self.subTest(needle=needle):
                self.assertIn(needle, doc)


class DecisionDocumentationTests(unittest.TestCase):
    """H3: a decisão é ``best``; ``ranking[0]`` pode estar vetado."""

    def test_h3a_ranking_head_can_be_vetoed_while_best_is_not(self):
        out = _rank()
        self.assertEqual(out["ranking"][0]["actions"], ["SHUTDOWN_SERVICE"])
        self.assertTrue(out["ranking"][0]["vetoed"])
        self.assertEqual(out["best"]["actions"], ["MONITOR"])

    def test_h3b_all_vetoed_means_best_none(self):
        out = scm().rank_alternatives(OBS, ["MONITOR"], [["SHUTDOWN_SERVICE"]], _world())
        self.assertIsNone(out["best"])
        self.assertTrue(out["ranking"])

    def test_h3c_rank_docstring_states_the_decision_rule(self):
        doc = scm().rank_alternatives.__doc__
        for needle in ("decisão é ``best``", "não ``ranking[0]``", "``best=None``", "todas vetadas"):
            with self.subTest(needle=needle):
                self.assertIn(needle, doc)


if __name__ == "__main__":
    unittest.main()
