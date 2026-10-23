"""Contrato CF10–CF13: plausibilidade informada pelo grafo causal.

Critérios pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit c77e57f) antes deste teste e antes da correção.
Na base 29a5711 a conta é só max(0.1, 0.7 - 0.05*|chain|): cadeia com
aresta e cadeia genérica do mesmo comprimento recebem o mesmo número,
e intervenção vazia não cai no piso.
"""
from __future__ import annotations

import contextlib
import io
import math
import unittest

from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from core.cognitivo.causal_world_model import RealCausalReasoning
from core.cognitivo.counterfactual import RealCounterfactualReasoningEngine

ABS = 1e-9
DO_STEP = "do(APPLY_PATCH): system_status=secure"
REL_STEP = "alarm causes patch"
HEURISTIC = [
    "1. system vulnerable",
    "2. Intervention: apply patch",
    "3. Immediate effect",
    "4. Secondary effect",
    "5. Final outcome",
]
DEPLOY_CHAIN = [
    "do(DEPLOY_FIX): system_status=secure",
    "host exposed causes DEPLOY_FIX",
]


def _engine(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealCounterfactualReasoningEngine(**kwargs)


def _length_score(n: int) -> float:
    """Oráculo da fórmula de comprimento, copiado do pré-registro. Não chama o módulo."""
    return max(0.1, 0.7 - 0.05 * n)


class _Graph:
    def __init__(self, adjacency):
        self.causal_graph = {"adjacency": adjacency}


def _close(testcase, score, expected):
    testcase.assertTrue(math.isclose(score, expected, rel_tol=0, abs_tol=ABS), score)


class CounterfactualPlausibilityContractTests(unittest.TestCase):
    def test_cf5_length_values_stay_monotone_without_graph(self):
        eng = _engine()
        lens = (1, 3, 5, 10)
        expected = (0.65, 0.55, 0.45, 0.20)
        scores = [eng._calculate_plausibility("x", ["a"] * n) for n in lens]
        for score, target in zip(scores, expected):
            _close(self, score, target)
        self.assertTrue(all(scores[i] > scores[i + 1] for i in range(len(scores) - 1)))
        _close(self, eng._calculate_plausibility("x", ["a", "b"]), 0.6)

    def test_heuristic_five_steps_stay_on_length_formula(self):
        eng = _engine()
        scenario = eng.generate_counterfactual_scenario("system vulnerable", "apply patch")
        self.assertEqual(scenario.causal_chain, HEURISTIC)
        _close(self, scenario.plausibility_score, 0.45)
        _close(self, scenario.plausibility_score, _length_score(5))

    def test_scm_chain_of_length_five_scores_above_length_formula(self):
        eng = _engine(causal_reasoning=_Graph({"APPLY_PATCH": {"system_status secure"}}))
        chain = [DO_STEP] * 5
        score = eng._calculate_plausibility("APPLY_PATCH", chain)
        _close(self, score, 0.65)
        self.assertFalse(math.isclose(score, 0.45, rel_tol=0, abs_tol=ABS))
        self.assertGreater(score, _length_score(5))

    def test_extracted_relation_chain_of_length_five_scores_above(self):
        eng = _engine(causal_reasoning=_Graph({"alarm": {"patch"}}))
        score = eng._calculate_plausibility("patch", [REL_STEP] * 5)
        _close(self, score, 0.65)
        self.assertGreater(score, _length_score(5))

    def test_support_fraction_is_not_a_flat_bonus(self):
        eng = _engine(causal_reasoning=_Graph({"alarm": {"patch"}}))
        chain = [REL_STEP, "noise", "noise", "noise", "noise"]
        score = eng._calculate_plausibility("patch", chain)
        _close(self, score, 0.49)
        self.assertFalse(math.isclose(score, 0.65, rel_tol=0, abs_tol=ABS))
        self.assertFalse(math.isclose(score, 0.45, rel_tol=0, abs_tol=ABS))

    def test_text_without_graph_is_not_evidence(self):
        eng = _engine()
        for chain in ([DO_STEP] * 5, [REL_STEP] * 5, HEURISTIC):
            with self.subTest(chain=chain[0]):
                score = eng._calculate_plausibility("patch", chain)
                _close(self, score, 0.45)

    def test_length_five_orders_unsupported_below_plain_below_attested(self):
        plain = _engine()._calculate_plausibility("patch", ["noise"] * 5)
        unsupported = _engine(
            causal_reasoning=_Graph({"alarm": {"patch"}})
        )._calculate_plausibility("patch", ["noise"] * 5)
        attested = _engine(
            causal_reasoning=_Graph({"alarm": {"patch"}})
        )._calculate_plausibility("patch", [REL_STEP] * 5)
        _close(self, unsupported, 0.25)
        _close(self, plain, 0.45)
        _close(self, attested, 0.65)
        self.assertLess(unsupported, plain)
        self.assertLess(plain, attested)

    def test_live_extracted_relation_is_not_the_old_formula(self):
        causal = RealCausalReasoning()
        eng = _engine(causal_reasoning=causal)
        scenario = eng.generate_counterfactual_scenario("alarm", "patch")
        self.assertEqual(scenario.causal_chain, [REL_STEP])
        _close(self, scenario.plausibility_score, 0.85)
        self.assertFalse(math.isclose(scenario.plausibility_score, 0.65, rel_tol=0, abs_tol=ABS))

    def test_live_scm_chain_is_not_the_old_formula(self):
        with contextlib.redirect_stdout(io.StringIO()):
            causal = RealCausalReasoning()
        causal.extract_causal_relations("DEPLOY_FIX causes system_status secure")
        eng = _engine(causal_reasoning=causal)
        scenario = eng.generate_counterfactual_scenario("host exposed", "DEPLOY_FIX")
        self.assertEqual(scenario.causal_chain, DEPLOY_CHAIN)
        _close(self, scenario.plausibility_score, 0.8)
        self.assertFalse(math.isclose(scenario.plausibility_score, 0.6, rel_tol=0, abs_tol=ABS))

    def test_empty_intervention_is_floor_even_with_support(self):
        graph = _Graph({"alarm": {"patch"}})
        for intervention in ("", "   ", False):
            for reasoning in (None, graph):
                with self.subTest(intervention=intervention, graph=reasoning is not None):
                    eng = _engine(causal_reasoning=reasoning)
                    score = eng._calculate_plausibility(intervention, [REL_STEP] * 5)
                    _close(self, score, 0.1)
                    self.assertLess(score, 0.45)

    def test_generate_empty_false_and_whitespace_store_floor(self):
        eng = _engine()
        blank = eng.generate_counterfactual_scenario("sit", "")
        spaces = eng.generate_counterfactual_scenario("sit", "   ")
        absent = eng.generate_counterfactual_scenario("sit", False)
        self.assertEqual(blank.counterfactual_outcome, "outcome_without_intervention")
        self.assertEqual(absent.counterfactual_outcome, "outcome_without_intervention")
        self.assertEqual(spaces.counterfactual_outcome, "outcome_with____")
        for scenario in (blank, spaces, absent):
            _close(self, scenario.plausibility_score, 0.1)

    def test_unsupported_chain_and_empty_chain_when_graph_has_edges(self):
        eng = _engine(causal_reasoning=_Graph({"alarm": {"patch"}}))
        _close(self, eng._calculate_plausibility("patch", ["noise"] * 5), 0.25)
        _close(self, eng._calculate_plausibility("x", []), 0.5)
        bare = _engine()
        _close(self, bare._calculate_plausibility("x", []), 0.7)
        string_edge = _engine(causal_reasoning=_Graph({"alarm": "patch"}))
        _close(self, string_edge._calculate_plausibility("patch", ["noise"] * 5), 0.45)

    def test_invalid_types_raise_and_numeric_string_does_not(self):
        eng = _engine()
        for bad in (True, 1, -1, float("nan"), float("inf"), float("-inf"), None):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    eng._calculate_plausibility(bad, ["a"])
        for bad_chain in ("abc", ("a",), None, True, ["a", 1], ["a", True], ["a", float("nan")]):
            with self.subTest(chain=bad_chain):
                with self.assertRaises(TypeError):
                    eng._calculate_plausibility("x", bad_chain)
        _close(self, eng._calculate_plausibility("-1", ["a"]), 0.65)
        _close(self, eng._calculate_plausibility("nan", ["a"]), 0.65)

    @given(
        st.lists(st.text(max_size=40), max_size=12),
        st.text(min_size=1, max_size=24),
    )
    @example([REL_STEP] * 5, "patch")
    @example([DO_STEP] * 5, "APPLY_PATCH")
    @example(["a"] * 10, "x")
    @example([], "x")
    @settings(max_examples=40, deadline=None)
    def test_empty_graph_matches_length_formula(self, chain, intervention):
        assume(intervention.strip() != "")
        eng = _engine()
        score = eng._calculate_plausibility(intervention, chain)
        _close(self, score, _length_score(len(chain)))


if __name__ == "__main__":
    unittest.main()
