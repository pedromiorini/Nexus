"""Contrato: observação e ações do SCM viram ``str`` exato antes de validar.

Fenômeno em ``26e38e5``: ``_actions`` chama ``action.strip()`` e guarda a
subclasse; ``ACTION_EFFECTS.get`` então chama ``__eq__``/``__hash__``.
``_canonical_observation`` entrega a subclasse a ``_schema_pair``, que chama
``strip``/``lower``. Métodos do chamador não são parte do contrato (o mesmo
padrão de ``str.__str__`` já usado em ``vetoes``).

Hipóteses:
- H1: a subclasse na ação já era ``str`` exato. Falsifica se ``strip``/``__eq__``
  levanta.
- H2: a subclasse na observação já era ``str`` exato. Falsifica se ``strip``
  levanta.
- H3: converter com ``str.__str__`` também apararia ou mudaria a caixa do nome.
  Falsifica se `` MONITOR `` e ``monitor`` continuam não modelados.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings, strategies as st

from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator
from core.cognitivo.scm_counterfactual import (
    ScmInputError,
    counterfactual_outcome,
    rank_alternatives,
)

OBS = {"system_status": "vulnerable"}


class HostileAction(str):
    """Subclasse cuja superfície de ``str`` levanta. Não pode ser consultada."""

    calls: list[str] = []

    def strip(self, *args, **kwargs):
        type(self).calls.append("strip")
        raise RuntimeError("strip")

    def lower(self, *args, **kwargs):
        type(self).calls.append("lower")
        raise RuntimeError("lower")

    def upper(self, *args, **kwargs):
        type(self).calls.append("upper")
        raise RuntimeError("upper")

    def __str__(self):
        type(self).calls.append("str")
        raise RuntimeError("str")

    def __eq__(self, other):
        type(self).calls.append("eq")
        raise RuntimeError("eq")

    def __hash__(self):
        type(self).calls.append("hash")
        raise RuntimeError("hash")


class HostileText(str):
    """Subclasse que pode ser chave de dict (``eq``/``hash`` herdados)."""

    calls: list[str] = []

    def strip(self, *args, **kwargs):
        type(self).calls.append("strip")
        raise RuntimeError("strip")

    def lower(self, *args, **kwargs):
        type(self).calls.append("lower")
        raise RuntimeError("lower")

    def upper(self, *args, **kwargs):
        type(self).calls.append("upper")
        raise RuntimeError("upper")

    def __str__(self):
        type(self).calls.append("str")
        raise RuntimeError("str")


def _code(call):
    try:
        call()
    except ScmInputError as exc:
        return exc.code
    except Exception as exc:
        return type(exc).__name__
    return None


class ExactStrActionTests(unittest.TestCase):
    def setUp(self):
        HostileAction.calls = []
        HostileText.calls = []

    def test_modeled_subclass_matches_plain_str_and_calls_nothing(self):
        action = HostileAction("APPLY_PATCH")
        seq = [action]
        got = counterfactual_outcome(OBS, [HostileAction("MONITOR")], seq)
        plain = counterfactual_outcome(OBS, ["MONITOR"], ["APPLY_PATCH"])
        self.assertEqual(got, plain)
        self.assertEqual(got.verified_status, "secure")
        self.assertEqual(got.utility, 3)
        self.assertEqual(HostileAction.calls, [])
        self.assertIs(seq[0], action)

    def test_rank_returns_exact_str_actions(self):
        action = HostileAction("APPLY_PATCH")
        seq = [action]
        out = rank_alternatives(OBS, ["MONITOR"], [seq])
        self.assertEqual(out["best"]["actions"], ["APPLY_PATCH"])
        self.assertIs(type(out["best"]["actions"][0]), str)
        self.assertEqual(HostileAction.calls, [])
        self.assertIs(seq[0], action)

    def test_declared_name_is_looked_up_as_exact_str(self):
        causal = RealCausalReasoning()
        self.assertEqual(
            causal.declare_intervention("CUSTOM_ACT", {"system_status": "offline"}),
            [],
        )
        world = RealWorldModelSimulator(causal)
        got = counterfactual_outcome(
            OBS, ["MONITOR"], [HostileAction("CUSTOM_ACT")], world,
        )
        self.assertEqual(got.verified_status, "offline")
        self.assertEqual(HostileAction.calls, [])

    def test_blank_subclass_is_scm_bad_action_without_calling_strip(self):
        for blank in (HostileAction(""), HostileAction("   "), HostileAction("\t")):
            with self.subTest(blank=str.__str__(blank)):
                self.assertEqual(
                    _code(lambda blank=blank: counterfactual_outcome(
                        OBS, [blank], ["MONITOR"])),
                    "SCM_BAD_ACTION",
                )
                self.assertEqual(
                    _code(lambda blank=blank: counterfactual_outcome(
                        OBS, ["MONITOR"], (blank,))),
                    "SCM_BAD_ACTION",
                )
        self.assertEqual(HostileAction.calls, [])

    def test_padding_and_case_stay_unmodeled(self):
        """H3: não aparar nem dobrar caixa. ``strip`` da subclasse não decide."""

        class Smuggle(HostileAction):
            def strip(self, *args, **kwargs):
                type(self).calls.append("strip")
                return "APPLY_PATCH"

        samples = (
            ("plain-pad", " MONITOR "),
            ("plain-case", "monitor"),
            ("plain-tokens", "apply patch"),
            ("smuggle", Smuggle(" MONITOR ")),
        )
        for kind, name in samples:
            with self.subTest(kind=kind):
                self.assertEqual(
                    _code(lambda name=name: counterfactual_outcome(
                        OBS, ["MONITOR"], [name])),
                    "SCM_UNMODELED",
                )
        self.assertEqual(HostileAction.calls, [])

    def test_non_str_actions_stay_type_error(self):
        bad = (
            True, False, 1, 0, 1.0, float("nan"), float("inf"), float("-inf"),
            None, b"MONITOR", bytearray(b"MONITOR"), object(),
        )
        for item in bad:
            with self.subTest(item=item):
                for seq in ((["MONITOR"], [item]), ([item], ["MONITOR"])):
                    with self.assertRaises(TypeError) as caught:
                        counterfactual_outcome(OBS, *seq)
                    self.assertIn("deve conter só str", str(caught.exception))


class ExactStrObservationTests(unittest.TestCase):
    def setUp(self):
        HostileText.calls = []

    def test_value_subclass_matches_plain_and_calls_nothing(self):
        loud = {"system_status": HostileText(" vulnerable ")}
        got = counterfactual_outcome(loud, ["MONITOR"], ["MONITOR"])
        plain = counterfactual_outcome(
            {"system_status": "vulnerable"}, ["MONITOR"], ["MONITOR"],
        )
        self.assertEqual(got, plain)
        self.assertEqual(got.verified_status, "vulnerable")
        self.assertEqual(HostileText.calls, [])

    def test_key_subclass_matches_plain_and_calls_nothing(self):
        loud = {HostileText(" System_Status "): "secure"}
        got = counterfactual_outcome(loud, ["MONITOR"], ["MONITOR"])
        plain = counterfactual_outcome(
            {"system_status": "secure"}, ["MONITOR"], ["MONITOR"],
        )
        self.assertEqual(got, plain)
        self.assertEqual(HostileText.calls, [])

    def test_bool_patch_flag_stays_strict(self):
        got = counterfactual_outcome(
            {HostileText("security_patch_applied"): True}, ["MONITOR"], ["MONITOR"],
        )
        self.assertIs(got.state["security_patch_applied"], True)
        self.assertEqual(HostileText.calls, [])
        for bad in (1, 0, "true", "True", None, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                self.assertEqual(
                    _code(lambda bad=bad: counterfactual_outcome(
                        {"security_patch_applied": bad}, ["MONITOR"], ["MONITOR"])),
                    "SCM_BAD_OBSERVATION",
                )

    def test_hostile_value_outside_schema_is_bad_observation(self):
        self.assertEqual(
            _code(lambda: counterfactual_outcome(
                {"system_status": HostileText(" melting ")}, ["MONITOR"], ["MONITOR"])),
            "SCM_BAD_OBSERVATION",
        )
        self.assertEqual(
            _code(lambda: counterfactual_outcome(
                {HostileText("colour"): HostileText("red")}, ["MONITOR"], ["MONITOR"])),
            "SCM_BAD_OBSERVATION",
        )
        self.assertEqual(HostileText.calls, [])

    def test_duplicate_canonical_key_still_conflicts(self):
        obs = {
            HostileText("system_status"): HostileText("vulnerable"),
            HostileText(" System_Status "): "secure",
        }
        self.assertEqual(
            _code(lambda: counterfactual_outcome(obs, ["MONITOR"], ["MONITOR"])),
            "SCM_BAD_OBSERVATION",
        )
        self.assertEqual(HostileText.calls, [])

    def test_non_str_observation_container_stays_type_error(self):
        for bad in (None, [], "vulnerable", True, 1, b"{}"):
            with self.subTest(bad=type(bad).__name__):
                if bad is None or bad == {}:
                    self.assertEqual(
                        _code(lambda bad=bad: counterfactual_outcome(
                            bad, ["MONITOR"], ["MONITOR"])),
                        "SCM_NO_OBSERVATION",
                    )
                else:
                    with self.assertRaises(TypeError):
                        counterfactual_outcome(bad, ["MONITOR"], ["MONITOR"])


class ExactStrPropertyTests(unittest.TestCase):
    """A subclasse reproduz o ``str`` puro e não registra chamada."""

    @settings(max_examples=40, derandomize=True, deadline=None, database=None)
    @given(st.text(max_size=24))
    def test_action_subclass_matches_plain_text(self, text):
        HostileAction.calls = []
        loud = HostileAction(text)
        left = _code(lambda: counterfactual_outcome(OBS, ["MONITOR"], [text]))
        right = _code(lambda: counterfactual_outcome(OBS, ["MONITOR"], [loud]))
        self.assertEqual(left, right)
        self.assertEqual(HostileAction.calls, [])

    @settings(max_examples=40, derandomize=True, deadline=None, database=None)
    @given(st.text(max_size=24))
    def test_observation_value_subclass_matches_plain_text(self, text):
        HostileText.calls = []
        left = _code(lambda: counterfactual_outcome(
            {"system_status": text}, ["MONITOR"], ["MONITOR"]))
        right = _code(lambda: counterfactual_outcome(
            {"system_status": HostileText(text)}, ["MONITOR"], ["MONITOR"]))
        self.assertEqual(left, right)
        self.assertEqual(HostileText.calls, [])
