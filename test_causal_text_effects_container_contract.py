"""Contrato: classify_text_effects não fatiar texto nem tratar bool/zero como vazio.

Medido em main f4c66c0, antes da correção. O efeito válido de controle é
``alert_level high``.

- H1. str, bytes e bytearray já são um efeito só. Falsificada se o retorno é
  vazio, sem TypeError (o texto é fatiado em caracteres ou inteiros).
- H2. False, 0 e 0.0 já falham fechado, como True, NaN e infinito. Falsificada
  se o retorno é ``({}, {}, [])`` por serem falsy em ``effects or ()``.
- H3. skip str já é um conjunto de chaves. Falsificada se a chave contida na
  string some (``in`` de substring) em vez de TypeError.
- H4. lista, tupla, set e frozenset com o mesmo texto continuam claim; None e
  coleções vazias continuam sem efeitos; skip por chave continua pulando.
  Controles que já valem antes da correção.
"""
from __future__ import annotations

import math
import unittest

from core.cognitivo.causal_world_model import classify_text_effects

_EFFECTS = "effects deve ser lista, tupla, set ou frozenset"
_SKIP = "skip deve ser lista, tupla, set, frozenset ou dict"
_CLAIM = ({"alert_level": "high"}, {}, [])
_EMPTY = ({}, {}, [])


class CausalTextEffectsContainerContractTests(unittest.TestCase):
    def test_text_and_bytes_are_not_character_sequences(self):
        for effects in (
            "alert_level high",
            "",
            "100%",
            "a_b",
            b"alert_level high",
            bytearray(b"alert_level high"),
        ):
            with self.subTest(effects=effects):
                with self.assertRaisesRegex(TypeError, _EFFECTS):
                    classify_text_effects(effects)

    def test_bool_zero_nan_and_infinity_are_not_empty_collections(self):
        for effects in (
            False,
            True,
            0,
            0.0,
            -1,
            math.nan,
            math.inf,
            -math.inf,
            {"alert_level": "high"},
            {"alert_level high": "ignored"},
            iter(["alert_level high"]),
            range(1),
        ):
            with self.subTest(effects=effects):
                with self.assertRaisesRegex(TypeError, _EFFECTS):
                    classify_text_effects(effects)

    def test_string_skip_is_not_substring_membership(self):
        effects = ["alert_level high", "system_status secure"]
        for skip in ("alert_level", "alert_level high", b"alert_level", bytearray(b"alert_level"), ""):
            with self.subTest(skip=skip):
                with self.assertRaisesRegex(TypeError, _SKIP):
                    classify_text_effects(effects, skip=skip)
        for skip in (False, True, 0, 0.0, -1, math.nan, math.inf, None):
            with self.subTest(skip=skip):
                with self.assertRaisesRegex(TypeError, _SKIP):
                    classify_text_effects(effects, skip=skip)

    def test_sequences_and_sets_still_classify_one_effect(self):
        for effects in (
            ["alert_level high"],
            ("alert_level high",),
            {"alert_level high"},
            frozenset({"alert_level high"}),
        ):
            with self.subTest(effects=type(effects).__name__):
                self.assertEqual(classify_text_effects(effects), _CLAIM)

    def test_none_and_empty_collections_stay_empty(self):
        for effects in (None, [], (), set(), frozenset()):
            with self.subTest(effects=effects):
                self.assertEqual(classify_text_effects(effects), _EMPTY)

    def test_key_skip_still_drops_only_that_key(self):
        effects = ["alert_level high", "system_status secure"]
        expected = ({"system_status": "secure"}, {}, [])
        for skip in (
            {"alert_level"},
            ["alert_level"],
            ("alert_level",),
            frozenset({"alert_level"}),
            {"alert_level": "declared"},
        ):
            with self.subTest(skip=type(skip).__name__):
                self.assertEqual(classify_text_effects(effects, skip=skip), expected)

    def test_conflict_and_schema_miss_stay_deterministic(self):
        claims, conflicts, ignored = classify_text_effects(
            ["alert_level low", "not_a_key high", "alert_level high"]
        )
        self.assertEqual(claims, {})
        self.assertEqual(conflicts, {"alert_level": ["high", "low"]})
        self.assertEqual(ignored, ["not_a_key=high"])

    def test_bool_item_inside_a_real_collection_is_not_a_schema_bool(self):
        self.assertEqual(classify_text_effects([True, False, 1]), _EMPTY)


if __name__ == "__main__":
    unittest.main()
