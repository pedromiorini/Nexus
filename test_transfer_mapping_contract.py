"""Contrato público de TransferMapping e de transfer_knowledge.

O nome TransferMapping, depois do import do módulo, é a forma do POST 48
(mapping_id, transferable_knowledge, transfer_quality). A forma do POST 38
(transfer_strength, shared_features, adaptation_required) é o retorno de
RealMetaLearningStrategies.transfer_knowledge e não pode ser a mesma classe:
os construtores são disjuntos. Números inválidos falham fechado, sem coerção.
"""

import math
import unittest

from hypothesis import given, settings, strategies as st

from core.constitutional_brain import (
    RealMetaLearningStrategies,
    RealTransferLearningEngine,
    TransferMapping,
)


def _meta_mapping_type():
    """Tipo capturado pela anotação de retorno de transfer_knowledge.

    A anotação é avaliada na definição do método e preserva a classe do POST 38
    mesmo quando uma definição posterior reutiliza o nome TransferMapping.
    """
    return RealMetaLearningStrategies.transfer_knowledge.__annotations__["return"]


def _meta_mapping(transfer_strength, adaptation_required, shared_features=None):
    return _meta_mapping_type()(
        source_domain="vision",
        target_domain="planning",
        transfer_strength=transfer_strength,
        shared_features=["structure"] if shared_features is None else shared_features,
        adaptation_required=adaptation_required,
    )


class TransferMappingContractTests(unittest.TestCase):
    def test_transfer_knowledge_returns_bounded_meta_mapping(self):
        engine = RealMetaLearningStrategies()
        mapping = engine.transfer_knowledge("visual_cortex", "visual_planning")

        self.assertIs(type(mapping), _meta_mapping_type())
        self.assertNotIsInstance(mapping, TransferMapping)
        self.assertEqual(mapping.source_domain, "visual_cortex")
        self.assertEqual(mapping.target_domain, "visual_planning")
        self.assertEqual(mapping.shared_features, ["visual"])
        self.assertEqual(mapping.transfer_strength, 0.1)
        self.assertAlmostEqual(mapping.adaptation_required, 0.9)
        self.assertIs(engine.transfer_mappings[-1], mapping)
        self.assertEqual(engine.total_transfers, 1)

    def test_transfer_knowledge_caps_strength_at_one(self):
        tokens = "_".join(f"f{i}" for i in range(12))
        mapping = RealMetaLearningStrategies().transfer_knowledge(tokens, tokens)

        self.assertEqual(mapping.transfer_strength, 1.0)
        self.assertEqual(mapping.adaptation_required, 0.0)
        self.assertEqual(len(mapping.shared_features), 12)

    def test_transfer_knowledge_uses_generic_features_without_overlap(self):
        mapping = RealMetaLearningStrategies().transfer_knowledge("alpha", "beta")

        self.assertEqual(mapping.shared_features, ["general_pattern", "structure"])
        self.assertEqual(mapping.transfer_strength, 0.2)
        self.assertAlmostEqual(mapping.adaptation_required, 0.8)

    def test_transfer_knowledge_rejects_non_string_domains_without_mutation(self):
        engine = RealMetaLearningStrategies()
        for source, target in ((True, "planning"), ("vision", None), (1, "planning")):
            with self.subTest(source=source, target=target):
                with self.assertRaises(TypeError):
                    engine.transfer_knowledge(source, target)
        self.assertEqual(engine.total_transfers, 0)
        self.assertEqual(engine.transfer_mappings, [])

    def test_meta_mapping_accepts_unit_interval_boundaries(self):
        for strength, adaptation in ((0.0, 1.0), (1.0, 0.0), (0, 1), (1, 0)):
            with self.subTest(strength=strength, adaptation=adaptation):
                mapping = _meta_mapping(strength, adaptation)
                self.assertEqual(mapping.transfer_strength, strength)
                self.assertEqual(mapping.adaptation_required, adaptation)

    def test_meta_mapping_rejects_bool_as_number(self):
        for field, value in (("transfer_strength", True), ("adaptation_required", False)):
            with self.subTest(field=field, value=value):
                kwargs = {"transfer_strength": 0.2, "adaptation_required": 0.8}
                kwargs[field] = value
                with self.assertRaisesRegex(TypeError, field):
                    _meta_mapping(**kwargs)

    def test_meta_mapping_rejects_nan_infinity_and_out_of_range(self):
        invalid = (float("nan"), float("inf"), float("-inf"), -0.01, 1.01)
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "transfer_strength"):
                    _meta_mapping(value, 0.0)
                with self.assertRaisesRegex(ValueError, "adaptation_required"):
                    _meta_mapping(0.0, value)

    def test_transfer_mapping_keeps_post48_fields(self):
        mapping = TransferMapping(
            mapping_id="m1",
            source_domain="vision",
            target_domain="planning",
            transferable_knowledge=["edge detection"],
            transfer_quality=0.5,
        )
        self.assertEqual(
            [item.name for item in mapping.__dataclass_fields__.values()],
            [
                "mapping_id",
                "source_domain",
                "target_domain",
                "transferable_knowledge",
                "transfer_quality",
            ],
        )
        self.assertEqual(mapping.transfer_quality, 0.5)

    def test_transfer_mapping_rejects_bool_nan_infinity_and_negative_quality(self):
        for value in (True, False):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "transfer_quality"):
                    TransferMapping("m", "a", "b", ["k"], value)
        for value in (float("nan"), float("inf"), float("-inf"), -0.1, "0.5", None):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    TransferMapping("m", "a", "b", ["k"], value)

    def test_transfer_mapping_still_accepts_quality_above_one(self):
        mapping = TransferMapping("m", "a", "b", ["k"], 1.5)
        self.assertEqual(mapping.transfer_quality, 1.5)

    def test_transfer_engine_path_remains_compatible(self):
        engine = RealTransferLearningEngine()
        engine.register_source_domain("vision", ["edge detection"], 0.8)
        mapping = engine.find_transferable_knowledge(
            "vision", "planning", ["edge detection"]
        )

        self.assertIsInstance(mapping, TransferMapping)
        self.assertEqual(mapping.source_domain, "vision")
        self.assertEqual(mapping.target_domain, "planning")
        self.assertIn("edge detection", mapping.transferable_knowledge)
        self.assertGreaterEqual(mapping.transfer_quality, 0.0)
        self.assertTrue(math.isfinite(mapping.transfer_quality))
        self.assertLessEqual(mapping.transfer_quality, 1.0)

        result = engine.execute_transfer(mapping, target_baseline=0.2)
        self.assertEqual(result.source_domain, "vision")
        self.assertEqual(result.target_domain, "planning")
        self.assertEqual(result.knowledge_transferred, mapping.transferable_knowledge)
        self.assertGreaterEqual(result.performance_gain, 0.0)
        self.assertEqual(engine.total_transfers, 1)

    def test_engine_quality_may_exceed_one_when_coverage_exceeds_requirements(self):
        engine = RealTransferLearningEngine()
        knowledge = [f"shared token{i}" for i in range(5)]
        engine.register_source_domain("vision", knowledge, 1.0)
        mapping = engine.find_transferable_knowledge("vision", "planning", ["shared"])

        self.assertIsInstance(mapping, TransferMapping)
        self.assertGreater(mapping.transfer_quality, 1.0)
        self.assertEqual(len(mapping.transferable_knowledge), 5)

    @settings(max_examples=30, deadline=None)
    @given(
        st.text(
            alphabet=st.characters(min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=8,
        ),
        st.text(
            alphabet=st.characters(min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=8,
        ),
    )
    def test_transfer_knowledge_properties(self, left, right):
        engine = RealMetaLearningStrategies()
        before = engine.total_transfers
        mapping = engine.transfer_knowledge(f"src_{left}", f"tgt_{right}")

        self.assertIs(type(mapping), _meta_mapping_type())
        self.assertNotIsInstance(mapping, TransferMapping)
        self.assertGreaterEqual(mapping.transfer_strength, 0.0)
        self.assertLessEqual(mapping.transfer_strength, 1.0)
        self.assertGreaterEqual(mapping.adaptation_required, 0.0)
        self.assertLessEqual(mapping.adaptation_required, 1.0)
        self.assertAlmostEqual(
            mapping.adaptation_required + mapping.transfer_strength, 1.0
        )
        self.assertTrue(mapping.shared_features)
        self.assertTrue(
            all(isinstance(item, str) and item for item in mapping.shared_features)
        )
        self.assertEqual(engine.total_transfers, before + 1)
        self.assertIs(engine.transfer_mappings[-1], mapping)

    @settings(max_examples=25, deadline=None)
    @given(
        st.one_of(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            st.just(0),
            st.just(1),
        ),
        st.one_of(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            st.just(0),
            st.just(1),
        ),
    )
    def test_meta_mapping_accepts_finite_unit_interval(self, strength, adaptation):
        mapping = _meta_mapping(strength, adaptation, shared_features=["structure"])
        self.assertEqual(mapping.transfer_strength, strength)
        self.assertEqual(mapping.adaptation_required, adaptation)

    @settings(max_examples=20, deadline=None)
    @given(
        st.one_of(
            st.just(True),
            st.just(False),
            st.just(float("nan")),
            st.just(float("inf")),
            st.just(float("-inf")),
            st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.0 + 1e-6, max_value=10.0, allow_nan=False, allow_infinity=False),
            st.integers(max_value=-1),
            st.integers(min_value=2, max_value=8),
            st.text(min_size=1, max_size=6),
            st.none(),
        )
    )
    def test_meta_mapping_rejects_invalid_strength(self, strength):
        with self.assertRaises((TypeError, ValueError)):
            _meta_mapping(strength, 0.0)

    @settings(max_examples=20, deadline=None)
    @given(
        st.one_of(
            st.just(True),
            st.just(False),
            st.just(float("nan")),
            st.just(float("inf")),
            st.just(float("-inf")),
            st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False),
            st.text(min_size=1, max_size=6),
            st.none(),
        )
    )
    def test_transfer_mapping_rejects_invalid_quality(self, quality):
        with self.assertRaises((TypeError, ValueError)):
            TransferMapping("m", "a", "b", ["k"], quality)

    @settings(max_examples=20, deadline=None)
    @given(
        st.one_of(
            st.floats(
                min_value=0.0,
                max_value=5.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            st.integers(min_value=0, max_value=5),
        )
    )
    def test_transfer_mapping_accepts_finite_non_negative_quality(self, quality):
        mapping = TransferMapping("m", "a", "b", ["k"], quality)
        self.assertEqual(mapping.transfer_quality, quality)
        self.assertNotIsInstance(mapping.transfer_quality, bool)


if __name__ == "__main__":
    unittest.main()
