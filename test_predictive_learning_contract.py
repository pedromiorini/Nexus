"""Contrato de aprendizado do nível sensorial (critérios PC1–PC3 de docs/cognitivo/AUDITORIA_2026-10-22.md)."""
import contextlib
import io
import unittest

from core.cognitivo.predictive_coding import RealPredictiveCodingSystem
from tools.cognitivo_eval import eval_predictive


def _system():
    with contextlib.redirect_stdout(io.StringIO()):
        return RealPredictiveCodingSystem()


class PredictiveLearningContractTests(unittest.TestCase):
    def test_preregistered_criteria_pass(self):
        checks = {c["id"]: c for c in eval_predictive()}
        for cid in ("PC1", "PC2", "PC3", "PC4"):
            self.assertTrue(checks[cid]["passed"], checks[cid])

    def test_initial_prediction_is_unchanged_before_any_observation(self):
        pred = _system().generate_prediction(0, {})
        self.assertEqual((pred.predicted_value, pred.confidence), ("sensory_features", 0.5))

    def test_level_zero_learns_observed_symbol_with_calibrated_confidence(self):
        system = _system()
        for _ in range(3):
            system.hierarchical_prediction_cycle({"sensory": "x"})
        result = system.hierarchical_prediction_cycle({"sensory": "x"})
        self.assertEqual(result["predictions"][-1].predicted_value, "x")
        self.assertAlmostEqual(result["predictions"][-1].confidence, 1.0)
        self.assertEqual(result["errors"][0].error_magnitude, 0.0)

    def test_distribution_is_normalized_and_tracks_change(self):
        system = _system()
        for _ in range(50):
            system.hierarchical_prediction_cycle({"sensory": "a"})
        for _ in range(30):
            system.hierarchical_prediction_cycle({"sensory": "b"})
        dist = system.sensory_distribution()
        self.assertAlmostEqual(sum(dist.values()), 1.0)
        self.assertGreater(dist["b"], dist["a"])
        self.assertEqual(system.generate_prediction(0, {}).predicted_value, "b")

    def test_higher_levels_keep_previous_behavior(self):
        system = _system()
        system.hierarchical_prediction_cycle({"sensory": "a"})
        pred = system.generate_prediction(1, {})
        self.assertEqual((pred.predicted_value, pred.confidence), ("detected_features", 0.6))


if __name__ == "__main__":
    unittest.main()
