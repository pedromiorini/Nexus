import unittest

from core.constitutional_brain import RealMultimodalSystem, RealSensorimotorIntegrationLayer


class SensorimotorMultimodalContractTests(unittest.TestCase):
    def test_multimodal_processor_is_used_when_available(self):
        layer = RealSensorimotorIntegrationLayer(multimodal=RealMultimodalSystem())
        processed = layer._process_modality("vision", {"frame": "sample"})
        self.assertEqual(processed["modality"], "vision")
        self.assertEqual(processed["source"], "multimodal_processor")
        self.assertIn("objects_detected", processed["features"])
        self.assertGreater(processed["confidence"], 0.0)

    def test_heuristic_fallback_remains_available(self):
        layer = RealSensorimotorIntegrationLayer()
        processed = layer._process_modality("unknown", {"feature": 1})
        self.assertEqual(processed["modality"], "unknown")
        self.assertEqual(processed["features"], ["feature"])
        self.assertNotIn("source", processed)


if __name__ == "__main__":
    unittest.main()
