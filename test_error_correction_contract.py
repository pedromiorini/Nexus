import unittest

from core.constitutional_brain import DetectedError, RealErrorDetectionCorrectionEngine


class ErrorCorrectionContractTests(unittest.TestCase):
    def setUp(self):
        self.engine = RealErrorDetectionCorrectionEngine.__new__(RealErrorDetectionCorrectionEngine)

    def test_malformed_bounds_preserve_original_content(self):
        below = DetectedError("e1", "constraint", 0.5, "below minimum not-a-number", "field")
        above = DetectedError("e2", "constraint", 0.5, "above maximum not-a-number", "field")
        self.assertEqual(self.engine._generate_correction(below, "original", "clamp_to_range"), "original")
        self.assertEqual(self.engine._generate_correction(above, "original", "clamp_to_range"), "original")

    def test_valid_bounds_are_clamped(self):
        below = DetectedError("e1", "constraint", 0.5, "below minimum 1.5", "field")
        above = DetectedError("e2", "constraint", 0.5, "above maximum 9.5", "field")
        self.assertEqual(self.engine._generate_correction(below, "original", "clamp_to_range"), 1.5)
        self.assertEqual(self.engine._generate_correction(above, "original", "clamp_to_range"), 9.5)


if __name__ == "__main__":
    unittest.main()
