import unittest

from core.constitutional_brain import RealHardwareAwareSelfOptimization


class HardwareTelemetryContractTests(unittest.TestCase):
    def test_gpu_reader_returns_conservative_zero_without_backend(self):
        optimizer = RealHardwareAwareSelfOptimization()
        value = optimizer._read_gpu_percent()
        self.assertEqual(value, 0.0)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 100.0)


if __name__ == "__main__":
    unittest.main()
