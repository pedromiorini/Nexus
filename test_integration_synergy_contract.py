import unittest

from core.constitutional_brain import RealIntegrationOrchestrationEngine


class IntegrationSynergyContractTests(unittest.TestCase):
    def test_average_synergy_uses_all_integrations(self):
        engine = RealIntegrationOrchestrationEngine()
        first = engine.integrate_modules("reasoning", "memory")
        second = engine.integrate_modules("unrelated_source", "unrelated_target")

        self.assertEqual(first.synergy_score, 0.9)
        self.assertEqual(second.synergy_score, 0.6)
        self.assertEqual(engine.total_integrations, 2)
        self.assertAlmostEqual(engine.avg_synergy_score, 0.75)

    def test_empty_engine_has_zero_average(self):
        engine = RealIntegrationOrchestrationEngine()
        self.assertEqual(engine.avg_synergy_score, 0.0)


if __name__ == "__main__":
    unittest.main()
