import unittest

from core.constitutional_brain import (
    PredictionError,
    RealPredictiveCodingSystem,
    RealRealTimeAdaptationEngine,
    RealTimeEvent,
)


class ExplicitGapContractTests(unittest.TestCase):
    def test_prediction_error_is_copied_to_target_level(self):
        system = RealPredictiveCodingSystem()
        error = PredictionError("source", 0, "expected", "observed", 1.0, 0.5)
        propagated = system._propagate_error_up(error, 2)
        self.assertEqual(propagated.level, 2)
        self.assertNotEqual(propagated.error_id, error.error_id)
        self.assertEqual(system.errors[2], [propagated])
        self.assertEqual(error.level, 0)

    def test_invalid_prediction_target_is_rejected(self):
        system = RealPredictiveCodingSystem()
        error = PredictionError("source", 0, "expected", "observed", 1.0, 0.5)
        with self.assertRaises(ValueError):
            system._propagate_error_up(error, 99)

    def test_adaptation_records_simulation_without_claiming_execution(self):
        engine = RealRealTimeAdaptationEngine()
        event = RealTimeEvent("event-1", "high_load", 1.0, {"load": 0.9}, 0.8)
        record = engine._execute_adaptation(event, "scale_up")
        self.assertEqual(record["event_id"], "event-1")
        self.assertEqual(record["adaptation_type"], "scale_up")
        self.assertFalse(record["executed"])
        self.assertEqual(engine.adaptation_history, [record])


if __name__ == "__main__":
    unittest.main()
