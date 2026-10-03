import unittest

from core.constitutional_brain import (
    PredictionError,
    RealPredictiveCodingSystem,
    RealRealTimeAdaptationEngine,
    RealTimeEvent,
    RealToolUse,
    ToolDefinition,
)


class ExplicitGapContractTests(unittest.TestCase):
    class _Log:
        def __init__(self):
            self.events = []

        def log_event(self, name, payload):
            self.events.append((name, payload))

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

    def test_generative_update_is_recorded_without_claiming_training(self):
        system = RealPredictiveCodingSystem()
        error = PredictionError("source", 1, "expected", "observed", 1.0, 0.5)
        update = system._update_generative_model(error, 0.08)
        self.assertEqual(update["error_id"], "source")
        self.assertEqual(update["learning_rate"], 0.08)
        self.assertFalse(update["applied"])
        self.assertEqual(system.generative_model_updates, [update])

    def test_adaptation_records_simulation_without_claiming_execution(self):
        engine = RealRealTimeAdaptationEngine()
        event = RealTimeEvent("event-1", "high_load", 1.0, {"load": 0.9}, 0.8)
        record = engine._execute_adaptation(event, "scale_up")
        self.assertEqual(record["event_id"], "event-1")
        self.assertEqual(record["adaptation_type"], "scale_up")
        self.assertFalse(record["executed"])
        self.assertEqual(engine.adaptation_history, [record])

    def test_registered_tool_without_executor_returns_structured_failure(self):
        log = self._Log()
        tool_use = RealToolUse(log)
        self.assertTrue(tool_use.registry.register_tool(ToolDefinition(
            name="future_tool",
            description="A registered future capability",
            parameters={},
            returns={"type": "object"},
        )))
        result = tool_use.execute_tool("future_tool", {})
        self.assertFalse(result.success)
        self.assertIn("not implemented", result.error)
        self.assertEqual(tool_use.failed_executions, 1)
        self.assertEqual(log.events[-1][0], "TOOL_EXECUTION_FAILED")


if __name__ == "__main__":
    unittest.main()
