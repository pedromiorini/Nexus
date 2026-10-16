import unittest
from unittest.mock import patch

from core.constitutional_brain import RealToolUse, ToolDefinition


class ToolDefinitionTimeoutContractTests(unittest.TestCase):
    class _Log:
        def __init__(self):
            self.events = []

        def log_event(self, name, payload):
            self.events.append((name, payload))

    def setUp(self):
        self.log = self._Log()
        self.tool_use = RealToolUse(self.log)

    @staticmethod
    def make_tool(name, timeout_seconds):
        return ToolDefinition(
            name=name,
            description="A safe registered capability",
            parameters={},
            returns={"type": "object"},
            timeout_seconds=timeout_seconds,
        )

    def test_timeout_must_be_an_integer_from_one_through_sixty(self):
        invalid_timeouts = (
            0,
            -1,
            61,
            True,
            1.5,
            float("nan"),
            float("inf"),
            "30",
            None,
        )
        for index, timeout in enumerate(invalid_timeouts):
            name = f"invalid_timeout_{index}"
            with self.subTest(timeout=timeout):
                self.assertFalse(
                    self.tool_use.registry.register_tool(
                        self.make_tool(name, timeout)
                    )
                )
                self.assertNotIn(name, self.tool_use.registry.tools)
                event_name, payload = self.log.events[-1]
                self.assertEqual(event_name, "TOOL_VALIDATION_FAILED")
                self.assertEqual(payload["tool"], name)
                self.assertEqual(
                    payload["timeout_type"], type(timeout).__name__
                )

    def test_inclusive_timeout_boundaries_are_valid(self):
        for timeout in (1, 60):
            name = f"boundary_timeout_{timeout}"
            with self.subTest(timeout=timeout):
                self.assertTrue(
                    self.tool_use.registry.register_tool(
                        self.make_tool(name, timeout)
                    )
                )
                self.assertIn(name, self.tool_use.registry.tools)

    def test_mutated_timeout_is_rejected_before_executor_runs(self):
        tool = self.tool_use.registry.get_tool("list_operations")
        tool.timeout_seconds = 0
        with patch.object(
            self.tool_use, "_execute_list_operations", side_effect=AssertionError
        ):
            result = self.tool_use.execute_tool(
                "list_operations", {"items": [2, 1], "operation": "sort"}
            )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Tool timeout validation failed")
        self.assertEqual(self.tool_use.failed_executions, 1)

    def test_over_budget_executor_logs_completion_without_cancelling_result(self):
        tool = self.tool_use.registry.get_tool("list_operations")
        tool.timeout_seconds = 1
        tool.cacheable = False
        with (
            patch.object(
                self.tool_use, "_execute_list_operations", return_value=[1, 2]
            ),
            patch(
                "core.constitutional_brain.time.monotonic",
                side_effect=(10.0, 11.25),
            ),
        ):
            result = self.tool_use.execute_tool(
                "list_operations", {"items": [2, 1], "operation": "sort"}
            )
        self.assertTrue(result.success)
        self.assertEqual(result.result, [1, 2])
        overruns = [
            payload
            for name, payload in self.log.events
            if name == "TOOL_EXECUTION_OVERRUN"
        ]
        self.assertEqual(len(overruns), 1)
        self.assertEqual(overruns[0]["timeout_seconds"], 1)
        self.assertEqual(overruns[0]["executor_duration_seconds"], 1.25)
        self.assertTrue(overruns[0]["execution_completed"])
        self.assertFalse(overruns[0]["cancelled"])
        self.assertEqual(self.tool_use.successful_executions, 1)

    def test_over_budget_executor_exception_still_logs_overrun(self):
        tool = self.tool_use.registry.get_tool("list_operations")
        tool.timeout_seconds = 1
        tool.cacheable = False
        with (
            patch.object(
                self.tool_use,
                "_execute_list_operations",
                side_effect=ValueError("executor failed"),
            ),
            patch(
                "core.constitutional_brain.time.monotonic",
                side_effect=(30.0, 31.5),
            ),
        ):
            result = self.tool_use.execute_tool(
                "list_operations", {"items": [2, 1], "operation": "sort"}
            )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "executor failed")
        overruns = [
            payload
            for name, payload in self.log.events
            if name == "TOOL_EXECUTION_OVERRUN"
        ]
        self.assertEqual(len(overruns), 1)
        self.assertEqual(overruns[0]["executor_duration_seconds"], 1.5)
        self.assertTrue(overruns[0]["execution_completed"])
        self.assertFalse(overruns[0]["cancelled"])
        self.assertEqual(self.tool_use.failed_executions, 1)

    def test_executor_within_budget_does_not_log_overrun(self):
        tool = self.tool_use.registry.get_tool("list_operations")
        tool.timeout_seconds = 1
        tool.cacheable = False
        with (
            patch.object(
                self.tool_use, "_execute_list_operations", return_value=[1, 2]
            ),
            patch(
                "core.constitutional_brain.time.monotonic",
                side_effect=(10.0, 10.999),
            ),
        ):
            result = self.tool_use.execute_tool(
                "list_operations", {"items": [2, 1], "operation": "sort"}
            )
        self.assertTrue(result.success)
        self.assertFalse(
            any(name == "TOOL_EXECUTION_OVERRUN" for name, _ in self.log.events)
        )


if __name__ == "__main__":
    unittest.main()
