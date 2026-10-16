import unittest
from unittest.mock import patch

from core.constitutional_brain import RealToolUse


class ToolCacheStoreFallbackContractTests(unittest.TestCase):
    class _Log:
        def __init__(self):
            self.events = []

        def log_event(self, name, payload):
            self.events.append((name, payload))

    def test_copy_failure_bypasses_cache_and_preserves_executor_success(self):
        log = self._Log()
        tool_use = RealToolUse(log)
        expected_result = {"payload": [1, 2, 3]}
        params = {"items": [3, 2, 1], "operation": "reverse"}

        with patch.object(
            tool_use, "_execute_list_operations", return_value=expected_result
        ) as executor, patch(
            "core.constitutional_brain.copy.deepcopy",
            side_effect=RecursionError("maximum recursion depth exceeded"),
        ) as deep_copy:
            result = tool_use.execute_tool("list_operations", params)

        executor.assert_called_once_with(params)
        deep_copy.assert_called_once()
        self.assertTrue(result.success, result.error)
        self.assertIs(result.result, expected_result)
        self.assertFalse(result.cached)
        self.assertEqual(tool_use.successful_executions, 1)
        self.assertEqual(tool_use.failed_executions, 0)
        self.assertEqual(tool_use.registry.execution_count["list_operations"], 1)
        self.assertEqual(tool_use.result_cache, {})
        self.assertEqual(tool_use._result_cache_bytes, 0)

        store_warnings = [
            payload
            for name, payload in log.events
            if name == "TOOL_CACHE_STORE_UNAVAILABLE"
        ]
        self.assertEqual(len(store_warnings), 1)
        self.assertEqual(store_warnings[0]["tool"], "list_operations")
        self.assertEqual(store_warnings[0]["reason"], "RecursionError")
        self.assertTrue(store_warnings[0]["cache_bypassed"])
        self.assertFalse(
            any(name == "TOOL_EXECUTION_FAILED" for name, _ in log.events)
        )
        self.assertTrue(
            any(name == "TOOL_EXECUTION_SUCCESS" for name, _ in log.events)
        )


if __name__ == "__main__":
    unittest.main()
