import copy
import unittest
from unittest.mock import patch

from core.constitutional_brain import RealToolUse


class ToolCacheHitFallbackContractTests(unittest.TestCase):
    class _Log:
        def __init__(self):
            self.events = []

        def log_event(self, name, payload):
            self.events.append((name, payload))

    def setUp(self):
        self.log = self._Log()
        self.tool_use = RealToolUse(self.log)

    @staticmethod
    def _copy_with_failures(failing_calls):
        real_deepcopy = copy.deepcopy
        calls = 0

        def copy_result(value, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls in failing_calls:
                raise RecursionError("injected cache copy failure")
            return real_deepcopy(value, *args, **kwargs)

        return copy_result

    def test_hit_copy_failure_falls_through_to_executor_without_hit_stats(self):
        params = {"items": [3, 2, 1], "operation": "sort"}
        executor_impl = self.tool_use._execute_list_operations
        copy_result = self._copy_with_failures({2})

        with patch.object(
            self.tool_use, "_execute_list_operations", wraps=executor_impl
        ) as executor, patch(
            "core.constitutional_brain.copy.deepcopy", side_effect=copy_result
        ) as deep_copy:
            first = self.tool_use.execute_tool("list_operations", params)
            second = self.tool_use.execute_tool("list_operations", params)

        self.assertEqual(executor.call_count, 2)
        self.assertEqual(deep_copy.call_count, 3)
        self.assertTrue(first.success, first.error)
        self.assertTrue(second.success, second.error)
        self.assertEqual(first.result, [1, 2, 3])
        self.assertEqual(second.result, [1, 2, 3])
        self.assertFalse(first.cached)
        self.assertFalse(second.cached)
        self.assertEqual(self.tool_use.total_executions, 2)
        self.assertEqual(self.tool_use.successful_executions, 2)
        self.assertEqual(self.tool_use.failed_executions, 0)
        self.assertEqual(self.tool_use.cached_executions, 0)
        self.assertEqual(self.tool_use.registry.execution_count["list_operations"], 2)

        hit_warnings = [
            payload
            for name, payload in self.log.events
            if name == "TOOL_CACHE_HIT_UNAVAILABLE"
        ]
        self.assertEqual(len(hit_warnings), 1)
        self.assertEqual(hit_warnings[0]["tool"], "list_operations")
        self.assertEqual(hit_warnings[0]["reason"], "RecursionError")
        self.assertTrue(hit_warnings[0]["cache_bypassed"])
        self.assertFalse(
            any(name == "TOOL_EXECUTION_FAILED" for name, _ in self.log.events)
        )

    def test_failed_hit_copy_evicts_entry_even_if_recache_copy_fails(self):
        params = {"items": [3, 2, 1], "operation": "sort"}
        executor_impl = self.tool_use._execute_list_operations
        copy_result = self._copy_with_failures({2, 3})

        with patch.object(
            self.tool_use, "_execute_list_operations", wraps=executor_impl
        ) as executor, patch(
            "core.constitutional_brain.copy.deepcopy", side_effect=copy_result
        ) as deep_copy:
            first = self.tool_use.execute_tool("list_operations", params)
            second = self.tool_use.execute_tool("list_operations", params)

        self.assertEqual(executor.call_count, 2)
        self.assertEqual(deep_copy.call_count, 3)
        self.assertTrue(first.success, first.error)
        self.assertTrue(second.success, second.error)
        self.assertEqual(second.result, [1, 2, 3])
        self.assertFalse(second.cached)
        self.assertEqual(self.tool_use.successful_executions, 2)
        self.assertEqual(self.tool_use.failed_executions, 0)
        self.assertEqual(self.tool_use.cached_executions, 0)
        self.assertEqual(self.tool_use.result_cache, {})
        self.assertEqual(self.tool_use._result_cache_bytes, 0)

        warnings = [
            (name, payload)
            for name, payload in self.log.events
            if name in {
                "TOOL_CACHE_HIT_UNAVAILABLE",
                "TOOL_CACHE_STORE_UNAVAILABLE",
            }
        ]
        self.assertEqual(
            [name for name, _ in warnings],
            ["TOOL_CACHE_HIT_UNAVAILABLE", "TOOL_CACHE_STORE_UNAVAILABLE"],
        )
        self.assertTrue(all(payload["cache_bypassed"] for _, payload in warnings))
        self.assertFalse(
            any(name == "TOOL_EXECUTION_FAILED" for name, _ in self.log.events)
        )


if __name__ == "__main__":
    unittest.main()
