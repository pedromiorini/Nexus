import unittest
from unittest.mock import patch

from core.constitutional_brain import RealToolUse


class ToolCacheKeyBoundaryContractTests(unittest.TestCase):
    class _Log:
        def __init__(self):
            self.events = []

        def log_event(self, name, payload):
            self.events.append((name, payload))

    def setUp(self):
        self.log = self._Log()
        self.tool_use = RealToolUse(self.log)

    def test_unserializable_extra_value_bypasses_cache_and_preserves_execution(self):
        params = {
            "items": [2, 1],
            "operation": "sort",
            "legacy_extension": object(),
        }
        executor_impl = self.tool_use._execute_list_operations
        with patch.object(
            self.tool_use, "_execute_list_operations", wraps=executor_impl
        ) as executor:
            for _ in range(2):
                result = self.tool_use.execute_tool("list_operations", params)
                self.assertTrue(result.success, result.error)
                self.assertEqual(result.result, [1, 2])
                self.assertFalse(result.cached)
            self.assertEqual(executor.call_count, 2)

        self.assertEqual(self.tool_use.result_cache, {})
        self.assertEqual(self.tool_use._result_cache_bytes, 0)
        cache_warnings = [
            payload
            for name, payload in self.log.events
            if name == "TOOL_CACHE_KEY_UNAVAILABLE"
        ]
        self.assertEqual(len(cache_warnings), 2)
        self.assertTrue(all(item["cache_bypassed"] for item in cache_warnings))
        self.assertTrue(all(item["reason"] == "TypeError" for item in cache_warnings))
        self.assertEqual(self.tool_use.successful_executions, 2)
        self.assertEqual(self.tool_use.cached_executions, 0)

    def test_non_cacheable_tool_does_not_serialize_parameters(self):
        tool = self.tool_use.registry.get_tool("list_operations")
        tool.cacheable = False
        with patch.object(
            self.tool_use,
            "_make_cache_key",
            side_effect=AssertionError("non-cacheable tool must skip key generation"),
        ) as make_key:
            result = self.tool_use.execute_tool(
                "list_operations", {"items": [2, 1], "operation": "sort"}
            )

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.result, [1, 2])
        self.assertFalse(result.cached)
        make_key.assert_not_called()
        self.assertEqual(self.tool_use.result_cache, {})


if __name__ == "__main__":
    unittest.main()
