import hashlib
import json
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

    def test_small_cache_keys_remain_byte_compatible_and_cacheable(self):
        params = {"items": [2, 1], "operation": "sort"}
        legacy_payload = json.dumps(params, sort_keys=True)
        expected_key = hashlib.sha256(
            f"list_operations:{legacy_payload}".encode()
        ).hexdigest()

        self.assertEqual(
            self.tool_use._make_cache_key("list_operations", params), expected_key
        )
        first = self.tool_use.execute_tool("list_operations", params)
        second = self.tool_use.execute_tool("list_operations", params)

        self.assertTrue(first.success, first.error)
        self.assertTrue(second.success, second.error)
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(second.result, [1, 2])

    def test_dict_subclass_remains_accepted_when_cache_preflight_bypasses(self):
        class Params(dict):
            pass

        result = self.tool_use.execute_tool(
            "list_operations", Params(items=[2, 1], operation="sort")
        )

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.result, [1, 2])
        self.assertFalse(result.cached)
        warning = next(
            payload
            for name, payload in self.log.events
            if name == "TOOL_CACHE_KEY_UNAVAILABLE"
        )
        self.assertEqual(warning["reason"], "TypeError")
        self.assertTrue(warning["cache_bypassed"])

    @staticmethod
    def _nested_lists(depth):
        value = "leaf"
        for _ in range(depth):
            value = [value]
        return value

    def test_over_budget_cache_keys_bypass_json_without_blocking_executor(self):
        cycle = []
        cycle.append(cycle)
        cases = (
            (
                "byte_limit",
                "\n" * (RealToolUse.MAX_CACHE_KEY_BYTES // 2 - 3),
            ),
            (
                "depth_limit",
                self._nested_lists(RealToolUse.MAX_CACHE_KEY_DEPTH + 1),
            ),
            (
                "node_limit",
                [[None] * 1000 for _ in range(5)],
            ),
            (
                "container_item_limit",
                [None] * (RealToolUse.MAX_CACHE_KEY_CONTAINER_ITEMS + 1),
            ),
            ("cycle", cycle),
        )

        for expected_reason, extension in cases:
            with self.subTest(reason=expected_reason):
                log = self._Log()
                tool_use = RealToolUse(log)
                params = {
                    "items": [2, 1],
                    "operation": "sort",
                    "legacy_extension": extension,
                }
                executor_impl = tool_use._execute_list_operations
                with patch.object(
                    tool_use, "_execute_list_operations", wraps=executor_impl
                ) as executor, patch(
                    "core.constitutional_brain.json.dumps",
                    side_effect=AssertionError("over-budget params reached JSON encoder"),
                ) as encoder:
                    results = [
                        tool_use.execute_tool("list_operations", params)
                        for _ in range(2)
                    ]

                encoder.assert_not_called()
                self.assertEqual(executor.call_count, 2)
                self.assertTrue(all(result.success for result in results))
                self.assertTrue(all(result.result == [1, 2] for result in results))
                self.assertTrue(all(not result.cached for result in results))
                self.assertEqual(tool_use.successful_executions, 2)
                self.assertEqual(tool_use.cached_executions, 0)
                self.assertEqual(tool_use.result_cache, {})
                self.assertEqual(tool_use._result_cache_bytes, 0)
                warnings = [
                    payload
                    for name, payload in log.events
                    if name == "TOOL_CACHE_KEY_UNAVAILABLE"
                ]
                self.assertEqual(len(warnings), 2)
                self.assertTrue(all(item["cache_bypassed"] for item in warnings))
                self.assertTrue(
                    all(item["reason"] == expected_reason for item in warnings)
                )


if __name__ == "__main__":
    unittest.main()
