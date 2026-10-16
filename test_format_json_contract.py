import json
import unittest

from core.constitutional_brain import RealToolUse


class FormatJSONStrictContractTests(unittest.TestCase):
    class _Log:
        def __init__(self):
            self.events = []

        def log_event(self, name, payload):
            self.events.append((name, payload))

    def setUp(self):
        self.log = self._Log()
        self.tool_use = RealToolUse(self.log)

    def test_finite_values_remain_pretty_printed_valid_json(self):
        data = {"z": 3, "a": {"items": [True, None, 1.25]}}

        result = self.tool_use.execute_tool("format_json", {"data": data})

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.result, json.dumps(data, indent=2, sort_keys=True))
        self.assertEqual(json.loads(result.result), data)
        self.assertEqual(self.tool_use.failed_executions, 0)

    def test_non_finite_numbers_return_structured_failure_without_caching(self):
        invalid_values = (float("nan"), float("inf"), float("-inf"))
        for value in invalid_values:
            with self.subTest(value=repr(value)):
                result = self.tool_use.execute_tool(
                    "format_json", {"data": {"nested": [value]}}
                )
                self.assertFalse(result.success)
                self.assertIsNone(result.result)
                self.assertTrue(result.error)
                self.assertFalse(result.cached)

        failures = [
            payload
            for name, payload in self.log.events
            if name == "TOOL_EXECUTION_FAILED" and payload.get("tool") == "format_json"
        ]
        self.assertEqual(len(failures), len(invalid_values))
        self.assertEqual(self.tool_use.failed_executions, len(invalid_values))
        self.assertEqual(self.tool_use.successful_executions, 0)
        self.assertEqual(self.tool_use.result_cache, {})
        self.assertEqual(self.tool_use._result_cache_bytes, 0)

    def test_non_serializable_data_returns_structured_failure(self):
        result = self.tool_use.execute_tool(
            "format_json", {"data": {"nested": [object()]}}
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertTrue(result.error)
        event_names = [name for name, _ in self.log.events]
        self.assertIn("TOOL_CACHE_KEY_UNAVAILABLE", event_names)
        self.assertIn("TOOL_EXECUTION_FAILED", event_names)
        self.assertEqual(self.tool_use.failed_executions, 1)


if __name__ == "__main__":
    unittest.main()
