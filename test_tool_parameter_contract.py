import unittest
from unittest.mock import patch

from core.constitutional_brain import RealToolUse


class ToolParameterBoundaryContractTests(unittest.TestCase):
    class _Log:
        def __init__(self):
            self.events = []

        def log_event(self, name, payload):
            self.events.append((name, payload))

    def setUp(self):
        self.log = self._Log()
        self.tool_use = RealToolUse(self.log)

    def test_non_dict_params_return_structured_failure_before_executor(self):
        invalid_params = (None, [], "itemsoperation")
        with patch.object(self.tool_use, "_execute_list_operations") as executor:
            for params in invalid_params:
                with self.subTest(params_type=type(params).__name__):
                    result = self.tool_use.execute_tool("list_operations", params)
                    self.assertFalse(result.success)
                    self.assertEqual(result.error, "Parameter validation failed")
                    self.assertIsNone(result.result)
            executor.assert_not_called()

        failures = [
            payload
            for name, payload in self.log.events
            if name == "TOOL_PARAM_TYPE_MISMATCH"
        ]
        self.assertEqual(len(failures), len(invalid_params))
        self.assertTrue(all(payload["parameter"] == "params" for payload in failures))
        self.assertTrue(all(payload["expected"] == "dict" for payload in failures))
        self.assertEqual(
            [payload["actual"] for payload in failures],
            ["NoneType", "list", "str"],
        )
        self.assertEqual(self.tool_use.failed_executions, len(invalid_params))

    def test_empty_dict_still_reports_missing_required_parameter(self):
        result = self.tool_use.execute_tool("list_operations", {})

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Parameter validation failed")
        event_name, payload = self.log.events[-1]
        self.assertEqual(event_name, "TOOL_PARAM_MISSING")
        self.assertEqual(payload["tool"], "list_operations")
        self.assertEqual(payload["parameter"], "items")
        self.assertEqual(self.tool_use.failed_executions, 1)

    def test_valid_dict_params_continue_to_execute(self):
        result = self.tool_use.execute_tool(
            "list_operations", {"items": [2, 1], "operation": "sort"}
        )

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.result, [1, 2])
        self.assertEqual(self.tool_use.successful_executions, 1)
        self.assertEqual(self.tool_use.failed_executions, 0)


if __name__ == "__main__":
    unittest.main()
