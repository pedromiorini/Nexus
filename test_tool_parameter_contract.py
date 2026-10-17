import unittest
from unittest.mock import patch

from core.constitutional_brain import RealToolUse, ToolDefinition


class ToolParameterBoundaryContractTests(unittest.TestCase):
    class _Log:
        def __init__(self):
            self.events = []

        def log_event(self, name, payload):
            self.events.append((name, payload))

    def setUp(self):
        self.log = self._Log()
        self.tool_use = RealToolUse(self.log)

    @staticmethod
    def typed_tool(expected_type):
        return ToolDefinition(
            name="typed_value_test",
            description="Validate a declared parameter type",
            parameters={
                "value": {"type": expected_type, "required": True},
            },
            returns={"type": "object"},
        )

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

    def test_list_subclass_is_accepted_and_executes_without_cache(self):
        class Items(list):
            pass

        result = self.tool_use.execute_tool(
            "list_operations", {"items": Items([2, 1]), "operation": "sort"}
        )

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.result, [1, 2])
        self.assertFalse(result.cached)
        self.assertEqual(self.tool_use.successful_executions, 1)
        self.assertEqual(self.tool_use.failed_executions, 0)
        self.assertEqual(self.tool_use.cached_executions, 0)
        self.assertEqual(self.tool_use.result_cache, {})
        warnings = [
            payload
            for name, payload in self.log.events
            if name == "TOOL_CACHE_KEY_UNAVAILABLE"
        ]
        self.assertEqual(len(warnings), 1)
        self.assertTrue(warnings[0]["cache_bypassed"])

    def test_builtin_scalar_subclass_is_accepted_but_bool_is_not_an_int(self):
        class Count(int):
            pass

        tool = self.typed_tool("int")
        self.assertTrue(
            self.tool_use.validator.validate_parameters(tool, {"value": Count(7)})
        )
        self.assertFalse(
            self.tool_use.validator.validate_parameters(tool, {"value": True})
        )
        event_name, payload = self.log.events[-1]
        self.assertEqual(event_name, "TOOL_PARAM_TYPE_MISMATCH")
        self.assertEqual(payload["expected"], "int")
        self.assertEqual(payload["actual"], "bool")

    def test_builtin_type_name_spoof_is_not_accepted_as_that_type(self):
        fake_list = type("list", (), {})()

        self.assertFalse(
            self.tool_use.validator.validate_parameters(
                self.typed_tool("list"), {"value": fake_list}
            )
        )
        event_name, payload = self.log.events[-1]
        self.assertEqual(event_name, "TOOL_PARAM_TYPE_MISMATCH")
        self.assertEqual(payload["expected"], "list")
        self.assertEqual(payload["actual"], "list")


if __name__ == "__main__":
    unittest.main()
