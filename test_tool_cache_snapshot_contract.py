import unittest
from unittest.mock import patch

from core.constitutional_brain import RealToolUse


class ToolCacheSnapshotContractTests(unittest.TestCase):
    class _Log:
        def __init__(self):
            self.events = []

        def log_event(self, name, payload):
            self.events.append((name, payload))

    def setUp(self):
        self.log = self._Log()
        self.tool_use = RealToolUse(self.log)

    def test_key_validation_and_executor_share_snapshot_if_source_changes(self):
        params = {"items": [2, 1], "operation": "sort"}
        original_items = params["items"]
        captured = {}
        key_impl = self.tool_use._make_cache_key
        validator_impl = self.tool_use.validator.validate_parameters
        executor_impl = self.tool_use._execute_list_operations

        def make_key(tool_name, snapshot, **kwargs):
            captured["key_params"] = snapshot
            key = key_impl(tool_name, snapshot, **kwargs)
            # Simulate the caller mutating its original object after validation
            # and cache-key preparation but before execution.
            params["items"] = "mutated-after-snapshot"
            return key

        def validate(tool, snapshot):
            captured["validation_params"] = snapshot
            return validator_impl(tool, snapshot)

        def execute(snapshot):
            captured["executor_params"] = snapshot
            result = executor_impl(snapshot)
            snapshot["items"].append(99)
            return result

        with patch.object(
            self.tool_use, "_make_cache_key", side_effect=make_key
        ) as make_key_spy, patch.object(
            self.tool_use.validator, "validate_parameters", side_effect=validate
        ), patch.object(
            self.tool_use, "_execute_list_operations", side_effect=execute
        ) as executor:
            first = self.tool_use.execute_tool("list_operations", params)

        snapshot = captured["key_params"]
        self.assertIsNot(snapshot, params)
        self.assertIsNot(snapshot["items"], original_items)
        self.assertIs(captured["validation_params"], snapshot)
        self.assertIs(captured["executor_params"], snapshot)
        self.assertEqual(params["items"], "mutated-after-snapshot")
        self.assertEqual(original_items, [2, 1])
        self.assertEqual(snapshot["items"], [2, 1, 99])
        self.assertTrue(first.success, first.error)
        self.assertEqual(first.result, [1, 2])
        self.assertFalse(first.cached)
        make_key_spy.assert_called_once()
        executor.assert_called_once()

        second = self.tool_use.execute_tool(
            "list_operations", {"items": [2, 1], "operation": "sort"}
        )
        self.assertTrue(second.success, second.error)
        self.assertEqual(second.result, [1, 2])
        self.assertTrue(second.cached)
        self.assertEqual(self.tool_use.successful_executions, 2)
        self.assertEqual(self.tool_use.failed_executions, 0)
        self.assertEqual(self.tool_use.cached_executions, 1)
        self.assertEqual(self.tool_use.registry.execution_count["list_operations"], 1)

    def test_shared_input_aliases_are_copied_as_independent_values(self):
        shared = [1, 2]
        params = {
            "items": [3, 4],
            "operation": "sort",
            "extension_one": shared,
            "extension_two": shared,
        }
        captured = {}
        key_impl = self.tool_use._make_cache_key

        def make_key(tool_name, snapshot, **kwargs):
            captured["snapshot"] = snapshot
            return key_impl(tool_name, snapshot, **kwargs)

        with patch.object(self.tool_use, "_make_cache_key", side_effect=make_key):
            result = self.tool_use.execute_tool("list_operations", params)

        snapshot = captured["snapshot"]
        self.assertTrue(result.success, result.error)
        self.assertEqual(snapshot["extension_one"], [1, 2])
        self.assertEqual(snapshot["extension_two"], [1, 2])
        self.assertIsNot(snapshot["extension_one"], shared)
        self.assertIsNot(snapshot["extension_two"], shared)
        self.assertIsNot(snapshot["extension_one"], snapshot["extension_two"])

    def test_non_json_values_bypass_cache_without_changing_executor_input(self):
        cases = (
            ("tuple", ("opaque", 1), "TypeError"),
            ("non-string key", {"nested": {1: "opaque"}}, "TypeError"),
            ("non-finite float", float("nan"), "non_finite_float"),
        )
        for label, extension, expected_reason in cases:
            with self.subTest(value=label):
                log = self._Log()
                tool_use = RealToolUse(log)
                params = {
                    "items": [2, 1],
                    "operation": "sort",
                    "legacy_extension": extension,
                }
                executor_impl = tool_use._execute_list_operations
                observed = {}

                def execute(original_params):
                    observed["params"] = original_params
                    return executor_impl(original_params)

                with patch.object(
                    tool_use, "_execute_list_operations", side_effect=execute
                ) as executor:
                    result = tool_use.execute_tool("list_operations", params)

                self.assertTrue(result.success, result.error)
                self.assertEqual(result.result, [1, 2])
                self.assertFalse(result.cached)
                self.assertIs(observed["params"], params)
                self.assertIs(observed["params"]["legacy_extension"], extension)
                executor.assert_called_once()
                self.assertEqual(tool_use.result_cache, {})
                warning = next(
                    payload
                    for name, payload in log.events
                    if name == "TOOL_CACHE_KEY_UNAVAILABLE"
                )
                self.assertEqual(warning["reason"], expected_reason)
                self.assertTrue(warning["cache_bypassed"])
                self.assertEqual(tool_use.successful_executions, 1)
                self.assertEqual(tool_use.failed_executions, 0)

    def test_builtin_subclass_bypasses_snapshot_without_losing_type_contract(self):
        class ItemList(list):
            pass

        log = self._Log()
        tool_use = RealToolUse(log)
        params = {"items": ItemList([2, 1]), "operation": "sort"}
        executor_impl = tool_use._execute_list_operations
        observed = {}

        def execute(original_params):
            observed["params"] = original_params
            return executor_impl(original_params)

        with patch.object(
            tool_use, "_execute_list_operations", side_effect=execute
        ) as executor:
            result = tool_use.execute_tool("list_operations", params)

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.result, [1, 2])
        self.assertFalse(result.cached)
        self.assertIs(observed["params"], params)
        self.assertIsInstance(observed["params"]["items"], ItemList)
        executor.assert_called_once()
        warning = next(
            payload
            for name, payload in log.events
            if name == "TOOL_CACHE_KEY_UNAVAILABLE"
        )
        self.assertEqual(warning["reason"], "TypeError")


if __name__ == "__main__":
    unittest.main()
