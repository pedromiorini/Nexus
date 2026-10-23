import unittest
from unittest.mock import patch

from core.constitutional_brain import RealToolUse


class ToolPermissionFailClosedContractTests(unittest.TestCase):
    class _Log:
        def __init__(self):
            self.events = []

        def log_event(self, name, payload):
            self.events.append((name, payload))

    def setUp(self):
        self.log = self._Log()
        self.tool_use = RealToolUse(self.log)

    def _arm_list_operations(self, *, requires_permission, cacheable):
        tool = self.tool_use.registry.get_tool("list_operations")
        tool.requires_permission = requires_permission
        tool.cacheable = cacheable
        return tool

    def test_ts1a_missing_grant_does_not_call_executor(self):
        self._arm_list_operations(requires_permission=True, cacheable=False)
        with patch.object(
            self.tool_use, "_execute_list_operations", return_value=[1, 2]
        ) as executor:
            result = self.tool_use.execute_tool(
                "list_operations", {"items": [2, 1], "operation": "sort"}
            )
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        folded = result.error.lower()
        self.assertIn("permission", folded)
        self.assertIn("grant", folded)
        executor.assert_not_called()
        self.assertEqual(self.tool_use.failed_executions, 1)
        denials = [
            payload
            for name, payload in self.log.events
            if name == "TOOL_PERMISSION_DENIED"
        ]
        self.assertEqual(len(denials), 1)
        self.assertEqual(denials[0]["tool"], "list_operations")

    def test_ts1b_explicit_grant_allows_executor(self):
        self._arm_list_operations(requires_permission=True, cacheable=False)
        self.tool_use.grant_permission("list_operations")
        with patch.object(
            self.tool_use, "_execute_list_operations", return_value=[1, 2]
        ) as executor:
            result = self.tool_use.execute_tool(
                "list_operations", {"items": [2, 1], "operation": "sort"}
            )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.result, [1, 2])
        executor.assert_called_once()

    def test_ts1c_revoke_blocks_cached_success(self):
        self._arm_list_operations(requires_permission=True, cacheable=True)
        params = {"items": [2, 1], "operation": "sort"}
        self.tool_use.grant_permission("list_operations")
        with patch.object(
            self.tool_use, "_execute_list_operations", return_value=[1, 2]
        ) as executor:
            first = self.tool_use.execute_tool("list_operations", params)
            self.assertTrue(first.success, first.error)
            self.assertEqual(executor.call_count, 1)
            self.tool_use.revoke_permission("list_operations")
            second = self.tool_use.execute_tool("list_operations", params)
        self.assertFalse(second.success)
        folded = second.error.lower()
        self.assertIn("permission", folded)
        self.assertIn("grant", folded)
        self.assertFalse(second.cached)
        self.assertEqual(executor.call_count, 1)

    def test_ts1d_empty_name_is_not_a_permission_error(self):
        with patch.object(
            self.tool_use, "_execute_list_operations", return_value=[1, 2]
        ) as executor:
            result = self.tool_use.execute_tool("", {"items": [2, 1], "operation": "sort"})
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Tool '' not found")
        self.assertNotIn("permission", result.error.lower())
        executor.assert_not_called()

    def test_ts1d_unregistered_name_is_not_a_permission_error(self):
        with patch.object(
            self.tool_use, "_execute_list_operations", return_value=[1, 2]
        ) as executor:
            result = self.tool_use.execute_tool("missing_tool", {})
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Tool 'missing_tool' not found")
        self.assertNotIn("permission", result.error.lower())
        executor.assert_not_called()

    def test_ts1e_default_builtin_runs_without_grant(self):
        tool = self.tool_use.registry.get_tool("list_operations")
        self.assertFalse(tool.requires_permission)
        result = self.tool_use.execute_tool(
            "list_operations", {"items": [2, 1], "operation": "sort"}
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.result, [1, 2])
        self.assertEqual(self.tool_use.failed_executions, 0)

    def test_grant_rejects_empty_unregistered_and_non_string(self):
        for name in ("", "missing_tool", None, True):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as ctx:
                    self.tool_use.grant_permission(name)
                self.assertEqual(str(ctx.exception), "permission grant rejected")

    def test_revoke_rejects_empty_and_non_string(self):
        for name in ("", None, True):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as ctx:
                    self.tool_use.revoke_permission(name)
                self.assertEqual(str(ctx.exception), "permission revoke rejected")

    def test_revoke_without_prior_grant_still_fail_closes(self):
        self._arm_list_operations(requires_permission=True, cacheable=False)
        self.tool_use.revoke_permission("list_operations")
        with patch.object(
            self.tool_use, "_execute_list_operations", return_value=[1, 2]
        ) as executor:
            result = self.tool_use.execute_tool(
                "list_operations", {"items": [2, 1], "operation": "sort"}
            )
        self.assertFalse(result.success)
        self.assertIn("permission", result.error.lower())
        executor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
