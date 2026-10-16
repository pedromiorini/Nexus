import time
import unittest

from core.constitutional_brain import RealToolUse


class MathCalculateResourceLimitTests(unittest.TestCase):
    class _Log:
        def __init__(self):
            self.events = []

        def log_event(self, name, payload):
            self.events.append((name, payload))

    def setUp(self):
        self.log = self._Log()
        self.tool_use = RealToolUse(self.log)

    def execute_math(self, expression):
        return self.tool_use.execute_tool("math_calculate", {"expression": expression})

    def test_normal_numeric_expressions_remain_supported(self):
        examples = (("2 + 2", 4), ("round(3.14159, 2)", 3.14), ("2 ** 8", 256))
        for expression, expected in examples:
            with self.subTest(expression=expression):
                result = self.execute_math(expression)
                self.assertTrue(result.success, result.error)
                self.assertEqual(result.result, expected)

    def test_string_repetition_and_non_numeric_constants_are_rejected(self):
        expressions = ('"x" * 1000000', "True + 1", "None", "1e999")
        for expression in expressions:
            with self.subTest(expression=expression):
                result = self.execute_math(expression)
                self.assertFalse(result.success)
                self.assertIsNone(result.result)

    def test_oversized_expression_is_rejected_before_cache_key_generation(self):
        expression = "1" * (RealToolUse.MAX_MATH_EXPRESSION_LENGTH + 1)

        def unexpected_cache_key(*_args, **_kwargs):
            self.fail("oversized expressions must be rejected before cache-key serialization")

        self.tool_use._make_cache_key = unexpected_cache_key
        result = self.execute_math(expression)
        self.assertFalse(result.success)
        self.assertIsNone(result.result)

    def test_exponent_that_would_exceed_numeric_budget_is_rejected(self):
        result = self.execute_math("2 ** 10000")
        self.assertFalse(result.success)
        self.assertIsNone(result.result)

    def test_expression_ast_complexity_is_bounded(self):
        expression = "+".join(["1"] * 70)
        result = self.execute_math(expression)
        self.assertFalse(result.success)
        self.assertIsNone(result.result)

    def test_oversized_success_result_is_returned_but_not_cached(self):
        text = "x" * (RealToolUse.MAX_CACHE_ENTRY_BYTES + 1)
        result = self.tool_use.execute_tool(
            "string_transform", {"text": text, "operation": "upper"}
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(result.result), len(text))
        self.assertEqual(self.tool_use.result_cache, {})
        self.assertEqual(self.tool_use._result_cache_bytes, 0)

    def test_expired_entries_are_removed_when_cache_is_accessed(self):
        first = self.execute_math("2 + 2")
        self.assertTrue(first.success)
        first_key = self.tool_use._make_cache_key("math_calculate", {"expression": "2 + 2"})
        cached = self.tool_use.result_cache[first_key]
        cached.timestamp = time.time() - self.tool_use.registry.get_tool("math_calculate").cache_ttl - 1

        second = self.execute_math("3 + 3")

        self.assertTrue(second.success)
        self.assertNotIn(first_key, self.tool_use.result_cache)
        self.assertLessEqual(self.tool_use._result_cache_bytes, self.tool_use.MAX_CACHE_BYTES)

    def test_mutating_returned_results_cannot_change_cached_payloads(self):
        self.tool_use._execute_list_operations = lambda _params: {"items": [1]}
        params = {"items": [1], "operation": "reverse"}
        cache_key = self.tool_use._make_cache_key("list_operations", params)

        original = self.tool_use.execute_tool("list_operations", params)
        self.assertTrue(original.success, original.error)
        stored = self.tool_use.result_cache[cache_key]
        original.result["items"].append("caller mutation")
        self.assertEqual(stored.result, {"items": [1]})

        cache_hit = self.tool_use.execute_tool("list_operations", params)
        self.assertTrue(cache_hit.cached)
        cache_hit.result["items"].append("cache-hit mutation")
        third_result = self.tool_use.execute_tool("list_operations", params)

        self.assertTrue(third_result.cached)
        self.assertEqual(third_result.result, {"items": [1]})

    def test_cache_entry_count_and_total_size_remain_bounded(self):
        for index in range(self.tool_use.MAX_CACHE_ENTRIES + 5):
            result = self.execute_math(f"{index} + 1")
            self.assertTrue(result.success, result.error)
        self.assertLessEqual(len(self.tool_use.result_cache), self.tool_use.MAX_CACHE_ENTRIES)

        payload = "x" * 30000
        for index in range(40):
            result = self.tool_use.execute_tool(
                "string_transform",
                {"text": f"{index}:" + payload, "operation": "upper"},
            )
            self.assertTrue(result.success, result.error)
        self.assertLessEqual(self.tool_use._result_cache_bytes, self.tool_use.MAX_CACHE_BYTES)
        self.assertLessEqual(len(self.tool_use.result_cache), self.tool_use.MAX_CACHE_ENTRIES)


if __name__ == "__main__":
    unittest.main()
