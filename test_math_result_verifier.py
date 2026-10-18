import unittest
from decimal import getcontext, localcontext
from unittest.mock import patch

from core.constitutional_brain import (
    METACOGNITIVE_SCORE_EVALUATOR,
    RealMetacognitionSystem,
    RealToolUse,
    ToolExecutionResult,
)
from core.math_result_verifier import TASK_TYPE, make_result_evaluator


class MathResultVerifierTests(unittest.TestCase):
    class Log:
        def log_event(self, *_args):
            pass

    def test_development_cases_run_through_real_tool_and_exact_oracle(self):
        tool = RealToolUse(self.Log())
        # Development fixtures only: never reused in the fresh holdout.
        for expression in ("0 + 0", "9 + 1", "123 + 456", "999999999999 + 999999999999"):
            with self.subTest(expression=expression):
                callback = make_result_evaluator(expression)
                result = tool.execute_tool("math_calculate", {"expression": expression})
                self.assertEqual(callback(result), 1.0)

    def test_oracle_does_not_call_executor_or_inherit_decimal_precision(self):
        with localcontext() as context:
            context.prec = 1
            with patch.object(RealToolUse, "_execute_math_calculate", side_effect=AssertionError):
                callback = make_result_evaluator("999999999999 + 1")
                self.assertEqual(callback(ToolExecutionResult("math_calculate", True, 1000000000000)), 1.0)
            self.assertEqual(getcontext().prec, 1)

    def test_unknown_domain_has_no_evaluator(self):
        for expression in (
            None, True, 1, "2+2", "2 + 2\n", "01 + 2", "-1 + 2", "1.0 + 2",
            "1e2 + 3", "1 + 2 + 3", "sum(1, 2)", "١ + ٢", "1 + 1000000000000",
            "calculate 1 + 2", "__import__('os')", " " * 10000,
        ):
            with self.subTest(expression=expression):
                self.assertIsNone(make_result_evaluator(expression))

    def test_self_reported_success_is_not_ground_truth(self):
        callback = make_result_evaluator("9 + 1")
        for value in (11, True, 10.0, "10", None, -1, 10 ** 100):
            with self.subTest(value=value):
                self.assertEqual(callback(ToolExecutionResult("math_calculate", True, value)), 0.0)
        self.assertEqual(callback(ToolExecutionResult("math_calculate", False, 10)), 0.0)
        self.assertEqual(callback(ToolExecutionResult("math_calculate", True, 10, error="bad")), 0.0)

    def test_wrong_tool_or_untrusted_container_fails_closed(self):
        callback = make_result_evaluator("9 + 1")
        for result in (
            {"success": True, "result": 10}, 10,
            ToolExecutionResult("other", True, 10),
            ToolExecutionResult("math_calculate", 1, 10),
        ):
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    callback(result)

    def test_bound_oracle_detects_faulty_executor(self):
        tool = RealToolUse(self.Log())
        with patch.object(tool, "_execute_math_calculate", return_value=11):
            result = tool.execute_tool("math_calculate", {"expression": "9 + 1"})
        self.assertTrue(result.success)
        self.assertEqual(make_result_evaluator("9 + 1")(result), 0.0)

    def test_harness_callback_has_provenance_but_no_strategy_credit(self):
        system = RealMetacognitionSystem()
        tool = RealToolUse(self.Log())
        result_evaluator = make_result_evaluator("9 + 1")
        result, assessment = system.execute_with_monitoring(
            TASK_TYPE, "analytical",
            lambda: tool.execute_tool("math_calculate", {"expression": "9 + 1"}),
            evaluator=result_evaluator,
        )
        self.assertEqual(result.result, 10)
        self.assertEqual(assessment.score_source, METACOGNITIVE_SCORE_EVALUATOR)
        self.assertEqual(system.monitor.completed_processes[-1].performance, 1.0)
        self.assertTrue(all(s.usage_count == 0 for s in system.strategy_selector.strategies.values()))

    def test_request_is_bound_at_construction_not_to_mutable_input(self):
        params = {"expression": "9 + 1"}
        callback = make_result_evaluator(params["expression"])
        params["expression"] = "9 + 2"
        self.assertEqual(callback(ToolExecutionResult("math_calculate", True, 11)), 0.0)


if __name__ == "__main__":
    unittest.main()
