import math
import unittest

from core.constitutional_brain import (
    METACOGNITIVE_SCORE_EVALUATOR,
    METACOGNITIVE_SCORE_HEURISTIC,
    RealMetacognitionSystem,
    StrategySelector,
)


class StrategyCreditAssignmentTests(unittest.TestCase):
    def test_explicit_score_for_incompatible_strategy_is_monitored_but_not_credited(self):
        system = RealMetacognitionSystem()
        strategy = system.strategy_selector.strategies["analytical"]
        initial_state = (
            strategy.usage_count,
            strategy.success_count,
            strategy.success_rate,
            strategy.avg_performance,
        )

        result, assessment = system.execute_with_monitoring(
            "learning",
            "analytical",
            lambda: "learned",
            evaluator=lambda _: 0.9,
        )

        self.assertEqual(result, "learned")
        self.assertEqual(assessment.process_evaluated, "learning")
        self.assertEqual(assessment.score_source, METACOGNITIVE_SCORE_EVALUATOR)
        process = system.monitor.completed_processes[-1]
        self.assertEqual(process.score_source, METACOGNITIVE_SCORE_EVALUATOR)
        self.assertEqual(process.performance, 0.9)
        metrics = system.monitor.process_metrics["learning"]
        self.assertEqual(metrics["count"], 1)
        self.assertEqual(metrics["verified_performance_count"], 1)
        self.assertEqual(metrics["avg_performance"], 0.9)
        self.assertEqual(
            (
                strategy.usage_count,
                strategy.success_count,
                strategy.success_rate,
                strategy.avg_performance,
            ),
            initial_state,
        )

    def test_compatible_reasoning_and_problem_solving_tasks_receive_credit(self):
        for task_type in ("reasoning", "problem_solving"):
            with self.subTest(task_type=task_type):
                system = RealMetacognitionSystem()
                result, assessment = system.execute_with_monitoring(
                    task_type,
                    "analytical",
                    lambda: True,
                    evaluator=lambda _: 0.9,
                )

                self.assertTrue(result)
                self.assertEqual(assessment.score_source, METACOGNITIVE_SCORE_EVALUATOR)
                strategy = system.strategy_selector.strategies["analytical"]
                self.assertEqual(strategy.usage_count, 1)
                self.assertEqual(strategy.success_count, 1)
                self.assertEqual(strategy.success_rate, 1.0)
                self.assertAlmostEqual(strategy.avg_performance, 0.9)

    def test_heuristic_fallback_does_not_train_even_a_compatible_strategy(self):
        system = RealMetacognitionSystem()

        result, assessment = system.execute_with_monitoring(
            "reasoning", "analytical", lambda: True
        )

        self.assertTrue(result)
        self.assertEqual(assessment.score_source, METACOGNITIVE_SCORE_HEURISTIC)
        self.assertEqual(
            system.monitor.completed_processes[-1].score_source,
            METACOGNITIVE_SCORE_HEURISTIC,
        )
        strategy = system.strategy_selector.strategies["analytical"]
        self.assertEqual(strategy.usage_count, 0)
        self.assertEqual(strategy.success_count, 0)

    def test_direct_incompatible_update_is_atomic_and_compatible_update_succeeds(self):
        selector = StrategySelector()
        strategy = selector.strategies["analytical"]
        initial_state = (
            strategy.usage_count,
            strategy.success_count,
            strategy.success_rate,
            strategy.avg_performance,
        )

        self.assertFalse(
            selector.update_strategy_performance(
                "analytical", 0.9, task_type="learning"
            )
        )
        self.assertEqual(
            (
                strategy.usage_count,
                strategy.success_count,
                strategy.success_rate,
                strategy.avg_performance,
            ),
            initial_state,
        )

        self.assertTrue(
            selector.update_strategy_performance(
                "analytical", 0.9, task_type="reasoning"
            )
        )
        self.assertEqual(strategy.usage_count, 1)
        self.assertEqual(strategy.success_count, 1)
        self.assertEqual(strategy.success_rate, 1.0)
        self.assertAlmostEqual(strategy.avg_performance, 0.9)

    def test_legacy_update_without_task_type_remains_supported(self):
        selector = StrategySelector()

        self.assertTrue(selector.update_strategy_performance("analytical", 0.9))
        strategy = selector.strategies["analytical"]
        self.assertEqual(strategy.usage_count, 1)
        self.assertEqual(strategy.success_count, 1)
        self.assertEqual(strategy.success_rate, 1.0)
        self.assertAlmostEqual(strategy.avg_performance, 0.9)

    def test_unknown_strategy_and_invalid_scores_return_false_without_mutation(self):
        selector = StrategySelector()
        strategy = selector.strategies["analytical"]
        initial_state = (
            strategy.usage_count,
            strategy.success_count,
            strategy.success_rate,
            strategy.avg_performance,
        )

        self.assertFalse(selector.update_strategy_performance("missing", 0.9))
        for score in (None, True, "0.8", math.nan, math.inf, -0.01, 1.01):
            with self.subTest(score=score):
                self.assertFalse(
                    selector.update_strategy_performance("analytical", score)
                )
                self.assertEqual(
                    (
                        strategy.usage_count,
                        strategy.success_count,
                        strategy.success_rate,
                        strategy.avg_performance,
                    ),
                    initial_state,
                )


if __name__ == "__main__":
    unittest.main()
