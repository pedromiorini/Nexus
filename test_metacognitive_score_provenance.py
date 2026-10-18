import math
import unittest

from core.constitutional_brain import (
    METACOGNITIVE_SCORE_EVALUATOR,
    METACOGNITIVE_SCORE_HEURISTIC,
    METACOGNITIVE_SCORE_UNAVAILABLE,
    MetacognitiveAssessment,
    RealMetacognitionSystem,
)


class TruthyFailure:
    success = False

    def __bool__(self):
        return True


class TruthinessRaises:
    def __bool__(self):
        raise RuntimeError("truthiness failed")


class AdversarialFloat(float):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.float_called = False
        return instance

    def __float__(self):
        self.float_called = True
        raise AssertionError("adversarial __float__ must not be called")


class AdversarialInt(int):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.int_called = False
        return instance

    def __int__(self):
        self.int_called = True
        raise AssertionError("adversarial __int__ must not be called")


class MetacognitiveScoreProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.system = RealMetacognitionSystem()

    def execute(self, result, *, evaluator=None, process_type="reasoning"):
        return self.system.execute_with_monitoring(
            process_type,
            "analytical",
            lambda: result,
            evaluator=evaluator,
        )

    def test_truthy_and_falsy_fallbacks_remain_heuristic_and_do_not_train(self):
        self.execute(True)
        self.execute(False)

        processes = self.system.monitor.completed_processes
        self.assertEqual([process.performance for process in processes], [0.8, 0.3])
        self.assertEqual(
            [process.score_source for process in processes],
            [METACOGNITIVE_SCORE_HEURISTIC, METACOGNITIVE_SCORE_HEURISTIC],
        )
        self.assertEqual(
            self.system.strategy_selector.strategies["analytical"].usage_count,
            0,
        )
        metrics = self.system.monitor.process_metrics["reasoning"]
        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["verified_performance_count"], 0)
        self.assertEqual(metrics["avg_performance"], 0.0)
        self.assertEqual(self.system.performance_evaluator.get_avg_confidence(), 0.5)
        self.assertEqual(self.system.performance_evaluator.get_avg_understanding(), 0.5)

    def test_reflection_distinguishes_missing_verified_scores_from_zero(self):
        initial = self.system.reflect_on_learning()
        self.assertIsNone(initial["reasoning_performance"])
        self.assertEqual(initial["reasoning_verified_performance_count"], 0)
        self.assertIsNone(initial["learning_performance"])
        self.assertEqual(initial["learning_verified_performance_count"], 0)

        self.execute(True)
        after_heuristic = self.system.reflect_on_learning()
        self.assertIsNone(after_heuristic["reasoning_performance"])
        self.assertEqual(after_heuristic["reasoning_verified_performance_count"], 0)

        self.execute(True, evaluator=lambda _: 0.0)
        after_verified_zero = self.system.reflect_on_learning()
        self.assertEqual(after_verified_zero["reasoning_performance"], 0.0)
        self.assertEqual(after_verified_zero["reasoning_verified_performance_count"], 1)

    def test_zero_is_a_valid_explicit_evaluator_score(self):
        result, assessment = self.execute(True, evaluator=lambda _: 0.0)

        self.assertTrue(result)
        self.assertEqual(assessment.score_source, METACOGNITIVE_SCORE_EVALUATOR)
        process = self.system.monitor.completed_processes[-1]
        self.assertEqual(process.performance, 0.0)
        self.assertEqual(process.outcome, "failure")
        strategy = self.system.strategy_selector.strategies["analytical"]
        self.assertEqual(strategy.usage_count, 1)
        self.assertEqual(strategy.success_count, 0)
        self.assertEqual(strategy.success_rate, 0.0)
        metrics = self.system.monitor.process_metrics["reasoning"]
        self.assertEqual(metrics["verified_performance_count"], 1)
        self.assertEqual(metrics["avg_performance"], 0.0)

    def test_evaluator_runs_before_truthiness_is_attempted(self):
        result_object = TruthinessRaises()
        evaluator_calls = []

        result, assessment = self.execute(
            result_object,
            evaluator=lambda value: evaluator_calls.append(value) or 0.4,
        )

        self.assertIs(result, result_object)
        self.assertEqual(evaluator_calls, [result_object])
        self.assertEqual(assessment.score_source, METACOGNITIVE_SCORE_EVALUATOR)
        process = self.system.monitor.completed_processes[-1]
        self.assertEqual(process.performance, 0.4)
        self.assertEqual(process.outcome, "failure")
        self.assertEqual(self.system.monitor.active_processes, {})

    def test_truthy_object_with_success_false_uses_evaluator_score(self):
        result_object = TruthyFailure()
        result, assessment = self.execute(
            result_object,
            evaluator=lambda value: 1.0 if value.success else 0.0,
        )

        self.assertIs(result, result_object)
        self.assertEqual(assessment.strategy_effectiveness, 0.0)
        self.assertEqual(assessment.score_source, METACOGNITIVE_SCORE_EVALUATOR)
        process = self.system.monitor.completed_processes[-1]
        self.assertEqual(process.performance, 0.0)
        self.assertEqual(process.outcome, "failure")
        self.assertEqual(
            self.system.strategy_selector.strategies["analytical"].usage_count,
            1,
        )

    def test_valid_evaluator_updates_verified_aggregates_not_heuristic_assessments(self):
        self.execute(True)  # Fallback confidence would be 0.9; it must be excluded.
        self.execute(False, evaluator=lambda _: 0.0)

        self.assertAlmostEqual(self.system.performance_evaluator.get_avg_confidence(), 0.3)
        metrics = self.system.monitor.process_metrics["reasoning"]
        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["verified_performance_count"], 1)
        self.assertEqual(metrics["avg_performance"], 0.0)
        self.assertEqual(
            self.system.strategy_selector.strategies["analytical"].usage_count,
            1,
        )

    def test_invalid_evaluator_scores_raise_without_fallback_or_active_state(self):
        invalid_scores = [None, True, "0.8", math.nan, math.inf, -0.01, 1.01]
        for score in invalid_scores:
            with self.subTest(score=score):
                with self.assertRaisesRegex(ValueError, "evaluator must return"):
                    self.execute(True, evaluator=lambda _, value=score: value)

                process = self.system.monitor.completed_processes[-1]
                self.assertIsNone(process.performance)
                self.assertEqual(process.score_source, METACOGNITIVE_SCORE_UNAVAILABLE)
                self.assertEqual(process.outcome, "invalid_evaluator_score")
                self.assertEqual(self.system.monitor.active_processes, {})

        self.assertEqual(
            self.system.strategy_selector.strategies["analytical"].usage_count,
            0,
        )
        metrics = self.system.monitor.process_metrics["reasoning"]
        self.assertEqual(metrics["verified_performance_count"], 0)
        self.assertEqual(metrics["avg_performance"], 0.0)

    def test_evaluator_exception_propagates_without_fallback_and_cleans_process(self):
        def broken_evaluator(_):
            raise RuntimeError("evaluator unavailable")

        with self.assertRaisesRegex(RuntimeError, "evaluator unavailable"):
            self.execute(True, evaluator=broken_evaluator)

        process = self.system.monitor.completed_processes[-1]
        self.assertIsNone(process.performance)
        self.assertEqual(process.score_source, METACOGNITIVE_SCORE_UNAVAILABLE)
        self.assertEqual(process.outcome, "evaluation_error")
        self.assertEqual(self.system.monitor.active_processes, {})
        self.assertEqual(
            self.system.strategy_selector.strategies["analytical"].usage_count,
            0,
        )
        self.assertEqual(self.system.performance_evaluator.assessments, [])

    def test_truthiness_exception_without_evaluator_closes_as_unscored(self):
        with self.assertRaisesRegex(RuntimeError, "truthiness failed"):
            self.execute(TruthinessRaises())

        process = self.system.monitor.completed_processes[-1]
        self.assertIsNone(process.performance)
        self.assertEqual(process.score_source, METACOGNITIVE_SCORE_UNAVAILABLE)
        self.assertEqual(process.outcome, "truthiness_error")
        self.assertEqual(self.system.monitor.active_processes, {})
        metrics = self.system.monitor.process_metrics["reasoning"]
        self.assertEqual(metrics["verified_performance_count"], 0)
        self.assertEqual(metrics["avg_performance"], 0.0)
        self.assertEqual(
            self.system.strategy_selector.strategies["analytical"].usage_count,
            0,
        )

    def test_public_end_monitoring_cannot_claim_verified_provenance(self):
        process_id = self.system.monitor.start_monitoring("reasoning", "analytical")

        with self.assertRaises(TypeError):
            self.system.monitor.end_monitoring(
                process_id,
                0.95,
                "success",
                score_source=METACOGNITIVE_SCORE_EVALUATOR,
            )

        self.system.monitor.end_monitoring(process_id, 0.95, "success")
        process = self.system.monitor.completed_processes[-1]
        self.assertEqual(process.performance, 0.95)
        self.assertEqual(process.score_source, METACOGNITIVE_SCORE_HEURISTIC)
        with self.assertRaisesRegex(AttributeError, "score_source"):
            process.score_source = METACOGNITIVE_SCORE_EVALUATOR
        self.assertEqual(
            self.system.monitor.process_metrics["reasoning"]["verified_performance_count"],
            0,
        )
        self.assertEqual(self.system.monitor.active_processes, {})

    def test_execution_exception_cleans_active_process_and_propagates(self):
        def broken_execution():
            raise RuntimeError("execution failed")

        with self.assertRaisesRegex(RuntimeError, "execution failed"):
            self.system.execute_with_monitoring(
                "reasoning", "analytical", broken_execution
            )

        self.assertEqual(self.system.monitor.active_processes, {})
        self.assertEqual(len(self.system.monitor.completed_processes), 1)
        process = self.system.monitor.completed_processes[0]
        self.assertEqual(process.outcome, "error")
        self.assertIsNone(process.performance)
        self.assertEqual(process.score_source, METACOGNITIVE_SCORE_UNAVAILABLE)
        self.assertEqual(
            self.system.strategy_selector.strategies["analytical"].usage_count,
            0,
        )

    def test_success_rate_uses_success_count_for_scores_08_then_05(self):
        self.execute(True, evaluator=lambda _: 0.8)
        self.execute(True, evaluator=lambda _: 0.5)

        strategy = self.system.strategy_selector.strategies["analytical"]
        self.assertEqual(strategy.usage_count, 2)
        self.assertEqual(strategy.success_count, 1)
        self.assertEqual(strategy.success_rate, 0.5)
        self.assertAlmostEqual(strategy.avg_performance, 0.65)
        metrics = self.system.monitor.process_metrics["reasoning"]
        self.assertEqual(metrics["verified_performance_count"], 2)
        self.assertAlmostEqual(metrics["avg_performance"], 0.65)


    def test_evaluator_rejects_adversarial_float_without_fallback(self):
        score = AdversarialFloat(0.9)
        execution_calls = []

        with self.assertRaisesRegex(ValueError, "evaluator must return"):
            self.system.execute_with_monitoring(
                "reasoning",
                "analytical",
                lambda: execution_calls.append(True) or True,
                evaluator=lambda _: score,
            )

        self.assertFalse(score.float_called)
        self.assertEqual(execution_calls, [True])
        self.assertEqual(self.system.monitor.active_processes, {})
        self.assertEqual(len(self.system.monitor.completed_processes), 1)
        process = self.system.monitor.completed_processes[0]
        self.assertEqual(process.outcome, "invalid_evaluator_score")
        self.assertIsNone(process.performance)
        self.assertEqual(process.score_source, METACOGNITIVE_SCORE_UNAVAILABLE)
        metrics = self.system.monitor.process_metrics["reasoning"]
        self.assertEqual(metrics["verified_performance_count"], 0)
        self.assertEqual(metrics["avg_performance"], 0.0)

    def test_public_end_monitoring_closes_on_adversarial_float(self):
        score = AdversarialFloat(0.9)
        process_id = self.system.monitor.start_monitoring(
            "reasoning", "analytical"
        )

        self.system.monitor.end_monitoring(process_id, score, "success")

        self.assertFalse(score.float_called)
        self.assertEqual(self.system.monitor.active_processes, {})
        self.assertEqual(len(self.system.monitor.completed_processes), 1)
        process = self.system.monitor.completed_processes[0]
        self.assertIsNone(process.performance)
        self.assertEqual(process.score_source, METACOGNITIVE_SCORE_UNAVAILABLE)
        metrics = self.system.monitor.process_metrics["reasoning"]
        self.assertEqual(metrics["verified_performance_count"], 0)
        self.assertEqual(metrics["avg_performance"], 0.0)


    def test_invalid_expected_score_inputs_are_rejected_before_process_start(self):
        invalid_scores = [
            ("nan", math.nan),
            ("infinity", math.inf),
            ("negative_infinity", -math.inf),
            ("below_range", -0.01),
            ("above_range", 1.01),
            ("bool", True),
            ("float_subclass", AdversarialFloat(0.5)),
            ("int_subclass", AdversarialInt(1)),
        ]
        for parameter_name in ("expected_performance", "difficulty"):
            for label, score in invalid_scores:
                with self.subTest(parameter=parameter_name, score=label):
                    execution_calls = []
                    kwargs = {
                        "expected_performance": 0.7,
                        "difficulty": 0.5,
                    }
                    kwargs[parameter_name] = score

                    with self.assertRaisesRegex(ValueError, parameter_name):
                        self.system.execute_with_monitoring(
                            "reasoning",
                            "analytical",
                            lambda: execution_calls.append(True) or True,
                            **kwargs,
                        )

                    self.assertEqual(execution_calls, [])
                    self.assertEqual(self.system.monitor.active_processes, {})
                    self.assertEqual(self.system.monitor.completed_processes, [])
                    self.assertEqual(
                        self.system.performance_evaluator.assessments, []
                    )
                    self.assertEqual(
                        self.system.strategy_selector.strategies[
                            "analytical"
                        ].usage_count,
                        0,
                    )
                    metrics = self.system.monitor.process_metrics["reasoning"]
                    self.assertEqual(metrics["count"], 0)
                    self.assertEqual(metrics["verified_performance_count"], 0)
                    self.assertEqual(metrics["avg_performance"], 0.0)
                    self.assertEqual(
                        self.system.performance_evaluator.get_avg_confidence(),
                        0.5,
                    )
                    self.assertEqual(
                        self.system.performance_evaluator.get_avg_understanding(),
                        0.5,
                    )

        subclass_scores = dict(invalid_scores)
        self.assertFalse(subclass_scores["float_subclass"].float_called)
        self.assertFalse(subclass_scores["int_subclass"].int_called)

    def test_public_evaluator_rejects_invalid_performance_score_inputs(self):
        invalid_scores = [
            ("nan", math.nan),
            ("infinity", math.inf),
            ("negative_infinity", -math.inf),
            ("below_range", -0.01),
            ("above_range", 1.01),
            ("bool", True),
            ("float_subclass", AdversarialFloat(0.5)),
            ("int_subclass", AdversarialInt(1)),
        ]
        for parameter_name in (
            "actual_performance",
            "expected_performance",
            "difficulty",
        ):
            for label, score in invalid_scores:
                with self.subTest(parameter=parameter_name, score=label):
                    kwargs = {
                        "actual_performance": 0.8,
                        "expected_performance": 0.7,
                        "difficulty": 0.5,
                    }
                    kwargs[parameter_name] = score

                    with self.assertRaisesRegex(ValueError, parameter_name):
                        self.system.performance_evaluator.evaluate_performance(
                            "reasoning",
                            **kwargs,
                        )

                    self.assertEqual(
                        self.system.performance_evaluator.assessments, []
                    )
                    self.assertEqual(
                        self.system.performance_evaluator.get_avg_confidence(),
                        0.5,
                    )
                    self.assertEqual(
                        self.system.performance_evaluator.get_avg_understanding(),
                        0.5,
                    )

        subclass_scores = dict(invalid_scores)
        self.assertFalse(subclass_scores["float_subclass"].float_called)
        self.assertFalse(subclass_scores["int_subclass"].int_called)

    def test_public_evaluator_cannot_claim_verified_provenance(self):
        evaluator = self.system.performance_evaluator
        with self.assertRaisesRegex(TypeError, "score_source"):
            evaluator.evaluate_performance(
                "reasoning",
                actual_performance=0.8,
                expected_performance=0.7,
                difficulty=0.5,
                score_source=METACOGNITIVE_SCORE_EVALUATOR,
            )

        self.assertEqual(evaluator.assessments, [])
        assessment = evaluator.evaluate_performance(
            "reasoning",
            actual_performance=0.8,
            expected_performance=0.7,
            difficulty=0.5,
        )
        self.assertEqual(assessment.score_source, METACOGNITIVE_SCORE_HEURISTIC)
        self.assertEqual(evaluator.get_avg_confidence(), 0.5)
        self.assertEqual(evaluator.get_avg_understanding(), 0.5)

    def test_public_assessment_and_list_mutation_cannot_forge_aggregates(self):
        _, verified = self.execute(True, evaluator=lambda _: 0.8)
        evaluator = self.system.performance_evaluator
        expected_confidence = evaluator.get_avg_confidence()
        expected_understanding = evaluator.get_avg_understanding()

        with self.assertRaisesRegex(AttributeError, "score_source"):
            verified.score_source = METACOGNITIVE_SCORE_HEURISTIC
        with self.assertRaisesRegex(AttributeError, "confidence_in_result"):
            verified.confidence_in_result = math.nan

        fabricated_fields = {
            "assessment_id": "fabricated",
            "process_evaluated": "reasoning",
            "confidence_in_result": 0.0,
            "understanding_level": 0.0,
            "difficulty_estimate": 0.0,
            "strategy_effectiveness": 0.0,
            "timestamp": 0.0,
        }
        with self.assertRaisesRegex(TypeError, "score_source"):
            MetacognitiveAssessment(
                **fabricated_fields,
                score_source=METACOGNITIVE_SCORE_EVALUATOR,
            )
        fabricated = MetacognitiveAssessment(**fabricated_fields)
        evaluator.assessments.append(fabricated)
        self.assertAlmostEqual(
            evaluator.get_avg_confidence(), expected_confidence
        )
        self.assertAlmostEqual(
            evaluator.get_avg_understanding(), expected_understanding
        )

        evaluator.assessments.clear()
        evaluator.assessments.extend([fabricated] * 20)
        self.assertEqual(len(evaluator.assessments), 20)
        statistics = self.system.get_statistics()
        self.assertEqual(statistics["assessments_made"], 1)
        self.assertAlmostEqual(statistics["metacognitive_awareness"], 0.05)
        self.assertAlmostEqual(
            self.system.reflect_on_learning()["metacognitive_awareness"],
            0.05,
        )

        evaluator.assessments.clear()
        statistics = self.system.get_statistics()
        self.assertEqual(statistics["assessments_made"], 1)
        self.assertAlmostEqual(statistics["metacognitive_awareness"], 0.05)
        self.assertAlmostEqual(
            evaluator.get_avg_confidence(), expected_confidence
        )
        self.assertAlmostEqual(
            evaluator.get_avg_understanding(), expected_understanding
        )


if __name__ == "__main__":
    unittest.main()
