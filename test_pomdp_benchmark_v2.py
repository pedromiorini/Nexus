import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.nexus_pomdp_benchmark_v2 import (
    BENCHMARK_VERSION,
    DEFAULT_CASE_COUNT,
    DEFAULT_SEED,
    SCHEMA_VERSION,
    build_taskset,
    main,
    run_benchmark,
)


class POMDPBeliefBenchmarkV2Tests(unittest.TestCase):
    def test_taskset_is_deterministic_variable_and_versioned(self):
        first = build_taskset(73)
        second = build_taskset(73)

        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], "nexus.pomdp-belief-taskset.v2")
        self.assertEqual(len(first["cases"]), DEFAULT_CASE_COUNT)
        self.assertEqual(
            {len(case["states"]) for case in first["cases"]},
            set(range(2, 9)),
        )
        self.assertEqual(len(first["rejection_controls"]), 4)

    def test_different_seed_changes_taskset(self):
        self.assertNotEqual(build_taskset(73), build_taskset(74))

    def test_report_is_reproducible_and_measures_posterior_accuracy(self):
        first = run_benchmark(19)
        second = run_benchmark(19)

        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], SCHEMA_VERSION)
        self.assertEqual(first["benchmark_version"], BENCHMARK_VERSION)
        result = first["tasks"]["belief_state_update"]
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["cases"], DEFAULT_CASE_COUNT)
        self.assertEqual(result["within_tolerance_cases"], DEFAULT_CASE_COUNT)
        self.assertEqual(result["within_tolerance_case_rate"], 1.0)
        self.assertLessEqual(result["maximum_absolute_error"], result["posterior_tolerance"])
        self.assertTrue(result["quality_gate"]["passed"])
        self.assertEqual(result["training_examples_used"], 0)
        self.assertEqual(first["case_counts"]["rejection_controls"], 4)

    def test_ci_seed_passes_accuracy_and_rejection_gates(self):
        result = run_benchmark(DEFAULT_SEED)["tasks"]["belief_state_update"]

        self.assertTrue(result["quality_gate"]["passed"])
        self.assertEqual(result["quality_gate"]["failed_conditions"], [])
        self.assertEqual(result["within_tolerance_cases"], DEFAULT_CASE_COUNT)
        self.assertEqual(result["rejection_controls"], {"cases": 4, "rejected": 4, "accepted": 0})

    def test_cli_returns_failure_and_writes_report_when_quality_gate_fails(self):
        def invalid_posterior(states, *_args):
            return {state: 0.0 for state in states}

        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "failed-report.json"
            with patch(
                "benchmarks.nexus_pomdp_benchmark_v2.RealPredictiveCodingSystem.update_belief_state",
                side_effect=invalid_posterior,
            ):
                exit_code = main(
                    ["--seed", str(DEFAULT_SEED), "--output", str(report_path)]
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        result = report["tasks"]["belief_state_update"]
        self.assertEqual(exit_code, 1)
        self.assertFalse(result["quality_gate"]["passed"])
        self.assertFalse(result["quality_gate"]["posterior_cases_passed"])
        self.assertFalse(result["quality_gate"]["rejection_controls_passed"])
        self.assertEqual(
            result["quality_gate"]["failed_conditions"],
            ["posterior_cases_outside_tolerance", "rejection_controls_not_all_rejected"],
        )
        self.assertEqual(result["rejection_controls"]["accepted"], 4)

    def test_invalid_controls_are_rejected(self):
        result = run_benchmark(31)["tasks"]["belief_state_update"]["rejection_controls"]
        self.assertEqual(result, {"cases": 4, "rejected": 4, "accepted": 0})

    def test_cli_writes_report_and_case_dump(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "report.json"
            cases_path = Path(temporary_directory) / "cases.json"
            exit_code = main(
                [
                    "--seed",
                    "41",
                    "--output",
                    str(report_path),
                    "--dump-cases",
                    str(cases_path),
                ]
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            taskset = json.loads(cases_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["metadata"]["seed"], 41)
        self.assertEqual(taskset["schema_version"], "nexus.pomdp-belief-taskset.v2")
        self.assertEqual(len(taskset["cases"]), DEFAULT_CASE_COUNT)

    def test_negative_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_taskset(-1)


if __name__ == "__main__":
    unittest.main()
