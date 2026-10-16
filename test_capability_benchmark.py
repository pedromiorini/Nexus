import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from benchmarks.nexus_capability_benchmark import (
    DEFAULT_SEED,
    build_taskset,
    run_benchmark,
)


class CapabilityBenchmarkContractTests(unittest.TestCase):
    def test_taskset_is_deterministic_and_versioned(self):
        first = build_taskset(DEFAULT_SEED)
        second = build_taskset(DEFAULT_SEED)
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], "nexus.capability-benchmark.v1")
        self.assertEqual(len(first["composition"]), 39)
        self.assertEqual(len(first["causal_extraction"]), 24)
        self.assertEqual(len(first["causal_intervention"]), 24)
        self.assertEqual(len(first["uncertainty"]), 9 * 128)
        self.assertEqual(len(first["partial_observation"]), 8)

    def test_report_is_reproducible_for_same_seed(self):
        self.assertEqual(run_benchmark(73), run_benchmark(73))
        root = Path(__file__).resolve().parent
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        completed = subprocess.run(
            [sys.executable, "-m", "benchmarks.nexus_capability_benchmark", "--seed", "73"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        cli_report = json.loads(completed.stdout)
        self.assertEqual(cli_report["metadata"]["seed"], 73)

    def test_report_exposes_metrics_and_limits_for_all_four_axes(self):
        report = run_benchmark(19)
        tasks = report["tasks"]
        self.assertEqual(report["schema_version"], "nexus.capability-benchmark.v1")
        self.assertEqual(tasks["compositional_generalization"]["status"], "measured")
        self.assertIn("held_out_length_3", tasks["compositional_generalization"])
        self.assertIn("f1", tasks["causal_relation_extraction"])
        self.assertIn("metrics", tasks["causal_intervention_effect_claims"])
        self.assertIn("brier_score", tasks["uncertainty"])
        self.assertIn("expected_calibration_error", tasks["uncertainty"])
        self.assertEqual(tasks["partial_observability"]["status"], "unsupported_interface")
        self.assertIsNone(tasks["partial_observability"]["posterior_accuracy"])
        self.assertIn("taskset_sha256", report["metadata"])

    def test_uncertainty_probe_reports_spread_order_and_controls(self):
        uncertainty = run_benchmark(31)["tasks"]["uncertainty"]
        self.assertTrue(uncertainty["uncertainty_spread_probe"]["nondecreasing_with_spread"])
        self.assertEqual(uncertainty["probability_vector_output"], False)
        self.assertGreaterEqual(uncertainty["brier_score"], 0.0)
        self.assertLessEqual(uncertainty["brier_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
