import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import targeted_mutation as mutation_tool
from tools.targeted_mutation import (
    TIMEOUT_SECONDS,
    classify_mutation_result,
    run_tests,
)


class TargetedMutationTimeoutContractTests(unittest.TestCase):
    def test_timeout_is_inconclusive_not_killed(self):
        timeout = subprocess.TimeoutExpired(
            cmd=["python", "-m", "unittest"], timeout=TIMEOUT_SECONDS
        )
        with patch("tools.targeted_mutation.subprocess.run", side_effect=timeout):
            exit_code, output = run_tests(Path("."), ("test_example.py",))

        self.assertIsNone(exit_code)
        self.assertIn("inconclusivo", output)
        self.assertEqual(classify_mutation_result(exit_code), "timeout")

    def test_nonzero_exit_kills_mutant(self):
        self.assertEqual(classify_mutation_result(1), "killed")

    def test_zero_exit_means_mutant_survived(self):
        self.assertEqual(classify_mutation_result(0), "survived")

    def test_main_timeout_is_inconclusive_and_fails_gate(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "source"
            root.mkdir()
            (root / "sample.py").write_text("original", encoding="utf-8")
            report_path = Path(scratch) / "report.json"
            mutations = [
                mutation_tool.Mutation(name, "sample.py", "original", "mutated", ("test_example.py",))
                for name in ("detected", "inconclusive")
            ]
            with (
                patch.object(mutation_tool, "ROOT", root),
                patch.object(mutation_tool, "REPORT", report_path),
                patch.object(mutation_tool, "MUTATIONS", mutations),
                patch.object(mutation_tool, "run_tests", side_effect=[
                    (0, "baseline passed"), (1, "assertion failed"), (None, "timeout"),
                ]),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(mutation_tool.main(), 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["summary"], {
            "declared": 2, "applicable": 1, "killed": 1, "survived": 0,
            "timed_out": 1, "not_applied": 0, "score_percent": 100.0,
        })
        self.assertEqual(report["mutations"][1]["status"], "timeout")
        self.assertIsNone(report["mutations"][1]["exit_code"])


if __name__ == "__main__":
    unittest.main()
