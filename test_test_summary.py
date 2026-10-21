import json
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from tools.test_summary import SummaryRunner


class TestSummaryAccountingTests(unittest.TestCase):
    def test_empty_suite_is_reported_as_failed_without_inventing_failures(self):
        result = SummaryRunner(stream=StringIO(), verbosity=0).run(unittest.TestSuite())

        self.assertEqual(
            result.summary,
            {
                "executed": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "unexpected_successes": 0,
                "result": "failed",
            },
        )

    def test_empty_discovery_cli_writes_failed_summary_and_exits_nonzero(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="nexus-empty-summary-") as temporary:
            output = Path(temporary) / "summary.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "tools" / "test_summary.py"),
                    "--pattern", "no_such_tests_*.py",
                    "--output", str(output),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            summary = json.loads(output.read_text(encoding="utf-8"))

        with self.subTest(contract="summary"):
            self.assertEqual(
                summary,
                {
                    "executed": 0,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "unexpected_successes": 0,
                    "result": "failed",
                },
            )
        with self.subTest(contract="exit_code"):
            self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_nonempty_success_skip_and_expected_failure_preserve_semantics(self):
        def ordinary_pass(test_case):
            test_case.assertTrue(True)

        def expected_failure(test_case):
            test_case.fail("still broken")

        for name, method, passed, skipped in (
            ("ordinary_pass", ordinary_pass, 1, 0),
            ("skip", unittest.skip("fixture skip")(expected_failure), 0, 1),
            ("expected_failure", unittest.expectedFailure(expected_failure), 1, 0),
        ):
            with self.subTest(case=name):
                fixture = type("SummaryFixture", (unittest.TestCase,), {"test_case": method})
                suite = unittest.defaultTestLoader.loadTestsFromTestCase(fixture)
                result = SummaryRunner(stream=StringIO(), verbosity=0).run(suite)
                self.assertEqual(
                    result.summary,
                    {
                        "executed": 1,
                        "passed": passed,
                        "failed": 0,
                        "skipped": skipped,
                        "unexpected_successes": 0,
                        "result": "passed",
                    },
                )

    def test_expected_failure_unexpected_success_subtests_and_skip_are_accounted(self):
        def expected_failure(test_case):
            test_case.fail("still broken")

        def unexpected_success(test_case):
            test_case.assertTrue(True)

        def skipped(test_case):
            test_case.fail("a skipped test must not run")

        def ordinary_pass(test_case):
            test_case.assertTrue(True)

        def multiple_subtest_failures(test_case):
            for index in range(3):
                with test_case.subTest(index=index):
                    test_case.fail(f"failure {index}")

        fixture = type(
            "SummaryFixture",
            (unittest.TestCase,),
            {
                "test_expected_failure": unittest.expectedFailure(expected_failure),
                "test_unexpected_success": unittest.expectedFailure(unexpected_success),
                "test_skipped": unittest.skip("fixture skip")(skipped),
                "test_ordinary_pass": ordinary_pass,
                "test_multiple_subtest_failures": multiple_subtest_failures,
            },
        )
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(fixture)
        result = SummaryRunner(stream=StringIO(), verbosity=0).run(suite)

        self.assertEqual(len(result.failures), 3)
        self.assertEqual(
            result.summary,
            {
                "executed": 5,
                "passed": 2,
                "failed": 2,
                "skipped": 1,
                "unexpected_successes": 1,
                "result": "failed",
            },
        )
        self.assertFalse(result.wasSuccessful())


if __name__ == "__main__":
    unittest.main()
