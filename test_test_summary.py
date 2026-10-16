import unittest
from io import StringIO

from tools.test_summary import SummaryRunner


class TestSummaryAccountingTests(unittest.TestCase):
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
