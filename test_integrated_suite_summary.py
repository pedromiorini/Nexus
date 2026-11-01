import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from core.integrated_suite_summary import IntegratedSuiteTally


class IntegratedSuiteTallyTests(unittest.TestCase):
    def test_tracks_passed_skipped_and_missing_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runner.py"
            source.write_text(
                'print("TEST 1 PASSED")\nprint("TEST 2 PASSED")\nprint("TEST 3 PASSED")\n',
                encoding="utf-8",
            )
            stream = StringIO()
            tally = IntegratedSuiteTally(stream)
            tally.write("✅ TEST 1 PASSED - first\n")
            tally.write("TEST 2: Optional component DISABLED\n")

            summary = tally.build_summary(source)
            output = Path(directory) / "integrated-summary.json"
            tally.write_summary(summary, output)

            self.assertEqual(summary["expected"], ["1", "2", "3"])
            self.assertEqual(summary["passed"], ["1"])
            self.assertEqual(summary["skipped"], ["2"])
            self.assertEqual(summary["missing"], ["3"])
            self.assertEqual(summary["result"], "failed")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), summary)
            self.assertEqual(stream.getvalue(), "✅ TEST 1 PASSED - first\nTEST 2: Optional component DISABLED\n")

    def test_passes_when_every_expected_check_passes_or_is_explicitly_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runner.py"
            source.write_text('"TEST 1 PASSED"\n"TEST 2 PASSED"\n', encoding="utf-8")
            tally = IntegratedSuiteTally(StringIO())
            tally.write("TEST 1 PASSED\nTEST 2: feature not enabled\n")

            summary = tally.build_summary(source)

            self.assertEqual(summary["missing"], [])
            self.assertEqual(summary["result"], "passed")

    def test_unexpected_and_conflicting_check_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runner.py"
            source.write_text('"TEST 1 PASSED"\n', encoding="utf-8")
            tally = IntegratedSuiteTally(StringIO())
            tally.write("TEST 1 PASSED\nTEST 1: feature DISABLED\nTEST 2 PASSED\n")

            summary = tally.build_summary(source)

            self.assertEqual(summary["unexpected"], ["2"])
            self.assertEqual(summary["conflicting"], ["1"])
            self.assertEqual(summary["result"], "failed")

    def test_repeated_terminal_results_fail_without_changing_existing_categories(self):
        outcomes = ("TEST 1 PASSED\n", "TEST 1: feature not enabled\n")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runner.py"
            source.write_text('"TEST 1 PASSED"\n', encoding="utf-8")
            for first in outcomes:
                for second in outcomes:
                    with self.subTest(first=first, second=second):
                        stream = StringIO()
                        tally = IntegratedSuiteTally(stream)
                        text = first + second + second
                        self.assertEqual(tally.write(text), len(text))
                        summary = tally.build_summary(source)

                        self.assertEqual(summary["result"], "failed")
                        self.assertEqual(summary["duplicates"], ["1"])
                        self.assertEqual(summary["passed"], ["1"] if outcomes[0] in text else [])
                        self.assertEqual(summary["skipped"], ["1"] if outcomes[1] in text else [])
                        self.assertEqual(summary["conflicting"], ["1"] if first != second else [])
                        self.assertEqual(summary["missing"], [])
                        self.assertEqual(summary["unexpected"], [])
                        self.assertEqual(stream.getvalue(), text)

    def test_duplicate_ids_include_unexpected_and_preserve_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runner.py"
            source.write_text('"TEST 1 PASSED"\n"TEST 2 PASSED"\n', encoding="utf-8")
            tally = IntegratedSuiteTally(StringIO())
            tally.write("TEST 1 PASSED\nTEST 1 PASSED\nTEST 10 PASSED\nTEST 10 PASSED\n")
            tally.write("TEST 0.5: feature DISABLED\nTEST 0.5: feature DISABLED\n")

            summary = tally.build_summary(source)

            self.assertEqual(summary["duplicates"], ["0.5", "1", "10"])
            self.assertEqual(summary["unexpected"], ["0.5", "10"])
            self.assertEqual(summary["missing"], ["2"])
            self.assertEqual(summary["conflicting"], [])
            self.assertEqual(summary["result"], "failed")

    def test_chunked_duplicate_and_unterminated_line_are_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runner.py"
            source.write_text('"TEST 0.5 PASSED"\n', encoding="utf-8")
            stream = StringIO()
            tally = IntegratedSuiteTally(stream)
            chunks = ("✅ TEST 0.", "5 PASS", "ED\nTEST ", "0.5 PASSED")
            for chunk in chunks:
                self.assertEqual(tally.write(chunk), len(chunk))
            tally.flush()

            summary = tally.build_summary(source)

            self.assertEqual(summary["duplicates"], ["0.5"])
            self.assertEqual(summary["result"], "failed")
            self.assertEqual(tally.build_summary(source), summary)
            self.assertEqual(stream.getvalue(), "".join(chunks))
            output = Path(directory) / "summary.json"
            tally.write_summary(summary, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), summary)

    def test_build_summary_and_diagnostics_do_not_create_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runner.py"
            source.write_text('"TEST 1 PASSED"\n', encoding="utf-8")
            tally = IntegratedSuiteTally(StringIO())
            tally.write("diagnostic: expected TEST 1 PASSED\nTEST 1 PASSED")

            summary = tally.build_summary(source)

            self.assertEqual(summary["duplicates"], [])
            self.assertEqual(summary["result"], "passed")
            self.assertEqual(tally.build_summary(source), summary)
            tally.write("\n")
            self.assertEqual(tally.build_summary(source), summary)

    def test_preserves_fractional_check_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runner.py"
            source.write_text('print("TEST 0.5 PASSED")\n', encoding="utf-8")
            tally = IntegratedSuiteTally(StringIO())
            tally.write("✅ TEST 0.5 PASSED - fractional check\n")

            summary = tally.build_summary(source)

            self.assertEqual(summary["expected"], ["0.5"])
            self.assertEqual(summary["passed"], ["0.5"])
            self.assertEqual(summary["result"], "passed")

    def test_rejects_source_without_expected_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runner.py"
            source.write_text("print('no markers')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no embedded check markers"):
                IntegratedSuiteTally(StringIO()).build_summary(source)

    def test_rejects_log_mentions_and_duplicate_source_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runner.py"
            source.write_text('print("TEST 1 PASSED")\n', encoding="utf-8")
            tally = IntegratedSuiteTally(StringIO())
            tally.write("diagnostic: not executed; expected TEST 1 PASSED\n")

            summary = tally.build_summary(source)

            self.assertEqual(summary["passed"], [])
            self.assertEqual(summary["missing"], ["1"])
            self.assertEqual(summary["result"], "failed")

            source.write_text('print("TEST 1 PASSED")\nprint("TEST 1 PASSED")\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate embedded check markers"):
                IntegratedSuiteTally(StringIO()).build_summary(source)


if __name__ == "__main__":
    unittest.main()
