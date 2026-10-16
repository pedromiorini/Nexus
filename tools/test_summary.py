"""Executa a suíte unittest e grava um resumo factual para o manifesto."""
from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class SummaryResult(unittest.TextTestResult):
    """Resultado unittest com contagens explícitas para consumo pelo CI."""

    @staticmethod
    def _case_id(test: object) -> int:
        return id(getattr(test, "test_case", test))

    @property
    def summary(self) -> dict[str, int | str]:
        failed_cases = {
            self._case_id(test)
            for test, _ in (*self.failures, *self.errors)
        }
        unexpected_successes = {self._case_id(test) for test in self.unexpectedSuccesses}
        failed_cases.update(unexpected_successes)
        skipped_cases = {self._case_id(test) for test, _ in self.skipped}
        skipped_cases.difference_update(failed_cases)
        failed = len(failed_cases)
        skipped = len(skipped_cases)
        return {
            "executed": self.testsRun,
            "passed": self.testsRun - failed - skipped,
            "failed": failed,
            "skipped": skipped,
            "unexpected_successes": len(unexpected_successes),
            "result": "passed" if self.wasSuccessful() else "failed",
        }


class SummaryRunner(unittest.TextTestRunner):
    resultclass = SummaryResult


def run_suite(pattern: str = "test_*.py") -> dict[str, int | str]:
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern=pattern)
    runner = SummaryRunner(stream=sys.stdout, verbosity=1)
    result = runner.run(suite)
    return result.summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="test-summary.json")
    parser.add_argument("--pattern", default="test_*.py")
    args = parser.parse_args(argv)
    summary = run_suite(args.pattern)
    output = ROOT / args.output
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
