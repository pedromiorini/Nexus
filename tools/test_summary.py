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

    @property
    def summary(self) -> dict[str, int | str]:
        return {
            "executed": self.testsRun,
            "passed": self.testsRun - len(self.failures) - len(self.errors) - len(self.skipped),
            "failed": len(self.failures) + len(self.errors),
            "skipped": len(self.skipped),
            "result": "passed" if not self.failures and not self.errors else "failed",
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
