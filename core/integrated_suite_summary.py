"""Capture structured outcomes for the embedded checks in the demo runner."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

_PASS_LINE = re.compile(r"^\s*(?:✅\s*)?TEST\s+(\d+(?:\.\d+)?)\s+PASSED\b")
_SKIP_LINE = re.compile(
    r"^\s*TEST\s+(\d+(?:\.\d+)?):.*(?:DISABLED|not\s+(?:available|enabled|active))",
    re.IGNORECASE,
)
_EXPECTED_IN_SOURCE = re.compile(r"TEST\s+(\d+(?:\.\d+)?)\s+PASSED")


def _sort_check_ids(values: set[str]) -> list[str]:
    return sorted(values, key=lambda value: tuple(int(part) for part in value.split(".")))


class IntegratedSuiteTally:
    """Tee stdout while tallying expected, passed and explicitly skipped checks."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._buffer = ""
        self.passed: set[str] = set()
        self.skipped: set[str] = set()
        self._terminal_counts: Counter[str] = Counter()

    def _record_line(self, line: str) -> None:
        passed = _PASS_LINE.search(line)
        if passed:
            self.passed.add(passed.group(1))
            self._terminal_counts[passed.group(1)] += 1
        skipped = _SKIP_LINE.search(line)
        if skipped:
            self.skipped.add(skipped.group(1))
            self._terminal_counts[skipped.group(1)] += 1

    def write(self, text: str) -> int:
        written = self._stream.write(text)
        self._buffer += text
        *complete_lines, self._buffer = self._buffer.split("\n")
        for line in complete_lines:
            self._record_line(line)
        return written if isinstance(written, int) else len(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def build_summary(self, source_path: str | Path) -> dict[str, Any]:
        if self._buffer:
            self._record_line(self._buffer)
            self._buffer = ""

        source = Path(source_path).read_text(encoding="utf-8")
        expected_markers = _EXPECTED_IN_SOURCE.findall(source)
        expected = set(expected_markers)
        if not expected:
            raise ValueError("no embedded check markers found in source")
        if len(expected_markers) != len(expected):
            raise ValueError("duplicate embedded check markers found in source")

        passed = self.passed & expected
        skipped = self.skipped & expected
        missing = expected - passed - skipped
        unexpected = (self.passed | self.skipped) - expected
        conflicting = passed & skipped
        duplicates = {check_id for check_id, count in self._terminal_counts.items() if count > 1}
        failed = bool(missing or unexpected or conflicting or duplicates)
        return {
            "schema_version": 1,
            "scope": "embedded checks of core/constitutional_brain.py (integrated suite)",
            "expected": _sort_check_ids(expected),
            "passed": _sort_check_ids(passed),
            "skipped": _sort_check_ids(skipped),
            "missing": _sort_check_ids(missing),
            "unexpected": _sort_check_ids(unexpected),
            "conflicting": _sort_check_ids(conflicting),
            "duplicates": _sort_check_ids(duplicates),
            "result": "failed" if failed else "passed",
        }

    @staticmethod
    def write_summary(summary: dict[str, Any], output_path: str | Path) -> None:
        Path(output_path).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
