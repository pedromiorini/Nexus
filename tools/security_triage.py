"""Gera uma triagem conservadora do relatório JSON do Bandit.

A classificação é apenas operacional: nenhum achado é apagado ou marcado como
corrigido automaticamente. O relatório exige revisão humana para qualquer
mudança de severidade ou supressão.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DISPOSITIONS = {
    "B101": "embedded_demo_assert_review",
    "B110": "silent_exception_review",
    "B311": "simulation_only_random_review",
    "B404": "controlled_subprocess_review",
    "B603": "controlled_subprocess_review",
    "B608": "sql_construction_high_priority_review",
}


def classify(finding: dict[str, Any]) -> str:
    return DISPOSITIONS.get(finding.get("test_id", ""), "manual_review")


def build_report(payload: dict[str, Any]) -> str:
    findings = payload.get("results", [])
    by_disposition = Counter(classify(item) for item in findings)
    by_severity = Counter(item.get("issue_severity", "UNKNOWN") for item in findings)
    lines = [
        "# Bandit Security Triage",
        "",
        "> This is a conservative inventory, not a clearance report. Findings remain open until reviewed and fixed or explicitly justified.",
        "",
        f"- Total findings: **{len(findings)}**",
        f"- Severity counts: {', '.join(f'{key}={value}' for key, value in sorted(by_severity.items())) or 'none'}",
        "",
        "## Dispositions",
        "",
        "| Disposition | Count | Meaning |",
        "|---|---:|---|",
    ]
    meanings = {
        "embedded_demo_assert_review": "Assertions inside the legacy __main__ demonstration block; keep them out of production contracts and migrate incrementally.",
        "silent_exception_review": "Silent exception handling; review whether fallback behavior hides failures.",
        "simulation_only_random_review": "Randomness reviewed as simulation/heuristic behavior; keep it out of secrets and security decisions.",
        "controlled_subprocess_review": "Mutation harness subprocess; keep inputs fixed and review execution boundaries.",
        "sql_construction_high_priority_review": "SQL construction in legacy core; highest-priority manual review for parameterization and trust boundaries.",
        "manual_review": "No project-specific disposition; manual review required.",
    }
    for disposition, count in sorted(by_disposition.items()):
        lines.append(f"| `{disposition}` | {count} | {meanings.get(disposition, meanings['manual_review'])} |")
    lines += ["", "## Findings", "", "| ID | Severity | File | Line | Disposition |", "|---|---|---|---:|---|"]
    for item in findings:
        lines.append(
            f"| `{item.get('test_id', '')}` | {item.get('issue_severity', '')} | "
            f"`{item.get('filename', '')}` | {item.get('line_number', '')} | `{classify(item)}` |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Bandit JSON report")
    parser.add_argument("output", help="Markdown triage report")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    Path(args.output).write_text(build_report(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
