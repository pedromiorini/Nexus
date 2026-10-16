"""Gera uma triagem conservadora do relatório JSON do Bandit.

A classificação é apenas operacional: nenhum achado é apagado ou marcado como
corrigido automaticamente. O relatório exige revisão humana para qualquer
mudança de severidade ou supressão.

Os achados são agrupados por escopo (``Classe.método`` ou bloco ``__main__``) em vez de
número de linha: números de linha mudam a cada edição do núcleo e deixavam o documento
desatualizado. Ao regenerar um arquivo existente, o bloco ``NEXUS-CURRENT-STATE`` dele é
preservado (ele é validado por ``tools/state_consistency_audit.py``).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
STATE_BLOCK_RE = re.compile(r"<!--\s*NEXUS-CURRENT-STATE\s*\n.*?\n\s*-->[ \t]*\n?", re.DOTALL)

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


def _is_main_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


@lru_cache(maxsize=None)
def _scope_spans(path: str) -> tuple[tuple[int, int, str], ...]:
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return ()
    spans: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                spans.append((child.lineno, child.end_lineno or child.lineno, f"{prefix}{child.name}"))
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.append((child.lineno, child.end_lineno or child.lineno, f"{prefix}{child.name}"))
                walk(child, f"{prefix}{child.name}.")
            else:
                if prefix == "" and _is_main_guard(child):
                    spans.append((child.lineno, child.end_lineno or child.lineno, "<__main__ block>"))
                walk(child, prefix)

    walk(tree, "")
    return tuple(spans)


def resolve_scope(filename: str, line: Any, root: Path | None = None) -> str:
    """Nome estável do escopo que contém a linha (o mais interno), ou ``<module>``."""
    spans = _scope_spans(str((root or ROOT) / filename))
    if not spans:
        return "<unresolved>"
    try:
        number = int(line)
    except (TypeError, ValueError):
        return "<unresolved>"
    containing = [span for span in spans if span[0] <= number <= span[1]]
    return min(containing, key=lambda span: span[1] - span[0])[2] if containing else "<module>"


def build_report(payload: dict[str, Any], root: Path | None = None) -> str:
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
    grouped = Counter(
        (
            item.get("test_id", ""),
            item.get("issue_severity", ""),
            item.get("filename", ""),
            resolve_scope(item.get("filename", ""), item.get("line_number"), root),
            classify(item),
        )
        for item in findings
    )
    lines += ["", "## Findings", "", "Agrupados por escopo (sem números de linha, que mudam a cada edição).", "",
              "| ID | Severity | File | Scope | Count | Disposition |", "|---|---|---|---|---:|---|"]
    for (test_id, severity, filename, scope, disposition), count in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2], item[0][3])):
        lines.append(f"| `{test_id}` | {severity} | `{filename}` | `{scope}` | {count} | `{disposition}` |")
    return "\n".join(lines) + "\n"


def merge_state_block(report: str, existing: str) -> str:
    """Reinsere, sob o título, o bloco NEXUS-CURRENT-STATE de um arquivo já existente."""
    block = STATE_BLOCK_RE.search(existing)
    if not block:
        return report
    title = "# Bandit Security Triage\n\n"
    return report.replace(title, title + block.group(0).rstrip("\n") + "\n\n", 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Bandit JSON report")
    parser.add_argument("output", help="Markdown triage report")
    parser.add_argument("--check", action="store_true",
                        help="não escreve: falha se o arquivo de saída existente divergir do relatório gerado")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output = Path(args.output)
    report = build_report(payload)
    existing = output.read_text(encoding="utf-8") if output.exists() else ""
    if existing:
        report = merge_state_block(report, existing)
    if args.check:
        if report.strip() != existing.strip():
            print(f"{output} está desatualizado; regenere com: python tools/security_triage.py {args.input} {args.output}", file=sys.stderr)
            return 1
        return 0
    output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
