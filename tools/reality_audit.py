#!/usr/bin/env python3
"""Auditoria conservadora das alegações e evidências do Nexus.

O relatório distingue evidência de execução, evidência estrutural e alegações não
comprovadas. Ele não tenta inferir capacidades cognitivas a partir de nomes de
classes ou de testes de demonstração.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
CORE = ROOT / "core" / "constitutional_brain.py"

CLAIMS = {
    "AGI/ASI": r"\bAGI\b|\bASI\b|Artificial General Intelligence|Superintelligence",
    "consciência/autoconsciência": r"consciên|consciousness|self-aware",
    "completude absoluta": r"100% Complete|Absolute Perfection|ZERO WARNINGS|ZERO EXCEPTIONS|62/62|80/80",
    "autonomia/soberania": r"sistema autônomo|autonomous|sovereign|soberania",
}


def parse_python(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))


def collect_static_evidence() -> dict:
    tree = parse_python(CORE)
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    passes = [node for node in functions if len([item for item in node.body if not (isinstance(item, ast.Expr) and isinstance(getattr(item, "value", None), ast.Constant) and isinstance(item.value.value, str))]) == 1 and isinstance([item for item in node.body if not (isinstance(item, ast.Expr) and isinstance(getattr(item, "value", None), ast.Constant) and isinstance(item.value.value, str))][0], ast.Pass)]
    constant_returns = []
    for node in functions:
        body = [item for item in node.body if not (isinstance(item, ast.Expr) and isinstance(getattr(item, "value", None), ast.Constant) and isinstance(item.value.value, str))]
        if len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(body[0].value, (ast.Constant, ast.Dict, ast.List, ast.Tuple)):
            constant_returns.append({"name": node.name, "line": node.lineno, "value": ast.unparse(body[0].value)})
    source = CORE.read_text(encoding="utf-8", errors="replace")
    embedded_tests = len(re.findall(r"TEST \d+ PASSED", source))
    return {
        "core_lines": len(source.splitlines()),
        "classes": len(classes),
        "functions": len(functions),
        "pass_only_functions": [{"name": node.name, "line": node.lineno} for node in passes],
        "constant_return_functions": constant_returns,
        "embedded_test_markers": embedded_tests,
        "optional_dependency_warnings": [
            dependency for dependency, marker in (("psutil", "psutil not available"), ("FAISS/SentenceTransformers", "FAISS/SentenceTransformers not available")) if marker in source
        ],
    }


def audit_claims() -> list[dict]:
    text = README.read_text(encoding="utf-8", errors="replace")
    findings = []
    for label, pattern in CLAIMS.items():
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        positive_matches = [
            match for match in matches
            if not re.search(r"não\s+(reivindica|afirma|demonstra)|não\s+constitui", text[max(0, match.start() - 180):match.start()], re.IGNORECASE)
        ]
        if positive_matches:
            findings.append({
                "claim": label,
                "occurrences": len(positive_matches),
                "status": "unsubstantiated_by_software_tests",
                "reason": "A documentação contém a alegação, mas os testes do repositório não constituem evidência suficiente dessa capacidade.",
            })
        elif matches:
            findings.append({
                "claim": label,
                "occurrences": len(matches),
                "status": "explicitly_disclaimed",
                "reason": "A documentação nega explicitamente que o projeto reivindique essa capacidade.",
            })
    return findings


def referenced_paths() -> list[dict]:
    text = README.read_text(encoding="utf-8", errors="replace")
    references = re.findall(r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_./-]+\.py)", text)
    return [{"path": item, "exists": (ROOT / item).exists()} for item in references]


def build_report() -> dict:
    evidence = collect_static_evidence()
    claims = audit_claims()
    paths = referenced_paths()
    return {
        "audit_version": "1.0",
        "scope": "conservative reality audit",
        "principle": "names, banners and demo asserts are not proof of general intelligence or production readiness",
        "static_evidence": evidence,
        "claims": claims,
        "referenced_paths": paths,
        "risks": [
            "README contains capability claims stronger than the measured evidence.",
            "Embedded demonstration tests are not independent contract or adversarial tests.",
            "Constant-return functions and pass-only functions require manual review.",
            "Optional dependency fallbacks can change behavior and guarantees.",
        ],
        "next_actions": [
            "Maintain independent contract tests for critical interfaces.",
            "Add property-based tests for queue invariants.",
            "Run Bandit and targeted mutation testing in CI.",
            "Replace absolute capability language with evidence-qualified documentation.",
        ],
    }


def markdown(report: dict) -> str:
    evidence = report["static_evidence"]
    lines = [
        "# Nexus — Auditoria conservadora de realidade",
        "",
        "> Este relatório separa o que foi observado no código e nos testes do que é apenas alegação documental. Nomes de módulos, banners e asserts de demonstração não são tratados como prova de AGI, ASI, consciência ou prontidão de produção.",
        "",
        "## Evidência estrutural",
        "",
        f"O núcleo analisado possui **{evidence['core_lines']} linhas**, **{evidence['classes']} classes** e **{evidence['functions']} funções**. Foram encontrados **{len(evidence['pass_only_functions'])}** métodos compostos apenas por `pass`, **{len(evidence['constant_return_functions'])}** funções com retorno constante simples e **{evidence['embedded_test_markers']}** marcadores de testes embutidos.",
        "",
        "| Área | Resultado | Interpretação |",
        "|---|---:|---|",
        f"| Testes embutidos marcados | {evidence['embedded_test_markers']} | Evidência de demos internas, não de capacidade geral |",
    ]
    lines.append(f"| Funções somente com `pass` | {len(evidence['pass_only_functions'])} | Lacunas explícitas que exigem revisão |")
    lines.append(f"| Funções com retorno constante | {len(evidence['constant_return_functions'])} | Possíveis heurísticas, stubs ou simplificações |")
    lines.append(f"| Dependências opcionais sinalizadas | {len(evidence['optional_dependency_warnings'])} | Fallbacks podem alterar o comportamento |")
    lines.extend(["", "## Alegações não comprovadas", ""])
    for claim in report["claims"]:
        lines.append(f"- **{claim['claim']}**: {claim['status']}. {claim['reason']}")
    lines.extend(["", "## Referências quebradas ou verificadas", ""])
    for item in report["referenced_paths"]:
        lines.append(f"- `{item['path']}`: {'existe' if item['exists'] else 'não encontrado'}.")
    lines.extend(["", "## Riscos prioritários", ""])
    for item in report["risks"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Próximas ações", ""])
    for item in report["next_actions"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    report = build_report()
    (ROOT / "AUDIT_REALITY_REPORT.md").write_text(markdown(report), encoding="utf-8")
    (ROOT / "audit_reality.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(markdown(report))
