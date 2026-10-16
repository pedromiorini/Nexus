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

# Frases assertivas no código-fonte (docstrings, prints, banners) que contradizem a postura
# conservadora do README. Nomes de classes e rótulos neutros não entram aqui.
CODE_OVERCLAIMS = {
    "autoconsciência afirmada": r"REAL\s+Self-?Awareness|is\s+self-aware|self-aware!",
    "funcionalidade total afirmada": r"SYSTEM\s+100%\s+FUNCTIONAL|100%\s+FUNCTIONAL\s+SYSTEM",
    "consciência genuína afirmada": r"genuine(?:ly)?\s+(?:conscious|sentient|aware)|truly\s+(?:conscious|sentient|self-aware)",
}

# Inventário por nível (não é gate por ocorrência; o limite de "strong_capability_claim" fica em
# test_reality_audit.py e só pode diminuir). Níveis:
#   strong_capability_claim  ênfase de capacidade ("REAL <Capacidade>", "REAL e FUNCIONAL", "100% funcional")
#   scientific_terminology   nomes de teorias/constructos modelados (informativo, não é alegação por si só)
#   neutral_identifier       nomes históricos de classe (renomear exige alias de compatibilidade)
CODE_CLAIM_LEVELS = (
    ("strong_capability_claim", "ênfase 'REAL <Capacidade>'", r"\bREAL\s+(?=[A-Z][a-z])"),
    ("strong_capability_claim", "'REAL e FUNCIONAL'", r"\bREAL\s+e\s+FUNCIONAL\b"),
    ("strong_capability_claim", "'100% funcional'", r"100%\s+(?:functional|funcional)"),
    ("strong_capability_claim", "máquina bayesiana afirmada", r"Bayesian prediction machine"),
    ("scientific_terminology", "consciência/consciousness", r"(?i)\b(?:consciousness|consci[êe]ncia)\b"),
    ("scientific_terminology", "teorias nomeadas", r"\b(?:Global Workspace|Higher[- ]Order Thought|Predictive Coding|Integrated Information|Theory of Mind)\b"),
    ("neutral_identifier", "classes com prefixo Real", r"\bclass\s+Real[A-Z]\w*"),
)
CLAIM_LEVELS = ("strong_capability_claim", "scientific_terminology", "neutral_identifier")

# Negação na mesma sentença que antecede a menção ("não reivindica", "não constituem prova de",
# "sem reivindicar", "without claiming"...).
NEGATION = re.compile(
    r"\b(?:não|nao|sem|nem|without|not|no|never)\b[^.\n;]{0,80}?\b(?:reivindic\w*|afirm\w*|demonstr\w*|constitu\w*|claim\w*|prova\w*|garant\w*)",
    re.IGNORECASE,
)


def parse_python(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))


def iter_functions(tree: ast.AST):
    """Percorre as funções com nome qualificado (Classe.metodo), estável a edições acima delas."""
    def walk(node: ast.AST, prefix: str):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                yield from walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield f"{prefix}{child.name}", child
                yield from walk(child, f"{prefix}{child.name}.<locals>.")
            else:
                yield from walk(child, prefix)
    yield from walk(tree, "")


def is_pure_literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


def collect_static_evidence() -> dict:
    tree = parse_python(CORE)
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    passes = [node for node in functions if len([item for item in node.body if not (isinstance(item, ast.Expr) and isinstance(getattr(item, "value", None), ast.Constant) and isinstance(item.value.value, str))]) == 1 and isinstance([item for item in node.body if not (isinstance(item, ast.Expr) and isinstance(getattr(item, "value", None), ast.Constant) and isinstance(item.value.value, str))][0], ast.Pass)]
    qualnames = {id(node): name for name, node in iter_functions(tree)}
    strict_constants = []
    dynamic_literals = []
    for node in functions:
        body = [item for item in node.body if not (isinstance(item, ast.Expr) and isinstance(getattr(item, "value", None), ast.Constant) and isinstance(item.value.value, str))]
        if len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(body[0].value, (ast.Constant, ast.Dict, ast.List, ast.Tuple)):
            if is_pure_literal(body[0].value):
                strict_constants.append({"qualname": qualnames[id(node)], "value": ast.unparse(body[0].value)})
            else:
                dynamic_literals.append(qualnames[id(node)])
    source = CORE.read_text(encoding="utf-8", errors="replace")
    embedded_tests = len(re.findall(r"TEST \d+(?:\.\d+)? PASSED", source))
    return {
        "core_lines": len(source.splitlines()),
        "classes": len(classes),
        "functions": len(functions),
        "pass_only_functions": [{"qualname": qualnames[id(node)]} for node in passes],
        "constant_return_functions": strict_constants,
        "dynamic_literal_return_functions": sorted(dynamic_literals),
        "embedded_test_markers": embedded_tests,
        "optional_dependency_warnings": [
            dependency for dependency, marker in (("psutil", "psutil not available"), ("FAISS/SentenceTransformers", "FAISS/SentenceTransformers not available")) if marker in source
        ],
    }


def sentence_start(text: str, position: int) -> int:
    return max(text.rfind(".", 0, position), text.rfind("\n", 0, position), text.rfind(";", 0, position)) + 1


def audit_claims(text: str | None = None) -> list[dict]:
    if text is None:
        text = README.read_text(encoding="utf-8", errors="replace")
    findings = []
    for label, pattern in CLAIMS.items():
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        positive_matches = [
            match for match in matches
            if not NEGATION.search(text[sentence_start(text, match.start()):match.start()])
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


def audit_code_overclaims(source: str | None = None) -> list[dict]:
    if source is None:
        source = CORE.read_text(encoding="utf-8", errors="replace")
    findings = []
    for label, pattern in CODE_OVERCLAIMS.items():
        occurrences = len(re.findall(pattern, source, flags=re.IGNORECASE))
        if occurrences:
            findings.append({
                "claim": label,
                "occurrences": occurrences,
                "status": "overclaim_in_code",
                "reason": "Texto do código afirma uma capacidade que os testes do repositório não demonstram.",
            })
    return findings


def audit_code_claim_inventory(source: str | None = None) -> dict:
    if source is None:
        source = CORE.read_text(encoding="utf-8", errors="replace")
    items = []
    totals = {level: 0 for level in CLAIM_LEVELS}
    for level, label, pattern in CODE_CLAIM_LEVELS:
        matches = list(re.finditer(pattern, source))
        if not matches:
            continue
        examples: list[str] = []
        for match in matches:
            start = source.rfind("\n", 0, match.start()) + 1
            end = source.find("\n", match.end())
            line = source[start:end if end != -1 else len(source)].strip()[:70]
            if line not in examples:
                examples.append(line)
            if len(examples) == 3:
                break
        items.append({"level": level, "label": label, "occurrences": len(matches), "examples": examples})
        totals[level] += len(matches)
    return {"by_level": totals, "items": items}


def referenced_paths() -> list[dict]:
    text = README.read_text(encoding="utf-8", errors="replace")
    references = re.findall(r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_./-]+\.py)", text)
    return [{"path": item, "exists": (ROOT / item).exists()} for item in references]


def build_report() -> dict:
    evidence = collect_static_evidence()
    claims = audit_claims()
    code_claims = audit_code_overclaims()
    inventory = audit_code_claim_inventory()
    paths = referenced_paths()
    risks = []
    if any(item["status"] == "unsubstantiated_by_software_tests" for item in claims):
        risks.append("README contains capability claims stronger than the measured evidence.")
    if code_claims:
        risks.append("Source strings assert capabilities stronger than the measured evidence (see code_claims).")
    risks.extend([
        "Embedded demonstration tests are not independent contract or adversarial tests.",
        "Constant-return functions and pass-only functions require manual review.",
        "Optional dependency fallbacks can change behavior and guarantees.",
    ])
    return {
        "audit_version": "1.1",
        "scope": "conservative reality audit",
        "principle": "names, banners and demo asserts are not proof of general intelligence or production readiness",
        "static_evidence": evidence,
        "claims": claims,
        "code_claims": code_claims,
        "code_claim_inventory": inventory,
        "referenced_paths": paths,
        "risks": risks,
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
    lines.append(f"| Funções com retorno constante (estrito: só literais) | {len(evidence['constant_return_functions'])} | Possíveis heurísticas, stubs ou simplificações |")
    lines.append(f"| Funções que retornam dict/list/tuple literal com valores dinâmicos | {len(evidence['dynamic_literal_return_functions'])} | Em geral getters e estatísticas; não são stubs por si só |")
    lines.append(f"| Padrões proibidos de alegação no código (gate: zero) | {sum(item['occurrences'] for item in report['code_claims'])} | Cobre só as frases listadas em CODE_OVERCLAIMS; não significa ausência de linguagem de capacidade |")
    levels = report["code_claim_inventory"]["by_level"]
    lines.append(f"| Frases fortes de capacidade no código (inventário; só pode diminuir) | {levels['strong_capability_claim']} | Ênfase \"REAL <Capacidade>\", \"REAL e FUNCIONAL\", \"100% funcional\"; limite em test_reality_audit.py |")
    lines.append(f"| Terminologia científica no código (informativo) | {levels['scientific_terminology']} | Nomes de constructos modelados; não são alegações por si só |")
    lines.append(f"| Classes com prefixo Real (identificadores) | {levels['neutral_identifier']} | Nome histórico; renomear exige alias de compatibilidade |")
    lines.append(f"| Dependências opcionais sinalizadas | {len(evidence['optional_dependency_warnings'])} | Fallbacks podem alterar o comportamento |")
    lines.extend(["", "## Alegações (README e código-fonte)", ""])
    for claim in report["claims"] + report["code_claims"]:
        lines.append(f"- **{claim['claim']}**: {claim['status']}. {claim['reason']}")
    if not any(item["status"] != "explicitly_disclaimed" for item in report["claims"] + report["code_claims"]):
        lines.append("- Nenhuma alegação assertiva sem ressalva foi encontrada no README nem no código-fonte.")
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
