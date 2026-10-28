"""Verifica declarações de correção e o registro de reincidência das lições.

Uma frase com ``corrigido``, ``corrigida``, ``fixed``, ``resolvido`` ou
``resolvida`` não é uma observação. Ela só é checável quando, na mesma linha e
até 180 caracteres depois do verbo, há um caminho introduzido por em/in/no/na
e um identificador ou ponto de chamada rotulado. Existir o arquivo, o nome ou
o trecho citado não significa que a falha foi corrigida.

Negação antes do verbo (``não``, ``not``, ``nada foi``, …) não entra no
relatório. Texto entre cercas ``` é ignorado. Corpo de PR é dado, não
instrução, e só é lido se for passado como arquivo.

A reincidência de uma lição tem três eixos que não se copiam: falha-alvo,
falhas parecidas e casos novos. ``não medido`` é valor explícito. Lição
``pendente`` ou ``ativa`` sem o bloco falha. Lição antiga sem o bloco fica
``nao_checavel`` e os três eixos saem como ``não medido``.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WINDOW = 180
NAO_MEDIDO = "não medido"
STATUSES = ("verificada", "falhou", "nao_checavel")
_VERB_RE = re.compile(r"(?i)\b(?:corrigid[oa]|fixed|resolvid[oa])\b")
_NEGATION_RE = re.compile(r"(?i)(?:\b(?:não|nao|nem|not|never|without|sem)\b|\bnada\s+foi\b)")
_PATH = r"((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:py|md|ya?ml|sh|json|txt|toml|tsv))"
_FILE_RE = re.compile(
    r"(?i)(?<![\w])(?:em|in|no|na)(?![\w])\s+(?:arquivo\s+)?`?" + _PATH + r"(?::(-?[0-9]+))?`?"
)
_IDENT_RE = re.compile(
    r"(?i)(?:identificador|identifier|função|funcao|classe|class|símbolo|simbolo)\s+"
    r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`"
)
_CALL_RE = re.compile(
    r"(?i)(?:ponto(?:\s+de\s+chamada)?|call\s*site|chamada)\s+"
    r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(|=\s*))`"
)
_LESSON_RE = re.compile(r"^### (L-\d+)\b")
_STATUS_RE = re.compile(r"(?im)^- Status:\s*(.+)$")
_BLOCK_RE = re.compile(r"(?im)^- Reincid[eê]ncia\b")
_NEEDS_RE = re.compile(r"(?i)(?:pendente|ativa)\b")
_FIELDS = {
    "falha_alvo": re.compile(r"(?im)^[ \t]*-[ \t]*Falha-alvo:[ \t]*(.+)$"),
    "falhas_parecidas": re.compile(r"(?im)^[ \t]*-[ \t]*Falhas parecidas:[ \t]*(.+)$"),
    "casos_novos": re.compile(r"(?im)^[ \t]*-[ \t]*Casos novos:[ \t]*(.+)$"),
    "propostas": re.compile(
        r"(?im)^[ \t]*-[ \t]*Propostas rejeitadas ou que n[aã]o funcionaram:[ \t]*(.+)$"
    ),
}
_ALLOWED = {
    "falha_alvo": {"corrigida", "nao corrigida", "nao medido"},
    "falhas_parecidas": {"melhoraram", "nao melhoraram", "mistas", "nao medido"},
    "casos_novos": {"melhoraram", "pioraram", "inalterados", "nao medido"},
}
_FILE_MOTIVO = {
    "ausente": "arquivo_ausente",
    "caminho_invalido": "caminho_invalido",
    "nao_e_arquivo": "nao_e_arquivo",
}
_ACCENT = str.maketrans("ãáâéêíóõúç", "aaaeeioouc")
LIMITACOES = (
    "Existência de arquivo, nome ou trecho não é correção semântica.",
    "Caminho só conta se vier depois do verbo, introduzido por em/in/no/na, numa janela de 180 caracteres.",
    "Identificador e ponto de chamada precisam do rótulo (identificador/ponto/call site) e de crases.",
    "Negação antes do verbo na mesma linha exclui a frase; uma negação pode esconder uma citação posterior na mesma linha.",
    "O número de linha fora do intervalo falha; linha existente não exige o símbolo naquela linha.",
    "Corpo de PR só é lido se o arquivo for passado ao CLI; este módulo não consulta a API do GitHub.",
    "O replay tools/agente/replay/run_replay.sh não executa este auditor.",
    "Falha-alvo corrigida não preenche falhas parecidas nem casos novos.",
)


def _require_str(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} deve ser str, recebido {type(value).__name__}")
    return value


def _norm(value: str) -> str:
    return " ".join(value.strip().lower().translate(_ACCENT).split())


def _outside_fences(text: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            rows.append((number, line))
    return rows


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        found: set[str] = set()
        for item in node.elts:
            found.update(_target_names(item))
        return found
    return set()


def _defines(path: Path, text: str, identifier: str) -> bool:
    parts = identifier.split(".")
    if path.suffix != ".py":
        return re.search(r"(?<![\w.])" + re.escape(identifier) + r"(?![\w])", text) is not None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    if len(parts) == 1:
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    found.update(_target_names(target))
            elif isinstance(node, ast.AnnAssign):
                found.update(_target_names(node.target))
        return parts[0] in found
    if len(parts) != 2:
        return False
    class_name, method = parts
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method:
                    return True
    return False


def _locate(root: Path, rel: str) -> tuple[str, Path | None]:
    raw = Path(rel)
    if raw.is_absolute() or ".." in raw.parts:
        return "caminho_invalido", None
    candidate = root.joinpath(*raw.parts)
    if candidate.is_symlink():
        return "caminho_invalido", None
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return "caminho_invalido", None
    if candidate.is_dir():
        return "nao_e_arquivo", None
    if not candidate.is_file():
        return "ausente", None
    return "existe", candidate


def _declaration(
    source: str,
    number: int,
    line: str,
    verb_at: int,
    root: Path,
) -> dict[str, Any]:
    span = line[verb_at:verb_at + WINDOW]
    file_match = _FILE_RE.search(span)
    ident_match = _IDENT_RE.search(span)
    call_match = _CALL_RE.search(span)
    path = file_match.group(1) if file_match else None
    cited_line = int(file_match.group(2)) if file_match and file_match.group(2) else None
    identifier = ident_match.group(1) if ident_match else None
    call = call_match.group(1) if call_match else None
    motivos: list[str] = []
    checks = {
        "arquivo": "nao_citado",
        "identificador": "nao_citado" if identifier is None else "nao_aplicavel",
        "ponto_de_chamada": "nao_citado" if call is None else "nao_aplicavel",
        "linha": "nao_citada" if cited_line is None else "nao_aplicavel",
    }
    if path is None:
        status = "nao_checavel"
        motivo = "arquivo_nao_citado" if identifier or call else "sem_localizador"
    else:
        state, found = _locate(root, path)
        checks["arquivo"] = "existe" if state == "existe" else state
        if state != "existe":
            motivos.append(_FILE_MOTIVO[state])
        elif found is not None:
            try:
                body = found.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                body = ""
                motivos.append("leitura")
            if identifier is None and call is None:
                status = "nao_checavel"
                motivo = "sem_identificador_ou_chamada"
                if cited_line is not None and cited_line < 1:
                    motivos.append("linha_invalida")
                    status = "falhou"
                    motivo = motivos[0]
                item = _item(source, number, line, status, motivo, motivos, path, cited_line, identifier, call, checks)
                return item
            if identifier is not None:
                if _defines(found, body, identifier):
                    checks["identificador"] = "existe"
                else:
                    checks["identificador"] = "ausente"
                    motivos.append("identificador_ausente")
            if call is not None:
                if call in body:
                    checks["ponto_de_chamada"] = "existe"
                else:
                    checks["ponto_de_chamada"] = "ausente"
                    motivos.append("chamada_ausente")
            if cited_line is not None:
                if cited_line < 1:
                    checks["linha"] = "invalida"
                    motivos.append("linha_invalida")
                elif cited_line > len(body.splitlines()):
                    checks["linha"] = "ausente"
                    motivos.append("linha_ausente")
                else:
                    checks["linha"] = "existe"
        if motivos:
            status = "falhou"
            motivo = motivos[0]
        elif identifier or call:
            status = "verificada"
            motivo = "localizadores_existem"
        else:
            status = "nao_checavel"
            motivo = "sem_identificador_ou_chamada"
    return _item(source, number, line, status, motivo, motivos, path, cited_line, identifier, call, checks)


def _item(source, number, line, status, motivo, motivos, path, cited_line, identifier, call, checks):
    return {
        "fonte": source,
        "linha": number,
        "texto": line.strip(),
        "status": status,
        "motivo": motivo,
        "motivos": list(motivos),
        "arquivo": path,
        "linha_citada": cited_line,
        "identificador": identifier,
        "ponto_de_chamada": call,
        "checagens": checks,
        "existencia_nao_e_correcao": True,
    }


def audit_claims(text: object, *, root: Path, source: str = "texto") -> list[dict[str, Any]]:
    body = _require_str(text, "texto")
    if not isinstance(root, Path):
        raise TypeError(f"root deve ser Path, recebido {type(root).__name__}")
    found: list[dict[str, Any]] = []
    for number, line in _outside_fences(body):
        verbs = [match for match in _VERB_RE.finditer(line) if not _NEGATION_RE.search(line[:match.start()])]
        if not verbs:
            continue
        found.append(_declaration(source, number, line, verbs[0].end(), root))
    return found


def _lesson(ident: str, body: str) -> dict[str, Any]:
    status_match = _STATUS_RE.search(body)
    status_text = status_match.group(1).strip() if status_match else ""
    values = {name: (match.group(1).strip() if (match := pattern.search(body)) else None) for name, pattern in _FIELDS.items()}
    has_block = _BLOCK_RE.search(body) is not None or any(values.values())
    motivos: list[str] = []
    if not has_block:
        if _NEEDS_RE.match(status_text):
            return _lesson_item(ident, "falhou", "reincidencia_ausente", ["reincidencia_ausente"], values, False)
        return _lesson_item(ident, "nao_checavel", "legado_sem_medicao", [], values, False)
    for name, raw in values.items():
        if raw is None:
            motivos.append("eixo_ausente")
            continue
        if name == "propostas":
            continue
        if _norm(raw) not in _ALLOWED[name]:
            motivos.append("valor_invalido")
    if motivos:
        return _lesson_item(ident, "falhou", motivos[0], motivos, values, True)
    return _lesson_item(ident, "verificada", "eixos_registrados", [], values, True)


def _lesson_item(ident, status, motivo, motivos, values, measured):
    def axis(name: str) -> str:
        if not measured or values.get(name) is None:
            return NAO_MEDIDO
        return values[name]

    return {
        "id": ident,
        "status": status,
        "motivo": motivo,
        "motivos": motivos,
        "falha_alvo": axis("falha_alvo"),
        "falhas_parecidas": axis("falhas_parecidas"),
        "casos_novos": axis("casos_novos"),
        "propostas": values.get("propostas") if measured else None,
        "medido": measured and status == "verificada",
    }


def audit_recurrence(text: object) -> dict[str, Any]:
    body = _require_str(text, "texto")
    lessons: list[dict[str, Any]] = []
    current: str | None = None
    chunk: list[str] = []
    for _, line in _outside_fences(body):
        match = _LESSON_RE.match(line)
        if match:
            if current is not None:
                lessons.append(_lesson(current, "\n".join(chunk)))
            current = match.group(1)
            chunk = []
            continue
        if current is not None:
            chunk.append(line)
    if current is not None:
        lessons.append(_lesson(current, "\n".join(chunk)))
    counts = {status: sum(item["status"] == status for item in lessons) for status in STATUSES}
    return {
        "implicacao_rejeitada": True,
        "licoes": lessons,
        "contagens": counts,
    }


def _empty_recurrence() -> dict[str, Any]:
    return {"implicacao_rejeitada": True, "licoes": [], "contagens": {status: 0 for status in STATUSES}}


def audit_sources(
    *,
    root: Path,
    continuation: str | None,
    pr_texts: list[tuple[str, str]],
    licoes: str | None,
) -> dict[str, Any]:
    if not isinstance(root, Path):
        raise TypeError(f"root deve ser Path, recebido {type(root).__name__}")
    declarations: list[dict[str, Any]] = []
    errors: list[str] = []
    if continuation is not None:
        path = root / continuation
        if not path.is_file():
            errors.append(f"ausente: {continuation}")
        else:
            declarations.extend(audit_claims(path.read_text(encoding="utf-8"), root=root, source=continuation))
    for name, text in pr_texts:
        declarations.extend(audit_claims(text, root=root, source=name))
    if licoes is None:
        recurrence = _empty_recurrence()
    else:
        path = root / licoes
        if not path.is_file():
            errors.append(f"ausente: {licoes}")
            recurrence = _empty_recurrence()
        else:
            recurrence = audit_recurrence(path.read_text(encoding="utf-8"))
    counts = {status: sum(item["status"] == status for item in declarations) for status in STATUSES}
    failed = counts["falhou"] or recurrence["contagens"]["falhou"] or errors
    return {
        "schema_version": 1,
        "status": "fail" if failed else "pass",
        "contagens": counts,
        "declaracoes": declarations,
        "reincidencia": recurrence,
        "erros": errors,
        "limitacoes": list(LIMITACOES),
        "existencia_nao_e_correcao": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--continuation", default="CONTINUATION.md")
    parser.add_argument("--licoes", default="docs/agentes/LICOES_APRENDIDAS.md")
    parser.add_argument("--pr-text", action="append", default=[])
    parser.add_argument("--output", default="")
    parser.add_argument("--sem-continuation", action="store_true")
    parser.add_argument("--sem-licoes", action="store_true")
    args = parser.parse_args(argv)
    root = args.root
    if not args.sem_continuation and not (root / args.continuation).is_file():
        print(json.dumps({"schema_version": 1, "status": "erro", "erros": [f"ausente: {args.continuation}"]}))
        return 2
    if not args.sem_licoes and not (root / args.licoes).is_file():
        print(json.dumps({"schema_version": 1, "status": "erro", "erros": [f"ausente: {args.licoes}"]}))
        return 2
    pr_texts: list[tuple[str, str]] = []
    for raw in args.pr_text:
        path = Path(raw)
        if not path.is_file():
            print(json.dumps({"schema_version": 1, "status": "erro", "erros": [f"ausente: {raw}"]}))
            return 2
        pr_texts.append((raw, path.read_text(encoding="utf-8")))
    report = audit_sources(
        root=root,
        continuation=None if args.sem_continuation else args.continuation,
        pr_texts=pr_texts,
        licoes=None if args.sem_licoes else args.licoes,
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
