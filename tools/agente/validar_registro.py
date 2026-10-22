"""Valida docs/agentes/REGISTRO_AGENTES.yaml e CLAIMS.yaml.

Falha (rc 1) em: campo obrigatório ausente, hash/versão de instruções divergente,
claim expirado (modo padrão) ou claims ativos sobrepostos no mesmo arquivo.
Avisos (rc 0): autor de commit não declarado; claim expirado com --expirados aviso.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = Path("docs/agentes/schema/registro_agentes.schema.json")
_GLOB_WORK_LIMIT = 100_000
_REGISTRY_GLOB_WORK_LIMIT = 1_000_000


class _GlobBudgetExceeded(ValueError):
    """An unresolved intersection must never be reported as disjoint."""


class _GlobBudget:
    def __init__(self, limit: int = _GLOB_WORK_LIMIT):
        self.remaining = limit

    def charge(self, amount: int) -> None:
        if amount > self.remaining:
            raise _GlobBudgetExceeded("orçamento de trabalho de globs excedido; reduza/divida os claims")
        self.remaining -= amount


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: raiz deve ser um mapeamento")
    return data


def _required(root: Path, name: str) -> list[str]:
    schema = json.loads((root / SCHEMA).read_text(encoding="utf-8"))
    node = schema if name == "raiz" else schema["$defs"][name]
    return list(node["required"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _glob_masks(pattern: str, alphabet: list[str], budget: _GlobBudget) -> list[int | None]:
    """One character mask per fnmatch atom; None is a star (epsilon/loop)."""
    masks: list[int | None] = []
    i = 0
    while i < len(pattern):
        start = i
        char = pattern[i]
        i += 1
        if char == "*":
            if not masks or masks[-1] is not None:
                masks.append(None)
            continue
        if char == "[":
            # Match fnmatch's closing-bracket rules, including [!]] and [[];
            # an unclosed '[' is literal. Translation handles ranges/negation.
            end = i
            if end < len(pattern) and pattern[end] == "!":
                end += 1
            if end < len(pattern) and pattern[end] == "]":
                end += 1
            while end < len(pattern) and pattern[end] != "]":
                budget.charge(1)  # Unclosed '[' atoms can repeatedly scan a suffix.
                end += 1
            if end < len(pattern):
                i = end + 1
        matcher = re.compile(fnmatch.translate(pattern[start:i]))
        masks.append(sum(1 << n for n, value in enumerate(alphabet) if matcher.fullmatch(value)))
    return masks


def _glob_intersects(a: str, b: str, budget: _GlobBudget | None = None) -> bool:
    # fnmatch atoms distinguish only literals and codepoint ranges. A boundary
    # and its successor represent every induced interval, including complements
    # and Unicode; this is a symbolic alphabet, not a sampled filename search.
    budget = budget if budget is not None else _GlobBudget()
    budget.charge(len(a) + len(b))
    points = {0}
    for char in a + b:
        points.add(ord(char))
        if ord(char) < sys.maxunicode:
            points.add(ord(char) + 1)
    alphabet = [chr(point) for point in sorted(points)]
    budget.charge((len(a) + len(b)) * len(alphabet))
    left, right = _glob_masks(a, alphabet, budget), _glob_masks(b, alphabet, budget)
    any_char = (1 << len(alphabet)) - 1
    pending = [(0, 0)]
    seen = set()
    while pending:
        budget.charge(1)
        i, j = pending.pop()
        if (i, j) in seen:
            continue
        seen.add((i, j))
        if i == len(left) and j == len(right):
            return True
        if i < len(left) and left[i] is None:
            pending.append((i + 1, j))
        if j < len(right) and right[j] is None:
            pending.append((i, j + 1))
        if i == len(left) or j == len(right):
            continue
        lm = any_char if left[i] is None else left[i]
        rm = any_char if right[j] is None else right[j]
        if lm & rm:
            pending.append((i if left[i] is None else i + 1,
                            j if right[j] is None else j + 1))
    return False


def overlaps(a: str, b: str, *, _budget: _GlobBudget | None = None) -> bool:
    """Intersect fnmatch globs (/** includes parent); raise ValueError on budget exhaustion."""
    budget = _budget if _budget is not None else _GlobBudget()
    left = (a, a[:-3]) if a.endswith("/**") else (a,)
    right = (b, b[:-3]) if b.endswith("/**") else (b,)
    return any(_glob_intersects(x, y, budget) for x in left for y in right)


def _parse_time(value: object) -> datetime:
    when = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def validate(root: Path, registro: Path, claims_path: Path, now: datetime,
             expired_as_warning: bool = False) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    reg = _load(root / registro)
    for key in _required(root, "raiz"):
        if key not in reg:
            errors.append(f"registro: campo obrigatório ausente: {key}")
    atual = reg.get("instrucoes_atuais") or {}
    instr_file = root / str(atual.get("arquivo", ""))
    current_hash = _sha256(instr_file) if instr_file.is_file() else None
    if current_hash is None:
        errors.append(f"registro: arquivo de instruções atual não existe: {atual.get('arquivo')}")
    ids: set[str] = set()
    for i, ag in enumerate(reg.get("agentes") or []):
        label = f"agente[{i}] {ag.get('id', '?')}"
        missing = [k for k in _required(root, "agente") if k not in ag]
        if missing:
            errors.append(f"{label}: campos obrigatórios ausentes: {', '.join(missing)}")
            continue
        if ag["id"] in ids:
            errors.append(f"{label}: id duplicado")
        ids.add(ag["id"])
        if "uid" in ag and (not isinstance(ag["uid"], str) or not ag["uid"]):
            errors.append(f"{label}: uid deve ser string não vazia")
        last_seen = ag["last_seen"]
        if not isinstance(last_seen, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", last_seen):
            errors.append(f"{label}: last_seen deve ser string no formato YYYY-MM-DD")
        else:
            try:
                date.fromisoformat(last_seen)
            except ValueError:
                errors.append(f"{label}: last_seen deve ser uma data real")
        ins = ag["instrucoes"] or {}
        if ins.get("arquivo") != atual.get("arquivo"):
            errors.append(f"{label}: instruções seguidas não são as atuais ({ins.get('arquivo')})")
        if current_hash and ins.get("sha256") != current_hash:
            errors.append(f"{label}: hash sha256 das instruções diverge do arquivo atual")
        if str(ins.get("versao_aceita")) != str(atual.get("versao")):
            errors.append(f"{label}: versão aceita {ins.get('versao_aceita')} != atual {atual.get('versao')}")
    active = []
    claims = _load(root / claims_path).get("claims") or []
    for i, cl in enumerate(claims):
        label = f"claim[{i}] {cl.get('agente', '?')}"
        missing = [k for k in _required(root, "claim") if k not in cl]
        if missing:
            errors.append(f"{label}: campos obrigatórios ausentes: {', '.join(missing)}")
            continue
        if cl["agente"] not in ids:
            errors.append(f"{label}: agente não declarado no registro")
        if _parse_time(cl["expira_em"]) <= now:
            (warnings if expired_as_warning else errors).append(f"{label}: claim expirado em {cl['expira_em']}")
            continue
        active.append((label, cl))
    budget = _GlobBudget(_REGISTRY_GLOB_WORK_LIMIT)
    for x in range(len(active)):
        for y in range(x + 1, len(active)):
            for a in active[x][1]["arquivos"]:
                for b in active[y][1]["arquivos"]:
                    try:
                        conflict = overlaps(a, b, _budget=budget)
                    except _GlobBudgetExceeded as exc:
                        errors.append(str(exc))
                        return errors, warnings
                    if conflict:
                        errors.append(f"{active[x][0]} e {active[y][0]}: claims sobrepostos em {a} / {b}")
    return errors, warnings


def undeclared_authors(root: Path, registro: Path, emails: list[str]) -> list[str]:
    reg = _load(root / registro)
    known = {s for ag in reg.get("agentes") or [] for s in ag.get("assinaturas_commit") or []}
    return sorted({e for e in emails if e and e not in known})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--registro", type=Path, default=Path("docs/agentes/REGISTRO_AGENTES.yaml"))
    p.add_argument("--claims", type=Path, default=Path("docs/agentes/CLAIMS.yaml"))
    p.add_argument("--expirados", choices=["erro", "aviso"], default="erro")
    p.add_argument("--autores", type=Path, help="arquivo com e-mails de autores de commit (um por linha) para avisos")
    p.add_argument("--agora", help="ISO 8601 (testes)")
    a = p.parse_args(argv)
    now = _parse_time(a.agora) if a.agora else datetime.now(timezone.utc)
    try:
        errors, warnings = validate(a.root, a.registro, a.claims, now, a.expirados == "aviso")
        if a.autores:
            emails = a.autores.read_text(encoding="utf-8").split()
            warnings += [f"autor de commit não declarado: {e}" for e in undeclared_authors(a.root, a.registro, emails)]
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
    for w in warnings:
        print(f"AVISO: {w}")
    for e in errors:
        print(f"ERRO: {e}", file=sys.stderr)
    print("registro de agentes: " + ("FALHOU" if errors else "OK"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
