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
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = Path("docs/agentes/schema/registro_agentes.schema.json")


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


def _prefix(pattern: str) -> str:
    return pattern[:-3] if pattern.endswith("/**") else pattern


def overlaps(a: str, b: str) -> bool:
    if a == b or fnmatch.fnmatch(a, b) or fnmatch.fnmatch(b, a):
        return True
    pa, pb = _prefix(a), _prefix(b)
    if a.endswith("/**") and (b == pa or b.startswith(pa + "/")):
        return True
    return b.endswith("/**") and (a == pb or a.startswith(pb + "/"))


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
    for x in range(len(active)):
        for y in range(x + 1, len(active)):
            for a in active[x][1]["arquivos"]:
                for b in active[y][1]["arquivos"]:
                    if overlaps(a, b):
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
