"""Valida docs/agentes/REGISTRO_AGENTES.yaml e CLAIMS.yaml.

Schema 1: aceite estrito das instruções atuais (compatibilidade).
Schema 2: versão aceita existe, não é futura e aponta ao artefato/SHA exato.
Versão anterior verificável gera aviso (rc 0); divergência/ausência gera erro.
Claims expirados falham salvo --expirados aviso; sobreposição ativa sempre falha.
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


def _version(value: object) -> tuple[int, ...]:
    if not isinstance(value, str) or not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", value):
        raise ValueError("versão deve ser string canônica MAJOR.MINOR.PATCH (sem sufixo)")
    return tuple(int(part) for part in value.split("."))


def _instruction_object(value: object, fields: tuple[str, ...]) -> dict:
    if not isinstance(value, dict) or any(field not in value for field in fields):
        raise ValueError(f"instruções devem conter {', '.join(fields)}")
    return value


def _instruction_file(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("arquivo de instruções deve ser caminho relativo não vazio")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError("arquivo de instruções deve ser caminho relativo canônico dentro da raiz")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"arquivo de instruções contém link simbólico: {value}")
    if not path.is_file():
        raise ValueError(f"arquivo de instruções não existe: {value}")
    return path


def _version_catalog(root: Path, reg: dict, errors: list[str]) -> dict:
    """Verifica inclusive artefatos não referenciados; nunca usa o arquivo mutável como fallback."""
    verified: dict = {}
    try:
        current = _instruction_object(reg.get("instrucoes_atuais"), ("arquivo", "versao"))
        current_version = _version(current["versao"])
        catalog = reg.get("instrucoes_versoes")
        if not isinstance(catalog, dict) or not catalog:
            raise ValueError("instrucoes_versoes deve ser catálogo não vazio no schema 2")
    except ValueError as exc:
        errors.append(f"registro: {exc}")
        return verified
    for version, artifact in catalog.items():
        try:
            parsed = _version(version)
            if parsed > current_version:
                raise ValueError(f"versão futura {version} > atual {current['versao']}")
            artifact = _instruction_object(artifact, ("arquivo", "sha256"))
            digest = artifact["sha256"]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("sha256 deve conter 64 caracteres hexadecimais minúsculos")
            expected = f"docs/agentes/versoes/{version}/{digest}.md"
            if artifact["arquivo"] != expected:
                raise ValueError(f"arquivo do artefato deve ser {expected}")
            if _sha256(_instruction_file(root, artifact["arquivo"])) != digest:
                raise ValueError("sha256 do artefato diverge do conteúdo")
            verified[version] = artifact
        except (OSError, ValueError) as exc:
            errors.append(f"registro: versão {version!r}: {exc}")
    artifact = verified.get(current["versao"])
    if artifact is None:
        errors.append("registro: versão atual ausente ou inválida no catálogo")
    try:
        path = _instruction_file(root, current["arquivo"])
        if artifact is not None and _sha256(path) != artifact["sha256"]:
            errors.append("registro: sha256 do arquivo atual diverge do artefato versionado")
    except (OSError, ValueError) as exc:
        errors.append(f"registro: arquivo atual: {exc}")
    return verified


def _accepted_version(ins: object, current: dict, catalog: dict, label: str,
                      errors: list[str], warnings: list[str]) -> None:
    try:
        ins = _instruction_object(ins, ("arquivo", "sha256", "versao_aceita"))
        version = ins["versao_aceita"]
        parsed = _version(version)
        current_version = _version(current.get("versao"))
        if parsed > current_version:
            raise ValueError(f"versão aceita futura {version} > atual {current['versao']}")
        artifact = catalog.get(version)
        if artifact is None:
            raise ValueError(f"versão aceita desconhecida ou artefato inválido: {version}")
        if ins["arquivo"] != artifact["arquivo"]:
            raise ValueError("arquivo seguido não é o artefato da versão aceita")
        if ins["sha256"] != artifact["sha256"]:
            raise ValueError("hash sha256 diverge do artefato da versão aceita")
        if parsed < current_version:
            warnings.append(f"{label}: versão aceita {version} anterior à atual {current['versao']} (artefato verificado)")
    except ValueError as exc:
        errors.append(f"{label}: {exc}")


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
    schema_version = reg.get("schema_version")
    if type(schema_version) is not int or schema_version not in (1, 2):
        errors.append("registro: schema_version deve ser inteiro 1 ou 2")
        return errors, warnings
    atual = reg.get("instrucoes_atuais")
    if not isinstance(atual, dict):
        errors.append("registro: instrucoes_atuais deve ser mapeamento")
        atual = {}
    catalog = _version_catalog(root, reg, errors) if schema_version == 2 else {}
    current_hash = None
    if schema_version == 1:
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
        if schema_version == 2:
            _accepted_version(ag["instrucoes"], atual, catalog, label, errors, warnings)
        else:
            ins = ag["instrucoes"] or {}
            if not isinstance(ins, dict):
                errors.append(f"{label}: instrucoes deve ser mapeamento")
                continue
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
