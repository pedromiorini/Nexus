"""Verifica drift entre o manifesto factual e os claims atuais dos documentos.

Os blocos ``NEXUS-CURRENT-STATE`` são a interface estável entre documentação
humana e o artefato JSON produzido pelo CI. Texto histórico fora desses blocos
não é considerado estado atual e não causa falha.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = (
    "NEXUS_KNOWLEDGE.md",
    "NEXUS_EVIDENCE.md",
    "SECURITY_TRIAGE.md",
    "CONTINUATION.md",
)
MARKER_RE = re.compile(
    r"<!--\s*NEXUS-CURRENT-STATE\s*\n(?P<body>.*?)\n\s*-->",
    re.DOTALL,
)
LINE_RE = re.compile(r"^\s*(?P<key>[a-z][a-z0-9_]*)\s*:\s*(?P<value>\S+)\s*$")
REQUIRED = {"commit", "ci_run", "tests", "bandit_low"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest inválido: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifesto deve ser um objeto JSON")
    return payload


def _parse_marker(text: str, document: str) -> dict[str, str]:
    match = MARKER_RE.search(text)
    if not match:
        raise ValueError(f"{document}: bloco NEXUS-CURRENT-STATE ausente")
    claims: dict[str, str] = {}
    for line in match.group("body").splitlines():
        parsed = LINE_RE.match(line)
        if parsed:
            claims[parsed.group("key")] = parsed.group("value")
        elif line.strip():
            raise ValueError(f"{document}: linha inválida no bloco atual: {line!r}")
    missing = REQUIRED - claims.keys()
    if missing:
        raise ValueError(f"{document}: campos ausentes: {', '.join(sorted(missing))}")
    return claims


def _as_int(value: str, field: str, document: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{document}: {field} não é inteiro: {value!r}") from exc


def _manifest_values(manifest: dict[str, Any]) -> dict[str, str]:
    try:
        repository = manifest["repository"]
        verification = manifest["verification"]
        tests = verification["tests"]
        security = manifest["security"]["bandit"]
        ci = manifest["ci"]
        commit = repository["commit"]
        run_id = ci["run_id"]
        test_count = tests["executed"]
        bandit_low = security["severity"]["LOW"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"manifesto sem campos obrigatórios: {exc}") from exc
    if not isinstance(commit, str) or not commit:
        raise ValueError("manifesto: repository.commit inválido")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("manifesto: ci.run_id inválido")
    return {
        "commit": commit,
        "ci_run": run_id,
        "tests": str(test_count),
        "bandit_low": str(bandit_low),
    }


def audit(manifest_path: Path, root: Path = ROOT) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    expected = _manifest_values(manifest)
    documents: dict[str, dict[str, str]] = {}
    conflicts: list[dict[str, str]] = []
    errors: list[str] = []
    for name in DOCUMENTS:
        path = root / name
        try:
            claims = _parse_marker(path.read_text(encoding="utf-8"), name)
            documents[name] = claims
            for field in sorted(REQUIRED):
                value = claims[field]
                resolved = expected[field] if value in {"HEAD", "CURRENT_RUN"} else value
                if resolved != expected[field]:
                    conflicts.append({
                        "document": name,
                        "field": field,
                        "document_value": value,
                        "manifest_value": expected[field],
                    })
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    status = "pass" if not conflicts and not errors else "fail"
    return {
        "schema_version": 1,
        "status": status,
        "manifest": expected,
        "documents": documents,
        "conflicts": conflicts,
        "errors": errors,
        "historical_text_ignored": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="STATE_MANIFEST.json")
    parser.add_argument("--output", default="state-consistency.json")
    args = parser.parse_args(argv)
    report = audit(ROOT / args.manifest, ROOT)
    output = ROOT / args.output
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
