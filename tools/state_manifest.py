"""Gera o manifesto factual de uma execução do Nexus.

O manifesto é um artefato de CI, não uma fonte manual. Ele registra o commit
que foi realmente verificado e separa fatos atuais de informações ausentes.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _test_count() -> int:
    count = 0
    for path in sorted(ROOT.glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        count += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test")
            for node in ast.walk(tree)
        )
    return count


def _document_commits() -> dict[str, str]:
    names = [
        "NEXUS_CONSTITUTION.md", "NEXUS_KNOWLEDGE.md", "NEXUS_ARCHITECTURE.md",
        "NEXUS_EVIDENCE.md", "NEXUS_DECISIONS.md", "NEXUS_ROADMAP.md",
        "NEXUS_BOOTSTRAP_AUDIT.md", "CONTINUATION.md",
    ]
    return {name: _git("log", "-1", "--format=%H", "--", name) for name in names}


def _bandit_summary() -> dict[str, Any]:
    path = ROOT / "bandit-report.json"
    if not path.exists():
        return {"status": "not_generated"}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "invalid_report"}
    results = report.get("results", [])
    severity = {level: sum(item.get("issue_severity") == level for item in results) for level in ("LOW", "MEDIUM", "HIGH")}
    return {"status": "generated", "findings": len(results), "severity": severity}


def _mutation_summary() -> dict[str, Any]:
    path = ROOT / "targeted_mutation_report.json"
    if not path.exists():
        return {"status": "not_generated"}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = report["summary"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return {"status": "invalid_report"}
    return {"status": "generated", **{key: summary.get(key) for key in ("declared", "applicable", "killed", "survived", "not_applied", "score_percent")}}


def _optional_dependencies() -> dict[str, str]:
    modules = {"psutil": "psutil", "faiss": "faiss", "sentence_transformers": "sentence_transformers"}
    return {name: "available" if importlib.util.find_spec(module) else "unavailable" for name, module in modules.items()}


def _ast_summary() -> dict[str, Any]:
    reality_path = ROOT / "audit_reality.json"
    gaps_path = ROOT / "implementation-gaps.json"
    summary: dict[str, Any] = {"status": "not_generated"}
    try:
        reality = json.loads(reality_path.read_text(encoding="utf-8"))
        evidence = reality["static_evidence"]
        summary = {
            "status": "generated",
            "core_lines": evidence.get("core_lines"),
            "classes": evidence.get("classes"),
            "functions": evidence.get("functions"),
            "constant_returns": len(evidence.get("constant_return_functions", [])),
            "pass_only": len(evidence.get("pass_only_functions", [])),
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass
    try:
        gaps = json.loads(gaps_path.read_text(encoding="utf-8"))
        summary["pass_only"] = gaps.get("pass_only_count")
        summary["not_implemented"] = gaps.get("not_implemented_count")
    except (OSError, json.JSONDecodeError, TypeError):
        summary.setdefault("not_implemented", None)
    return summary


def build_manifest(test_result: str = "not_run", test_command: str | None = None, ci_status: str | None = None) -> dict[str, Any]:
    commit = os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    branch = os.environ.get("GITHUB_REF_NAME") or _git("branch", "--show-current")
    run_id = os.environ.get("GITHUB_RUN_ID")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": {"branch": branch, "commit": commit, "worktree": "clean_required"},
        "verification": {
            "tests": {"count": _test_count(), "command": test_command or "not_run", "result": test_result},
            "ast": _ast_summary(),
        },
        "security": {"bandit": _bandit_summary()},
        "mutation": _mutation_summary(),
        "ci": {"run_id": run_id or "not_ci", "commit": os.environ.get("GITHUB_SHA", commit), "status": ci_status or "not_ci"},
        "optional_dependencies": _optional_dependencies(),
        "documents": {"latest_commits": _document_commits()},
        "provenance": {"generator": "tools/state_manifest.py", "source_of_truth": "generated execution artifact"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="STATE_MANIFEST.json")
    parser.add_argument("--test-result", default="not_run")
    parser.add_argument("--test-command")
    parser.add_argument("--ci-status")
    args = parser.parse_args(argv)
    manifest = build_manifest(args.test_result, args.test_command, args.ci_status)
    output = ROOT / args.output
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
