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
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
_INTEGRATED_CHECK_ID = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z")


def _git(*args: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        return "unknown"
    try:
        return subprocess.check_output([executable, *args], cwd=ROOT, text=True).strip()
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


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _test_summary() -> dict[str, Any]:
    path = ROOT / "test-summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except FileNotFoundError:
        return {"status": "not_generated"}
    except (OSError, ValueError):  # Includes JSONDecodeError and UnicodeDecodeError.
        return {"status": "invalid_summary"}

    if not isinstance(summary, dict):
        return {"status": "invalid_summary"}
    required = ("executed", "passed", "failed", "skipped", "unexpected_successes", "result")
    if not all(key in summary for key in required):
        return {"status": "invalid_summary"}
    counts = ("executed", "passed", "failed", "skipped", "unexpected_successes")
    if any(type(summary[key]) is not int or summary[key] < 0 for key in counts):
        return {"status": "invalid_summary"}
    if summary["unexpected_successes"] > summary["failed"]:
        return {"status": "invalid_summary"}
    if summary["passed"] + summary["failed"] + summary["skipped"] != summary["executed"]:
        return {"status": "invalid_summary"}
    if not isinstance(summary["result"], str) or summary["result"] not in {"passed", "failed"}:
        return {"status": "invalid_summary"}
    if (summary["result"] == "passed") != (summary["failed"] == 0):
        return {"status": "invalid_summary"}
    return {"status": "generated", **{key: summary[key] for key in required}}


def _integrated_summary() -> dict[str, Any]:
    """Read and validate the embedded-suite summary as separate evidence."""
    path = ROOT / "integrated-summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "not_generated"}
    except (OSError, json.JSONDecodeError):
        return {"status": "invalid_summary"}

    if not isinstance(summary, dict):
        return {"status": "invalid_summary"}
    required = (
        "schema_version", "scope", "expected", "passed", "skipped",
        "missing", "unexpected", "conflicting", "result",
    )
    if not all(key in summary for key in required):
        return {"status": "invalid_summary"}
    if type(summary["schema_version"]) is not int or summary["schema_version"] != 1:
        return {"status": "invalid_summary"}
    if not isinstance(summary["scope"], str) or not summary["scope"].strip():
        return {"status": "invalid_summary"}

    list_fields = ("expected", "passed", "skipped", "missing", "unexpected", "conflicting")
    values = {key: summary[key] for key in list_fields}
    if any(
        not isinstance(items, list)
        or any(not isinstance(item, str) or not _INTEGRATED_CHECK_ID.fullmatch(item) for item in items)
        or len(set(items)) != len(items)
        for items in values.values()
    ):
        return {"status": "invalid_summary"}
    if not values["expected"]:
        return {"status": "invalid_summary"}

    expected = set(values["expected"])
    passed = set(values["passed"])
    skipped = set(values["skipped"])
    missing = set(values["missing"])
    unexpected = set(values["unexpected"])
    conflicting = set(values["conflicting"])
    if (
        missing != expected - passed - skipped
        or unexpected != (passed | skipped) - expected
        or conflicting != passed & skipped
    ):
        return {"status": "invalid_summary"}
    if not isinstance(summary["result"], str) or summary["result"] not in {"passed", "failed"}:
        return {"status": "invalid_summary"}
    expected_result = "failed" if missing or unexpected or conflicting else "passed"
    if summary["result"] != expected_result:
        return {"status": "invalid_summary"}

    return {
        "status": "generated",
        "scope": summary["scope"],
        "discovered_static": len(expected),
        "passed": len(passed & expected),
        "skipped": len(skipped & expected),
        "failed": len(missing) + len(conflicting),
        "unexpected": len(unexpected),
        "result": summary["result"],
    }


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
            "strong_capability_claims": reality.get("code_claim_inventory", {}).get("by_level", {}).get("strong_capability_claim"),
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


def build_manifest(test_result: str = "not_run", test_command: str | None = None, gates_status: str | None = None) -> dict[str, Any]:
    commit = os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    branch = os.environ.get("GITHUB_REF_NAME") or _git("branch", "--show-current")
    run_id = os.environ.get("GITHUB_RUN_ID")
    test_summary = _test_summary()
    tests = {
        "scope": "unittest discover (test_*.py); embedded checks are under verification.integrated",
        "discovered_static": _test_count(),
        **test_summary,
        "command": test_command or "not_run",
        "result": test_summary.get("result", test_result),
    }
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": {"branch": branch, "commit": commit, "worktree": "clean_required"},
        "verification": {
            "tests": tests,
            "integrated": _integrated_summary(),
            "ast": _ast_summary(),
        },
        "security": {"bandit": _bandit_summary()},
        "mutation": _mutation_summary(),
        "ci": {
            "run_id": run_id or "not_ci",
            "commit": os.environ.get("GITHUB_SHA", commit),
            # Refere-se ao job de gates já concluído, não à conclusão final deste workflow.
            "gates_status": gates_status or "not_ci",
        },
        "optional_dependencies": _optional_dependencies(),
        "documents": {"latest_commits": _document_commits()},
        "provenance": {"generator": "tools/state_manifest.py", "source_of_truth": "generated execution artifact"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="STATE_MANIFEST.json")
    parser.add_argument("--test-result", default="not_run")
    parser.add_argument("--test-command")
    parser.add_argument("--gates-status")
    args = parser.parse_args(argv)
    manifest = build_manifest(args.test_result, args.test_command, args.gates_status)
    output = ROOT / args.output
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
