"""Mutation testing direcionado para contratos críticos do Nexus.

As mutações são explícitas, pequenas e restauradas sempre. O objetivo é medir
se os testes detectam alterações semânticas nos módulos críticos, não mutar o
núcleo monolítico inteiro.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "targeted_mutation_report.json"
TESTS = [
    "test_deferred_task_queue.py",
    "test_deferred_task_properties.py",
    "test_recovery_diagnostics_contract.py",
]


@dataclass(frozen=True)
class Mutation:
    name: str
    target: str
    needle: str
    replacement: str


MUTATIONS = [
    Mutation("queue_capacity_guard", "core/deferred_task_queue.py", "max_attempts < 1 or max_size < 1", "max_attempts < 0 or max_size < 1"),
    Mutation("queue_priority_direction", "core/deferred_task_queue.py", "(-self.priority, self.created_at)", "(self.priority, self.created_at)"),
    Mutation("queue_pressure_guard", "core/deferred_task_queue.py", "if pressure_critical or not self._heap:", "if pressure_critical and not self._heap:"),
    Mutation("queue_retry_limit", "core/deferred_task_queue.py", "task.attempts >= task.max_attempts", "task.attempts > task.max_attempts"),
    Mutation("queue_success_branch", "core/deferred_task_queue.py", "if ok:\n                self.complete", "if not ok:\n                self.complete"),
    Mutation("queue_batch_validation", "core/deferred_task_queue.py", "if max_batch < 1:", "if max_batch < 0:"),
    Mutation("vita_failure_threshold_boundary", "vita/nexus_constitutional_bridge_v3.py", "failure_rate >= self.reprocessing_thresholds[\"failure_rate\"]", "failure_rate > self.reprocessing_thresholds[\"failure_rate\"]"),
    Mutation("vita_audit_order", "vita/nexus_constitutional_bridge_v3.py", "reversed(rows)", "rows"),
    Mutation("schema_version_compatibility", "vita/nexus_constitutional_bridge_v3.py", "schema == self.RECOVERY_SCHEMA", "schema != self.RECOVERY_SCHEMA"),
    Mutation("schema_severity_membership", "vita/nexus_constitutional_bridge_v3.py", "diagnostics.get(\"severity\") not in {\"info\", \"warning\", \"critical\"}", "diagnostics.get(\"severity\") in {\"info\", \"warning\", \"critical\"}"),
]


def run_tests() -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", *TESTS],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
    )
    return completed.returncode, completed.stdout[-3000:]


def main() -> int:
    originals: dict[Path, str] = {}
    results = []
    baseline_code, baseline_output = run_tests()
    if baseline_code != 0:
        raise SystemExit(f"Baseline failed before mutation:\n{baseline_output}")
    try:
        for mutation in MUTATIONS:
            target = ROOT / mutation.target
            original = originals.setdefault(target, target.read_text(encoding="utf-8"))
            matches = original.count(mutation.needle)
            if matches != 1:
                results.append({"name": mutation.name, "target": mutation.target, "status": "not_applied", "matches": matches})
                continue
            target.write_text(original.replace(mutation.needle, mutation.replacement, 1), encoding="utf-8")
            started = time.time()
            try:
                code, output = run_tests()
                results.append({
                    "name": mutation.name,
                    "target": mutation.target,
                    "status": "killed" if code != 0 else "survived",
                    "exit_code": code,
                    "duration_seconds": round(time.time() - started, 3),
                    "output_tail": output,
                })
            finally:
                target.write_text(original, encoding="utf-8")
    finally:
        for target, original in originals.items():
            target.write_text(original, encoding="utf-8")
    killed = sum(item.get("status") == "killed" for item in results)
    survived = sum(item.get("status") == "survived" for item in results)
    not_applied = sum(item.get("status") == "not_applied" for item in results)
    applicable = killed + survived
    score = round(100 * killed / applicable, 2) if applicable else 0.0
    report = {
        "tool": "nexus-targeted-mutation",
        "tests": TESTS,
        "mutations": results,
        "summary": {
            "declared": len(MUTATIONS),
            "applicable": applicable,
            "killed": killed,
            "survived": survived,
            "not_applied": not_applied,
            "score_percent": score,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0 if applicable == len(MUTATIONS) and survived == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
