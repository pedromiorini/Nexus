"""Mutation testing direcionado para contratos críticos do Nexus.

As mutações são explícitas, pequenas e sempre revertidas. O objetivo é medir se os
testes detectam alterações semânticas em pontos críticos (fila, ponte Vita, guard de
VRAM, log imutável e detector de jailbreak), não mutar o núcleo monolítico inteiro.

Segurança operacional: as mutações são aplicadas em uma CÓPIA TEMPORÁRIA do repositório;
a árvore de trabalho nunca é modificada (antes, um processo interrompido podia deixar
código mutado no lugar). Mutante que excede o tempo limite conta como morto.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "targeted_mutation_report.json"
DEFAULT_TESTS = (
    "test_deferred_task_queue.py",
    "test_deferred_task_properties.py",
    "test_recovery_diagnostics_contract.py",
)
VRAM = ("test_vram_defense_guard.py",)
LOG = ("test_immutable_log_contract.py",)
JB = ("test_jailbreak_detector_contract.py",)
COPY_IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", ".venv", "venv", ".hypothesis", ".ruff_cache", ".mypy_cache",
    "*.sqlite3", "*.db", ".coverage*", "coverage.xml", "targeted_mutation_report.json",
)
TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class Mutation:
    name: str
    target: str
    needle: str
    replacement: str
    tests: tuple[str, ...] = DEFAULT_TESTS


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
    Mutation('vram_hard_boundary', 'core/vram_defense_guard.py', 'elif current.utilization >= self.hard_limit:', 'elif current.utilization > self.hard_limit:', tests=VRAM),
    Mutation('vram_soft_boundary', 'core/vram_defense_guard.py', 'elif current.utilization >= self.soft_limit:', 'elif current.utilization > self.soft_limit:', tests=VRAM),
    Mutation('vram_limit_validation', 'core/vram_defense_guard.py', 'if not 0 < soft_limit < hard_limit <= 1:', 'if not 0 < soft_limit < hard_limit < 1:', tests=VRAM),
    Mutation('vram_utilization_cap', 'core/vram_defense_guard.py', 'return min(1.0, self.reserved_bytes / self.total_bytes)', 'return min(2.0, self.reserved_bytes / self.total_bytes)', tests=VRAM),
    Mutation('vram_success_property', 'core/vram_defense_guard.py', 'return not self.errors', 'return bool(self.errors)', tests=VRAM),
    Mutation('log_chain_previous_link', 'core/constitutional_brain.py', 'if previous_hash != expected_previous:', 'if previous_hash == expected_previous:', tests=LOG),
    Mutation('log_chain_content_hash', 'core/constitutional_brain.py', 'if recomputed != content_hash:', 'if recomputed == content_hash:', tests=LOG),
    Mutation('log_resume_head', 'core/constitutional_brain.py', '"SELECT content_hash FROM log_entries ORDER BY sequence DESC LIMIT 1"', '"SELECT content_hash FROM log_entries ORDER BY sequence ASC LIMIT 1"', tests=LOG),
    Mutation('log_head_read_from_db', 'core/constitutional_brain.py', 'previous_hash = self._load_head()\n                content = f', 'previous_hash = self.current_hash\n                content = f', tests=LOG),
    Mutation('log_append_only_update', 'core/constitutional_brain.py', "BEFORE UPDATE ON log_entries\n            BEGIN\n                SELECT RAISE(ABORT, 'log_entries is append-only');", 'BEFORE UPDATE ON log_entries\n            BEGIN\n                SELECT 1;', tests=LOG),
    Mutation('log_append_only_delete', 'core/constitutional_brain.py', "BEFORE DELETE ON log_entries\n            BEGIN\n                SELECT RAISE(ABORT, 'log_entries is append-only');", 'BEFORE DELETE ON log_entries\n            BEGIN\n                SELECT 1;', tests=LOG),
    Mutation('jailbreak_threshold_formula', 'core/constitutional_brain.py', '1.3 - self.sensitivity', '1.0 - self.sensitivity', tests=JB),
    Mutation('jailbreak_unicode_normalization', 'core/constitutional_brain.py', 'unicodedata.normalize("NFKC", "" if text is None else str(text))', '"" if text is None else str(text)', tests=JB),
    Mutation('jailbreak_zero_width_removal', 'core/constitutional_brain.py', 'unicodedata.category(ch) not in ("Cf", "Cc")', 'unicodedata.category(ch) not in ("Cc",)', tests=JB),
    Mutation('jailbreak_leet_view', 'core/constitutional_brain.py', 'deleet = norm.translate(self._LEET)', 'deleet = norm', tests=JB),
    Mutation('jailbreak_squash_view', 'core/constitutional_brain.py', 'squash = re.sub(r"[^a-z0-9]+", "", deleet)', 'squash = ""', tests=JB),
    Mutation('jailbreak_strong_weight', 'core/constitutional_brain.py', '("override_en_qualified", D, 1.0,', '("override_en_qualified", D, 0.1,', tests=JB),
    Mutation('jailbreak_detection_counter', 'core/constitutional_brain.py', 'self.detections_count += 1', 'self.detections_count += 2', tests=JB),
]


def run_tests(workdir: Path, tests: tuple[str, ...]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", *tests],
            cwd=workdir,
            env={**os.environ, "PYTHONPATH": str(workdir), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return 124, f"timeout após {TIMEOUT_SECONDS}s (mutante considerado morto)"
    return completed.returncode, completed.stdout[-3000:]


def main() -> int:
    results = []
    with tempfile.TemporaryDirectory(prefix="nexus-mutation-") as scratch:
        workdir = Path(scratch) / "repo"
        shutil.copytree(ROOT, workdir, ignore=COPY_IGNORE)
        for tests in dict.fromkeys(mutation.tests for mutation in MUTATIONS):
            code, output = run_tests(workdir, tests)
            if code != 0:
                raise SystemExit(f"Baseline failed before mutation ({', '.join(tests)}):\n{output}")
        for mutation in MUTATIONS:
            target = workdir / mutation.target
            original = target.read_text(encoding="utf-8")
            matches = original.count(mutation.needle)
            if matches != 1:
                results.append({"name": mutation.name, "target": mutation.target, "status": "not_applied", "matches": matches})
                continue
            target.write_text(original.replace(mutation.needle, mutation.replacement, 1), encoding="utf-8")
            started = time.time()
            try:
                code, output = run_tests(workdir, mutation.tests)
            finally:
                target.write_text(original, encoding="utf-8")
            results.append({
                "name": mutation.name,
                "target": mutation.target,
                "tests": list(mutation.tests),
                "status": "killed" if code != 0 else "survived",
                "exit_code": code,
                "duration_seconds": round(time.time() - started, 3),
                "output_tail": output,
            })
    killed = sum(item.get("status") == "killed" for item in results)
    survived = sum(item.get("status") == "survived" for item in results)
    not_applied = sum(item.get("status") == "not_applied" for item in results)
    applicable = killed + survived
    score = round(100 * killed / applicable, 2) if applicable else 0.0
    report = {
        "tool": "nexus-targeted-mutation",
        "isolated_copy": True,
        "tests": sorted({test for mutation in MUTATIONS for test in mutation.tests}),
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
    for item in results:
        if item["status"] != "killed":
            print(f"  {item['status']}: {item['name']} ({item['target']})", file=sys.stderr)
    return 0 if applicable == len(MUTATIONS) and survived == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
