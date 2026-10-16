"""Contratos de higiene do núcleo (auditoria de 2026-10-08).

Cobre: fechamento determinístico de conexões SQLite (sem ResourceWarning no Python >= 3.13),
traces de depuração silenciosos por padrão, recusa de ``python -O`` pela suíte integrada e
guardas estáticos contra padrões que já causaram problemas (ramo morto e prints de debug).
"""
import ast
import contextlib
import gc
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import core.constitutional_brain as brain
from core.constitutional_brain import CompleteNexusBrain, RealEpisodicMemory, RealHierarchicalMemory, RealImmutableLog
from vita.nexus_constitutional_bridge_v3 import NexusConstitutionalBridge

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = sorted(path for folder in ("core", "vita", "tools") for path in (ROOT / folder).glob("*.py"))


class ConnectionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        bridge_path = str(Path(self.tmp.name) / "bridge.sqlite3")
        # (nome, fábrica, função que devolve a conexão sqlite3 do objeto)
        self.cases = (
            ("RealHierarchicalMemory", lambda: RealHierarchicalMemory(":memory:"), lambda o: o.conn),
            ("RealEpisodicMemory", lambda: RealEpisodicMemory(":memory:"), lambda o: o.conn),
            ("RealImmutableLog", lambda: RealImmutableLog(":memory:"), lambda o: o.conn),
            ("NexusConstitutionalBridge", lambda: NexusConstitutionalBridge(audit_db_path=bridge_path), lambda o: o._audit_db),
        )

    def test_close_is_idempotent_and_releases_the_connection(self):
        for name, factory, connection_of in self.cases:
            with self.subTest(name=name):
                instance = factory()
                connection = connection_of(instance)
                instance.close()
                instance.close()
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")

    def test_context_manager_closes_the_connection(self):
        for name, factory, connection_of in self.cases:
            with self.subTest(name=name):
                with factory() as instance:
                    connection = connection_of(instance)
                    connection.execute("SELECT 1")
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")

    def test_garbage_collection_closes_the_connection_without_resource_warning(self):
        for name, factory, connection_of in self.cases:
            with self.subTest(name=name):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    instance = factory()
                    connection = connection_of(instance)
                    del instance
                    gc.collect()
                self.assertEqual([item for item in caught if issubclass(item.category, ResourceWarning)], [])
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")


class DebugTraceTests(unittest.TestCase):
    def test_dbg_is_silent_by_default_and_prints_when_enabled(self):
        for enabled, expected in ((False, ""), (True, "trace\n")):
            with self.subTest(enabled=enabled):
                buffer = io.StringIO()
                with mock.patch.object(brain, "_NEXUS_DEBUG", enabled), contextlib.redirect_stdout(buffer):
                    brain._dbg("trace")
                self.assertEqual(buffer.getvalue(), expected)

    def test_think_emits_no_debug_trace_by_default(self):
        buffer = io.StringIO()
        with mock.patch.object(brain, "_NEXUS_DEBUG", False), contextlib.redirect_stdout(buffer):
            instance = CompleteNexusBrain(enable_router=True)
            instance.think("Estou ajudando minha mãe com a mudança")
        output = buffer.getvalue()
        self.assertNotIn("DEBUG v3", output)
        self.assertNotIn("TRACE", output)
        self.assertNotIn("NOT in reasoning_result", output)


class IntegratedSuiteGuardTests(unittest.TestCase):
    def test_integrated_suite_refuses_python_optimize(self):
        # Com -O os asserts desaparecem e a suíte imprimiria "PASSED" sem verificar nada.
        completed = subprocess.run(
            [sys.executable, "-O", str(ROOT / "core" / "constitutional_brain.py")],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("PYTHONOPTIMIZE", completed.stderr + completed.stdout)
        self.assertNotIn("PASSED", completed.stdout)


class IntegratedSuiteRunTests(unittest.TestCase):
    def test_integrated_suite_runs_clean_in_dev_mode_and_writes_a_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            completed = subprocess.run(
                [sys.executable, "-X", "dev", "-W", "error", str(ROOT / "core" / "constitutional_brain.py")],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT), "NEXUS_AUDIT_DB": str(Path(directory) / "audit.sqlite3"),
                     "NEXUS_INTEGRATED_SUMMARY": str(summary_path)},
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
            # ResourceWarning em __del__/encerramento não muda o código de saída: verificar o texto.
            self.assertNotIn("ResourceWarning", completed.stderr)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["result"], "passed")
        self.assertEqual(summary["missing"], [])
        self.assertEqual(set(summary["passed"]) | set(summary["skipped"]), set(summary["expected"]))
        self.assertGreaterEqual(len(summary["passed"]), 29)
        self.assertIn("EMBEDDED CHECKS", completed.stdout)


class StaticHygieneTests(unittest.TestCase):
    def parse(self, path):
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_else_branch_repeats_its_own_condition(self):
        # `if C: ... else: if C: ...` cria um ramo inalcançável (e escondeu um NameError em think()).
        offenders = []
        for path in SOURCE_FILES:
            for node in ast.walk(self.parse(path)):
                if isinstance(node, ast.If):
                    for inner in node.orelse:
                        if isinstance(inner, ast.If) and ast.dump(inner.test) == ast.dump(node.test):
                            offenders.append(f"{path.name}:{inner.lineno}")
        self.assertEqual(offenders, [])

    def test_debug_and_trace_prints_go_through_dbg(self):
        offenders = []
        for path in SOURCE_FILES:
            for node in ast.walk(self.parse(path)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                    text = ast.unparse(node)
                    if "DEBUG v" in text or "TRACE" in text or "NOT in reasoning_result" in text:
                        offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [], "use _dbg(...) para traces (silencioso sem NEXUS_DEBUG=1)")


if __name__ == "__main__":
    unittest.main()
