"""Contratos locais do replay e da dependência da atestação final."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parent
REPLAY = ROOT / "tools/agente/replay/run_replay.sh"
PASS = "controle\tgh\tpass CLEAN\t0"


class ReplayContract(unittest.TestCase):
    def run_replay(self, path, env=None):
        return subprocess.run(
            ["bash", str(REPLAY), str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )

    def run_table(self, text):
        with tempfile.TemporaryDirectory(prefix="nexus-replay-contract-") as tmp:
            path = Path(tmp) / "casos.tsv"
            path.write_text(text, encoding="utf-8")
            return self.run_replay(path)

    def test_successful_case(self):
        result = self.run_table(PASS + "\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASSOU controle rc=0", result.stdout)

    def test_documented_runner_is_directly_executable(self):
        with tempfile.TemporaryDirectory(prefix="nexus-replay-executable-") as tmp:
            path = Path(tmp) / "casos.tsv"
            path.write_text(PASS + "\n", encoding="utf-8")
            try:
                result = subprocess.run(
                    [str(REPLAY), str(path)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except OSError as exc:
                self.fail(f"Invocacao direta do runner falhou: {exc}")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Executados: 1", result.stdout)
        self.assertIn("PASSOU controle rc=0", result.stdout)
        self.assertEqual(result.stdout.count("PASSOU "), 1, result.stdout)
        self.assertNotIn("FALHOU", result.stdout + result.stderr)

    def test_expected_guard_failure_is_a_successful_replay(self):
        result = self.run_table("negativo\tgh\tfail CLEAN\t1\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASSOU negativo rc=1", result.stdout)

    def test_mismatch_fails_even_after_a_success(self):
        result = self.run_table(PASS + "\nerro\tgh\tpass CLEAN\t9\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FALHOU erro rc=0 esperado=9", result.stdout)

    def test_empty_or_comment_only_tables_fail(self):
        for text in ("", "\n\n", "# apenas cabecalho\n", "# cabecalho\n\n"):
            with self.subTest(text=text):
                result = self.run_table(text)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("PASSOU", result.stdout)

    def test_missing_input_fails(self):
        with tempfile.TemporaryDirectory(prefix="nexus-replay-missing-") as tmp:
            result = self.run_replay(Path(tmp) / "ausente.tsv")
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_directory_input_fails(self):
        with tempfile.TemporaryDirectory(prefix="nexus-replay-directory-") as tmp:
            result = self.run_replay(Path(tmp))
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipIf(os.geteuid() == 0, "permissao negada requer usuario nao-root")
    def test_unreadable_input_fails(self):
        with tempfile.TemporaryDirectory(prefix="nexus-replay-unreadable-") as tmp:
            path = Path(tmp) / "sem-leitura.tsv"
            path.write_text(PASS + "\n", encoding="utf-8")
            path.chmod(0)
            try:
                with self.assertRaises(PermissionError):
                    path.read_text(encoding="utf-8")
                result = self.run_replay(path)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            finally:
                path.chmod(0o600)

    def test_read_error_after_partial_data_fails_before_execution(self):
        with tempfile.TemporaryDirectory(prefix="nexus-replay-read-error-") as tmp:
            root = Path(tmp)
            path = root / "casos.tsv"
            path.write_text(PASS + "\n", encoding="utf-8")
            # Injecao local de EIO: leitor entrega um caso valido e depois falha.
            reader = root / "cat"
            reader.write_text(
                "#!/bin/sh\nprintf 'controle\\tgh\\tpass CLEAN\\t0\\n'\nexit 1\n",
                encoding="utf-8",
            )
            reader.chmod(0o700)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"])
            result = self.run_replay(path, env=env)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("PASSOU", result.stdout)

    def test_last_case_without_newline_is_executed(self):
        result = self.run_table(PASS)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASSOU controle rc=0", result.stdout)

    def test_unknown_case_cannot_reuse_previous_success(self):
        result = self.run_table(PASS + "\nignorado\tdesconhecido\tpass CLEAN\t0\n")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)


class AttestationDependencyContract(unittest.TestCase):
    def setUp(self):
        self.jobs = yaml.safe_load(
            (ROOT / ".github/workflows/nexus-contract-gate.yml").read_text(encoding="utf-8")
        )["jobs"]

    def test_attestation_waits_for_both_jobs(self):
        needs = self.jobs["state-manifest"]["needs"]
        if isinstance(needs, str):
            needs = [needs]
        self.assertEqual(set(needs), {"contract-gate", "agent-guard-replay"})

    def test_attestation_requires_both_success_results(self):
        # Contrato estrutural; nao simula o scheduler hospedado do GitHub.
        condition = " ".join(self.jobs["state-manifest"]["if"].split())
        self.assertEqual(
            condition,
            "needs.contract-gate.result == 'success' && needs.agent-guard-replay.result == 'success'",
        )

    def test_replay_failures_are_not_optional(self):
        replay = self.jobs["agent-guard-replay"]
        self.assertFalse(replay.get("continue-on-error", False))
        for step in replay["steps"]:
            self.assertFalse(step.get("continue-on-error", False))


if __name__ == "__main__":
    unittest.main()
