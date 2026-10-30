"""Contrato hermético da guarda `tools/agente/artefatos_drift.sh` (L-010; AD3–AD7).

Pré-registro: docs/agentes/ARTEFATOS_DRIFT.md (commit anterior a este arquivo). O
repositório é falso: `security_triage.py` e `reality_audit.py` de mentira, e um `PY`
que simula só `-m bandit`. Não depende de bandit nem de rede; o resultado real
(AD1) é medido à parte com `tools/agente/replay/casos_drift.tsv`.
"""
import hashlib
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GUARD = ROOT / "tools/agente/artefatos_drift.sh"
REPLAY = ROOT / "tools/agente/replay/run_replay.sh"

TRIAGE_STUB = (
    "import os, sys\n"
    "sys.exit(1 if os.environ.get('STUB_TRIAGE') == 'fail' else 0)\n"
)
REALITY_STUB = (
    "import os, sys\n"
    "mode = os.environ.get('STUB_REALITY', 'ok')\n"
    "if mode == 'fail':\n"
    "    sys.exit(1)\n"
    "if mode == 'modify':\n"
    "    with open('AUDIT_REALITY_REPORT.md', 'a', encoding='utf-8') as f:\n"
    "        f.write('drift\\n')\n"
)
PY_WRAPPER = """#!/usr/bin/env bash
if [[ "${1:-}" == "-m" && "${2:-}" == "bandit" ]]; then
  while [[ $# -gt 0 ]]; do [[ $1 == -o ]] && { out=$2; break; }; shift; done
  [[ "${STUB_BANDIT:-ok}" == "none" ]] || echo '{"results": []}' > "$out"
  exit 0
fi
exec python3 "$@"
"""


def _git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo, capture_output=True, text=True, check=True,
    )


def _tree_hash(repo):
    digest = hashlib.sha256()
    for path in sorted(Path(repo).rglob("*")):
        if ".git" in path.relative_to(repo).parts or not path.is_file():
            continue
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class DriftGuardBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nexus-drift-")
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.py = self.base / "py.sh"
        self.py.write_text(PY_WRAPPER, encoding="utf-8")
        self.py.chmod(0o755)

    def make_repo(self, triage=True, reality=True, git=True):
        repo = self.base / "repo"
        (repo / "tools").mkdir(parents=True)
        for name in ("core", "vita"):
            (repo / name).mkdir()
            (repo / name / "x.py").write_text("x = 1\n", encoding="utf-8")
        if triage:
            (repo / "tools/security_triage.py").write_text(TRIAGE_STUB, encoding="utf-8")
        if reality:
            (repo / "tools/reality_audit.py").write_text(REALITY_STUB, encoding="utf-8")
        for name in ("SECURITY_TRIAGE.md", "AUDIT_REALITY_REPORT.md", "audit_reality.json"):
            (repo / name).write_text("base\n", encoding="utf-8")
        if git:
            _git(repo, "init", "-q")
            _git(repo, "add", "-A")
            _git(repo, "commit", "-q", "-m", "base")
        return repo

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("STUB_")}
        env["PY"] = str(self.py)
        env.update(extra)
        return env

    def guard(self, *args, **stubs):
        return subprocess.run(
            ["bash", str(GUARD), *args], capture_output=True, text=True,
            timeout=60, env=self.env(**stubs),
        )


class AD4Matrix(DriftGuardBase):
    def test_clean_repo_exits_0(self):
        repo = self.make_repo()
        result = self.guard(str(repo))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.strip(), "")

    def test_triage_drift_exits_7(self):
        repo = self.make_repo()
        result = self.guard(str(repo), STUB_TRIAGE="fail")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("DRIFT: SECURITY_TRIAGE.md", result.stderr)
        self.assertNotIn("relatorios de realidade", result.stderr)

    def test_reality_drift_exits_7(self):
        repo = self.make_repo()
        result = self.guard(str(repo), STUB_REALITY="modify")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("DRIFT: relatorios de realidade", result.stderr)
        self.assertNotIn("SECURITY_TRIAGE", result.stderr)

    def test_both_drift_exits_7_with_both_messages(self):
        repo = self.make_repo()
        result = self.guard(str(repo), STUB_TRIAGE="fail", STUB_REALITY="modify")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("DRIFT: SECURITY_TRIAGE.md", result.stderr)
        self.assertIn("DRIFT: relatorios de realidade", result.stderr)

    def test_reality_failure_is_invalid_input_not_drift(self):
        repo = self.make_repo()
        result = self.guard(str(repo), STUB_REALITY="fail")
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertNotIn("DRIFT", result.stderr)

    def test_bandit_without_report_is_invalid_input(self):
        repo = self.make_repo()
        result = self.guard(str(repo), STUB_BANDIT="none")
        self.assertEqual(result.returncode, 3, result.stderr)


class AD5InvalidInput(DriftGuardBase):
    def test_missing_or_empty_argument(self):
        for args in ((), ("",)):
            with self.subTest(args=args):
                self.assertEqual(self.guard(*args).returncode, 3)

    def test_nonexistent_directory(self):
        self.assertEqual(self.guard(str(self.base / "nao-existe")).returncode, 3)

    def test_file_instead_of_directory(self):
        path = self.base / "arquivo.txt"
        path.write_text("x", encoding="utf-8")
        self.assertEqual(self.guard(str(path)).returncode, 3)

    def test_directory_that_is_not_a_git_repo(self):
        repo = self.make_repo(git=False)
        self.assertEqual(self.guard(str(repo)).returncode, 3)

    def test_repo_without_triage_tool(self):
        repo = self.make_repo(triage=False)
        self.assertEqual(self.guard(str(repo)).returncode, 3)

    def test_repo_without_reality_tool(self):
        repo = self.make_repo(reality=False)
        self.assertEqual(self.guard(str(repo)).returncode, 3)

    def test_invalid_input_never_exits_0_or_7(self):
        for result in (self.guard(), self.guard(str(self.base / "x"))):
            self.assertNotIn(result.returncode, (0, 7))


class AD3AD6Properties(DriftGuardBase):
    def test_guard_is_directly_executable(self):
        mode = GUARD.stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR, "artefatos_drift.sh perdeu o bit executável")
        result = subprocess.run(
            [str(GUARD), str(self.base / "nao-existe")], capture_output=True, text=True,
            timeout=30, env=self.env(),
        )
        self.assertEqual(result.returncode, 3)

    def test_original_repo_is_never_written(self):
        for stubs in ({}, {"STUB_TRIAGE": "fail"}, {"STUB_REALITY": "modify"},
                      {"STUB_TRIAGE": "fail", "STUB_REALITY": "modify"}):
            with self.subTest(stubs=stubs):
                repo = self.make_repo()
                before = _tree_hash(repo)
                self.guard(str(repo), **stubs)
                self.assertEqual(_tree_hash(repo), before)
                self.assertEqual(_git(repo, "status", "--porcelain").stdout, "")
                subprocess.run(["rm", "-rf", str(repo)], check=True)


class AD7ReplayType(DriftGuardBase):
    def replay(self, table_text, **stubs):
        table = self.base / "casos.tsv"
        table.write_text(table_text, encoding="utf-8")
        return subprocess.run(
            ["bash", str(REPLAY), str(table)], cwd=ROOT, capture_output=True,
            text=True, timeout=120, env=self.env(**stubs),
        )

    def test_absolute_path_cases_follow_the_guard(self):
        repo = self.make_repo()
        result = self.replay(
            f"limpo\tdrift\t{repo}\t0\n"
            f"incompat\tdrift\t{self.base}/nao-existe\t3\n"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASSOU limpo rc=0", result.stdout)
        self.assertIn("PASSOU incompat rc=3", result.stdout)

    def test_drift_case_with_wrong_expectation_fails(self):
        repo = self.make_repo()
        result = self.replay(f"limpo\tdrift\t{repo}\t7\n")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("FALHOU limpo rc=0 esperado=7", result.stdout)

    def test_divergent_repo_is_7_in_the_replay(self):
        repo = self.make_repo()
        result = self.replay(f"div\tdrift\t{repo}\t7\n", STUB_TRIAGE="fail")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_sha_is_invalid_input_of_the_replay(self):
        result = self.replay("sem-sha\tdrift\tdeadbeefdeadbeef\t0\n")
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertIn("entrada invalida", result.stdout)

    def test_unknown_type_still_fails_closed(self):
        result = self.replay("x\tdriftz\tfoo\t0\n")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
