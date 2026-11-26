"""Contrato da guarda `tools/agente/base_viva.sh`.

A branch ainda mescla limpo sobre a main atual?
Uso: ``base_viva.sh [HEAD] [MAIN]`` (padrão ``HEAD`` e ``origin/main``;
``R`` é o diretório do repositório, padrão ``.``).

Saídas: 0 mescla limpo; 9 conflito com a main (até 20 arquivos); 3 SHA/ref
inexistente ou ``merge-tree`` falhou. Git sem ``merge-tree --write-tree``
(git < 2.38) pula este contrato em vez de falhar.
"""
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GUARD = ROOT / "tools" / "agente" / "base_viva.sh"


def _write_tree_supported() -> bool:
    proc = subprocess.run(
        ["git", "merge-tree", "-h"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "--write-tree" in (proc.stdout + proc.stderr)


def _git(repo: Path, *args: str) -> None:
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.name=base-viva",
            "-c",
            "user.email=base-viva@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"git {args!r} rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        )


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "R": str(repo),
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }
    return subprocess.run(
        [os.fspath(GUARD), *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


class BaseVivaGuardContract(unittest.TestCase):
    def setUp(self) -> None:
        if not _write_tree_supported():
            self.skipTest(
                "git sem merge-tree --write-tree (git < 2.38); contrato pulado"
            )
        self.assertTrue(GUARD.is_file(), "base_viva.sh ausente")
        mode = GUARD.stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR, "base_viva.sh sem bit executável")

    def _repo(self) -> Path:
        tmp = tempfile.TemporaryDirectory(prefix="nexus-base-viva-")
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name)
        _git(repo, "init", "-q", "-b", "main")
        return repo

    def _commit_file(self, repo: Path, name: str, text: str, message: str) -> None:
        (repo / name).write_text(text, encoding="utf-8")
        _git(repo, "add", "--", name)
        _git(repo, "commit", "-q", "-m", message)

    def test_conflict_exits_9_and_names_the_file(self) -> None:
        repo = self._repo()
        self._commit_file(repo, "conflito.txt", "base\n", "base")
        _git(repo, "checkout", "-q", "-b", "topic")
        self._commit_file(repo, "conflito.txt", "topic\n", "topic")
        _git(repo, "checkout", "-q", "main")
        self._commit_file(repo, "conflito.txt", "main\n", "main")

        proc = _run(repo, "topic", "main")

        self.assertEqual(proc.returncode, 9, proc.stdout + proc.stderr)
        self.assertIn("conflito.txt", proc.stdout.splitlines())
        self.assertNotIn("mescla limpo", proc.stdout)

    def test_clean_merge_exits_0(self) -> None:
        repo = self._repo()
        (repo / "a.txt").write_text("base\n", encoding="utf-8")
        (repo / "b.txt").write_text("base\n", encoding="utf-8")
        _git(repo, "add", "--", "a.txt", "b.txt")
        _git(repo, "commit", "-q", "-m", "base")
        _git(repo, "checkout", "-q", "-b", "topic")
        self._commit_file(repo, "b.txt", "topic\n", "topic")
        _git(repo, "checkout", "-q", "main")
        self._commit_file(repo, "a.txt", "main\n", "main")

        proc = _run(repo, "topic", "main")

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("mescla limpo", proc.stdout)
        self.assertNotIn("CONFLITO", proc.stdout)

    def test_missing_ref_exits_3(self) -> None:
        repo = self._repo()
        self._commit_file(repo, "a.txt", "base\n", "base")

        proc = _run(repo, "ref-que-nao-existe", "main")

        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertIn("SHA ausente", proc.stderr)
        self.assertIn("ref-que-nao-existe", proc.stderr)

    def test_missing_sha_exits_3(self) -> None:
        repo = self._repo()
        self._commit_file(repo, "a.txt", "base\n", "base")
        missing = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

        proc = _run(repo, missing, "main")

        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertIn(missing, proc.stderr)

    def test_defaults_head_and_origin_main_when_clean(self) -> None:
        repo = self._repo()
        (repo / "a.txt").write_text("base\n", encoding="utf-8")
        (repo / "b.txt").write_text("base\n", encoding="utf-8")
        _git(repo, "add", "--", "a.txt", "b.txt")
        _git(repo, "commit", "-q", "-m", "base")
        _git(repo, "checkout", "-q", "-b", "topic")
        self._commit_file(repo, "b.txt", "topic\n", "topic")
        _git(repo, "checkout", "-q", "main")
        self._commit_file(repo, "a.txt", "main\n", "main")
        _git(repo, "update-ref", "refs/remotes/origin/main", "main")
        _git(repo, "checkout", "-q", "topic")

        proc = _run(repo)

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("origin/main", proc.stdout)

    def test_conflict_list_is_capped_at_20(self) -> None:
        repo = self._repo()
        names = [f"c{index:02d}.txt" for index in range(21)]
        for name in names:
            (repo / name).write_text("base\n", encoding="utf-8")
        _git(repo, "add", "--", *names)
        _git(repo, "commit", "-q", "-m", "base")
        _git(repo, "checkout", "-q", "-b", "topic")
        for name in names:
            (repo / name).write_text("topic\n", encoding="utf-8")
        _git(repo, "add", "--", *names)
        _git(repo, "commit", "-q", "-m", "topic")
        _git(repo, "checkout", "-q", "main")
        for name in names:
            (repo / name).write_text("main\n", encoding="utf-8")
        _git(repo, "add", "--", *names)
        _git(repo, "commit", "-q", "-m", "main")

        proc = _run(repo, "topic", "main")

        listed = [line for line in proc.stdout.splitlines() if line.endswith(".txt")]
        self.assertEqual(proc.returncode, 9, proc.stdout + proc.stderr)
        self.assertEqual(listed, names[:20])
        self.assertNotIn(names[20], listed)
