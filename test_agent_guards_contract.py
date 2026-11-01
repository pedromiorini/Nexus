"""Contract tests for agent shell guards; every git invocation is intercepted."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
PREDELEG = ROOT / "tools/agente/predeleg.sh"
PREFLIGHT = ROOT / "tools/agente/preflight_rotina.sh"
TREE = "0123456789abcdef0123456789abcdef01234567"


FAKE_GIT = r'''#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_GIT_LOG"
if [[ $1 == -C ]]; then shift 2; fi
case ${1:-} in
  cat-file)
    [[ ${FAKE_CAT_RC:-0} == 0 ]] && exit 0
    exit "${FAKE_CAT_RC}"
    ;;
  merge-tree)
    printf '%s' "${FAKE_MERGE_OUTPUT:-}"
    exit "${FAKE_MERGE_RC:-0}"
    ;;
  status)
    printf '%s' "${FAKE_STATUS_OUTPUT:-}"
    exit "${FAKE_STATUS_RC:-0}"
    ;;
  fetch)
    exit "${FAKE_FETCH_RC:-0}"
    ;;
  *)
    echo "unexpected fake git command: $*" >&2
    exit 97
    ;;
esac
'''


class AgentGuardsContract(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        bindir = self.root / "bin"
        bindir.mkdir()
        fake = bindir / "git"
        fake.write_text(FAKE_GIT, encoding="utf-8")
        fake.chmod(0o755)
        self.log = self.root / "git.log"
        self.env = os.environ.copy()
        self.env.update({
            "PATH": f"{bindir}{os.pathsep}{self.env['PATH']}",
            "FAKE_GIT_LOG": str(self.log),
            "R": "/isolated/repo",
            "REPO": "/isolated/repo",
            "FAKE_MERGE_RC": "0",
            "FAKE_MERGE_OUTPUT": TREE,
            "FAKE_STATUS_RC": "0",
            "FAKE_STATUS_OUTPUT": "",
            "FAKE_FETCH_RC": "0",
        })

    def run_script(self, script: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(script), *args], cwd=self.root, env=self.env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )

    def git_calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def test_predeleg_accepts_successful_merge_tree(self) -> None:
        result = self.run_script(PREDELEG, "head", "base")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK: sem conflito", result.stdout)

    def test_predeleg_accepts_documented_conflict_with_tree_and_paths(self) -> None:
        self.env.update(FAKE_MERGE_RC="1", FAKE_MERGE_OUTPUT=f"{TREE}\nREADME.md\n")
        result = self.run_script(PREDELEG, "head", "base")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("README.md", result.stdout)

    def test_predeleg_preserves_code_conflict_exit_2(self) -> None:
        self.env.update(FAKE_MERGE_RC="1", FAKE_MERGE_OUTPUT=f"{TREE}\nsrc/main.py\n")
        result = self.run_script(PREDELEG, "head", "base")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_predeleg_missing_hash_preserves_exit_3(self) -> None:
        self.env["FAKE_CAT_RC"] = "1"
        result = self.run_script(PREDELEG, "missing", "base")
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertFalse(any("merge-tree" in call for call in self.git_calls()))

    def test_predeleg_merge_tree_operational_128_empty_fails_closed(self) -> None:
        for rc in ("2", "128"):
            for output in ("", f"{TREE}\nREADME.md\n"):
                with self.subTest(rc=rc, output=output):
                    self.env.update(FAKE_MERGE_RC=rc, FAKE_MERGE_OUTPUT=output)
                    result = self.run_script(PREDELEG, "head", "base")
                    self.assertEqual(result.returncode, 3, result.stdout + result.stderr)

    def test_predeleg_exit_1_without_documented_tree_and_conflict_list_fails(self) -> None:
        for output in ("", TREE, "not-a-tree\nREADME.md\n", f"{TREE}\n../README.md\n",
                       f"{TREE}\n/README.md\n", f"{TREE}\n./README.md\n"):
            with self.subTest(output=output):
                self.env.update(FAKE_MERGE_RC="1", FAKE_MERGE_OUTPUT=output)
                result = self.run_script(PREDELEG, "head", "base")
                self.assertEqual(result.returncode, 3, result.stdout + result.stderr)

    def test_preflight_accepts_ok_after_successful_status_and_fetch(self) -> None:
        self.env["NUVEM_STATUS"] = "ok"
        result = self.run_script(PREFLIGHT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("MODO=nuvem", result.stdout)
        self.assertTrue(any("fetch" in call for call in self.git_calls()))

    def test_preflight_preserves_blocked_exit_10(self) -> None:
        self.env["NUVEM_STATUS"] = "blocked"
        result = self.run_script(PREFLIGHT)
        self.assertEqual(result.returncode, 10, result.stdout + result.stderr)
        self.assertIn("MODO=local", result.stdout)

    def test_preflight_status_128_empty_aborts_before_fetch(self) -> None:
        self.env.update(FAKE_STATUS_RC="128", FAKE_STATUS_OUTPUT="")
        result = self.run_script(PREFLIGHT)
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertIn("MODO=abortar", result.stdout)
        self.assertFalse(any("fetch" in call for call in self.git_calls()))

    def test_preflight_unexpected_status_aborts_without_fetch(self) -> None:
        self.env["NUVEM_STATUS"] = "unexpected"
        result = self.run_script(PREFLIGHT)
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertIn("MODO=abortar", result.stdout)
        self.assertFalse(any("fetch" in call for call in self.git_calls()))

    def test_preflight_dirty_checkout_preserves_exit_3(self) -> None:
        self.env["FAKE_STATUS_OUTPUT"] = " M file.txt"
        result = self.run_script(PREFLIGHT)
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertIn("checkout_sujo", result.stdout)
        self.assertFalse(any("fetch" in call for call in self.git_calls()))

    def test_preflight_fetch_error_preserves_exit_3(self) -> None:
        self.env["FAKE_FETCH_RC"] = "1"
        result = self.run_script(PREFLIGHT)
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertIn("motivo=fetch", result.stdout)


if __name__ == "__main__":
    unittest.main()
