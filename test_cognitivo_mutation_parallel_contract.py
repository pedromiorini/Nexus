"""Contrato do mutation gate paralelo do core/cognitivo (M1-M7; ver docs/agentes/MUTACAO_COGNITIVO_PARALELA.md).

Usa um repositório sintético temporário e testes reais em subprocessos; não
executa o catálogo real (isso é feito pelo CI e pela validação M2/M3).
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools import cognitivo_mutation as cm  # noqa: E402
from tools.targeted_mutation import Mutation  # noqa: E402

FIELDS = ("name", "target", "tests", "status", "exit_code", "matches")


def norm(results):
    return [{k: r.get(k) for k in FIELDS} for r in results]


def make_project(root: Path, *, broken_baseline=False):
    (root / "mod.py").write_text(
        "MARK = 0\n\n\ndef f(x):\n    return x + 1\n\n\ndef g():\n    return 1\n", encoding="utf-8"
    )
    expected = "3" if broken_baseline else "2"
    (root / "test_mod.py").write_text(
        "import unittest\nimport mod\n\n\nclass T(unittest.TestCase):\n"
        f"    def test_f(self):\n        self.assertEqual(mod.f(1), {expected})\n\n\n"
        "if __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8",
    )


TESTS = ("test_mod.py",)


def synthetic_catalog():
    return [
        Mutation("kills-f", "mod.py", "return x + 1", "return x - 1", TESTS),
        Mutation("survives-g", "mod.py", "return 1\n", "return 2\n", TESTS),
        Mutation("absent", "mod.py", "NAO_EXISTE", "x", TESTS),
        Mutation("ambiguous", "mod.py", "return", "return  ", TESTS),
        Mutation("syntax-error", "mod.py", "def g():", "def g(:", TESTS),
        Mutation("kills-f-again", "mod.py", "x + 1", "x + 2", TESTS),
    ]


class Project:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="cm-par-")
        self.root = Path(self._tmp.name)
        make_project(self.root)
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()


class ParseJobsContract(unittest.TestCase):
    def test_valid_forms(self):
        self.assertEqual(cm.parse_jobs(["--jobs", "1"]), 1)
        self.assertEqual(cm.parse_jobs(["--jobs=4"]), 4)
        self.assertEqual(cm.parse_jobs(["--jobs", "32"]), 32)

    def test_default_is_bounded_cpu_count(self):
        for cpus, want in ((4, 4), (2, 2), (64, 8), (None, 1), (0, 1)):
            with mock.patch.object(cm.os, "cpu_count", return_value=cpus):
                self.assertEqual(cm.parse_jobs([]), want, cpus)

    def test_invalid_inputs_exit_2(self):
        bad = [
            ["--jobs"], ["--jobs", ""], ["--jobs", "0"], ["--jobs", "-1"], ["--jobs", "abc"],
            ["--jobs", "1.5"], ["--jobs", "01"], ["--jobs", "33"], ["--jobs", "100"], ["--jobs", "1e1"],
            ["--jobs", " 2"], ["--jobs", "2 "], ["--jobs", "2\n"], ["--jobs", "\u0663"],
            ["--jobs=0"], ["--jobs="], ["--jobs", "2", "--jobs", "3"], ["--jobs=2", "--jobs", "3"],
            ["extra"], ["--unknown"], ["-j", "2"], ["--job", "2"], ["--jobs", "2", "extra"],
        ]
        for argv in bad:
            with self.subTest(argv=argv):
                with mock.patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as ctx:
                        cm.parse_jobs(argv)
                self.assertEqual(ctx.exception.code, 2)


class MainInvalidInputRunsNothing(unittest.TestCase):
    def test_invalid_jobs_executes_nothing_and_writes_no_report(self):
        with Project() as p:
            report = p.root / "report.json"
            with mock.patch("tools.cognitivo_mutation.run_tests") as run, mock.patch("sys.stderr"):
                code = cm.main(["--jobs", "0"], mutations=synthetic_catalog(), root=p.root, report=report)
            self.assertEqual(code, 2)
            run.assert_not_called()
            self.assertFalse(report.exists())

    def test_cli_invalid_jobs_exits_2_fast_without_report_change(self):
        report = ROOT / "cognitivo_mutation_report.json"
        before = report.read_bytes() if report.exists() else None
        done = subprocess.run(
            [sys.executable, str(ROOT / "tools/cognitivo_mutation.py"), "--jobs", "0"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn("--jobs", done.stderr)
        after = report.read_bytes() if report.exists() else None
        self.assertEqual(before, after)


class EquivalenceAcrossJobs(unittest.TestCase):
    def test_serial_and_parallel_report_are_identical_and_ordered(self):
        runs = {}
        for jobs in (1, 3, 8):
            with Project() as p:
                report = p.root / "report.json"
                code = cm.main(["--jobs", str(jobs)], mutations=synthetic_catalog(), root=p.root, report=report)
                data = json.loads(report.read_text(encoding="utf-8"))
                runs[jobs] = (code, norm(data["mutations"]), data["summary"], list(data))
        base = runs[1]
        names = [m.name for m in synthetic_catalog()]
        self.assertEqual([r["name"] for r in base[1]], names)
        self.assertEqual(
            {r["name"]: r["status"] for r in base[1]},
            {"kills-f": "killed", "survives-g": "survived", "absent": "not_applied",
             "ambiguous": "not_applied", "syntax-error": "killed", "kills-f-again": "killed"},
        )
        self.assertEqual(base[2], {"declared": 6, "killed": 3, "survived": 1, "not_applied": 2})
        self.assertEqual(base[0], 1)
        for jobs in (3, 8):
            self.assertEqual(runs[jobs], base, jobs)

    def test_report_shape_is_unchanged(self):
        with Project() as p:
            report = p.root / "report.json"
            cm.main(["--jobs", "2"], mutations=synthetic_catalog(), root=p.root, report=report)
            data = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(list(data), ["tool", "mutations", "summary"])
        self.assertEqual(data["tool"], "nexus-cognitivo-mutation")
        killed = data["mutations"][0]
        self.assertEqual(
            set(killed), {"name", "target", "tests", "status", "exit_code", "duration_seconds", "output_tail"}
        )
        self.assertEqual(set(data["mutations"][2]), {"name", "target", "status", "matches"})

    def test_repeated_parallel_runs_are_deterministic(self):
        seen = []
        for _ in range(2):
            with Project() as p:
                report = p.root / "report.json"
                cm.main(["--jobs", "4"], mutations=synthetic_catalog(), root=p.root, report=report)
                seen.append(norm(json.loads(report.read_text(encoding="utf-8"))["mutations"]))
        self.assertEqual(seen[0], seen[1])


class GateIsNotWeakened(unittest.TestCase):
    def run_main(self, mutations, jobs="2", **kw):
        with Project() as p:
            if kw.get("broken_baseline"):
                make_project(p.root, broken_baseline=True)
            report = p.root / "report.json"
            err = None
            try:
                code = cm.main(["--jobs", jobs], mutations=mutations, root=p.root, report=report)
            except SystemExit as exc:
                code, err = None, exc.code
            data = json.loads(report.read_text(encoding="utf-8")) if report.exists() else None
        return code, err, data

    def test_all_killed_exits_0(self):
        cat = [m for m in synthetic_catalog() if m.name in ("kills-f", "syntax-error", "kills-f-again")]
        code, _, data = self.run_main(cat)
        self.assertEqual(code, 0)
        self.assertEqual(data["summary"], {"declared": 3, "killed": 3, "survived": 0, "not_applied": 0})

    def test_survivor_exits_1(self):
        cat = [m for m in synthetic_catalog() if m.name in ("kills-f", "survives-g")]
        code, _, data = self.run_main(cat)
        self.assertEqual(code, 1)
        self.assertEqual(data["summary"]["survived"], 1)

    def test_not_applied_exits_1_for_zero_and_multiple_matches(self):
        for name, matches in (("absent", 0), ("ambiguous", 2)):
            with self.subTest(name=name):
                cat = [m for m in synthetic_catalog() if m.name in ("kills-f", name)]
                code, _, data = self.run_main(cat)
                self.assertEqual(code, 1)
                self.assertEqual(data["mutations"][1]["matches"], matches)

    def test_broken_baseline_aborts_before_report(self):
        code, err, data = self.run_main(synthetic_catalog(), broken_baseline=True)
        self.assertIsNone(code)
        self.assertIn("Baseline failed before mutation", str(err))
        self.assertIsNone(data)

    def test_timeout_still_counts_as_killed(self):
        with Project() as p, mock.patch(
            "tools.targeted_mutation.subprocess.run",
            side_effect=subprocess.TimeoutExpired("x", 1),
        ) as run:
            res = cm.evaluate_mutation(p.root, synthetic_catalog()[0])
        self.assertEqual((res["status"], res["exit_code"]), ("killed", 124))
        self.assertTrue(run.called)

    def test_catalog_and_timeout_are_untouched(self):
        from tools import targeted_mutation as tm

        self.assertGreaterEqual(len(cm.MUTATIONS), 483)  # o catálogo nunca encolhe (M7 compara o diff)
        self.assertEqual(tm.TIMEOUT_SECONDS, 180)
        workflow = (ROOT / ".github/workflows/nexus-contract-gate.yml").read_text(encoding="utf-8")
        self.assertIn("timeout-minutes: 20", workflow)
        self.assertIn("run: python tools/cognitivo_mutation.py", workflow)


class WorkerIsolationAndConcurrency(unittest.TestCase):
    def test_each_worker_has_its_own_copy(self):
        """Mutantes na mesma linha do mesmo arquivo não se enxergam: todos mortos, nenhum not_applied."""
        n = 6
        with tempfile.TemporaryDirectory(prefix="cm-iso-") as tmp:
            root = Path(tmp)
            make_project(root)
            muts = []
            for i in range(1, n + 1):
                (root / f"test_iso_{i}.py").write_text(
                    "import time\nfrom pathlib import Path\nimport unittest\n\n\n"
                    "class T(unittest.TestCase):\n    def test_own_mutation_only(self):\n"
                    "        time.sleep(0.4)\n"
                    "        text = Path('mod.py').read_text()\n"
                    f"        self.assertNotIn('MARK = {i}\\n', text)\n",
                    encoding="utf-8",
                )
                muts.append(Mutation(f"iso-{i}", "mod.py", "MARK = 0", f"MARK = {i}", (f"test_iso_{i}.py",)))
            for _ in range(2):
                results = cm.run_catalog(muts, root, jobs=n)
                self.assertEqual([r["status"] for r in results], ["killed"] * n, results)

    def test_workers_really_run_concurrently(self):
        """Barreira: o teste só passa (mutante sobrevive) se 3 mutantes rodarem ao mesmo tempo."""
        with tempfile.TemporaryDirectory(prefix="cm-bar-") as tmp, tempfile.TemporaryDirectory(prefix="cm-bar-d-") as bar:
            root = Path(tmp)
            make_project(root)
            muts = []
            for i in range(3):
                (root / f"test_bar_{i}.py").write_text(
                    "import os, time\nfrom pathlib import Path\nimport unittest\n\n\n"
                    "class T(unittest.TestCase):\n    def test_barrier(self):\n"
                    "        d = Path(os.environ['CM_BARRIER_DIR'])\n"
                    f"        (d / 'w{i}').write_text('x')\n"
                    "        end = time.time() + 20\n"
                    "        while time.time() < end:\n"
                    "            if len(list(d.iterdir())) >= 3:\n"
                    "                return\n"
                    "            time.sleep(0.05)\n"
                    "        self.fail('sem concorrência')\n",
                    encoding="utf-8",
                )
                muts.append(Mutation(f"bar-{i}", "mod.py", "MARK = 0", f"MARK = {i + 1}", (f"test_bar_{i}.py",)))
            with mock.patch.dict(os.environ, {"CM_BARRIER_DIR": bar}):
                # baseline passaria só com 3 simultâneos também; todos concorrentes -> survived
                results = cm.run_catalog(muts, root, jobs=3, skip_baseline=True)
            self.assertEqual([r["status"] for r in results], ["survived"] * 3, results)


if __name__ == "__main__":
    unittest.main()
