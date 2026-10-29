"""Contrato da guarda de margem do timeout do CI (TM1-TM9; ver docs/agentes/MARGEM_TIMEOUT_CI.md)."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
GUARD = ROOT / "tools/agente/timeout_margin.sh"
BACKTEST = ROOT / "tools/agente/timeout_margin_backtest.py"
SNAPSHOT = ROOT / "docs/agentes/timeout_margin_historico.json"
REPLAY = ROOT / "tools/agente/replay/run_replay.sh"


def run_guard(timeout, content, *, name="d.txt", raw=None):
    """Roda a guarda com `content` (str/bytes) num arquivo; `raw` troca o argumento do arquivo."""
    with tempfile.TemporaryDirectory(prefix="tm-guard-") as tmp:
        path = Path(tmp) / name
        if content is not None:
            data = content.encode() if isinstance(content, str) else content
            path.write_bytes(data)
        arg = str(path) if raw is None else raw
        args = [] if timeout is None else [timeout]
        if timeout is not None:
            args.append(arg)
        return subprocess.run(["bash", str(GUARD), *args], capture_output=True, text=True, timeout=20)


def run_backtest(*args, env=None):
    return subprocess.run([sys.executable, str(BACKTEST), *args], capture_output=True, text=True,
                          cwd=ROOT, timeout=120, env=env)


class GuardDecisionTests(unittest.TestCase):
    def rc(self, timeout, content):
        return run_guard(timeout, content).returncode

    def test_proposal_cases(self):
        fx = ROOT / "tools/agente/replay/fixtures"
        def g(t, name):
            return subprocess.run(["bash", str(GUARD), t, str(fx / name)], capture_output=True, text=True).returncode
        self.assertEqual(g("12", "tmo_origem.txt"), 6)
        self.assertEqual(g("12", "tmo_anterior.txt"), 0)
        self.assertEqual(g("20", "tmo_atual.txt"), 0)
        self.assertEqual(g("12", "tmo_texto.txt"), 3)
        self.assertEqual(g("12", "tmo_vazio.txt"), 3)
        self.assertEqual(g("12", "nao_existe.txt"), 3)

    def test_boundary_is_inclusive_at_85_percent(self):
        self.assertEqual(self.rc("12", "612\n"), 0)
        self.assertEqual(self.rc("12", "613\n"), 6)
        self.assertEqual(self.rc("1", "51\n"), 0)
        self.assertEqual(self.rc("1", "52\n"), 6)
        self.assertEqual(self.rc("20", "1020\n"), 0)
        self.assertEqual(self.rc("20", "1100\n"), 6)

    def test_max_decides_not_last_or_mean(self):
        self.assertEqual(self.rc("12", "700\n100\n100\n"), 6)
        self.assertEqual(self.rc("12", "100\n100\n612\n"), 0)

    def test_last_line_without_newline_and_blank_lines(self):
        self.assertEqual(self.rc("12", "500"), 0)
        self.assertEqual(self.rc("12", "700"), 6)
        self.assertEqual(self.rc("12", "\n500\n\n600\n\n"), 0)

    def test_message_names_limit(self):
        r = run_guard("12", "700\n")
        self.assertEqual(r.returncode, 6)
        self.assertIn("612", r.stdout)
        ok = run_guard("12", "600\n")
        self.assertIn("OK", ok.stdout)


class GuardInvalidInputTests(unittest.TestCase):
    def assert_invalid(self, r):
        self.assertEqual(r.returncode, 3, (r.stdout, r.stderr))
        self.assertIn("entrada invalida", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_invalid_timeout(self):
        for t in ("abc", "", "0", "-5", "12.5", "012", "1e1", "999999", " 12", "12 "):
            with self.subTest(timeout=t):
                self.assert_invalid(run_guard(t, "500\n"))

    def test_no_arguments_and_one_argument(self):
        r = subprocess.run(["bash", str(GUARD)], capture_output=True, text=True)
        self.assert_invalid(r)
        r = subprocess.run(["bash", str(GUARD), "12"], capture_output=True, text=True)
        self.assert_invalid(r)

    def test_missing_directory_empty_blank_only(self):
        self.assert_invalid(run_guard("12", None, raw="/nonexistent/duracoes.txt"))
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run(["bash", str(GUARD), "12", tmp], capture_output=True, text=True)
            self.assert_invalid(r)
        self.assert_invalid(run_guard("12", ""))
        self.assert_invalid(run_guard("12", "\n\n\n"))

    def test_unparseable_lines(self):
        bad = ["abc", "0", "-3", "3.5", "1e3", " 5", "5 ", "5\r", "1234567890",
               "99999999999999999999999", "0x10", "+5", "٣"]
        for line in bad:
            with self.subTest(line=line):
                self.assert_invalid(run_guard("12", f"500\n{line}\n"))

    def test_invalid_line_wins_over_a_blocking_value(self):
        self.assert_invalid(run_guard("12", "9999\nabc\n"))

    def test_unreadable_file(self):
        if os.geteuid() == 0:
            self.skipTest("root le qualquer arquivo")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.txt"
            p.write_text("500\n")
            p.chmod(0)
            r = subprocess.run(["bash", str(GUARD), "12", str(p)], capture_output=True, text=True)
            p.chmod(0o600)
            self.assert_invalid(r)

    def test_invalid_code_is_distinct_from_pass_and_block(self):
        codes = {run_guard("12", "500\n").returncode, run_guard("12", "700\n").returncode,
                 run_guard("12", "x\n").returncode}
        self.assertEqual(codes, {0, 6, 3})


class ReplayIntegrationTests(unittest.TestCase):
    def test_default_table_includes_timeout_cases_and_passes(self):
        # So os casos `tmo` da tabela padrao: os demais exigem historico git completo (o checkout do
        # contract-gate e raso); a tabela inteira roda no job `Agent guard replay` (fetch-depth: 0).
        rows = [ln for ln in (ROOT / "tools/agente/replay/casos.tsv").read_text().splitlines()
                if ln.split("\t")[1:2] == ["tmo"]]
        self.assertEqual(len(rows), 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "casos.tsv"
            path.write_text("\n".join(rows) + "\n")
            r = subprocess.run(["bash", str(REPLAY), str(path)], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for case in ("T-origem rc=6", "T-anterior rc=0", "T-incompat-ausente rc=3",
                     "T-incompat-vazio rc=3", "T-incompat-texto rc=3", "T-incompat-timeout rc=3"):
            self.assertIn("PASSOU " + case, r.stdout)

    def test_inverted_expectation_fails_the_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "casos.tsv"
            path.write_text("T-origem\ttmo\t12 fixtures/tmo_origem.txt\t0\n")
            r = subprocess.run(["bash", str(REPLAY), str(path)], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 1)
            self.assertIn("FALHOU T-origem rc=6 esperado=0", r.stdout)


class BacktestInvalidDataTests(unittest.TestCase):
    def check(self, content):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "h.json"
            if content is not None:
                p.write_text(content)
            r = run_backtest("avaliar", "--dados", str(p))
        self.assertEqual(r.returncode, 3, (r.stdout, r.stderr))
        self.assertIn("dados invalidos", r.stderr)

    def test_missing_empty_and_malformed_snapshot(self):
        self.check(None)
        self.check("")
        self.check("{not json")
        self.check("[]")
        self.check(json.dumps({"schema": 1, "jobs": []}))
        self.check(json.dumps({"schema": 1}))
        self.check(json.dumps({"schema": 2, "jobs": [{}]}))

    def test_job_missing_field_or_bad_time(self):
        job = {"run": 1, "attempt": 1, "conclusion": "success", "started": "2026-10-28T10:00:00Z",
               "completed": "2026-10-28T10:05:00Z", "sha": "a", "event": "push", "branch": "main", "timeout": 12}
        for broken in ({k: v for k, v in job.items() if k != "completed"},
                       dict(job, started="ontem"), dict(job, completed="2026-10-28T09:00:00Z")):
            with self.subTest(job=broken):
                self.check(json.dumps({"schema": 1, "jobs": [broken]}))

    def test_unknown_subcommand_and_no_subcommand(self):
        self.assertEqual(run_backtest("nada").returncode, 3)
        self.assertEqual(run_backtest().returncode, 3)

    def test_collect_fails_closed_without_gh(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "gh"
            fake.write_text("#!/bin/sh\necho 'not json'\nexit 0\n")
            fake.chmod(0o755)
            out = Path(tmp) / "h.json"
            env = dict(os.environ, PATH=f"{tmp}:{os.environ['PATH']}")
            r = run_backtest("coletar", "--saida", str(out), env=env)
            self.assertEqual(r.returncode, 3, (r.stdout, r.stderr))
            self.assertFalse(out.exists())


class BacktestRealHistoryTests(unittest.TestCase):
    """TM1-TM5 e TM9 sobre o instantaneo congelado do historico real."""

    @classmethod
    def setUpClass(cls):
        cls.r = run_backtest("avaliar", "--json")
        try:
            cls.out = json.loads(cls.r.stdout)
        except ValueError:
            cls.out = {}

    def test_exit_zero_and_all_criteria_pass(self):
        self.assertEqual(self.r.returncode, 0, self.r.stdout + self.r.stderr)
        for key in ("TM1", "TM2", "TM3", "TM4", "TM5"):
            self.assertTrue(self.out["criterios"][key]["passou"], key)

    def test_known_incidents_are_flagged(self):
        tm1 = self.out["criterios"]["TM1"]
        self.assertGreaterEqual(tm1["incidentes"], 3)
        self.assertEqual(tm1["sinalizados"], tm1["incidentes"])
        self.assertTrue(tm1["conhecidos_presentes"])

    def test_trend_is_reported(self):
        self.assertTrue(self.out["tendencia"])
        self.assertIn("mediana_s", self.out["tendencia"][0])

    def test_current_main_report_is_report_only(self):
        r = run_backtest("atual")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("guarda", r.stdout)


if __name__ == "__main__":
    unittest.main()
