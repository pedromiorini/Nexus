"""Contrato da folga do Contract gate: `--shard K/N` da mutação cognitiva e a estrutura do workflow.

Pré-registro: docs/agentes/CI_GATE_PARALELO.md (CG3, CG4, CG5, CG7, CG8). Usa repositório sintético
temporário; não executa o catálogo real nem o CI.
"""

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from test_cognitivo_mutation_parallel_contract import Project, norm, synthetic_catalog  # noqa: E402
from tools import cognitivo_mutation as cm  # noqa: E402

WORKFLOW = ROOT / ".github/workflows/nexus-contract-gate.yml"


def load_jobs():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def run_cli_main(argv, catalog, root, report):
    return cm.main(argv, mutations=catalog, root=root, report=report)


class ParseShardContract(unittest.TestCase):
    def test_valid_forms(self):
        self.assertEqual(cm.parse_args(["--shard", "2/3"]), (cm.parse_jobs([]), (2, 3)))
        self.assertEqual(cm.parse_args(["--shard=2/3", "--jobs", "4"]), (4, (2, 3)))
        self.assertEqual(cm.parse_args(["--jobs=2", "--shard", "16/16"]), (2, (16, 16)))
        self.assertEqual(cm.parse_args(["--shard", "1/1"])[1], (1, 1))
        self.assertIsNone(cm.parse_args(["--jobs", "2"])[1])
        self.assertEqual(cm.parse_args(["--jobs", "2"])[0], 2)

    def test_invalid_inputs_exit_2(self):
        bad = [
            ["--shard"], ["--shard", ""], ["--shard", "0/3"], ["--shard", "4/3"], ["--shard", "1/0"],
            ["--shard", "1/17"], ["--shard", "17/17"], ["--shard", "a/b"], ["--shard", "1"], ["--shard", "1/"],
            ["--shard", "/3"], ["--shard", "1/3/5"], ["--shard", "01/3"], ["--shard", "1/03"],
            ["--shard", " 1/3"], ["--shard", "1/3 "], ["--shard", "1/3\n"], ["--shard", "-1/3"],
            ["--shard", "1.5/3"], ["--shard", "1e0/3"], ["--shard", "\u0661/3"], ["--shard", "1/\u0663"],
            ["--shard", "1/3", "--shard", "2/3"], ["--shard=1/3", "--shard", "2/3"], ["--shard="],
            ["--shards", "1/3"], ["--shard", "1/3", "extra"],
        ]
        for argv in bad:
            with self.subTest(argv=argv):
                with mock.patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as ctx:
                        cm.parse_args(argv)
                self.assertEqual(ctx.exception.code, 2)

    def test_parse_jobs_is_unchanged_for_jobs_only(self):
        self.assertEqual(cm.parse_jobs(["--jobs", "3"]), 3)
        self.assertEqual(cm.parse_jobs(["--jobs=3", "--shard", "1/2"]), 3)

    def test_usage_mentions_shard(self):
        self.assertIn("--shard", cm.USAGE)


class InvalidShardRunsNothing(unittest.TestCase):
    def test_invalid_shard_executes_nothing_and_writes_no_report(self):
        with Project() as p:
            report = p.root / "report.json"
            with mock.patch("tools.cognitivo_mutation.run_tests") as run, mock.patch("sys.stderr"):
                code = run_cli_main(["--shard", "4/3"], synthetic_catalog(), p.root, report)
            self.assertEqual(code, 2)
            run.assert_not_called()
            self.assertFalse(report.exists())

    def test_cli_invalid_shard_exits_2_fast_without_report_change(self):
        report = ROOT / "cognitivo_mutation_report.json"
        before = report.read_bytes() if report.exists() else None
        done = subprocess.run(
            [sys.executable, str(ROOT / "tools/cognitivo_mutation.py"), "--shard", "4/3"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn("--shard", done.stderr)
        after = report.read_bytes() if report.exists() else None
        self.assertEqual(before, after)


class PartitionContract(unittest.TestCase):
    def test_round_robin_by_catalog_index(self):
        cat = synthetic_catalog()
        self.assertEqual([m.name for m in cm.shard_catalog(cat, 1, 3)], ["kills-f", "ambiguous"])
        self.assertEqual([m.name for m in cm.shard_catalog(cat, 2, 3)], ["survives-g", "syntax-error"])
        self.assertEqual([m.name for m in cm.shard_catalog(cat, 3, 3)], ["absent", "kills-f-again"])
        self.assertEqual([m.name for m in cm.shard_catalog(cat, 1, 1)], [m.name for m in cat])

    def test_real_catalog_is_partitioned_exactly_for_every_count(self):
        names = [m.name for m in cm.MUTATIONS]
        self.assertEqual(len(names), len(set(names)))
        for count in range(1, cm.MAX_SHARDS + 1):
            with self.subTest(count=count):
                parts = [cm.shard_catalog(cm.MUTATIONS, k, count) for k in range(1, count + 1)]
                flat = [m.name for part in parts for m in part]
                self.assertEqual(len(flat), len(names))
                self.assertEqual(sorted(flat), sorted(names))
                for part in parts:
                    got = [m.name for m in part]
                    self.assertEqual(got, [n for n in names if n in set(got)])  # ordem do catálogo
                    self.assertLessEqual(abs(len(part) - len(names) / count), 1)

    def test_shard_catalog_does_not_mutate_the_catalog(self):
        before = [m.name for m in cm.MUTATIONS]
        cm.shard_catalog(cm.MUTATIONS, 2, 3)
        self.assertEqual([m.name for m in cm.MUTATIONS], before)


class ShardedRunsAreEquivalent(unittest.TestCase):
    def run_main(self, argv):
        with Project() as p:
            report = p.root / "report.json"
            code = run_cli_main(argv, synthetic_catalog(), p.root, report)
            data = json.loads(report.read_text(encoding="utf-8"))
        return code, data

    def test_union_of_four_shards_equals_unsharded_run(self):
        code, whole = self.run_main(["--jobs", "2"])
        order = {m.name: i for i, m in enumerate(synthetic_catalog())}
        merged, codes = [], []
        for k in (1, 2, 3, 4):
            c, data = self.run_main(["--jobs", "2", "--shard", f"{k}/4"])
            codes.append(c)
            merged.extend(data["mutations"])
        merged.sort(key=lambda r: order[r["name"]])
        self.assertEqual(norm(merged), norm(whole["mutations"]))
        self.assertEqual(code, 1)
        self.assertEqual(codes, [0, 1, 1, 1])  # só a fatia 1 (kills-f, syntax-error) é limpa

    def test_one_of_one_matches_unsharded(self):
        code, whole = self.run_main(["--jobs", "2"])
        code1, one = self.run_main(["--jobs", "2", "--shard", "1/1"])
        self.assertEqual(code, code1)
        self.assertEqual(norm(one["mutations"]), norm(whole["mutations"]))
        self.assertEqual(one["summary"], whole["summary"])

    def test_sharded_report_carries_completeness_fields(self):
        _, data = self.run_main(["--jobs", "2", "--shard", "2/3"])
        self.assertEqual(list(data), ["tool", "mutations", "summary", "shard"])
        self.assertEqual(data["shard"], {"index": 2, "count": 3, "catalog_total": 6})
        self.assertEqual(data["summary"]["declared"], 2)

    def test_unsharded_report_shape_is_unchanged(self):
        _, data = self.run_main(["--jobs", "2"])
        self.assertEqual(list(data), ["tool", "mutations", "summary"])

    def test_survivor_in_a_shard_fails_only_that_shard(self):
        code, data = self.run_main(["--shard", "2/4"])
        self.assertEqual(code, 1)
        self.assertEqual(data["summary"]["survived"], 1)
        code, data = self.run_main(["--shard", "1/4"])
        self.assertEqual((code, data["summary"]["survived"]), (0, 0))


class WorkflowStructureContract(unittest.TestCase):
    def setUp(self):
        self.jobs = load_jobs()
        self.text = WORKFLOW.read_text(encoding="utf-8")

    def steps_text(self, job):
        return "\n".join(str(step.get("run", "")) for step in self.jobs[job]["steps"])

    def test_cognitivo_gate_runs_only_in_the_shard_job(self):
        self.assertIn("cognitivo-mutation", self.jobs)
        self.assertNotIn("cognitivo_mutation", self.steps_text("contract-gate"))
        for name in ("agent-guard-replay", "state-manifest"):
            self.assertNotIn("cognitivo_mutation", self.steps_text(name))
        runs = [s["run"] for s in self.jobs["cognitivo-mutation"]["steps"] if "cognitivo_mutation" in str(s.get("run", ""))]
        self.assertEqual(len(runs), 1)

    def test_matrix_covers_every_shard_exactly_once_and_matches_count(self):
        job = self.jobs["cognitivo-mutation"]
        shards = job["strategy"]["matrix"]["shard"]
        self.assertEqual(sorted(shards), list(range(1, len(shards) + 1)))
        self.assertFalse(job["strategy"].get("fail-fast", True))
        run = [s["run"] for s in job["steps"] if "cognitivo_mutation" in str(s.get("run", ""))][0]
        self.assertEqual(run.strip(), f"python tools/cognitivo_mutation.py --shard ${{{{ matrix.shard }}}}/{len(shards)}")
        self.assertLessEqual(len(shards), cm.MAX_SHARDS)

    def test_shard_job_has_no_escape_hatches(self):
        job = self.jobs["cognitivo-mutation"]
        self.assertFalse(job.get("continue-on-error", False))
        self.assertNotIn("if", job)
        self.assertEqual(job["timeout-minutes"], 20)
        for step in job["steps"]:
            self.assertFalse(step.get("continue-on-error", False), step.get("name"))
            self.assertNotIn("if", step, step.get("name"))
        self.assertNotIn("needs", job)

    def test_attestation_still_requires_every_gate(self):
        jobs = self.jobs
        needs = jobs["state-manifest"]["needs"]
        self.assertEqual(set([needs] if isinstance(needs, str) else needs), {"contract-gate", "agent-guard-replay"})
        self.assertEqual(
            " ".join(jobs["state-manifest"]["if"].split()),
            "needs.contract-gate.result == 'success' && needs.agent-guard-replay.result == 'success'",
        )
        replay_needs = jobs["agent-guard-replay"]["needs"]
        self.assertIn("cognitivo-mutation", [replay_needs] if isinstance(replay_needs, str) else replay_needs)
        self.assertNotIn("if", jobs["agent-guard-replay"])

    def test_contract_gate_keeps_every_other_threshold_and_gate(self):
        gate = self.steps_text("contract-gate")
        for needle in (
            "--include='core/deferred_task_queue.py' --fail-under=90",
            "--include='core/vram_defense_guard.py' --fail-under=90",
            "--include='core/temporal_facts.py' --fail-under=90",
            "--include='vita/nexus_constitutional_bridge_v3.py' --fail-under=70",
            "--include='core/constitutional_brain.py' --fail-under=62",
            "--include='core/cognitivo/*' --fail-under=90",
            "python tools/targeted_mutation.py",
            "bandit -r core vita tools",
            "python tools/reality_audit.py",
            "python tools/test_summary.py --output test-summary.json",
        ):
            self.assertIn(needle, gate)
        self.assertEqual(self.jobs["contract-gate"]["timeout-minutes"], 20)
        self.assertIn("timeout-minutes: 20", self.text)
        self.assertIn("run: python tools/cognitivo_mutation.py", self.text)
        for step in self.jobs["contract-gate"]["steps"]:
            self.assertFalse(step.get("continue-on-error", False), step.get("name"))

    def test_shard_job_retains_its_report(self):
        uploads = [s for s in self.jobs["cognitivo-mutation"]["steps"] if "upload-artifact" in str(s.get("uses", ""))]
        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0]["with"]["if-no-files-found"], "error")
        self.assertIn("${{ matrix.shard }}", uploads[0]["with"]["name"])


if __name__ == "__main__":
    unittest.main()
