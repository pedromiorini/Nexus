"""Contrato estrutural do job matricial de compatibilidade Python do CI."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parent
WORKFLOW = ROOT / ".github/workflows/nexus-contract-gate.yml"


class PythonCompatibilityGateContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        cls.job = workflow["jobs"]["python-compatibility"]
        cls.steps = cls.job["steps"]

    def test_matrix_covers_supported_interpreter_versions(self):
        strategy = self.job["strategy"]
        self.assertFalse(strategy["fail-fast"])
        self.assertEqual(strategy["matrix"]["python-version"], ["3.12", "3.13"])
        self.assertEqual(self.job["timeout-minutes"], 10)

    def test_job_is_independent_and_has_repository_environment(self):
        self.assertNotIn("needs", self.job)
        self.assertEqual(self.job["env"]["PYTHONPATH"], ".")
        self.assertEqual(self.job["runs-on"], "ubuntu-24.04")

    def test_each_matrix_job_uses_locked_dependencies(self):
        install = next(step for step in self.steps if step["name"] == "Install runtime dependencies")
        self.assertIn("--require-hashes", install["run"])
        self.assertIn("requirements-ci.lock", install["run"])

    def test_each_matrix_job_compiles_and_runs_unittest(self):
        compile_step = next(step for step in self.steps if step["name"] == "Compile Python modules")
        test_step = next(step for step in self.steps if step["name"] == "Run unittest compatibility suite")
        self.assertIn("python -m compileall", compile_step["run"])
        self.assertIn("python -m unittest discover -p 'test_*.py'", test_step["run"])

    def test_job_keeps_generated_databases_outside_workspace(self):
        database_step = next(
            step for step in self.steps if step["name"] == "Keep generated databases out of the workspace"
        )
        self.assertIn("RUNNER_TEMP", database_step["run"])
        self.assertIn("NEXUS_AUDIT_DB", database_step["run"])
