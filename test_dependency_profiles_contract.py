"""Contratos dos manifests locais; não instalam pacotes nem acessam a rede."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parent


def requirement_lines(name):
    """Entradas ativas dos manifests simples mantidos pelo projeto."""
    return [
        line.split("#", 1)[0].strip()
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]


class DependencyProfilesContractTests(unittest.TestCase):
    def test_runtime_remains_minimal_and_constitutional_profile_is_an_alias(self):
        self.assertEqual(requirement_lines("requirements.txt"), ["numpy>=1.26,<3"])
        self.assertEqual(
            requirement_lines("requirements_constitutional.txt"),
            ["-r requirements.txt"],
        )

    def test_dev_inherits_shared_profile_without_duplicating_test_pins(self):
        lines = requirement_lines("requirements-dev.txt")
        self.assertEqual(lines[0], "-r requirements-ci.in")
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[1], r"^zizmor==[0-9]+\.[0-9]+\.[0-9]+$")

    def test_shared_profile_inherits_runtime_and_pins_all_direct_test_dependencies(self):
        lines = requirement_lines("requirements-ci.in")
        self.assertEqual(lines[0], "-r requirements.txt")
        names = []
        for line in lines[1:]:
            match = re.fullmatch(r"([a-z0-9-]+)==([0-9]+\.[0-9]+\.[0-9]+)", line)
            self.assertIsNotNone(match, line)
            names.append(match.group(1))
        self.assertEqual(len(names), len(set(names)), "Dependência direta duplicada")
        self.assertEqual(
            set(names), {"numpy", "coverage", "hypothesis", "bandit", "ruff", "networkx"}
        )

    def test_lock_matches_every_direct_pin_and_provides_hashes(self):
        lock = (ROOT / "requirements-ci.lock").read_text(encoding="utf-8")
        blocks = re.findall(
            r"^([a-z0-9-]+)==([^\s]+) \\\n((?:[ \t]+[^\n]*\n)+)",
            lock,
            re.MULTILINE,
        )
        locked = {name: (version, body) for name, version, body in blocks}
        self.assertTrue(locked, "Lock vazio ou formato inesperado")
        for line in requirement_lines("requirements-ci.in")[1:]:
            name, version = line.split("==")
            with self.subTest(package=name):
                self.assertIn(name, locked)
                self.assertEqual(locked[name][0], version)
        for name, (_, body) in locked.items():
            with self.subTest(package=name):
                self.assertRegex(body, r"--hash=sha256:[0-9a-f]{64}")

    def test_dev_workflow_linter_version_matches_ci_binary(self):
        workflow = (ROOT / ".github/workflows/nexus-contract-gate.yml").read_text(encoding="utf-8")
        version = requirement_lines("requirements-dev.txt")[1].split("==")[1]
        self.assertRegex(
            workflow,
            rf"name: Install zizmor[^\n]*\n\s+run: \|\n\s+version={re.escape(version)}\n",
        )

    def test_ci_installs_hash_locked_profile_and_requires_graph_backend(self):
        workflow = (ROOT / ".github/workflows/nexus-contract-gate.yml").read_text(encoding="utf-8")
        install = "python -m pip install --require-hashes -r requirements-ci.lock"
        backend = 'python -c "import networkx;'
        self.assertIn(install, workflow)
        self.assertIn(backend, workflow)
        self.assertLess(workflow.index(install), workflow.index(backend))
        self.assertLess(workflow.index(backend), workflow.index("coverage run --branch"))


if __name__ == "__main__":
    unittest.main()
