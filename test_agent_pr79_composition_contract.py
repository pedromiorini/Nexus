"""Contrato de composição #79/schema 2; não representa aceite ou publicação."""
import contextlib
import copy
import hashlib
import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tools.agente.validar_registro import ROOT, main, validate

REG = Path("docs/agentes/REGISTRO_AGENTES.yaml")
CLAIMS = Path("docs/agentes/CLAIMS.yaml")
CURRENT = Path("docs/agentes/INSTRUCOES_AGENTE_NEXUS_v3.md")
OLD_SHA = "6778552adef5433c85bf9209b494456f4f497ffdf6f8e190b19eab45d54c20a6"
NEW_SHA = "8f03762d2a2d128eea955597f1b55210e387c08385bdc5544514ea2f39701b5d"
OLD_PATH = f"docs/agentes/versoes/3.4.1/{OLD_SHA}.md"
NEW_PATH = f"docs/agentes/versoes/3.4.2/{NEW_SHA}.md"
NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


class PR79CompositionContract(unittest.TestCase):
    def setUp(self):
        self.reg = yaml.safe_load((ROOT / REG).read_text(encoding="utf-8"))

    def fixture(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for relative in (REG, CLAIMS, CURRENT, Path(OLD_PATH), Path(NEW_PATH),
                         Path("docs/agentes/schema/registro_agentes.schema.json")):
            source = ROOT / relative
            if source.is_file():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_bytes(source.read_bytes())
        return root

    def save(self, root, reg):
        (root / REG).write_text(yaml.safe_dump(reg), encoding="utf-8")

    def test_current_and_snapshot_are_exact_pr79_bytes(self):
        self.assertEqual(self.reg["schema_version"], 2)
        self.assertEqual(self.reg["instrucoes_atuais"],
                         {"arquivo": str(CURRENT), "versao": "3.4.2"})
        self.assertEqual(self.reg["instrucoes_versoes"]["3.4.2"],
                         {"arquivo": NEW_PATH, "sha256": NEW_SHA})
        current = (ROOT / CURRENT).read_bytes()
        self.assertEqual(hashlib.sha256(current).hexdigest(), NEW_SHA)
        self.assertEqual(current, (ROOT / NEW_PATH).read_bytes())

    def test_axis_keeps_historical_acceptance_and_grok_accepts_only_own_version(self):
        self.assertEqual(self.reg["instrucoes_versoes"]["3.4.1"],
                         {"arquivo": OLD_PATH, "sha256": OLD_SHA})
        self.assertEqual(hashlib.sha256((ROOT / OLD_PATH).read_bytes()).hexdigest(), OLD_SHA)
        agents = {agent["id"]: agent for agent in self.reg["agentes"]}
        self.assertEqual(agents["axis"]["instrucoes"],
                         {"arquivo": OLD_PATH, "sha256": OLD_SHA, "versao_aceita": "3.4.1"})
        self.assertEqual(agents["cursor-grok-4-7"]["instrucoes"],
                         {"arquivo": OLD_PATH, "sha256": OLD_SHA, "versao_aceita": "3.4.1"})
        self.assertEqual(agents["grok-bot"]["instrucoes"],
                         {"arquivo": NEW_PATH, "sha256": NEW_SHA, "versao_aceita": "3.4.2"})

    def test_real_composition_warns_for_older_agents_and_never_writes(self):
        root = self.fixture()
        before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        errors, warnings = validate(root, REG, CLAIMS, NOW)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 2)
        warning_text = "\n".join(warnings)
        self.assertIn("axis", warning_text)
        self.assertIn("cursor-grok-4-7", warning_text)
        self.assertNotIn("grok-bot", warning_text)
        self.assertTrue(all("3.4.1" in warning and "3.4.2" in warning
                            for warning in warnings))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main(["--root", str(root), "--expirados", "aviso", "--agora", NOW.isoformat()])
        self.assertEqual(rc, 0)
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(out.getvalue().count("AVISO:"), 2)
        self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_schema2_rejects_naive_pr79_registry_and_missing_artifacts(self):
        for case in ("no_catalog", "no_snapshot", "mutable_agent", "wrong_hash", "unknown", "future"):
            with self.subTest(case=case):
                root = self.fixture()
                reg = copy.deepcopy(self.reg)
                if case == "no_catalog":
                    reg["instrucoes_versoes"].pop("3.4.2", None)
                elif case == "no_snapshot":
                    path = root / NEW_PATH
                    if path.exists():
                        path.unlink()
                elif case == "mutable_agent":
                    reg["agentes"][0]["instrucoes"] = {
                        "arquivo": str(CURRENT), "sha256": NEW_SHA, "versao_aceita": "3.4.2"}
                elif case == "wrong_hash":
                    reg["agentes"][0]["instrucoes"]["sha256"] = "0" * 64
                elif case == "unknown":
                    reg["agentes"][0]["instrucoes"]["versao_aceita"] = "3.3.9"
                else:
                    reg["agentes"][0]["instrucoes"]["versao_aceita"] = "3.4.3"
                self.save(root, reg)
                self.assertTrue(validate(root, REG, CLAIMS, NOW)[0])

    def test_claim_policy_present_without_introducing_active_claims(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("no próprio branch do PR", agents)
        self.assertIn("último commit", agents)
        self.assertIn("PRs abertos", agents)
        body = (ROOT / CLAIMS).read_bytes().split(b"\n", 1)[1]
        # #117 removed the expired Grok claim; this PR must not restore it.
        self.assertEqual(yaml.safe_load(body)["claims"], [])
        self.assertIn("v3.4.2", (ROOT / "docs/agentes/CHANGELOG_INSTRUCOES.md").read_text())


if __name__ == "__main__":
    unittest.main()
