import hashlib
import json
import shutil
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

from tools.agente.validar_registro import ROOT, main, overlaps, undeclared_authors, validate

NOW = datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc)
REG = Path("docs/agentes/REGISTRO_AGENTES.yaml")
CLA = Path("docs/agentes/CLAIMS.yaml")


class AgentRegistryContract(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "docs/agentes/schema").mkdir(parents=True)
        shutil.copy(ROOT / "docs/agentes/schema/registro_agentes.schema.json", self.tmp / "docs/agentes/schema/")
        instr = self.tmp / "docs/agentes/INSTR.md"
        instr.write_text("instrucoes v9\n", encoding="utf-8")
        self.reg = {
            "schema_version": 1,
            "instrucoes_atuais": {"arquivo": "docs/agentes/INSTR.md", "versao": "9"},
            "agentes": [self._agent("a"), self._agent("b")],
        }
        self.claims = {"schema_version": 1, "claims": [
            self._claim("a", ["core/x.py"]), self._claim("b", ["docs/y.md"])]}

    def _agent(self, ident):
        digest = hashlib.sha256(b"instrucoes v9\n").hexdigest()
        return {"id": ident, "operador": "op", "ferramenta": "t", "modelo": "m",
                "instrucoes": {"arquivo": "docs/agentes/INSTR.md", "sha256": digest, "versao_aceita": "9"},
                "skills": [], "escopo_arquivos": [], "limites": [], "contato": "c", "last_seen": "2026-10-21"}

    @staticmethod
    def _claim(agent, files, expires="2026-10-22T00:00:00+00:00"):
        return {"agente": agent, "tarefa": "t", "arquivos": files, "expira_em": expires}

    def _run(self, expired_as_warning=False):
        (self.tmp / REG).write_text(yaml.safe_dump(self.reg), encoding="utf-8")
        (self.tmp / CLA).write_text(yaml.safe_dump(self.claims), encoding="utf-8")
        return validate(self.tmp, REG, CLA, NOW, expired_as_warning)

    def test_valid_registry_passes(self):
        self.assertEqual(self._run(), ([], []))

    def test_last_seen_valid_calendar_dates_pass(self):
        for value in ("2026-10-21", "2024-02-29", "0001-01-01", "9999-12-31"):
            with self.subTest(value=value):
                self.reg["agentes"][0]["last_seen"] = value
                self.assertEqual(self._run(), ([], []))

    def test_last_seen_rejects_invalid_values_without_coercion(self):
        invalid = ("2026-02-30", "2025-02-29", "2026-13-01", "0000-01-01",
                   "invalid", "", True, False, [], ["2026-10-21"], 20261021, None,
                   date(2026, 10, 21), datetime(2026, 10, 21))
        for value in invalid:
            with self.subTest(value=value):
                self.reg["agentes"][0]["last_seen"] = value
                errors, _ = self._run()
                self.assertTrue(any("last_seen" in e for e in errors), errors)

    def test_last_seen_rejects_alternative_formats(self):
        for value in ("20261021", "2026-W43-3", "2026-1-01", "2026-10-1",
                      "21/10/2026", "2026/10/21", "2026-10-21T00:00:00",
                      " 2026-10-21", "2026-10-21 ", "2026-10-21\n", "２０２６-１０-２１"):
            with self.subTest(value=value):
                self.reg["agentes"][0]["last_seen"] = value
                errors, _ = self._run()
                self.assertTrue(any("last_seen" in e for e in errors), errors)

    def test_last_seen_is_required(self):
        del self.reg["agentes"][0]["last_seen"]
        errors, _ = self._run()
        self.assertTrue(any("last_seen" in e for e in errors), errors)

    def test_uid_is_optional_and_preserved_exactly(self):
        uid = "hAVRxhnxpHyGN2av2zCZQn"
        self.reg["agentes"][0]["uid"] = uid
        self.assertEqual(self._run(), ([], []))
        saved = yaml.safe_load((self.tmp / REG).read_text(encoding="utf-8"))
        self.assertEqual(saved["agentes"][0]["uid"], uid)
        self.assertNotIn("uid", saved["agentes"][1])

    def test_uid_when_present_must_be_a_nonempty_string(self):
        for value in (None, True, 123, [], {}, ""):
            with self.subTest(value=value):
                self.reg["agentes"][0]["uid"] = value
                errors, _ = self._run()
                self.assertTrue(any("uid" in e for e in errors), errors)

    def test_schema_declares_optional_uid_and_required_date(self):
        schema = json.loads((self.tmp / "docs/agentes/schema/registro_agentes.schema.json").read_text())
        agent = schema["$defs"]["agente"]
        self.assertNotIn("uid", agent["required"])
        self.assertEqual(agent["properties"]["uid"], {"type": "string", "minLength": 1})
        self.assertIn("last_seen", agent["required"])
        self.assertEqual(agent["properties"]["last_seen"]["type"], "string")
        self.assertEqual(agent["properties"]["last_seen"]["format"], "date")

    def test_repository_axis_identity_and_legacy_control(self):
        agents = yaml.safe_load((ROOT / REG).read_text(encoding="utf-8"))["agentes"]
        axis = [agent for agent in agents if agent["id"] == "axis"]
        self.assertEqual(len(axis), 1)
        self.assertEqual(axis[0]["uid"], "hAVRxhnxpHyGN2av2zCZQn")
        grok = [agent for agent in agents if agent["id"] == "grok-bot"]
        self.assertEqual(len(grok), 1)
        self.assertEqual(grok[0]["last_seen"], "2026-10-21")
        self.assertNotIn("uid", grok[0])

    def test_overlapping_claims_fail(self):
        self.claims["claims"][1]["arquivos"] = ["core/**"]
        errors, _ = self._run()
        self.assertTrue(any("sobrepostos" in e for e in errors), errors)

    def test_hash_mismatch_fails(self):
        self.reg["agentes"][0]["instrucoes"]["sha256"] = "0" * 64
        errors, _ = self._run()
        self.assertTrue(any("hash" in e for e in errors), errors)

    def test_version_mismatch_fails(self):
        self.reg["agentes"][1]["instrucoes"]["versao_aceita"] = "8"
        errors, _ = self._run()
        self.assertTrue(any("versão" in e for e in errors), errors)

    def test_missing_required_field_fails(self):
        del self.reg["agentes"][0]["contato"]
        errors, _ = self._run()
        self.assertTrue(any("contato" in e for e in errors), errors)

    def test_expired_claim_fails_by_default_and_warns_in_ci_mode(self):
        self.claims["claims"][0]["expira_em"] = "2026-10-20T00:00:00+00:00"
        errors, _ = self._run()
        self.assertTrue(any("expirado" in e for e in errors), errors)
        errors, warnings = self._run(expired_as_warning=True)
        self.assertEqual(errors, [])
        self.assertTrue(any("expirado" in w for w in warnings), warnings)

    def test_expired_claim_does_not_count_as_overlap(self):
        self.claims["claims"].append(self._claim("b", ["core/x.py"], "2026-10-01T00:00:00+00:00"))
        errors, _ = self._run(expired_as_warning=True)
        self.assertEqual(errors, [])

    def test_claim_by_undeclared_agent_fails(self):
        self.claims["claims"][0]["agente"] = "fantasma"
        errors, _ = self._run()
        self.assertTrue(any("não declarado" in e for e in errors), errors)

    def test_overlap_helper(self):
        self.assertTrue(overlaps("core/**", "core/a/b.py"))
        self.assertTrue(overlaps("docs/*.md", "docs/x.md"))
        self.assertFalse(overlaps("core/**", "corex/a.py"))
        self.assertFalse(overlaps("a.py", "b.py"))

    def test_undeclared_author_is_only_a_warning(self):
        self.reg["agentes"][0]["assinaturas_commit"] = ["a@x"]
        self._run()
        self.assertEqual(undeclared_authors(self.tmp, REG, ["a@x", "outro@y"]), ["outro@y"])
        autores = self.tmp / "autores.txt"
        autores.write_text("outro@y\n", encoding="utf-8")
        rc = main(["--root", str(self.tmp), "--registro", str(REG), "--claims", str(CLA),
                   "--agora", NOW.isoformat(), "--autores", str(autores)])
        self.assertEqual(rc, 0)

    def test_repository_registry_is_valid(self):
        self.assertEqual(main(["--expirados", "aviso"]), 0)


if __name__ == "__main__":
    unittest.main()
