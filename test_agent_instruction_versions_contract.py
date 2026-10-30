"""Aceite gradual: origem corrigida, controle vigente e semelhantes inválidos.

Os casos 3.4.1/3.4.2 são fixtures sintéticas, não aceites de agentes reais.
"""
import contextlib
import copy
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tools.agente.validar_registro import ROOT, main, validate

REG = Path("docs/agentes/REGISTRO_AGENTES.yaml")
CLA = Path("docs/agentes/CLAIMS.yaml")
NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


class InstructionVersionsContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "docs/agentes/schema").mkdir(parents=True)
        shutil.copy(ROOT / "docs/agentes/schema/registro_agentes.schema.json",
                    self.root / "docs/agentes/schema/")
        self.catalog = {}
        self.old = self.artifact("3.4.1", b"old instructions\n")
        self.current = self.artifact("3.4.2", b"new instructions\n")
        self.entry = self.root / "docs/agentes/INSTR.md"
        self.entry.write_bytes(b"new instructions\n")
        self.reg = {"schema_version": 2,
                    "instrucoes_atuais": {"arquivo": "docs/agentes/INSTR.md", "versao": "3.4.2"},
                    "instrucoes_versoes": self.catalog,
                    "agentes": [{"id": "fixture", "operador": "test", "ferramenta": "test",
                                 "modelo": "test", "skills": [], "escopo_arquivos": [],
                                 "limites": [], "contato": "test", "last_seen": "2026-10-23",
                                 "instrucoes": {**self.current, "versao_aceita": "3.4.2"}}]}
        (self.root / CLA).write_text("claims: []\n", encoding="utf-8")

    def artifact(self, version, data):
        digest = hashlib.sha256(data).hexdigest()
        relative = f"docs/agentes/versoes/{version}/{digest}.md"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        result = {"arquivo": relative, "sha256": digest}
        self.catalog[version] = result
        return result

    def accept(self, version, artifact):
        self.reg["agentes"][0]["instrucoes"] = {**artifact, "versao_aceita": version}

    def run_validation(self):
        (self.root / REG).write_text(yaml.safe_dump(self.reg), encoding="utf-8")
        return validate(self.root, REG, CLA, NOW)

    def test_current_exact_acceptance_passes_without_warning(self):
        self.assertEqual(self.run_validation(), ([], []))

    def test_older_verified_acceptance_warns_without_using_current_hash(self):
        self.accept("3.4.1", self.old)
        errors, warnings = self.run_validation()
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("3.4.1", warnings[0])
        self.assertIn("3.4.2", warnings[0])

    def test_older_wrong_hash_is_error_not_staleness_warning(self):
        self.accept("3.4.1", {**self.old, "sha256": self.current["sha256"]})
        errors, warnings = self.run_validation()
        self.assertTrue(any("sha256" in e for e in errors), errors)
        self.assertEqual(warnings, [])

    def test_future_version_is_error_even_when_catalogued(self):
        future = self.artifact("3.4.3", b"future\n")
        self.accept("3.4.3", future)
        errors, warnings = self.run_validation()
        self.assertTrue(any("futura" in e for e in errors), errors)
        self.assertEqual(warnings, [])

    def test_uncatalogued_future_is_explicitly_rejected_for_agent(self):
        self.accept("3.4.3", self.current)
        errors, warnings = self.run_validation()
        self.assertTrue(any("agente[0]" in e and "futura" in e for e in errors), errors)
        self.assertEqual(warnings, [])

    def test_unaccepted_future_catalog_entry_also_fails(self):
        self.artifact("3.4.3", b"future\n")
        errors, _ = self.run_validation()
        self.assertTrue(any("registro:" in e and "futura" in e for e in errors), errors)

    def test_unknown_older_version_is_error(self):
        self.accept("3.3.9", self.old)
        errors, warnings = self.run_validation()
        self.assertTrue(any("desconhecida" in e for e in errors), errors)
        self.assertEqual(warnings, [])

    def test_numeric_order_not_lexicographic(self):
        self.reg["instrucoes_atuais"]["versao"] = "3.4.10"
        self.reg["instrucoes_versoes"]["3.4.10"] = self.artifact("3.4.10", b"new instructions\n")
        errors, warnings = self.run_validation()
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)

    def test_missing_accepted_artifact_fails(self):
        self.accept("3.4.1", self.old)
        (self.root / self.old["arquivo"]).unlink()
        errors, _ = self.run_validation()
        self.assertTrue(any("não existe" in e for e in errors), errors)

    def test_missing_current_artifact_and_entrypoint_fail(self):
        for path in (self.root / self.current["arquivo"], self.entry):
            with self.subTest(path=path):
                data = path.read_bytes()
                path.unlink()
                errors, _ = self.run_validation()
                self.assertTrue(errors)
                path.write_bytes(data)

    def test_mutating_historical_artifact_fails_despite_unchanged_entrypoint(self):
        self.accept("3.4.1", self.old)
        (self.root / self.old["arquivo"]).write_bytes(b"tampered\n")
        errors, _ = self.run_validation()
        self.assertTrue(any("sha256" in e for e in errors), errors)

    def test_entrypoint_drift_fails_even_if_all_agents_accept_old(self):
        self.accept("3.4.1", self.old)
        self.entry.write_bytes(b"unpublished edit\n")
        errors, _ = self.run_validation()
        self.assertTrue(any("atual" in e and "sha256" in e for e in errors), errors)

    def test_agent_must_point_to_accepted_artifact_not_mutable_entrypoint(self):
        self.reg["agentes"][0]["instrucoes"]["arquivo"] = "docs/agentes/INSTR.md"
        errors, _ = self.run_validation()
        self.assertTrue(any("arquivo" in e for e in errors), errors)

    def test_catalog_requires_canonical_content_addressed_path(self):
        for path in ("docs/agentes/INSTR.md", "../outside.md", str(self.entry), "", None, True):
            with self.subTest(path=path):
                self.catalog["3.4.2"] = {**self.current, "arquivo": path}
                self.reg["agentes"][0]["instrucoes"]["arquivo"] = path
                errors, _ = self.run_validation()
                self.assertTrue(errors)

    def test_symlink_artifact_or_parent_is_rejected(self):
        path = self.root / self.old["arquivo"]
        data = path.read_bytes()
        target = self.root / "other.md"
        target.write_bytes(data)
        path.unlink()
        path.symlink_to(target)
        errors, _ = self.run_validation()
        self.assertTrue(any("simbólico" in e for e in errors), errors)
        path.unlink()
        path.write_bytes(data)
        directory = path.parent
        moved = directory.with_name("moved")
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
        errors, _ = self.run_validation()
        self.assertTrue(any("simbólico" in e for e in errors), errors)

    def test_missing_or_malformed_catalog_never_falls_back_to_legacy(self):
        for value in (None, {}, [], True, "catalog"):
            with self.subTest(value=value):
                self.reg["instrucoes_versoes"] = value
                errors, _ = self.run_validation()
                self.assertTrue(errors)
        del self.reg["instrucoes_versoes"]
        self.assertTrue(self.run_validation()[0])

    def test_current_version_must_exist_in_catalog(self):
        del self.catalog["3.4.2"]
        self.assertTrue(self.run_validation()[0])

    def test_invalid_version_types_and_formats_fail_without_coercion(self):
        for field in ("current", "accepted", "catalog"):
            original = copy.deepcopy(self.reg)
            for value in (None, True, 3, 3.4, [], {}, "3.04.2", "3.4", "3.4.2-rc1", " 3.4.2", "３.４.２"):
                with self.subTest(field=field, value=value):
                    self.reg = copy.deepcopy(original)
                    if field == "current":
                        self.reg["instrucoes_atuais"]["versao"] = value
                    elif field == "accepted":
                        self.reg["agentes"][0]["instrucoes"]["versao_aceita"] = value
                    elif not isinstance(value, (dict, list)):
                        self.reg["instrucoes_versoes"][value] = self.old
                    else:
                        continue
                    self.assertTrue(self.run_validation()[0])
            self.reg = original

    def test_malformed_instruction_objects_fail_cleanly(self):
        original = copy.deepcopy(self.reg)
        for field in ("current", "accepted", "artifact"):
            for value in (None, [], True, "bad", {}):
                with self.subTest(field=field, value=value):
                    self.reg = copy.deepcopy(original)
                    if field == "current":
                        self.reg["instrucoes_atuais"] = value
                    elif field == "accepted":
                        self.reg["agentes"][0]["instrucoes"] = value
                    else:
                        self.reg["instrucoes_versoes"]["3.4.1"] = value
                    self.assertTrue(self.run_validation()[0])

    def test_hash_types_and_formats_fail(self):
        for value in (None, True, 123, [], {}, "", "F" * 64, "0" * 63):
            with self.subTest(value=value):
                self.reg["agentes"][0]["instrucoes"]["sha256"] = value
                self.assertTrue(self.run_validation()[0])

    def test_unknown_schema_versions_fail(self):
        for value in (None, True, "2", 2.0, 0, 3, [], {}):
            with self.subTest(value=value):
                self.reg["schema_version"] = value
                self.assertTrue(self.run_validation()[0])

    def test_validation_and_cli_never_rewrite_inputs(self):
        self.accept("3.4.1", self.old)
        self.run_validation()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main(["--root", str(self.root), "--agora", NOW.isoformat()])
        self.assertEqual(rc, 0)
        self.assertIn("AVISO:", out.getvalue())
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_cli_invalid_hash_unknown_future_missing_return_one(self):
        original = copy.deepcopy(self.reg)
        for case in ("hash", "unknown", "future", "missing"):
            with self.subTest(case=case):
                self.reg = copy.deepcopy(original)
                if case == "hash":
                    self.reg["agentes"][0]["instrucoes"]["sha256"] = "0" * 64
                elif case == "unknown":
                    self.reg["agentes"][0]["instrucoes"]["versao_aceita"] = "3.3.0"
                elif case == "future":
                    self.reg["agentes"][0]["instrucoes"]["versao_aceita"] = "3.5.0"
                else:
                    (self.root / self.current["arquivo"]).unlink()
                self.run_validation()
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    rc = main(["--root", str(self.root), "--agora", NOW.isoformat()])
                self.assertEqual(rc, 1)
                self.assertIn("ERRO:", err.getvalue())

    def test_repository_preserves_341_bytes_and_axis_acceptance(self):
        reg = yaml.safe_load((ROOT / REG).read_text())
        self.assertEqual(reg["schema_version"], 2)
        expected = "6778552adef5433c85bf9209b494456f4f497ffdf6f8e190b19eab45d54c20a6"
        artifact = reg["instrucoes_versoes"]["3.4.1"]
        self.assertEqual(artifact["sha256"], expected)
        self.assertEqual(hashlib.sha256((ROOT / artifact["arquivo"]).read_bytes()).hexdigest(),
                         expected)
        current = reg["instrucoes_versoes"][reg["instrucoes_atuais"]["versao"]]
        self.assertEqual((ROOT / current["arquivo"]).read_bytes(),
                         (ROOT / reg["instrucoes_atuais"]["arquivo"]).read_bytes())
        axis = next(agent for agent in reg["agentes"] if agent["id"] == "axis")
        self.assertEqual(axis["instrucoes"], {**artifact, "versao_aceita": "3.4.1"})

    def test_schema_exposes_explicit_v2_catalog(self):
        schema = json.loads((ROOT / "docs/agentes/schema/registro_agentes.schema.json").read_text())
        self.assertIn("instrucoes_versoes", schema["properties"])
        self.assertEqual(schema["properties"]["schema_version"]["enum"], [1, 2])


class LegacyInstructionCompatibility(unittest.TestCase):
    def setUp(self):
        self.fixture = InstructionVersionsContract()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.reg["schema_version"] = 1
        del self.fixture.reg["instrucoes_versoes"]
        self.fixture.reg["agentes"][0]["instrucoes"]["arquivo"] = "docs/agentes/INSTR.md"

    def test_legacy_current_still_passes(self):
        self.assertEqual(self.fixture.run_validation(), ([], []))

    def test_legacy_stale_still_fails_without_historical_catalog(self):
        self.fixture.reg["agentes"][0]["instrucoes"]["versao_aceita"] = "3.4.1"
        errors, warnings = self.fixture.run_validation()
        self.assertTrue(errors)
        self.assertEqual(warnings, [])

    def test_schema_unknown_is_not_treated_as_legacy(self):
        for value in (None, True, "1", 1.0, 0, 3, [], {}):
            with self.subTest(value=value):
                self.fixture.reg["schema_version"] = value
                self.assertTrue(self.fixture.run_validation()[0])


if __name__ == "__main__":
    unittest.main()
