import fnmatch
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import yaml

from tools.agente import validar_registro as registry
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

    def test_intersecting_globs_have_explicit_witnesses(self):
        cases = (
            ("src/a?.py", "src/?b.py", "src/ab.py"),
            ("src/a*.py", "src/*b.py", "src/ab.py"),
            ("src/[a-c]?.py", "src/?[b-d].py", "src/ab.py"),
            ("src/[!a].py", "src/[b-d].py", "src/b.py"),
            ("src/[α-ω].py", "src/[β-δ].py", "src/γ.py"),
            ("src/[a-z].py", "src/[!az].py", "src/m.py"),
            ("src/[]].py", "src/?.py", "src/].py"),
            ("src/[[]?.py", "src/?a.py", "src/[a.py"),
            ("src/[?.py", "src/?a.py", "src/[a.py"),
            ("*a", "b*", "ba"),
            ("a*", "*a", "a"),
            ("*", "", ""),
            ("a?b", "a*b", "a/b"),
            ("[a-c]", "[!ac]", "b"),
            ("[!\x00-\U0010fffe]", "?", "\U0010ffff"),
        )
        for a, b, witness in cases:
            with self.subTest(a=a, b=b, witness=witness):
                self.assertTrue(fnmatch.fnmatchcase(witness, a))
                self.assertTrue(fnmatch.fnmatchcase(witness, b))
                self.assertTrue(overlaps(a, b))
                self.assertTrue(overlaps(b, a))

    def test_disjoint_globs_are_not_conflicts(self):
        cases = (("src/a?.py", "src/b?.py"),
                 ("src/[a-c].py", "src/[d-f].py"),
                 ("src/[!a].py", "src/a.py"),
                 ("src/?.py", "src/??.py"),
                 ("a*b", "a*c"), ("", "?"),
                 ("[z-a]", "*"), ("[z-a]", "[z-a]"),
                 ("core/**", "corex/*"))
        for a, b in cases:
            with self.subTest(a=a, b=b):
                self.assertFalse(overlaps(a, b))
                self.assertFalse(overlaps(b, a))

    def test_recursive_claim_keeps_parent_convention(self):
        for a, b in (("core/**", "core"), ("src/a*/**", "src/*b"),
                     ("core/**", "core/**"), ("core/**", "core/x/**")):
            with self.subTest(a=a, b=b):
                self.assertTrue(overlaps(a, b))
                self.assertTrue(overlaps(b, a))

    def test_small_glob_intersections_match_bounded_fnmatch_oracle(self):
        # For these <=2-token patterns a shortest witness has <=4 characters.
        # a,b,c,] represent all character classes induced by this grammar.
        atoms = ("a", "b", "?", "*", "[ab]", "[!a]", "[!ab]", "[]]")
        patterns = [""] + list(atoms) + [a + b for a in atoms for b in atoms]
        words = ["".join(w) for n in range(5) for w in itertools.product("abc]", repeat=n)]
        matches = {p: {w for w in words if fnmatch.fnmatchcase(w, p)} for p in patterns}
        for a in patterns:
            for b in patterns:
                with self.subTest(a=a, b=b):
                    self.assertEqual(overlaps(a, b), bool(matches[a] & matches[b]))

    def test_intersecting_glob_claims_fail_without_materialized_files(self):
        self.claims["claims"][0]["arquivos"] = ["src/a?.py"]
        self.claims["claims"][1]["arquivos"] = ["src/?b.py"]
        self.assertFalse((self.tmp / "src/ab.py").exists())
        errors, warnings = self._run()
        self.assertEqual(warnings, [])
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("claims sobrepostos em src/a?.py / src/?b.py", errors[0])

    def test_cli_rejects_intersecting_globs_and_accepts_disjoint_control(self):
        self.claims["claims"][0]["arquivos"] = ["src/a?.py"]
        for second, expected in (("src/?b.py", 1), ("src/b?.py", 0)):
            with self.subTest(second=second):
                self.claims["claims"][1]["arquivos"] = [second]
                self._run()
                result = subprocess.run(
                    [sys.executable, str(ROOT / "tools/agente/validar_registro.py"),
                     "--root", str(self.tmp), "--agora", NOW.isoformat()],
                    capture_output=True, text=True, check=False, timeout=20)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                if expected:
                    self.assertIn("claims sobrepostos", result.stderr)
                    self.assertIn("FALHOU", result.stdout)
                else:
                    self.assertEqual(result.stderr, "")
                    self.assertIn("OK", result.stdout)

    def test_expired_intersecting_glob_claim_is_not_active(self):
        self.claims["claims"][0]["arquivos"] = ["src/a?.py"]
        self.claims["claims"][1]["arquivos"] = ["src/?b.py"]
        self.claims["claims"][1]["expira_em"] = NOW.isoformat()
        errors, warnings = self._run(expired_as_warning=True)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("expirado", warnings[0])

    def test_overlap_work_budget_rejects_oversized_input(self):
        with self.assertRaisesRegex(ValueError, "orçamento.*globs"):
            overlaps("a" * 100001, "*")

    def test_overlap_work_budget_rejects_quadratic_product(self):
        with self.assertRaisesRegex(ValueError, "orçamento.*globs"):
            overlaps("*a" * 1000, "*b" * 1000)

    def test_overlap_work_budget_covers_unclosed_class_scanning(self):
        with self.assertRaisesRegex(ValueError, "orçamento.*globs"):
            overlaps("[" * 1000, "*")

    def test_validation_shares_budget_across_claim_pairs(self):
        self.claims["claims"] = [self._claim("a", ["a"]) for _ in range(20)]
        with patch.object(registry, "_REGISTRY_GLOB_WORK_LIMIT", 100):
            errors, warnings = self._run()
        self.assertEqual(warnings, [])
        self.assertTrue(any("orçamento" in error for error in errors), errors)

    def test_cli_rejects_exhausted_glob_budget(self):
        # Many distinct boundaries exceed the mask budget before compilation.
        huge = "".join(chr(0x400 + i) for i in range(1100))
        self.claims["claims"][0]["arquivos"] = [huge]
        self.claims["claims"][1]["arquivos"] = [huge]
        errors, _ = self._run()
        self.assertTrue(any("orçamento" in error for error in errors), errors)
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/agente/validar_registro.py"),
             "--root", str(self.tmp), "--agora", NOW.isoformat()],
            capture_output=True, text=True, check=False, timeout=20)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("orçamento", result.stderr)
        self.assertIn("FALHOU", result.stdout)

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
