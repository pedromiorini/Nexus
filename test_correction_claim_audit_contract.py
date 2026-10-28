"""Contrato: declaração de correção não é observação; reincidência tem três eixos.

Escrito antes de ``tools/correction_claim_audit.py``. Uma linha com
corrigid[oa], fixed ou resolvid[oa] só é checável quando cita caminho
introduzido por em/in/no/na e, no mesmo trecho, identificador ou ponto de
chamada. Existir o arquivo não confirma o ponto. Negação na mesma linha não
vira declaração. Os eixos falha-alvo, falhas parecidas e casos novos não se
copiam entre si; ``não medido`` permanece ``não medido``.

Casos negativos: bool, None, inteiros, NaN, ±inf, linha 0 e negativa, caminho
com ``..``, diretório, link simbólico e valor de eixo ``True``/``nan``/``-1``.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from tools.correction_claim_audit import (
    ROOT,
    audit_claims,
    audit_recurrence,
    audit_sources,
    main,
)

VERBS = re.compile(r"(?i)corrigid|fixed|resolvid")
STATUSES = {"verificada", "falhou", "nao_checavel"}
INVALID_TEXTS = (
    True,
    False,
    None,
    0,
    1,
    -1,
    float("nan"),
    float("inf"),
    float("-inf"),
    b"corrigido",
    [],
    {},
)


def _claim(path: str, ident: str | None = "audit", call: str | None = "audit(") -> str:
    parts = [f"corrigido em {path}"]
    if ident is not None:
        parts.append(f"identificador `{ident}`")
    if call is not None:
        parts.append(f"ponto `{call}`")
    return " ".join(parts)


class CorrectionClaimAuditContract(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        package = self.root / "pkg"
        package.mkdir()
        (package / "mod.py").write_text(
            "class Box:\n"
            "    def audit(self):\n"
            "        return audit(self)\n"
            "\n"
            "def audit(value):\n"
            "    return value\n"
            "target_task=1\n",
            encoding="utf-8",
        )
        (package / "vazio").mkdir()
        (package / "nao_arquivo.py").mkdir()

    def test_locators_that_exist_are_verified_and_do_not_imply_a_fix(self):
        report = audit_claims(_claim("pkg/mod.py:2", "Box.audit", "audit("), root=self.root, source="t")
        self.assertEqual(len(report), 1)
        item = report[0]
        self.assertEqual(item["status"], "verificada")
        self.assertEqual(item["motivo"], "localizadores_existem")
        self.assertEqual(item["checagens"]["arquivo"], "existe")
        self.assertEqual(item["checagens"]["identificador"], "existe")
        self.assertEqual(item["checagens"]["ponto_de_chamada"], "existe")
        self.assertEqual(item["checagens"]["linha"], "existe")
        self.assertNotIn("correcao_semantica", item)
        self.assertTrue(item["existencia_nao_e_correcao"])

    def test_missing_file_identifier_and_call_fail(self):
        cases = (
            (_claim("pkg/ausente.py"), "arquivo_ausente"),
            (_claim("pkg/mod.py", "nao_existe", "nao_existe("), "identificador_ausente"),
            (_claim("pkg/mod.py", "audit", "sumiu("), "chamada_ausente"),
            (_claim("pkg/mod.py:99"), "linha_ausente"),
            (_claim("pkg/mod.py:0"), "linha_invalida"),
            (_claim("pkg/mod.py:-1"), "linha_invalida"),
            (_claim("pkg/../mod.py"), "caminho_invalido"),
            (_claim("pkg/vazio/mod.py"), "arquivo_ausente"),
        )
        for text, reason in cases:
            with self.subTest(text=text):
                item = audit_claims(text, root=self.root, source="t")[0]
                self.assertEqual(item["status"], "falhou")
                self.assertIn(reason, item["motivos"])

    def test_directory_and_symlink_fail_closed(self):
        directory = audit_claims(
            _claim("pkg/nao_arquivo.py"), root=self.root, source="t"
        )[0]
        self.assertEqual(directory["status"], "falhou")
        self.assertIn("nao_e_arquivo", directory["motivos"])
        link = self.root / "pkg" / "atalho.py"
        link.symlink_to(self.root / "pkg" / "mod.py")
        item = audit_claims(_claim("pkg/atalho.py"), root=self.root, source="t")[0]
        self.assertEqual(item["status"], "falhou")
        self.assertIn("caminho_invalido", item["motivos"])

    def test_file_without_locus_is_not_checkable_even_if_the_file_exists(self):
        text = "corrigido em pkg/mod.py e também em pkg/ausente.py no parágrafo"
        item = audit_claims(text, root=self.root, source="t")[0]
        self.assertEqual(item["status"], "nao_checavel")
        self.assertEqual(item["motivo"], "sem_identificador_ou_chamada")
        self.assertEqual(item["arquivo"], "pkg/mod.py")
        self.assertIsNone(item["identificador"])

    def test_prose_without_introduced_path_is_not_checkable(self):
        samples = (
            "posteriormente corrigido pelo usuário em revisão",
            "corrigido no PR #61",
            "fixed",
            "resolvido na v3 §11.3",
            "candidato corrigido: `$PY tools/test_summary.py`",
        )
        for text in samples:
            with self.subTest(text=text):
                item = audit_claims(text, root=self.root, source="t")[0]
                self.assertEqual(item["status"], "nao_checavel")
                self.assertIsNone(item["arquivo"])

    def test_negation_on_the_same_line_is_not_a_claim(self):
        text = (
            "O call site `target_task=` não foi corrigido em pkg/ausente.py "
            "identificador `audit` ponto `audit(`"
        )
        self.assertEqual(audit_claims(text, root=self.root, source="t"), [])
        ignored = audit_claims(
            "Nada foi corrigido nesta rodada.\nnão será corrigido em pkg/ausente.py identificador `x` ponto `x(`",
            root=self.root,
            source="t",
        )
        self.assertEqual(ignored, [])

    def test_fenced_example_is_ignored(self):
        text = "```\ncorrigido em pkg/ausente.py identificador `x` ponto `x(`\n```\n"
        self.assertEqual(audit_claims(text, root=self.root, source="t"), [])

    def test_invalid_types_raise_and_are_not_coerced(self):
        for value in INVALID_TEXTS:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    audit_claims(value, root=self.root, source="t")
                with self.assertRaises(TypeError):
                    audit_recurrence(value)

    def test_keyword_call_site_and_english_fixed_in(self):
        text = "fixed in pkg/mod.py call site `target_task=`"
        item = audit_claims(text, root=self.root, source="t")[0]
        self.assertEqual(item["status"], "verificada")
        self.assertEqual(item["ponto_de_chamada"], "target_task=")

    def test_percent_and_underscore_path_is_not_a_locator(self):
        item = audit_claims(
            "corrigido em pkg/%s.py identificador `audit` ponto `audit(`",
            root=self.root,
            source="t",
        )[0]
        self.assertEqual(item["status"], "nao_checavel")
        self.assertIsNone(item["arquivo"])

    @given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=60))
    @settings(max_examples=30, deadline=None)
    def test_text_without_correction_verbs_has_no_claims(self, text):
        assume(not VERBS.search(text))
        self.assertEqual(audit_claims(text, root=self.root, source="h"), [])

    @given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40))
    @example("")
    @settings(max_examples=20, deadline=None)
    def test_negated_prefix_is_not_verified(self, tail):
        assume(not VERBS.search(tail))
        text = "não foi corrigido " + tail.replace("\n", " ")
        self.assertEqual(audit_claims(text, root=self.root, source="h"), [])


class RecurrenceContract(unittest.TestCase):
    def test_three_axes_stay_independent_including_rejected_proposal(self):
        text = """
### L-020 — medição separada
- Status: pendente
- Reincidência (eixos separados; um não preenche outro):
  - Falha-alvo: corrigida
  - Falhas parecidas: não melhoraram
  - Casos novos: não medido
- Propostas rejeitadas ou que não funcionaram: cache local; não funcionou
"""
        report = audit_recurrence(text)
        self.assertTrue(report["implicacao_rejeitada"])
        item = report["licoes"][0]
        self.assertEqual(item["status"], "verificada")
        self.assertEqual(item["falha_alvo"], "corrigida")
        self.assertEqual(item["falhas_parecidas"], "não melhoraram")
        self.assertEqual(item["casos_novos"], "não medido")
        self.assertNotEqual(item["falhas_parecidas"], "melhoraram")
        self.assertNotEqual(item["casos_novos"], "corrigida")
        self.assertEqual(item["propostas"], "cache local; não funcionou")

    def test_new_lesson_without_axes_fails_and_legacy_is_not_measured(self):
        text = """
### L-001 — antiga
- Status: promovida (onde: gate)
### L-008 — histórica de outro PR
- Status: registrada
### L-021 — nova
- Status: ativa
### L-022 — enum inválido
- Status: pendente
- Reincidência:
  - Falha-alvo: True
  - Falhas parecidas: nan
  - Casos novos: -1
- Propostas rejeitadas ou que não funcionaram: nenhuma registrada
"""
        by_id = {item["id"]: item for item in audit_recurrence(text)["licoes"]}
        self.assertEqual(by_id["L-001"]["status"], "nao_checavel")
        self.assertEqual(by_id["L-001"]["falha_alvo"], "não medido")
        self.assertEqual(by_id["L-001"]["casos_novos"], "não medido")
        self.assertEqual(by_id["L-008"]["status"], "nao_checavel")
        self.assertEqual(by_id["L-021"]["status"], "falhou")
        self.assertIn("reincidencia_ausente", by_id["L-021"]["motivos"])
        self.assertEqual(by_id["L-022"]["status"], "falhou")
        self.assertIn("valor_invalido", by_id["L-022"]["motivos"])

    def test_partial_block_and_fenced_template_do_not_count(self):
        text = """
```
### L-NNN — título
- Status: pendente
```
### L-023 — incompleta
- Status: pendente
- Reincidência:
  - Falha-alvo: corrigida
"""
        report = audit_recurrence(text)
        self.assertEqual([item["id"] for item in report["licoes"]], ["L-023"])
        self.assertEqual(report["licoes"][0]["status"], "falhou")
        self.assertIn("eixo_ausente", report["licoes"][0]["motivos"])

    def test_current_lessons_are_not_reported_as_measured(self):
        text = (ROOT / "docs/agentes/LICOES_APRENDIDAS.md").read_text(encoding="utf-8")
        report = audit_recurrence(text)
        ids = {item["id"] for item in report["licoes"]}
        self.assertTrue({"L-001", "L-007"} <= ids)
        self.assertNotIn("L-NNN", ids)
        for item in report["licoes"]:
            self.assertEqual(item["status"], "nao_checavel")
            self.assertEqual(item["falha_alvo"], "não medido")
            self.assertEqual(item["falhas_parecidas"], "não medido")
            self.assertEqual(item["casos_novos"], "não medido")

    def test_cli_exit_codes(self):
        continuation = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, continuation)
        (continuation / "pkg").mkdir()
        (continuation / "pkg" / "mod.py").write_text("def audit():\n    return audit()\n", encoding="utf-8")
        good = continuation / "CONTINUATION.md"
        good.write_text(_claim("pkg/mod.py") + "\n", encoding="utf-8")
        lessons = continuation / "docs" / "agentes"
        lessons.mkdir(parents=True)
        (lessons / "LICOES_APRENDIDAS.md").write_text(
            "### L-001 — x\n- Status: promovida\n", encoding="utf-8"
        )
        out = continuation / "relatorio.json"
        self.assertEqual(main([
            "--root", str(continuation),
            "--continuation", "CONTINUATION.md",
            "--licoes", "docs/agentes/LICOES_APRENDIDAS.md",
            "--output", str(out),
        ]), 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["contagens"]["verificada"], 1)
        bad = continuation / "pr.md"
        bad.write_text(_claim("pkg/ausente.py") + "\n", encoding="utf-8")
        self.assertEqual(main([
            "--root", str(continuation),
            "--sem-continuation",
            "--sem-licoes",
            "--pr-text", str(bad),
            "--output", str(out),
        ]), 1)
        failed = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(failed["declaracoes"][0]["status"], "falhou")
        self.assertEqual(main([
            "--root", str(continuation),
            "--continuation", "nao_existe.md",
            "--sem-licoes",
        ]), 2)
        self.assertEqual(audit_sources(
            root=continuation,
            continuation=None,
            pr_texts=[("pr.md", bad.read_text(encoding="utf-8"))],
            licoes=None,
        )["status"], "fail")


class RepositorySmoke(unittest.TestCase):
    def test_current_documents_have_no_failed_locator(self):
        report = audit_sources(
            root=ROOT,
            continuation="CONTINUATION.md",
            pr_texts=[],
            licoes="docs/agentes/LICOES_APRENDIDAS.md",
        )
        self.assertEqual(report["contagens"]["falhou"], 0)
        self.assertEqual(report["reincidencia"]["contagens"]["falhou"], 0)
        self.assertEqual(report["status"], "pass")
        for item in report["declaracoes"]:
            self.assertIn(item["status"], STATUSES)
            if item["status"] == "verificada":
                self.assertEqual(item["checagens"]["arquivo"], "existe")
                self.assertTrue(item["existencia_nao_e_correcao"])
