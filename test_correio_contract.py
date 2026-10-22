import contextlib
import io
import os
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import yaml

from tools.agente import correio
from tools.agente.correio import PEDRO, main, renderizar_resumo, validar

NOW = datetime(2026, 10, 22, 15, 0, tzinfo=timezone.utc)
AGENTES = ("grok-bot", "axis")
REGISTRO = {
    "schema_version": 1,
    "agentes": [{"id": "grok-bot", "limites": ["sem merge sem aprovação"]},
                {"id": "axis", "limites": ["somente escopo local, sem push ou PR ou merge"]}],
}


@unittest.skipUnless(shutil.which("ssh-keygen") and shutil.which("git"), "requer git e ssh-keygen")
class CorreioBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        patcher = mock.patch.dict(os.environ, {
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org",
            "GIT_TERMINAL_PROMPT": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.keys: dict[str, Path] = {}
        for nome in (PEDRO, *AGENTES, "intruso"):
            chave = self.tmp / f"key-{nome}"
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(chave),
                            "-C", nome], check=True)
            self.keys[nome] = chave
        self.root = self.tmp / "main"
        (self.root / "docs/agentes").mkdir(parents=True)
        self.escrever_registro(REGISTRO)
        self.escrever_assinantes((PEDRO, *AGENTES))

    def escrever_registro(self, dados):
        (self.root / correio.REGISTRO).write_text(yaml.safe_dump(dados), encoding="utf-8")

    def escrever_assinantes(self, nomes):
        linhas = []
        for nome in nomes:
            pub = (self.keys[nome].with_suffix(".pub")).read_text().split()[:2]
            linhas.append(f'{nome} namespaces="git" {pub[0]} {pub[1]}')
        (self.root / correio.ASSINANTES).write_text("\n".join(linhas) + "\n", encoding="utf-8")

    # --- repositório de correio (órfão) ---
    def novo_repo(self, nome="mail") -> Path:
        repo = self.tmp / nome
        subprocess.run(["git", "init", "-q", "-b", correio.BRANCH, str(repo)], check=True)
        return repo

    def git(self, repo, *args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                              text=True).stdout

    def commitar(self, repo, agente, escrever=None, apagar=(), trailer="auto", assinar="auto",
                 msg="m", extra=()):
        for rel, texto in (escrever or {}).items():
            alvo = repo / rel
            alvo.parent.mkdir(parents=True, exist_ok=True)
            alvo.write_text(texto, encoding="utf-8")
        for rel in apagar:
            (repo / rel).unlink()
        self.git(repo, "add", "-A")
        cmd = ["commit", "-q", "--allow-empty-message", "-m", msg]
        trailer = agente if trailer == "auto" else trailer
        if trailer:
            cmd += ["--trailer", f"Agent-Id: {trailer}"]
        quem = agente if assinar == "auto" else assinar
        pre = []
        if quem:
            pre = ["-c", "gpg.format=ssh", "-c", f"user.signingkey={self.keys[quem]}"]
            cmd.append("-S")
        subprocess.run(["git", "-C", str(repo), *pre, *cmd, *extra], check=True, capture_output=True)
        return self.git(repo, "rev-parse", "HEAD").strip()

    def raiz(self, repo, **kw):
        return self.commitar(repo, PEDRO, {"LEIAME.md": "correio\n"}, **kw)

    @staticmethod
    def dados(i, de="grok-bot", para="axis", tipo="pergunta", resp=None, thread=None,
              evid=(), pp=False, assunto="olá", corpo="texto"):
        ident = f"20261022T1200{i:02d}Z-{i:08x}"
        return {"id": ident, "thread": thread or ident, "de": de, "para": para, "tipo": tipo,
                "em_resposta_a": resp, "assunto": assunto, "corpo": corpo,
                "evidencia": list(evid), "precisa_pedro": pp}

    def postar(self, repo, i, texto=None, trailer="auto", assinar="auto", **campos):
        d = self.dados(i, **campos)
        caminho = f"mensagens/2026-10-22/{d['id']}_{d['de']}_{d['para']}.yaml"
        conteudo = texto if texto is not None else yaml.safe_dump(d, sort_keys=False, allow_unicode=True)
        self.commitar(repo, d["de"], {caminho: conteudo}, trailer=trailer, assinar=assinar)
        return d, caminho

    def rodar(self, repo, ref="HEAD", **kw):
        kw.setdefault("verificar_sha", False)
        return validar(repo, ref, root=self.root, agora=NOW, **kw)

    def assertErro(self, res, trecho):
        self.assertTrue(any(trecho in e for e in res.erros), f"esperava erro com {trecho!r}: {res.erros}")


class HistoricoValido(CorreioBase):
    def test_conversa_completa_e_valida(self):
        repo = self.novo_repo()
        sha = self.raiz(repo)
        a, _ = self.postar(repo, 1, evid=[f"SHA:{sha[:7]}", "PR#70"], tipo="proposta")
        b, _ = self.postar(repo, 2, de="axis", para="grok-bot", resp=a["id"], thread=a["thread"])
        self.postar(repo, 3, tipo="confirmação", resp=b["id"], thread=a["thread"])
        res = self.rodar(repo, verificar_sha=True)
        self.assertEqual(res.erros, [])
        self.assertEqual(len(res.mensagens), 3)
        self.assertTrue(all(m.assinada for m in res.mensagens.values()))

    def test_branch_deve_ser_orfa(self):
        repo = self.novo_repo()
        self.git(repo, "branch", "-m", "main")
        self.commitar(repo, PEDRO, {"README.md": "x\n"})
        self.git(repo, "checkout", "-q", "-b", correio.BRANCH)
        self.commitar(repo, PEDRO, {"LEIAME.md": "x\n"})
        self.assertErro(self.rodar(repo, main_ref="main"), "órfã")


class AppendOnly(CorreioBase):
    def test_editar_mensagem_e_rejeitado(self):
        repo = self.novo_repo()
        self.raiz(repo)
        _, caminho = self.postar(repo, 1)
        self.commitar(repo, "grok-bot", {caminho: "corpo: reescrito\n"})
        self.assertErro(self.rodar(repo), "exatamente um arquivo")

    def test_apagar_mensagem_e_rejeitado(self):
        repo = self.novo_repo()
        self.raiz(repo)
        _, caminho = self.postar(repo, 1)
        self.commitar(repo, "grok-bot", apagar=[caminho])
        self.assertErro(self.rodar(repo), "exatamente um arquivo")

    def test_dois_arquivos_no_mesmo_commit(self):
        repo = self.novo_repo()
        self.raiz(repo)
        d1, d2 = self.dados(1), self.dados(2)
        arq = {f"mensagens/2026-10-22/{d['id']}_grok-bot_axis.yaml": yaml.safe_dump(d)
               for d in (d1, d2)}
        self.commitar(repo, "grok-bot", arq)
        self.assertErro(self.rodar(repo), "exatamente um arquivo")

    def test_arquivo_fora_do_padrao(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.commitar(repo, "grok-bot", {"docs/x.md": "oi\n"})
        self.assertErro(self.rodar(repo), "fora do padrão")

    def test_merge_commit_e_rejeitado(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1)
        self.git(repo, "checkout", "-q", "-b", "lado", "HEAD~1")
        self.postar(repo, 2)
        self.git(repo, "checkout", "-q", correio.BRANCH)
        self.git(repo, "merge", "-q", "--no-ff", "-m", "merge", "lado")
        self.assertErro(self.rodar(repo), "merge")

    def test_historico_reescrito_viola_ancora(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1)
        antigo = self.git(repo, "rev-parse", "HEAD").strip()
        (self.root / correio.ANCORA).write_text(yaml.safe_dump(
            {"schema_version": 1, "head_verificado": antigo, "aceitas_com_falha": []}))
        self.assertEqual(self.rodar(repo).erros, [])
        self.git(repo, "reset", "-q", "--hard", "HEAD~1")  # simula force-push que descarta a mensagem
        self.postar(repo, 2)
        self.assertErro(self.rodar(repo), "histórico reescrito")

    def test_ancora_pode_aceitar_falha_conhecida(self):
        repo = self.novo_repo()
        self.raiz(repo)
        _, caminho = self.postar(repo, 1)
        ruim = self.commitar(repo, "grok-bot", {caminho: "x: 1\n"})
        self.assertTrue(self.rodar(repo).erros)
        (self.root / correio.ANCORA).write_text(yaml.safe_dump(
            {"schema_version": 1, "head_verificado": None, "aceitas_com_falha": [ruim]}))
        res = self.rodar(repo)
        self.assertEqual(res.erros, [])
        self.assertTrue(any("aceita na âncora" in a for a in res.avisos))


class Identidade(CorreioBase):
    def test_remetente_fora_do_registro(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, de="intruso", para="axis")
        res = self.rodar(repo)
        self.assertErro(res, "não está no registro")
        self.assertFalse(res.mensagens)

    def test_trailer_diverge_do_remetente(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, de="axis", para="grok-bot", trailer="grok-bot", assinar="grok-bot")
        self.assertErro(self.rodar(repo), "diverge do remetente")

    def test_assinatura_de_outro_agente_e_forja(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, de="axis", para="grok-bot", assinar="grok-bot")
        self.assertErro(self.rodar(repo), "assinatura não verificada")

    def test_chave_desconhecida_e_rejeitada(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, assinar="intruso")
        self.assertErro(self.rodar(repo), "assinatura não verificada")

    def test_sem_assinatura_exigir_erro_aviso_so_avisa(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, assinar=None)
        self.assertErro(self.rodar(repo), "assinatura não verificada")
        res = self.rodar(repo, assinaturas="aviso")
        self.assertEqual(res.erros, [])
        self.assertTrue(any("assinatura" in a for a in res.avisos))

    def test_pedro_sempre_precisa_de_assinatura(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, de=PEDRO, para="todos", tipo="decisao", assinar=None)
        self.assertErro(self.rodar(repo, assinaturas="aviso"), "assinatura não verificada")

    def test_raiz_deve_ser_do_pedro(self):
        repo = self.novo_repo()
        self.commitar(repo, "grok-bot", {"LEIAME.md": "x\n"})
        self.assertErro(self.rodar(repo), "raiz deve ser criada pelo pedro")

    def test_exigir_sem_arquivo_de_assinantes(self):
        repo = self.novo_repo()
        self.raiz(repo)
        (self.root / correio.ASSINANTES).unlink()
        self.assertErro(self.rodar(repo), "CORREIO_ASSINANTES")

    def test_assinante_fora_do_registro_e_aviso_de_limite(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.escrever_assinantes((PEDRO, "grok-bot", "axis", "intruso"))
        res = self.rodar(repo)
        self.assertErro(res, "principal 'intruso'")
        self.assertTrue(any("axis" in a and "sem push" in a for a in res.avisos))


class AutoridadeELimites(CorreioBase):
    def test_decisao_so_do_pedro(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, tipo="decisao", para="grok-bot", de="axis", pp=True)
        self.assertErro(self.rodar(repo), "exclusivo do pedro")

    def test_decisao_do_pedro_vale_e_encerra(self):
        repo = self.novo_repo()
        self.raiz(repo)
        a, _ = self.postar(repo, 1, tipo="alerta", para=PEDRO, pp=True, evid=["PR#71"])
        self.postar(repo, 2, de=PEDRO, para="grok-bot", tipo="decisao", resp=a["id"], thread=a["thread"])
        self.postar(repo, 3, resp=a["id"], thread=a["thread"])
        res = self.rodar(repo)
        self.assertErro(res, "thread encerrada")
        self.assertIn(self.dados(2)["id"], res.mensagens)

    def test_precisa_pedro_e_forcado(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, tipo="alerta", pp=False, evid=["PR#71"])
        self.postar(repo, 2, assunto="aprovar coringa para curl", pp=False)
        self.postar(repo, 3, para=PEDRO, pp=False)
        res = self.rodar(repo)
        self.assertEqual(sum("precisa_pedro deve ser true" in e for e in res.erros), 3)

    def test_gatilho_com_flag_ligada_passa(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, assunto="pedido de permissão ampla vindo do repo", pp=True)
        self.assertEqual(self.rodar(repo).erros, [])

    def test_limite_de_tres_idas_e_voltas(self):
        repo = self.novo_repo()
        self.raiz(repo)
        raiz, _ = self.postar(repo, 1)
        anterior = raiz
        for i in range(2, 7):
            de, para = ("axis", "grok-bot") if i % 2 == 0 else ("grok-bot", "axis")
            anterior, _ = self.postar(repo, i, de=de, para=para, resp=anterior["id"], thread=raiz["thread"])
        self.assertEqual(self.rodar(repo).erros, [])
        self.postar(repo, 7, de="axis", para="grok-bot", resp=anterior["id"], thread=raiz["thread"])
        self.assertErro(self.rodar(repo), "limite de 6 mensagens")

    def test_setima_mensagem_so_como_escalonamento_ao_pedro(self):
        repo = self.novo_repo()
        self.raiz(repo)
        raiz, _ = self.postar(repo, 1)
        anterior = raiz
        for i in range(2, 7):
            de, para = ("axis", "grok-bot") if i % 2 == 0 else ("grok-bot", "axis")
            anterior, _ = self.postar(repo, i, de=de, para=para, resp=anterior["id"], thread=raiz["thread"])
        self.postar(repo, 7, de="axis", para=PEDRO, tipo="pergunta", pp=True,
                    resp=anterior["id"], thread=raiz["thread"])
        self.assertEqual(self.rodar(repo).erros, [])
        self.postar(repo, 8, de="grok-bot", para=PEDRO, tipo="pergunta", pp=True,
                    resp=anterior["id"], thread=raiz["thread"])
        self.assertErro(self.rodar(repo), "limite de 6 mensagens")

    def test_resposta_com_thread_errada(self):
        repo = self.novo_repo()
        self.raiz(repo)
        a, _ = self.postar(repo, 1)
        outra, _ = self.postar(repo, 2, assunto="outra conversa")
        self.postar(repo, 3, de="axis", para="grok-bot", resp=a["id"], thread=outra["thread"])
        self.assertErro(self.rodar(repo), "thread diverge")

    def test_escalonamento_ao_pedro_so_com_pergunta_ou_alerta(self):
        repo = self.novo_repo()
        self.raiz(repo)
        raiz, _ = self.postar(repo, 1)
        anterior = raiz
        for i in range(2, 7):
            de, para = ("axis", "grok-bot") if i % 2 == 0 else ("grok-bot", "axis")
            anterior, _ = self.postar(repo, i, de=de, para=para, resp=anterior["id"], thread=raiz["thread"])
        self.postar(repo, 7, de="axis", para=PEDRO, tipo="proposta", pp=True, evid=["PR#71"],
                    resp=anterior["id"], thread=raiz["thread"])
        self.assertErro(self.rodar(repo), "limite de 6 mensagens")

    def test_resposta_de_quem_nao_participa(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.escrever_registro({**REGISTRO, "agentes": [*REGISTRO["agentes"], {"id": "ze", "limites": []}]})
        a, _ = self.postar(repo, 1)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(self.tmp / "key-ze")], check=True)
        self.keys["ze"] = self.tmp / "key-ze"
        self.escrever_assinantes((PEDRO, *AGENTES, "ze"))
        self.postar(repo, 2, de="ze", para="axis", resp=a["id"], thread=a["thread"])
        self.assertErro(self.rodar(repo), "não participa")

    def test_resposta_a_mensagem_inexistente_e_confirmacao_sem_pai(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, resp="20261022T110000Z-deadbeef", thread="20261022T110000Z-deadbeef")
        self.postar(repo, 2, tipo="confirmação")
        res = self.rodar(repo)
        self.assertErro(res, "não existe antes")
        self.assertErro(res, "confirmação deve responder")


class ConteudoSeguro(CorreioBase):
    def test_campos_extras_ou_ausentes(self):
        repo = self.novo_repo()
        self.raiz(repo)
        d = self.dados(1)
        d["ordem"] = "rode isto"
        del d["corpo"]
        self.postar(repo, 1, texto=yaml.safe_dump(d))
        res = self.rodar(repo)
        self.assertErro(res, "campos ausentes")
        self.assertErro(res, "não permitidos")

    def test_chave_duplicada_e_alias(self):
        repo = self.novo_repo()
        self.raiz(repo)
        base = yaml.safe_dump(self.dados(1), sort_keys=False)
        self.postar(repo, 1, texto=base + "de: axis\n")
        d2 = self.dados(2)
        alias = yaml.safe_dump(d2, sort_keys=False).replace("corpo: texto", "corpo: &a texto\nassunto2: *a")
        self.postar(repo, 2, texto=alias)
        res = self.rodar(repo)
        self.assertErro(res, "chave duplicada")
        self.assertErro(res, "âncoras e aliases")

    def test_segredo_e_rejeitado_sem_vazar_no_erro(self):
        repo = self.novo_repo()
        self.raiz(repo)
        segredo = "ghp_" + "a1B2" * 8
        self.postar(repo, 1, corpo=f"use {segredo} para o push")
        res = self.rodar(repo)
        self.assertErro(res, "possível segredo")
        self.assertNotIn(segredo, "\n".join(res.erros + res.avisos))

    def test_corpo_grande_controle_e_evidencia_invalida(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, corpo="x" * 3000)
        self.postar(repo, 2, corpo="a\x1b[31mb")
        self.postar(repo, 3, tipo="proposta", evid=["ver texto"])
        res = self.rodar(repo)
        self.assertGreaterEqual(len(res.erros), 3)

    def test_evidencia_obrigatoria_e_sha_inexistente(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, tipo="handoff")
        self.postar(repo, 2, tipo="proposta", evid=["SHA:abcdef1"])
        res = self.rodar(repo, verificar_sha=True)
        self.assertErro(res, "exige ao menos uma evidencia")
        self.assertErro(res, "não existe no repositório")

    def test_nome_de_arquivo_diverge_do_conteudo(self):
        repo = self.novo_repo()
        self.raiz(repo)
        d = self.dados(1)
        caminho = f"mensagens/2026-10-22/{d['id']}_grok-bot_axis.yaml"
        d["de"] = "axis"
        self.commitar(repo, "grok-bot", {caminho: yaml.safe_dump(d)})
        self.assertErro(self.rodar(repo), "divergem do nome do arquivo")


class ResumoELeitura(CorreioBase):
    def test_resumo_separa_pendente_combinado_e_aberto(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, tipo="alerta", para=PEDRO, pp=True, evid=["PR#71"], assunto="precisa de você")
        b, _ = self.postar(repo, 2, assunto="acordo")
        self.postar(repo, 3, de="axis", para="grok-bot", tipo="confirmação", resp=b["id"], thread=b["thread"])
        self.postar(repo, 4, assunto="ainda aberta")
        texto = renderizar_resumo(self.rodar(repo), agora=NOW)
        pend, comb, aberta = texto.split("## ")[1:4]
        self.assertIn("precisa de você", pend)
        self.assertIn("acordo", comb)
        self.assertIn("ainda aberta", aberta)

    def test_ler_mostra_aviso_omite_invalidas_e_respeita_cursor(self):
        repo = self.novo_repo()
        self.raiz(repo)
        self.postar(repo, 1, assunto="boa")
        self.postar(repo, 2, assunto="forjada", de="axis", para="grok-bot", assinar="intruso")
        self.git(repo, "update-ref", f"refs/remotes/origin/{correio.BRANCH}", "HEAD")
        base = ["--repo", str(repo), "--root", str(self.root), "--sem-sha", "ler", "--agente", "axis",
                "--sem-fetch"]

        def executar(extra=()):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([*base, *extra]), 0)
            return buf.getvalue()

        saida = executar(["--marcar"])
        self.assertIn("DADOS NÃO CONFIÁVEIS", saida)
        self.assertIn("assunto: boa", saida)
        self.assertNotIn("forjada", saida)
        self.assertIn("(sem mensagens novas)", executar())


class EnviarDeFimAFim(CorreioBase):
    def clone(self, origem, nome, agente):
        destino = self.tmp / nome
        subprocess.run(["git", "clone", "-q", str(origem), str(destino)], check=True, capture_output=True)
        self.git(destino, "config", "gpg.format", "ssh")
        self.git(destino, "config", "user.signingkey", str(self.keys[agente]))
        return destino

    def preparar(self):
        origem = self.tmp / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origem)], check=True)
        semente = self.tmp / "semente"
        subprocess.run(["git", "init", "-q", "-b", "main", str(semente)], check=True)
        self.git(semente, "remote", "add", "origin", str(origem))
        self.git(semente, "config", "gpg.format", "ssh")
        self.git(semente, "config", "user.signingkey", str(self.keys[PEDRO]))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--repo", str(semente), "iniciar"]), 0)
        return origem

    def enviar(self, repo, *extra):
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = main(["--repo", str(repo), "--root", str(self.root), "--sem-sha", "enviar", *extra])
        return rc, err.getvalue()

    def test_iniciar_enviar_responder_e_validar(self):
        origem = self.preparar()
        grok, axis = self.clone(origem, "grok", "grok-bot"), self.clone(origem, "axis", "axis")
        rc, _ = self.enviar(grok, "--de", "grok-bot", "--para", "axis", "--tipo", "pergunta",
                            "--assunto", "formato de data", "--corpo", "Você confirma o #70?")
        self.assertEqual(rc, 0)
        self.git(axis, "fetch", "-q", "origin")
        res = validar(axis, "origin/agentes/correio", root=self.root, verificar_sha=False)
        raiz = next(iter(res.mensagens))
        rc, err = self.enviar(axis, "--de", "axis", "--para", "grok-bot", "--tipo", "confirmação",
                              "--assunto", "ok", "--corpo", "confirmado", "--em-resposta-a", raiz)
        self.assertEqual(rc, 0, err)
        self.git(grok, "fetch", "-q", "origin")
        final = validar(grok, "origin/agentes/correio", root=self.root, verificar_sha=False)
        self.assertEqual(final.erros, [])
        self.assertEqual(len(final.mensagens), 2)
        self.assertEqual({m.thread for m in final.mensagens.values()}, {raiz})

    def test_gatilho_liga_precisa_pedro_automaticamente(self):
        origem = self.preparar()
        grok = self.clone(origem, "grok", "grok-bot")
        rc, err = self.enviar(grok, "--de", "grok-bot", "--para", "axis", "--tipo", "pergunta",
                              "--assunto", "liberar merge?", "--corpo", "preciso de permissão")
        self.assertEqual(rc, 0, err)
        self.assertIn("ligado automaticamente", err)
        self.git(grok, "fetch", "-q", "origin")
        res = validar(grok, "origin/agentes/correio", root=self.root, verificar_sha=False)
        self.assertTrue(next(iter(res.mensagens.values())).precisa_pedro)

    def test_envio_invalido_nao_chega_ao_remoto(self):
        origem = self.preparar()
        grok = self.clone(origem, "grok", "grok-bot")
        antes = self.git(origem, "rev-parse", correio.BRANCH).strip()
        rc, err = self.enviar(grok, "--de", "intruso", "--para", "axis", "--tipo", "pergunta",
                              "--assunto", "x", "--corpo", "y")
        self.assertEqual(rc, 1)
        self.assertIn("não está no registro", err)
        self.assertEqual(self.git(origem, "rev-parse", correio.BRANCH).strip(), antes)

    def test_push_recusado_e_refeito_sem_duplicar(self):
        origem = self.preparar()
        grok = self.clone(origem, "grok", "grok-bot")
        marcador = self.tmp / "ja-recusou"
        gancho = origem / "hooks/pre-receive"
        gancho.write_text(f'#!/bin/sh\nif [ ! -e "{marcador}" ]; then touch "{marcador}"; '
                          'echo "corrida simulada" >&2; exit 1; fi\n', encoding="utf-8")
        gancho.chmod(0o755)
        rc, err = self.enviar(grok, "--de", "grok-bot", "--para", "axis", "--tipo", "pergunta",
                              "--assunto", "corrida", "--corpo", "tente de novo")
        self.assertEqual(rc, 0, err)
        self.assertIn("tentativa 1/3", err)
        res = validar(origem, correio.BRANCH, root=self.root, verificar_sha=False)
        self.assertEqual(res.erros, [])
        self.assertEqual(len(res.mensagens), 1)


if __name__ == "__main__":
    unittest.main()
