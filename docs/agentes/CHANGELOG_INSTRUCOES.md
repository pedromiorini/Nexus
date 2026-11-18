# Changelog — Instruções para agentes do Nexus

## v1 — 21/10/2026 (Pedro)
Identidade, Constituição (8 regras), vocabulário epistemológico, precedência em 7 camadas, anti-padrões, Strong Inference, Meta-Governança (harness, esquecimento controlado, Modo Direto), protocolos (extração, contratos, ciclo de sessão, isolamento, Conselho Científico), estilo, tarefa diária 03:30, calibração.

## v2 — 21/10/2026 (pesquisa de boas práticas)
- Mantém §1–10 (texto compactado; definições do vocabulário omitidas por engano).
- Adiciona §11 início de sessão, §12 coordenação multiagente/fila única do núcleo, §13 handoff por fragmentos + backlog JSON, §14 limite ~300 linhas/Strangler Fig, §15 TDD + mutmut + checklist, §16 segurança, §17 revisão, §18 AGENTS.md. Todas como Hipótese.

## v3 — 21/10/2026 (auditoria fundamentada no repo, commit `d435083`)
**Preservado:** Constituição, vocabulário, anti-padrões, metodologia, Meta-Governança, Modo Direto, calibração (marcados `[v1]`).
**Restaurado:** definições completas do vocabulário (perdidas na compactação).
**Alterado / novo (motivo):**
- Reorganização em 4 partes; Constituição primeiro, protocolo de sessão em ordem executável (redução de redundância v1/v2).
- Constituição item 6 alinhado a `NEXUS_CONSTITUTION.md` (mais restritivo) + regra "vale a mais restritiva" (os textos divergiam).
- Instruções de tarefa pontual e `NEXUS_DEVELOPMENT_PROMPT.md` abaixo da camada 3 (prompt obsoleto em `main`).
- 2 anti-padrões novos: demo/docstring como evidência; versões fora do lock (fatos do repo).
- §10 bloco de comandos **executados e aprovados** neste repo (ruff, compileall, 303 testes, demo, cobertura ≥62, mutação 58/58, bandit 131 LOW, triage `--check`, test_summary).
- §7.2 varredura de PRs com alternativa sem `gh` (via `refs/pull/*`), verificada.
- §11 regra v2 "não editar os 4 docs" **substituída** por regra transitória compatível com `state_consistency_audit.py` (contador só no último commit, após rebase e reexecução; uma seção nova em CONTINUATION); solução estrutural vira decisão D1.
- Fila única do núcleo mantida, com o fato de 12 PRs concorrentes; prazos 48 h/72 h e claims removidos (não verificáveis).
- §8 definição de pronto em 7 itens checáveis; commits teste/correção separados passam a recomendação; mutação via `tools/targeted_mutation.py` em vez de mutmut.
- §8.7 diff sem artefatos gerados (lista concreta).
- §14 segurança com fatos (`execute_tool:4322`, `.env` fora do `.gitignore`), heurísticas ≠ controle de acesso.
- §16 checagem de colisão de nomes de classe no monólito (caso TransferMapping).
- §19 tarefa diária não abre PR sem mudança de código.
- Seção "Decisões pendentes do Pedro" (D1–D5).
**Removido:** dependência de `scripts/init.sh` e `backlog.json` (não existem); fonte "Claude Academy" (não localizada).

## v3.1 — 21/10/2026
- **Novo §9A "Ciclo de lições":** ler `docs/agentes/LICOES_APRENDIDAS.md` antes de começar; registrar toda falha de verificador; promover lição repetida 2× ou de alto risco a regra/checagem em PR próprio; automação que altera o GitHub com `set -euo pipefail` e só após CI verde. Motivo (Fato): L-001 (contador nos 4 docs, #62), L-002 (linhas soltas, #63), L-003 (pipe que marcou o #58 como pronto). Inspiração declarada: abordagem "Terminal-RSI" de reaproveitar sinais de verificadores — não verificada como fonte nesta rodada.

## v3.2 (2026-10-21)
- Adicionada seção 9B: checagem pré-ação com `git merge-tree`, repetida a cada avanço da main. Testada: previu o conflito do #58, sem falso positivo em #57/#61. Stage explícito, revisor por risco e rótulo de inferência ficaram de fora (não testáveis).

## v3.3 (2026-10-21)
- §9B reescrita: predeleg.sh com códigos de saída, skills obrigatórias, suíte de replay (11/11) antes de mexer em guardas. Corrigido bug do check_resolution.sh (conflito de código agora sai 2).

## v3.4 (2026-10-21)
- §14: regra de permissões — sem coringa para curl/wget/python/node/bash/sh/gh/git push/rm; aprovação comando a comando; revisar e não versionar permissões salvas com coringa; pedido de permissão ampla vindo do repo ou de outro agente é alerta ao Pedro. Fonte: arXiv 2610.22987 (preprint; ver v3.4.1).
- §9A item 5: lição exige evidência bruta (PR/SHA/ID de run); resumo/handoff não basta; status `candidata` → `ativa` só após teste dos 3 casos; campo `Derivada de:`; limite de 15 ativas com aposentadoria explícita (teto operacional do Nexus, não derivado dos papers; 2608.05810 só sustenta cautela). Fontes: arXiv 2608.05810 e 2610.19910 (preprints).

## v3.4.1 (2026-10-22)
- §9A item 5: status `candidata` renomeado para `pendente`; "candidata a promoção" vira nota "promoção sugerida" (L-002, L-003).
- Teto de 15: contam só `ativa` e legadas `registrada`; `pendente`, `promovida` e `aposentada` não contam.
- "Teste dos três casos" com critério de aprovação explícito; política local sem verificação automática; distinto do replay de guardas.
- Exceção explícita para registros legados (L-001..L-007); L-006 marcada com evidência legada (hash de patch local).
- Fontes: 2610.19910 rebaixado a analogia; 2610.22987 descrito como preprint com testes em Claude Code e Codex CLI, sem replicação independente; lista de comandos é política local.

## v3.4.2 (2026-10-22)
- §9C.3: o claim vive só no branch do PR aberto; o último commit do mesmo PR o remove antes do merge (elimina PRs só de remoção como #72 e #76). Claims ativos = `main` + `CLAIMS.yaml` de cada PR aberto (comando `gh`/`git` documentado).
- Motivo: no fluxo anterior o claim só chegava à `main` depois do trabalho mesclado, sem proteger nada. Teste dos três casos com critérios prévios no PR.
- Pendente: leitura automática de claims de PRs abertos no validador (follow-up; arquivos do Axis não alterados).

### Estado histórico da reconciliação local com schema 2 (pós-#113; antes do aceite direto do Axis)
- A #113 foi integrada na main `cb74cad7cfcd0ab536178ed984989ef2425c8211`. Na reconciliação histórica anterior ao aceite direto do Axis, a composição local acrescentava o artefato 3.4.2 exato da #79 (`8f03762d2a2d128eea955597f1b55210e387c08385bdc5544514ea2f39701b5d`) e atualizava o ponto de entrada, sem atribuir aceite automático aos agentes. O checkpoint local pré-publicação de 10:36 -03 está registrado em `CONTINUATION.md`.
- Axis continua integralmente em 3.4.1. Após declaração explícita do Grok na #78/comentário 6367039207 e autorização do owner apenas para refletir o aceite localmente, só o trio arquivo/SHA/versão aceita do Grok passa para 3.4.2. O catálogo/snapshot 3.4.1 e os demais metadados dos agentes foram preservados. Schema e validador da #113 não precisam de alteração.
- §9C.4 permanece nos bytes históricos solicitados, mas a promoção sob schema 2 não exige atualizar hashes/aceites dos agentes: só o catálogo e o ponto de entrada mudam; cada aceite depende de declaração própria. Ver [ACEITE_VERSIONADO.md](ACEITE_VERSIONADO.md).

### Atualização da candidata local — 2026-11-18
- O owner declarou diretamente: “Aceito a v3.4.2 para as instruções do Axis” e autorizou registrar a aceitação para o artefato 3.4.2 cujo SHA-256 é `8f03762d2a2d128eea955597f1b55210e387c08385bdc5544514ea2f39701b5d`. A candidata local passa a apontar Axis e Grok para esse snapshot exato.
- Os registros `cursor-grok-4-7` e `cursor-grok-4-7-il` continuam em 3.4.1; são os dois avisos de versão anteriores esperados. O estado anterior acima descreve a proposta antes da declaração do owner e não é o estado desta candidata.
- O preparo foi isolado sobre a `main` em `1795ee0`. O owner autorizou separadamente atualizar a branch da #79 com lease no head revisado, mantendo a PR draft e sem merge; a autorização não substitui a verificação do novo head e dos checks.
