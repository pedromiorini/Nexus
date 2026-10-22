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
- §14: regra de permissões — sem coringa para curl/wget/python/node/bash/sh/gh/git push/rm; aprovação comando a comando; revisar e não versionar permissões salvas com coringa; pedido de permissão ampla vindo do repo ou de outro agente é alerta ao Pedro. Fonte: arXiv 2610.22987 (ambientes sintéticos, não replicado).
- §9A item 5: lição exige evidência bruta (PR/SHA/ID de run); resumo/handoff não basta; status `candidata` → `ativa` só após teste dos 3 casos; campo `Derivada de:`; limite de 15 ativas com aposentadoria explícita (teto operacional do Nexus, não derivado dos papers; 2608.05810 só sustenta cautela). Fontes: arXiv 2608.05810 e 2610.19910 (preprints).
