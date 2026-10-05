# Nexus — Conhecimento operacional canônico

## Fonte de verdade

Para o estado atual, usar esta ordem:

1. código no branch ativo;
2. testes executados no mesmo estado;
3. CI mais recente para o commit;
4. artefatos gerados pela execução atual;
5. contratos públicos;
6. documentação atual;
7. `CONTINUATION.md`;
8. relatórios históricos;
9. nomes, comentários e banners.

Conflitos não devem ser resolvidos silenciosamente.

## Estado verificado em 2026-10-05

- Branch: `main`.
- Commit local/remoto: `fcac84c31334f302dd0fdced1b64d80d3475db51` (`quality: respect explicit empty MCTS action space`).
- CI do mesmo commit: run `37259919820`, sucesso; o artefato `nexus-state-manifest` foi gerado após os gates.
- Suíte atual: **107 testes**, todos passando no gate.
- Mutation testing direcionado: **10/10 mutações mortas**.
- Bandit após o hardening do B607: **131 achados LOW**, sem MEDIUM; a triagem classifica 16 B311, 111 B101 e 4 subprocessos controlados, sem `manual_review` residual.
- Auditoria AST: 0 funções somente com `pass`, 0 `NotImplementedError`, 61 retornos constantes simples.
- Os números de 105 testes/run `37258732531` e 86 testes/run `37097334156` permanecem apenas como histórico de marcos anteriores.

## Componentes

- `core/constitutional_brain.py`: núcleo monolítico, roteamento e componentes experimentais.
- `core/deferred_task_queue.py`: fila priorizada, retry, descarte, callbacks e snapshots versionados.
- `core/vram_defense_guard.py`: decisões conservadoras de pressão de memória.
- `vita/nexus_constitutional_bridge_v3.py`: telemetria, auditoria SQLite e diagnósticos de recuperação.
- `tools/`: auditoria de realidade, triagem de segurança, mutation testing, auditoria AST e dashboard.
- `tools/state_manifest.py`: gera o manifesto factual por execução; o arquivo `STATE_MANIFEST.json` é artefato de CI, não snapshot manual versionado.

## Contratos independentes atuais

Fila, snapshots, recuperação, SQL, CentralRouter, memória episódica, working memory, telemetria, contrafactuais, multimodalidade, swarm, auditoria, segurança e lacunas explícitas.

## Dependências opcionais

`psutil` e `FAISS/SentenceTransformers` não estavam disponíveis na execução local da auditoria. O código usa fallbacks documentados; isso reduz garantias e deve permanecer explícito.

## Manifesto factual

O estado operacional atual deve ser lido do artefato `nexus-state-manifest` do CI correspondente ao commit. Ele registra commit, branch, contagem de testes, mutation testing, Bandit, métricas AST, dependências opcionais e commits dos documentos. A ausência de um relatório no manifesto significa `not_generated`, não zero.

## Regra de atualização

Números de testes, cobertura, findings e commits devem ser confirmados por execução. Markdown histórico não substitui a fonte primária atual.
