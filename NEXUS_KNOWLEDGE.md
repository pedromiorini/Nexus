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

## Estado verificado em 2026-10-04

- Branch: `main`.
- Commit local/remoto: `5f261fcaabffe4e265372bfbc1e5b7c7aa303d3f` (`docs: record episodic statistics CI checkpoint`).
- Árvore de trabalho estava limpa antes da auditoria; a execução do auditor gerou apenas atualização legítima de relatórios e `__pycache__` temporário.
- Suíte atual: **86 testes**, todos passando.
- Mutation testing direcionado: **10/10 mutações mortas**.
- Bandit atual: **129 achados LOW**, sem MEDIUM; triagem preserva 2 B311? Não: são 16 B311, 111 B101 e 2 achados de subprocesso controlado.
- Auditoria AST: 0 funções somente com `pass`, 0 `NotImplementedError`, 61 retornos constantes simples.
- CI do commit de código anterior `e523d9f`: run `37097334156`, sucesso.

## Componentes

- `core/constitutional_brain.py`: núcleo monolítico, roteamento e componentes experimentais.
- `core/deferred_task_queue.py`: fila priorizada, retry, descarte, callbacks e snapshots versionados.
- `core/vram_defense_guard.py`: decisões conservadoras de pressão de memória.
- `vita/nexus_constitutional_bridge_v3.py`: telemetria, auditoria SQLite e diagnósticos de recuperação.
- `tools/`: auditoria de realidade, triagem de segurança, mutation testing, auditoria AST e dashboard.

## Contratos independentes atuais

Fila, snapshots, recuperação, SQL, CentralRouter, memória episódica, working memory, telemetria, contrafactuais, multimodalidade, swarm, auditoria, segurança e lacunas explícitas.

## Dependências opcionais

`psutil` e `FAISS/SentenceTransformers` não estavam disponíveis na execução local da auditoria. O código usa fallbacks documentados; isso reduz garantias e deve permanecer explícito.

## Regra de atualização

Números de testes, cobertura, findings e commits devem ser confirmados por execução. Markdown histórico não substitui a fonte primária atual.
