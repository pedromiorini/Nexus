# Nexus — Conhecimento operacional canônico

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 203
bandit_low: 131
-->

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

## Estado histórico verificado em 2026-10-05

- Branch: `main`.
- Commit local/remoto: `HEAD` — o SHA exato é resolvido pelo manifesto do CI desta execução.
- CI do mesmo commit: o run associado ao manifesto, com sucesso; o artefato `nexus-state-manifest` foi gerado após os gates.
- Suíte daquela referência: **117 testes**, todos passando no gate.
- Mutation testing direcionado: **10/10 mutações mortas**.
- Bandit após o hardening do B607: **131 achados LOW**, sem MEDIUM; a triagem classifica 16 B311, 111 B101 e 4 subprocessos controlados, sem `manual_review` residual.
- Auditoria AST: 0 funções somente com `pass`, 0 `NotImplementedError`, 61 retornos constantes simples.
- Os números de 105 testes/run `37258732531` e 86 testes/run `37097334156` permanecem apenas como histórico de marcos anteriores.

## Validação local da branch de acompanhamento em 2026-10-16

- A suíte unitária executada nesta branch passou com **169 testes**.
- A suíte integrada registrou **30 verificações aprovadas e 1 explicitamente ignorada** por dependência FAISS opcional indisponível.
- O Bandit encontrou **131 achados LOW**, sem MEDIUM/HIGH; a triagem e o snapshot de realidade foram regenerados após as mudanças.
- `RealEpisodicMemory.close_episode` incrementa `episodes_closed` somente quando o SQLite confirma a transição de um episódio aberto para fechado; repetição e ID inexistente não incrementam.
- Esses resultados são locais; o estado oficial do commit é determinado pelo manifesto de CI associado ao `HEAD`.

## Benchmark científico sintético — branch de avaliação de 2026-10-16

- Base exata: `5f53a9922b71d6ef3c565a31630f56091a454671`; protocolo `nexus.capability-benchmark.v1`, seed `20261016`, taskset SHA-256 `f2cbc785470b5e4018c08512c436e65f98fb26ab2d6b62e2f2c23d8bfd08c5c1`.
- Execução local: 173/173 testes; cobertura da fila 96% e Vita 74%; suíte integrada 30 aprovados e 1 skip opcional.
- Resultados medidos: replay composicional 39/39 exato (inclui holdout de comprimento 3); extração lexical de relações F1=0,667 e taxa de falso positivo em controles negativos=0,5; alegações de efeito downstream F1=0,667 com 8 falsos positivos; Brier=0,1849 e ECE=0,01684 em probabilidades fornecidas pelo próprio taskset.
- Observação parcial: 8 tarefas, nenhum output de distribuição de crença e posterior sem nota (`unsupported_interface`); a API atual não oferece um estado de crença comparável.
- Resultados são locais e específicos ao protocolo. Não há treino composicional, previsão probabilística gerada pelo Nexus, descoberta causal ou validação de teorias; o CI do commit é a referência oficial.

## Componentes

- `core/constitutional_brain.py`: núcleo monolítico, roteamento e componentes experimentais.
- `core/deferred_task_queue.py`: fila priorizada, retry, descarte, callbacks e snapshots versionados.
- `core/vram_defense_guard.py`: decisões conservadoras de pressão de memória.
- `vita/nexus_constitutional_bridge_v3.py`: telemetria, auditoria SQLite e diagnósticos de recuperação.
- `tools/`: auditoria de realidade, triagem de segurança, mutation testing, auditoria AST e dashboard.
- `tools/state_manifest.py`: gera o manifesto factual por execução; o arquivo `STATE_MANIFEST.json` é artefato de CI, não snapshot manual versionado.

## Contratos independentes atuais

Fila, snapshots, recuperação, SQL, CentralRouter, memória episódica, working memory, telemetria, contrafactuais, multimodalidade, swarm, auditoria, segurança, consistência de estado e lacunas explícitas.

## Dependências opcionais

`psutil` e `FAISS/SentenceTransformers` não estavam disponíveis na execução local da auditoria. O código usa fallbacks documentados; isso reduz garantias e deve permanecer explícito.

## Manifesto factual

O estado operacional atual deve ser lido do artefato `nexus-state-manifest` do CI correspondente ao commit. Ele registra commit, branch, contagem de testes, mutation testing, Bandit, métricas AST, dependências opcionais e commits dos documentos. A ausência de um relatório no manifesto significa `not_generated`, não zero.

O resumo `unittest` contabiliza sucessos inesperados (`@expectedFailure` que passa, XPASS) como falhas, seguindo `TestResult.wasSuccessful()`, e expõe sua contagem em `unexpected_successes`; o manifesto e o gate de consistência validam esse campo sem somá-lo duas vezes.

## Regra de atualização

Números de testes, cobertura, findings e commits devem ser confirmados por execução. Markdown histórico não substitui a fonte primária atual.

O `state_consistency_audit.py` considera apenas o bloco estruturado `NEXUS-CURRENT-STATE` como claim atual; referências históricas fora dele são preservadas e ignoradas pelo gate.

O manifesto distingue `discovered_static` (inventário AST) de `executed`, `passed`, `failed` e `skipped` obtidos pelo resumo da suíte final. O gate usa `executed` para comparar o claim atual de testes.
