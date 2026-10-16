# Nexus — Conhecimento operacional canônico

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 222
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

## Validação local da branch anterior em 2026-10-16

- A suíte unitária executada nesta branch passou com **169 testes**.
- A suíte integrada registrou **30 verificações aprovadas e 1 explicitamente ignorada** por dependência FAISS opcional indisponível.
- O Bandit encontrou **131 achados LOW**, sem MEDIUM/HIGH; a triagem e o snapshot de realidade foram regenerados após as mudanças.
- `RealEpisodicMemory.close_episode` incrementa `episodes_closed` somente quando o SQLite confirma a transição de um episódio aberto para fechado; repetição e ID inexistente não incrementam.
- Esses resultados são locais; o estado oficial do commit é determinado pelo manifesto de CI associado ao `HEAD`.

## Auditoria local após a PR #21 — 2026-10-16

- Base: `main` no commit `b14123593861b1406e6f6f7b0204fe50630b0f8c`; suíte `unittest` 213/213, sem falhas, skips ou XPASS.
- A suíte integrada teve 31 marcadores: 30 aprovados e 1 skip opcional por FAISS; os fallbacks de `psutil`, FAISS e SentenceTransformers permaneceram ativos.
- Cobertura local: fila 95%, guard VRAM 94%, Vita 76%, núcleo 67%; mutation gate: 28/28 aplicáveis mortos; Bandit: 131 LOW, nenhum MEDIUM/HIGH.
- O roteador ignora/remova o `_deferred_reprocess` fornecido no contexto externo; somente o método interno de reprocessamento pode suprimir um novo enqueue.
- Vínculos episódicos verificam `memory_id` quando um backend resolutor está configurado; sem backend, o ID é armazenado como referência opaca e não é alegado como validado.
- Relatórios de realidade locais: 26.946 linhas no núcleo, 267 classes, 859 funções, 0 funções `pass`-only e 2 retornos constantes simples.
- POMDP permanece `unsupported_interface` no benchmark v1. A atualização Bayesiana exigiria contrato explícito de modelo fornecido e nova versão de protocolo; a API atual não é aprendizagem do modelo nem planejamento.
- Estes resultados são locais; o manifesto de CI associado ao `HEAD` permanece a fonte de verdade para o estado oficial.

## Benchmark científico sintético — branch de avaliação de 2026-10-16

- Base exata: `5f53a9922b71d6ef3c565a31630f56091a454671`; protocolo `nexus.capability-benchmark.v1`, seed `20261016`, taskset SHA-256 `f2cbc785470b5e4018c08512c436e65f98fb26ab2d6b62e2f2c23d8bfd08c5c1`.
- Execução local: 173/173 testes; cobertura da fila 96% e Vita 74%; suíte integrada 30 aprovados e 1 skip opcional.
- Resultados medidos: replay composicional 39/39 exato (inclui holdout de comprimento 3); extração lexical de relações F1=0,667 e taxa de falso positivo em controles negativos=0,5; alegações de efeito downstream F1=0,667 com 8 falsos positivos; Brier=0,1849 e ECE=0,01684 em probabilidades fornecidas pelo próprio taskset.
- Observação parcial: 8 tarefas, nenhum output de distribuição de crença e posterior sem nota (`unsupported_interface`); a API atual não oferece um estado de crença comparável.
- Resultados são locais e específicos ao protocolo. Não há treino composicional, previsão probabilística gerada pelo Nexus, descoberta causal ou validação de teorias; o CI do commit é a referência oficial.

## Autoauditoria de recursos do `RealToolUse` — 2026-10-16

- Base local: `main` pós-PR #22, commit `27299281b636deddb5555173a9f191dd01e65013`; branch isolada `fix/math-calculate-resource-limits-2026-10-16`.
- Achado reproduzido antes da correção: `math_calculate` aceitava literais string e repetição, podia devolver/cachear um milhão de caracteres, aceitava potências grandes e mantinha entradas expiradas.
- Mitigação implementada: expressões numéricas somente, até 256 caracteres/64 nós AST, limites de expoente/inteiro/magnitude, TTL expurgado e cache limitado a 128 entradas, 1 MiB agregado estimado e 64 KiB por payload. O cache guarda uma cópia profunda e devolve cópias nos hits.
- Validação local: 222/222 testes, 31 marcadores integrados (30 aprovados, 1 skip FAISS opcional), cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; 28/28 mutações direcionadas mortas; Bandit 131 LOW, 0 MEDIUM/HIGH; Ruff, actionlint, zizmor e compileall passaram.
- Estes resultados são locais e ainda não são CI do head da PR. `timeout_seconds` segue sem execução/cancelamento genérico; o limite de bytes do cache usa `sys.getsizeof` e não é um limite global de memória do processo.

## Componentes

- `core/constitutional_brain.py`: núcleo monolítico, roteamento e componentes experimentais.
- `core/deferred_task_queue.py`: fila priorizada, retry, descarte, callbacks e snapshots versionados.
- `core/vram_defense_guard.py`: decisões conservadoras de pressão de memória.
- `RealToolUse`: calculadora numérica com budgets explícitos e cache com expiração/capacidade limitadas.
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

O `state_consistency_audit.py` exige exatamente um bloco estruturado `NEXUS-CURRENT-STATE` e rejeita chaves duplicadas; referências históricas fora dele são preservadas e ignoradas pelo gate.

O manifesto distingue `discovered_static` (inventário AST) de `executed`, `passed`, `failed` e `skipped` obtidos pelo resumo da suíte final. O gate usa `executed` para comparar o claim atual de testes.
