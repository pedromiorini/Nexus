# Nexus — Conhecimento operacional canônico

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 257
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
- Estes resultados são locais e não eram CI do head da PR naquela rodada. `timeout_seconds` ainda não interrompia executores; o limite de bytes do cache usa `sys.getsizeof` e não é um limite global de memória do processo. A validação do campo e a telemetria advisory de overrun foram adicionadas na seção subsequente.

## Filtro de crença POMDP e benchmark v2 — 2026-10-16

- Base local: `main` pós-PR #23, commit `e3cc760597aa1782aef9879a8f88ee6f48032348`; branch isolada `feat/pomdp-belief-filter-v2-post-pr23-2026-10-16`.
- `RealPredictiveCodingSystem.update_belief_state` implementa uma atualização pura de um passo, valida entradas estocásticas, limita o espaço a 128 estados, usa log-sum-exp e rejeita evidência realmente nula.
- Protocolo separado `nexus.pomdp-belief-benchmark.v2`: 128 modelos determinísticos de 2–8 estados e 635 probabilidades comparadas com uma referência independente; erro absoluto médio `3.001891167837976e-17`, máximo `2.7755575615628914e-16`, 128/128 dentro de `1e-12`; 4/4 controles inválidos rejeitados. Taskset SHA-256: `232371d0e8e948f4b0a004261e3d271b1003aed8d0de8c931d3439f9d370d025`.
- Validação local: 240/240 testes, sem falhas/skips/XPASS; suíte integrada 30 aprovadas e 1 skip FAISS opcional; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation 28/28 mortas, 0 sobreviventes/não aplicadas; Ruff, compileall, actionlint, zizmor, Bandit (131 LOW, nenhum MEDIUM/HIGH) e auditorias AST passaram.
- O gate verifica a seed do workflow (`20261016`) e a CLI retorna código 1 se qualquer caso exceder `1e-12` ou um controle inválido for aceito; o relatório da falha é preservado pelo workflow quando gerado.
- O protocolo v1 permaneceu inalterado: a trilha de observação parcial continua `unsupported_interface`. O v2 recebe o modelo completo e não mede aprendizagem, descoberta de estados ou planejamento. A PR #24 foi mesclada em `main` no commit `dd0a73c6e57dfcccf9b522b3d2dd3a4623cdc732`, com ambos os checks oficiais aprovados.

## Validação do budget de ferramentas — 2026-10-16

- Base local: `main` pós-PR #24, commit `dd0a73c6e57dfcccf9b522b3d2dd3a4623cdc732`; branch isolada `feat/post-pr24-contract-hardening-2026-10-16`.
- Falha reproduzida: o registro aceitava `timeout_seconds` igual a zero, negativo, booleano, float ou `NaN`; valor string causava `TypeError`. A definição também podia ser mutada após o registro.
- Correção: validação exige inteiro não booleano no intervalo inclusivo `[1, 60]`, tanto ao registrar quanto antes de cada execução. `time.monotonic()` mede somente o executor; `TOOL_EXECUTION_OVERRUN` é emitido quando ele termina, por retorno ou exceção. Resultados bem-sucedidos são preservados; falhas continuam falhas estruturadas; o evento informa `execution_completed=true` e `cancelled=false`.
- Validação local em Python 3.12.3: 246/246 testes; suíte integrada 30 aprovados e 1 skip FAISS opcional; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation gate 33/33 mortas, 0 sobreviventes/não aplicadas; Ruff, actionlint, zizmor, compileall e auditorias AST passaram. Bandit: 131 LOW, nenhum MEDIUM/HIGH.
- Limite deliberado: isso é diagnóstico pós-execução, não deadline nem cancelamento. `Future.cancel()` não cancela uma chamada já em execução; `Process.terminate()` pode corromper pipes/queues ou deixar locks indisponíveis ([documentação oficial de `concurrent.futures`](https://docs.python.org/3.12/library/concurrent.futures.html), [documentação oficial de `multiprocessing`](https://docs.python.org/3.12/library/multiprocessing.html)). Um executor isolado ou API cooperativa continua necessário para interrupção segura.
- Estes resultados são locais; o CI do head candidato permanece a referência oficial.

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


## Validação da fronteira de parâmetros de ferramentas — 2026-10-16

- Base local: `main` pós-PR #25, commit `675f5aa95416c3a92998a34dd2f905f529b90980`; branch isolada `feat/post-pr25-hardening-2026-10-16`.
- Falha reproduzida: `RealToolUse.execute_tool("list_operations", None)` levantava `TypeError` na verificação de membership dos parâmetros obrigatórios, antes do bloco que converte exceções do executor em `ToolExecutionResult`.
- Correção: `ToolValidator.validate_parameters` verifica primeiro se a entrada externa é uma instância de `dict`; caso contrário registra `TOOL_PARAM_TYPE_MISMATCH` com o tipo observado e retorna `False`. A chamada pública responde com falha estruturada sem acessar cache nem executor.
- Três testes independentes cobrem `None`/lista/string, dicionário sem parâmetros obrigatórios e execução normal com dicionário válido. Um mutante que remove a guarda é morto pelo mutation gate.
- Validação local em Python 3.12.3: 249/249 testes; suíte integrada 30 aprovados e 1 skip FAISS opcional; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation gate 34/34 mortas, sem sobreviventes/não aplicadas; Ruff, actionlint, zizmor, compileall, auditorias AST e consistência documental passaram. Bandit: 131 LOW, nenhum MEDIUM/HIGH.
- Limite: a alteração valida apenas o tipo externo; não restringe subclasses de `dict`, valores aninhados nem chaves extras. Os resultados são locais; a CI do head candidato permanece a referência oficial.

## JSON estrito e resiliência da chave de cache — 2026-10-16

- Base local: `main` pós-PR #26, commit `508020d77d5705f99c76eaca4df05e8e265662f7`; branch isolada `feat/post-pr26-tool-input-boundary-2026-10-16`.
- Reproduções antes da correção: `format_json` retornava sucesso contendo `NaN`, `Infinity` ou `-Infinity`; a geração de cache key levantava `TypeError` fora do `try` do executor quando `params` incluía um valor Python não serializável, inclusive uma extensão extra ignorada pelo schema. Ferramentas marcadas como não cacheáveis também tentavam gerar chave.
- Correção: `format_json` passa `allow_nan=False`, convertendo valores não finitos em falha estruturada. Para ferramentas cacheáveis, erros esperados de serialização (`TypeError`, `ValueError`, `OverflowError`, `RecursionError`) emitem `TOOL_CACHE_KEY_UNAVAILABLE` e ignoram lookup/store sem impedir a execução normal; ferramentas não cacheáveis não geram chave. Chaves extras ainda ignoradas permanecem compatíveis.
- Contratos: `test_format_json_contract.py` verifica valores finitos, rejeição de `NaN`/infinitos e falha estruturada para dados não serializáveis. `test_tool_cache_key_contract.py` prova que a extensão não serializável não bloqueia um executor válido, não é cacheada e é reexecutada; também protege a ausência de serialização para ferramentas não cacheáveis.
- Validação local: 254/254 testes; suíte integrada 30 aprovados e 1 skip FAISS opcional; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation gate 37/37 mortas, sem sobreviventes/não aplicadas; Bandit 131 LOW e nenhum MEDIUM/HIGH; Ruff, actionlint, zizmor, compileall, auditorias AST e benchmarks de capacidade e POMDP v2 passaram.
- Base normativa: [documentação Python 3.12 do módulo `json`](https://docs.python.org/3.12/library/json.html) e [RFC 8259, §6](https://www.rfc-editor.org/rfc/rfc8259) determinam que NaN/Infinity não são números JSON válidos; `allow_nan=False` produz `ValueError` em modo estrito.
- Limites: não há validação recursiva de schema nem rejeição de chaves desconhecidas; o bypass não limita o custo de serializar entradas grandes que são serializáveis, nem transforma o orçamento estimado do cache em limite de memória global. Os resultados são locais, não CI oficial.


## Pré-verificação limitada da geração de cache key — 2026-10-16

- Base local: `main` pós-PR #27, merge commit `c4375d9f42703b11d2632b3a701303c5a69563a4`; branch isolada `feat/post-pr27-cache-key-resource-bounds-2026-10-16`.
- Lacuna reproduzida por inspeção: `execute_tool` chamava `json.dumps(params, sort_keys=True)` antes do executor. Os limites existentes de payload cacheado não restringiam o custo da serialização de parâmetros grandes, profundos ou com muitos nós.
- Correção: uma pré-verificação iterativa limita a estimativa conservadora do JSON+prefixo a 64 KiB, profundidade a 64, travessia a 4.096 ocorrências e cada contêiner a 1.024 itens; ciclos são identificados sem recursão. A estimativa conta escapes de strings e usa limite superior para inteiros antes do encoder.
- Se qualquer budget for excedido, ou o valor não pertencer aos tipos Python nativos aceitos pelo encoder JSON padrão e percorridos com segurança, `TOOL_CACHE_KEY_UNAVAILABLE` registra um motivo e `cache_bypassed=true`; lookup/store são pulados, mas o executor segue normalmente. Subclasses de `dict` continuam aceitas/executadas e apenas bypassam a otimização. Ferramentas não cacheáveis continuam sem pré-verificação.
- Para valores elegíveis, `json.dumps(..., sort_keys=True)` e a fórmula SHA-256 foram preservados; o teste compara a chave nova com a fórmula anterior e verifica que um hit ainda ocorre na repetição. `test_tool_cache_key_contract.py` protege também bypass de bytes, profundidade, nós, itens por contêiner, ciclos e valores não serializáveis, sem impedir execução.
- Validação local: **257/257** testes; suíte integrada 30 aprovados e 1 skip opcional FAISS; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation gate **42/42** mortas, sem sobreviventes/não aplicadas; Bandit 131 LOW, nenhum MEDIUM/HIGH; Ruff, actionlint, zizmor, compileall, auditoria de realidade e auditorias AST passaram. Benchmarks de capacidade e POMDP v2 passaram com seed `20261016`.
- Fundamentação: a [documentação oficial Python 3.12 do módulo `json`](https://docs.python.org/3.12/library/json.html) descreve `iterencode` como gerador de fragmentos disponíveis e não documenta um tamanho máximo para cada fragmento nem budgets gerais de tamanho/profundidade; por isso o preflight é iterativo e ocorre antes da serialização. Os resultados permanecem locais até a CI do commit candidato.
- Limite epistemológico: este contrato restringe somente o trabalho adicional da chave de cache e a elegibilidade ao cache. Não limita custo/memória próprios do executor, não valida schema recursivamente, não rejeita chaves extras e não converte o budget de cache em limite global de memória.
