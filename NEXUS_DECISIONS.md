# Nexus — Registro de decisões

## D-001 — Evolução incremental em vez de grande refatoração

- **Decisão:** priorizar patches pequenos e contratos independentes.
- **Contexto:** o núcleo é monolítico e possui alto risco de efeitos laterais.
- **Alternativas:** reescrita ampla ou modularização imediata.
- **Motivo:** mudanças menores são mais reversíveis e auditáveis.
- **Evidência:** suíte de contratos, mutation testing direcionado e CI existente.
- **Reversibilidade:** fácil a moderada.

## D-002 — Fallbacks devem ser explicitamente classificados

- **Decisão:** distinguir backend real, fallback degradado, heurística, simulação, mock e indisponibilidade.
- **Contexto:** dependências opcionais alteram comportamento.
- **Motivo:** impedir que fallback seja apresentado como capacidade completa.
- **Evidência:** contratos multimodais, telemetria GPU e busca SQL.
- **Reversibilidade:** fácil.

## D-003 — Snapshots são transporte, não durabilidade

- **Decisão:** `DeferredTaskQueue` exporta/restaura schema versionado, mas não promete persistência física, locking distribuído ou exactly-once.
- **Motivo:** manter o contrato honesto e a responsabilidade de armazenamento no consumidor.
- **Evidência:** testes de round-trip, validação atômica e documentação do README.
- **Reversibilidade:** moderada.

## D-004 — Auditoria de segurança conservadora

- **Decisão:** preservar findings LOW no relatório e registrar disposições sem suprimi-los automaticamente.
- **Motivo:** triagem contextual não equivale a correção ou clearance.
- **Evidência:** `SECURITY_TRIAGE.md`, Bandit atual e auditor AST B101.
- **Reversibilidade:** fácil.

## D-005 — Métricas observadas substituem constantes de demonstração

- **Decisão:** calcular estatísticas de swarm e memória episódica a partir de decisões e registros SQLite reais.
- **Motivo:** reduzir teatro cognitivo e aumentar observabilidade verificável.
- **Evidência:** `test_swarm_statistics_contract.py` e `test_episodic_memory_statistics_contract.py`.
- **Reversibilidade:** fácil.

## D-006 — Documentação histórica não é fonte de estado atual

- **Decisão:** manter resultados antigos, mas separar estado atual em conhecimento canônico.
- **Motivo:** auditorias históricas registraram 132, 130 e 129 findings; o manifesto CI final anterior registrava 132 LOW; após o hardening do B607, a triagem versionada registra 131 LOW sem `manual_review` residual.
- **Evidência:** comparação entre `NEXUS_AUDIT_FINAL.md`, `SECURITY_TRIAGE.md` e Bandit executado.
- **Reversibilidade:** fácil.

## D-007 — Sinais de atenção opcionais, com fallback explícito

- **Decisão:** `StimulusItem` aceita `contrast` e `motion` opcionais; `BottomUpAttention` usa sinais fornecidos e mantém valores heurísticos somente quando ausentes.
- **Contexto:** contraste e movimento eram constantes sem uma entrada observável.
- **Motivo:** aumentar observabilidade sem quebrar construtores existentes ou inventar um backend perceptual.
- **Evidência:** `test_attention_signal_contract.py` cobre sinais presentes, contexto, limites e fallback.
- **Reversibilidade:** fácil.

## D-008 — Média de sinergia calculada sobre todos os registros

- **Decisão:** `RealIntegrationOrchestrationEngine` calcula `avg_synergy_score` pela média aritmética das integrações registradas.
- **Contexto:** a atualização anterior fazia uma média recursiva sem peso, distorcendo o resultado após mais de uma integração.
- **Motivo:** preservar uma métrica estatística correta sem alterar a API pública.
- **Evidência:** `test_integration_synergy_contract.py` detecta os valores 0.9 e 0.6 e exige média 0.75.
- **Limitação:** os scores individuais continuam heurísticos baseados nos nomes dos módulos; a correção não demonstra sinergia cognitiva.
- **Reversibilidade:** fácil.

## D-009 — Entidades do grafo são identificadas por ID

- **Decisão:** re-adicionar um `entity_id` atualiza a entidade existente, não aumenta `total_entities`, e move o ID entre índices quando o tipo muda.
- **Contexto:** a implementação substituía o índice, mas acumulava a contagem de inserções e deixava o tipo antigo indexado.
- **Motivo:** alinhar estatísticas e índices com a identidade observada no grafo.
- **Evidência:** `test_knowledge_graph_contract.py` cobre duplicata, média de confiança, contagem única e reclassificação.
- **Limitação:** o grafo continua local; isso não define identidade distribuída, consistência concorrente ou conhecimento válido externamente.
- **Reversibilidade:** fácil.

## D-010 — Lista de ações vazia é uma entrada válida no MCTS

- **Decisão:** `available_actions=None` seleciona o espaço de ações padrão; `available_actions=[]` representa deliberadamente um espaço vazio.
- **Contexto:** o planejador usava truthiness e substituía uma lista vazia pelas ações padrão.
- **Motivo:** evitar que uma entrada explícita seja silenciosamente reinterpretada e manter resultados previsíveis para consumidores da API.
- **Evidência:** `test_mcts_action_contract.py` cobre ambos os caminhos e confirma árvore sem filhos para o espaço vazio.
- **Limitação:** isso corrige a semântica de entrada, mas não valida a qualidade dos rollouts ou das recompensas heurísticas do MCTS.
- **Reversibilidade:** fácil.

## D-011 — Consistência do estado atual é um gate de CI

- **Decisão:** documentos canônicos expõem um bloco `NEXUS-CURRENT-STATE`; `tools/state_consistency_audit.py` compara seus claims atuais com o manifesto produzido na mesma execução.
- **Contexto:** histórico legítimo e estado atual coexistem em documentos long-lived, e números de testes/segurança podem evoluir em commits diferentes.
- **Motivo:** falhar de forma determinística em commit, contagem de testes, run de CI ou findings divergentes, sem apagar histórico nem interpretar texto histórico como estado atual.
- **Evidência:** `test_state_consistency_audit.py` cobre sincronização, histórico ignorado, conflitos, marcador ausente e manifesto inválido.
- **Limitação:** o gate não prova a validade semântica dos testes ou das disposições de segurança; ele verifica apenas a consistência dos claims estruturados.
- **Reversibilidade:** fácil.

## D-012 — Manifesto registra execução final e severidade bloqueante

- **Decisão:** `verification.tests` separa `discovered_static`, `executed`, `passed`, `failed` e `skipped`; a suíte completa é executada antes da geração do manifesto. Bandit pode continuar informativo para LOW, mas MEDIUM/HIGH bloqueiam o workflow.
- **Contexto:** inventário AST não é necessariamente a quantidade executada, e `bandit ... || true` não deve transformar uma severidade alta em sucesso silencioso.
- **Motivo:** tornar a proveniência do estado factual e a política de segurança explícitas.
- **Evidência:** `tools/test_summary.py`, `test_state_manifest.py` e o passo `Enforce security severity policy` no workflow.
- **Limitação:** a contagem executada continua dependente do runner unittest; não mede cobertura semântica nem substitui revisão manual.
- **Reversibilidade:** fácil.

## D-013 — Benchmark sintético externo ao núcleo e limitado por interface

- **Decisão:** manter tarefas, métricas e adapters em `benchmarks/`, sem alterar o núcleo legado; versionar o protocolo e executar a seed fixa no CI.
- **Contexto:** o próximo passo de maior valor é testar generalização composicional, intervenções causais, incerteza e observação parcial, com atenção às contribuições teóricas que inspiram cada pergunta.
- **Motivo:** obter resultados reproduzíveis e delimitar cada evidência; quando não existe saída de crença ou distribuição probabilística, registrar a lacuna em vez de fabricar uma nota.
- **Evidência:** `benchmarks/nexus_capability_benchmark.py`, `test_capability_benchmark.py` e `benchmarks/README.md`.
- **Limitação:** benchmark sintético específico às APIs atuais; não valida aprendizagem composicional, descoberta/identificação causal, geração de probabilidades ou teorias cognitivas.
- **Reversibilidade:** fácil.

## D-014 — Calculadora e cache têm budgets de recursos explícitos

- **Decisão:** `math_calculate` aceita apenas expressões numéricas finitas com limites de tamanho, complexidade AST, expoente e magnitude; o cache expurga entradas vencidas, limita entradas/payload estimado e isola resultados mutáveis por cópias profundas.
- **Contexto:** um literal string repetido e potências grandes podiam produzir resultados arbitrariamente grandes; entradas expiradas permaneciam retidas.
- **Motivo:** reduzir consumo controlável de CPU/memória sem alegar isolamento de processo ou execução com timeout.
- **Evidência:** `test_math_calculate_resource_limits.py` e reprodução local antes do patch; 222 testes passaram localmente nesta branch.
- **Limitação:** o payload usa estimativa `sys.getsizeof`; não é limite global do processo. `ToolDefinition.timeout_seconds` continua sem cancelamento genérico.
- **Reversibilidade:** fácil.

## D-015 — Filtro POMDP de um passo com modelo explícito

- **Decisão:** `RealPredictiveCodingSystem.update_belief_state` é uma operação pura que recebe estados, prior, transições e likelihoods; limita o espaço a 128 estados, valida as distribuições e calcula o posterior com log-sum-exp. Evidência realmente nula é rejeitada, sem posterior uniforme.
- **Contexto:** o benchmark v1 exercita `hierarchical_prediction_cycle`, que não expõe distribuição sobre estados ocultos; adicionar uma nova saída ao ciclo legado mudaria seu contrato.
- **Motivo:** tornar mensurável a atualização de crença por modelo fornecido sem atribuir ao Nexus aprendizagem de modelo ou planejamento, e sem alterar o protocolo v1.
- **Evidência:** `test_pomdp_belief_filter_contract.py`, `test_pomdp_benchmark_v2.py` (inclui seed `20261016` e teste de falha injetada), `benchmarks/nexus_pomdp_benchmark_v2.py` e o artefato v2 do workflow. A CLI falha se qualquer posterior ultrapassar `1e-12` ou algum controle inválido for aceito.
- **Limitação:** o filtro não aprende o prior, a matriz de transição ou o modelo de observação; não resolve políticas nem demonstra inferência em ambiente real. A lacuna da interface de observação parcial do v1 permanece.
- **Reversibilidade:** fácil; protocolo e schema v2 são independentes do benchmark v1.

## D-016 — Budget de timeout é validado e observado, não apresentado como cancelamento

- **Decisão:** `ToolDefinition.timeout_seconds` aceita somente inteiros não booleanos em `[1, 60]`, com validação no registro e imediatamente antes da execução. A duração do executor síncrono é medida com `time.monotonic()`; se ele termina acima do budget, por retorno ou exceção, emite `TOOL_EXECUTION_OVERRUN`. O resultado de sucesso é preservado, uma exceção continua sendo falha e o evento marca `execution_completed=true` e `cancelled=false`.
- **Contexto:** a validação anterior rejeitava apenas valores acima de 60; zero, negativos, booleanos, floats e `NaN` eram aceitos, enquanto string podia lançar `TypeError`. O campo não interrompia a execução.
- **Motivo:** fechar a validação dos limites e observar violações sem sugerir que um timeout de espera cancela trabalho já iniciado nem descartar um resultado concluído.
- **Evidência:** seis testes em `test_tool_definition_contract.py` cobrem valores inválidos, limites inclusivos, alteração pós-registro, execução dentro do orçamento e overrun após sucesso ou exceção; `tools/targeted_mutation.py` mata cinco mutantes desse contrato.
- **Base técnica:** `Future.cancel()` falha em cancelar uma chamada já em execução; `Process.terminate()` pode corromper pipes/queues ou deixar locks indisponíveis ([`concurrent.futures`, Python 3.12](https://docs.python.org/3.12/library/concurrent.futures.html), [`multiprocessing`, Python 3.12](https://docs.python.org/3.12/library/multiprocessing.html)).
- **Limitação:** o evento é pós-execução, não uma deadline nem isolamento. Interrupção segura exige um contrato cooperativo ou runner/processo isolado sem recursos compartilhados perigosos.
- **Reversibilidade:** fácil; não altera os protocolos dos benchmarks nem o resultado dos executores built-in.


## D-017 — `execute_tool` exige parâmetros externos do tipo dict

- **Decisão:** `ToolValidator.validate_parameters` aceita somente uma instância de `dict` (incluindo subclasses) como contêiner externo de `params`; outros tipos geram `TOOL_PARAM_TYPE_MISMATCH` com tipo esperado/observado e são convertidos em falha estruturada por `execute_tool` antes da iteração do schema.
- **Contexto:** `execute_tool` anuncia um resultado `ToolExecutionResult`, mas `params=None` fazia a validação de campos obrigatórios levantar `TypeError` antes do tratamento de exceções do executor.
- **Motivo:** preservar a fronteira de falha estruturada para entrada malformada sem alterar o processamento de dicionários válidos nem permitir que a exceção escape ao chamador.
- **Evidência:** `test_tool_parameter_contract.py` cobre contêineres não-dict, dicionário incompleto e execução válida; `tools/targeted_mutation.py` mata um mutante que remove a guarda.
- **Limitação:** não valida recursivamente valores, serialização JSON ou chaves desconhecidas; esses contratos permanecem separados.
- **Reversibilidade:** simples; nenhuma forma válida de parâmetro `dict` muda de comportamento.

## D-018 — `format_json` emite apenas JSON estrito

- **Decisão:** serializar `format_json` com `json.dumps(..., allow_nan=False)`; valores não finitos resultam em falha estruturada, sem conversão silenciosa para texto ou `null`.
- **Contexto:** o padrão permissivo do encoder Python produz os tokens `NaN`, `Infinity` e `-Infinity`, que não pertencem à gramática numérica JSON.
- **Motivo:** a ferramenta anuncia formato/validação JSON, então seu resultado textual deve ser JSON interoperável; entradas finitas mantêm exatamente a formatação anterior.
- **Evidência:** `test_format_json_contract.py` cobre valores finitos, NaN, os dois infinitos e dados não serializáveis; a mutação que restaura `allow_nan=True` é morta.
- **Base normativa:** [documentação Python 3.12 de `json`](https://docs.python.org/3.12/library/json.html) e [RFC 8259, §6](https://www.rfc-editor.org/rfc/rfc8259).
- **Limitação:** a mudança não implementa validação profunda de schema nem resolve questões de interoperabilidade de números finitos fora da faixa de consumidores.
- **Reversibilidade:** fácil; somente entradas que antes geravam tokens inválidos deixam de retornar sucesso.

## D-019 — Falha de serialização da cache key é cache miss, não interrupção

- **Decisão:** construir cache key apenas para ferramentas cacheáveis. Se a serialização de parâmetros falhar com `TypeError`, `ValueError`, `OverflowError` ou `RecursionError`, emitir `TOOL_CACHE_KEY_UNAVAILABLE`, ignorar lookup/store e executar o executor normalmente.
- **Contexto:** a chave era gerada antes do `try` do executor; parâmetros não serializáveis podiam lançar ao chamador, inclusive uma extensão extra que o schema ignorava. A mesma serialização era tentada mesmo com cache desativado.
- **Motivo:** a infraestrutura opcional de cache não deve alterar a semântica do executor nem rejeitar chaves extras que a API legada aceita; entradas não representáveis no cache simplesmente não são cacheadas.
- **Evidência:** `test_tool_cache_key_contract.py` verifica execução válida repetida sem cache e ausência de serialização para ferramenta não cacheável; mutações que removem o bypass e a guarda são mortas.
- **Limitação:** não restringe o custo da serialização de parâmetros muito grandes que são representáveis nem valida recursivamente os valores.
- **Reversibilidade:** fácil; chamadas afetadas só deixam de obter cache hit/store.


## D-020 — Orçamento prévio para geração de cache key

- **Decisão:** antes de `json.dumps`, percorrer iterativamente os parâmetros de ferramentas cacheáveis, estimando o texto JSON no formato padrão mais o prefixo do nome da ferramenta. Limites: 64 KiB, profundidade 64, 4.096 ocorrências/nós e 1.024 itens por contêiner; detectar ciclos. A pré-verificação só percorre tipos Python nativos exatos aceitos pelo encoder padrão (`dict`, `list`, `tuple`, strings, números, booleanos e `None`).
- **Falha/bypass:** excesso, ciclo ou tipo não percorrido em segurança emite `TOOL_CACHE_KEY_UNAVAILABLE`, ignora lookup/store e deixa o executor rodar; não rejeita a chamada nem valida schema recursivamente. `cacheable=False` continua evitando toda a geração da chave.
- **Compatibilidade:** para parâmetros elegíveis, manter `json.dumps(params, sort_keys=True)` e a fórmula SHA-256 existentes, preservando a identidade de chaves pequenas. Subclasses de `dict` continuam aceitas pela fronteira externa e pelo executor, mas bypassam cache; chaves extras continuam ignoradas conforme o contrato legado.
- **Contexto/motivo:** os limites de entrada do validator eram superficiais, e o orçamento de resultado armazenado não protegia o custo proporcional da ordenação/serialização da chave. A documentação Python 3.12 não define limites gerais de tamanho/profundidade do encoder nem limite máximo para fragmentos de `iterencode`; o preflight precisa ocorrer antes dele.
- **Evidência:** `test_tool_cache_key_contract.py` cobre compatibilidade de hash/hit, bypass sem chamar o encoder para bytes, profundidade, nós, cardinalidade de contêiner e ciclos, preservação de execução, subclass de `dict`, valor não serializável e ferramenta não cacheável. Cinco mutantes das guardas/detecção são mortos; validação local: 257/257 testes e mutation 42/42.
- **Limitação:** o budget cobre somente trabalho de formação da cache key; não restringe memória/custo do executor, nem limita a memória total do processo. Os dois checks oficiais da PR #28 passaram antes do merge em `main` no commit `fafcd95e6b40c5d51b34d344f164a9045279d8bb`; as métricas listadas são da validação local daquela branch.
- **Fonte:** [documentação oficial Python 3.12 do módulo `json`](https://docs.python.org/3.12/library/json.html).
- **Reversibilidade:** simples; entradas sobre o budget deixam de obter cache hit/store, mas continuam seguindo a execução normal.


## D-021 — Falha ao copiar resultado para cache não altera o sucesso do executor

- **Decisão:** se `copy.deepcopy` falhar ao preparar o armazenamento de um resultado elegível, emitir `TOOL_CACHE_STORE_UNAVAILABLE`, pular o armazenamento e devolver o `ToolExecutionResult` concluído; contabilizar a execução como sucesso.
- **Contexto:** resultado profundo dentro do limite estimado de bytes podia ultrapassar o limite de recursão do Python. A cópia era feita dentro do bloco amplo de tratamento de `execute_tool`, que convertia a falha da otimização em falha da ferramenta e descartava o resultado já calculado.
- **Motivo:** cache é uma otimização; falha ao copiar uma representação interna não deve alterar a semântica observável da execução original.
- **Evidência:** reprodução local com listas aninhadas de profundidade 600 (38.445 bytes estimados), teste `test_tool_cache_store_contract.py` e mutante que remove o fallback; métricas locais 258/258 testes e mutation 43/43.
- **Limitação:** a captura se restringe à cópia profunda antes do armazenamento; não limita o executor nem o uso global de memória. A intenção de aceitar subclasses em valores tipados aninhados permanece questão separada.
- **Reversibilidade:** fácil; em caso de falha apenas se perde o armazenamento/cache hit desse resultado.

## D-022 — Parâmetros built-in aceitam subclasses reais, com exceção bool/int

- **Decisão:** resolver nomes built-in conhecidos para classes reais e aceitar subclasses verdadeiras no schema do parâmetro. Para `int`, rejeitar `bool` explicitamente. Manter `any` e `object` como curingas e preservar a comparação legada por nome para tipos de schema desconhecidos.
- **Contexto:** `ToolValidator.validate_parameters` comparava `type(value).__name__` exatamente. Isso rejeitava uma subclasse `list` que o executor `list_operations` consegue usar; ao mesmo tempo, a comparação por nome podia confundir uma classe não relacionada chamada `list` com o built-in.
- **Motivo:** alinhar a compatibilidade com a semântica real das classes Python, evitando uma aceitação acidental de booleanos como inteiros e evitando que apenas um nome de classe satisfaça um schema built-in.
- **Evidência:** `test_tool_parameter_contract.py` cobre subclasse de lista de ponta a ponta, subclasse escalar, rejeição de `bool` para `int` e classe de nome falsificado; dois mutantes protegem os limites de subclasse e booleano.
- **Fundamentação:** a documentação oficial Python 3.12 de [`issubclass`](https://docs.python.org/3.12/library/functions.html#isinstance) define compatibilidade por subclasse; a documentação de [`bool`](https://docs.python.org/3.12/library/stdtypes.html#boolean-type-bool) nota que `bool` deriva de `int`, apesar de não ser semanticamente válido para esse parâmetro.
- **Limitações:** só o valor declarado é validado; não há schema recursivo dos membros de listas/dicionários. O preflight de cache tem política independente e pode ignorar o cache para subclasses não nativas, sem impedir a execução. Tipos de schema desconhecidos ainda usam o nome exato legado.
- **Reversibilidade:** fácil; a regra está isolada em `ToolValidator._matches_parameter_type` e pode ser ajustada sem alterar o formato público de `ToolDefinition`.


## D-023 — Limite de memória deve ser inteiro não negativo antes de qualquer backend

- **Decisão:** `RealHierarchicalMemory.retrieve` aceita `int` não booleano; valores negativos geram `ValueError`, tipos inválidos geram `TypeError` e `limit=0` retorna `[]` sem consultar FAISS ou SQLite.
- **Contexto:** SQLite interpreta `LIMIT` negativo como ausência de teto; o caminho FAISS podia rejeitar `k` negativo e cair no fallback SQL, que também retornava sem limite.
- **Motivo:** o parâmetro chamado `limit` não deve ampliar inadvertidamente o resultado para todo o corpus. A validação antes de selecionar backend mantém a mesma fronteira nos modos semântico e fallback.
- **Evidência:** reprodução com quatro registros: `limit=0` devolveu 0, `limit=1` devolveu 1 e `limit=-1` devolveu 4. `test_memory_sql_contract.py` cobre consultas vazias/por palavra, o ramo FAISS simulado, tipos não inteiros/booleanos e zero; dois mutantes protegem os guardrails.
- **Fonte:** [SQLite Language Reference — SELECT, seção LIMIT](https://www.sqlite.org/lang_select.html), que declara que limite negativo não impõe teto.
- **Limitação:** não foi imposto máximo para valores positivos altos, que podem devolver todos os registros existentes se o chamador pedir isso; também não é uma limitação global de memória.
- **Reversibilidade:** simples; callers inválidos recebem erro explícito, `limit=0` permanece válido e os demais limites positivos preservam o fluxo existente.

## D-024 — Falha ao copiar um cache hit deve virar bypass, não falha da ferramenta

- **Decisão:** se `copy.deepcopy(cached_result)` falhar durante um hit, remover a entrada e seu tamanho contabilizado, registrar `TOOL_CACHE_HIT_UNAVAILABLE` com o tipo da exceção e `cache_bypassed=true`, e continuar pelo executor. Só uma cópia concluída atualiza LRU e contadores de cache/sucesso.
- **Contexto:** antes da correção, a cópia ocorria fora do `try` do executor; uma exceção escapava de `execute_tool` e impedia a nova execução, embora o cache seja opcional.
- **Motivo:** uma falha interna da otimização não deve transformar uma chamada executável em exceção nem contar hit não entregue; descartar a entrada evita repetição do caminho defeituoso.
- **Evidência:** `test_tool_cache_hit_contract.py` injeta falha no hit e na tentativa de re-cache; valida execução real, contadores, evento e remoção de bytes/entrada. Dois mutantes protegem fallback e eviction. A documentação oficial de [`copy.deepcopy`, Python 3.12](https://docs.python.org/3.12/library/copy.html) descreve cópia recursiva e limitações envolvendo objetos recursivos; a regressão reproduz a falha de forma determinística por injeção.
- **Validação histórica pré-merge da branch sincronizada:** cache focado 9/9; unittest 271/271 (0 falhas/skips/XPASS); integração 30 aprovados + 1 skip FAISS opcional; cobertura 95/94/76/68%; mutation 55/55, sem sobreviventes/não aplicadas; Bandit 131 LOW, sem MEDIUM/HIGH. Ruff, compileall, actionlint, zizmor, auditorias AST/realidade/gaps e CLIs dos benchmarks de capacidade/POMDP v2 passaram; o POMDP v2 manteve 128/128 casos dentro de `1e-12` e rejeitou 4/4 controles inválidos.
- **Limites:** cobre especificamente a cópia do resultado no cache hit; o fallback de armazenamento é independente. A corrida TOCTOU de parâmetros foi resolvida pelo snapshot registrado em D-025/PR #33; falhas do logger continuam fora do escopo.
- **Reversibilidade:** simples; basta restaurar o caminho de erro anterior, removendo também o evento/eviction, para reverter o comportamento.

## D-025 — Um snapshot JSON limitado alinha a chave e a execução da ferramenta

- **Decisão:** para parâmetros `dict` de ferramentas cacheáveis, construir um snapshot limitado por valor somente de estruturas JSON estritas reconhecidas. O mesmo snapshot é usado para validação do schema, guards semânticos, cache key e executor. A ordem é snapshot → validação/guards → cache key → lookup → executor; por exemplo, o limite de tamanho de `math_calculate` deve rejeitar a entrada antes da serialização da chave.
- **Contexto:** formar a chave a partir de `params` mutável e depois validar/executar o objeto original permite que uma mutação entre essas fases associe resultado a uma chave que descreve outro valor.
- **Motivo:** tornar explícita a semântica por valor e remover a divergência temporal entre a representação cacheada e o valor efetivamente validado/executado, sem deixar que a otimização altere a execução de valores que não podem ser snapshotados com segurança.
- **Fallback:** ciclo, valor não JSON, subclass/container não reconhecido ou exceder os budgets faz bypass da cache e mantém validação/execução com o objeto original. Se a criação da chave falhar depois de um snapshot válido, somente lookup/store são ignorados; validação e execução continuam usando o snapshot.
- **Evidência:** `test_tool_cache_snapshot_contract.py` verifica snapshot idêntico nas três fases, mutação do original depois da preparação da chave, mutação in-place isolada, aliases compartilhados copiados como valores independentes e bypass fora do JSON estrito; `test_tool_cache_key_contract.py` cobre budgets/ciclos em `dict` e `list`. `test_math_calculate_resource_limits.py` protege a rejeição de expressão superdimensionada antes da cache key. Validação local do head original da PR #33 (pré-sincronização da PR #32): suítes focadas 9/9; unittest 269/269; integração 30 aprovações e 1 skip opcional FAISS; cobertura 95/94/76/68%; mutation gate 53/53 mortas, 0 sobreviventes/não aplicadas; Bandit 131 LOW e 0 MEDIUM/HIGH, triagem verificada; Ruff, compileall, actionlint, zizmor, auditorias AST/realidade/gaps e benchmarks de capacidade/POMDP v2 passaram.
- **Limitações:** identidade/aliasing e mutações in-place do objeto original não são preservados. A travessia não sincroniza o produtor: substituição de valor com o mesmo tamanho durante a própria cópia pode produzir um snapshot misto. Depois de construído, esse mesmo snapshot é usado para chave, validação e executor; consumidores que precisem de um instante atomicamente consistente devem sincronizar a origem. O limite de cache não é teto de memória global.
- **Convivência de branches:** a PR #33 foi mesclada primeiro em `main` no commit `62f6af14a2ad1d3659a4a5940e713c588513bf9b`; a PR #32 foi mesclada em seguida no commit `b07254307a23cdaef68f7c2b0d23010ac0827580`. D-024 e D-025 coexistem em `main`.
- **Base técnica:** [documentação oficial Python 3.12 do módulo `json`](https://docs.python.org/3.12/library/json.html).
- **Reversibilidade:** fácil; entradas fora do snapshot limitado apenas deixam de obter cache hit/store e continuam no caminho normal de validação/execução.

## D-026 — `search_episodes` exige limite inteiro não negativo

- **Status:** implementada e incorporada à `main` pela PR #34, merge commit `51373dc77179e4edd094df14290f91aa9c8d8060`.
- **Decisão:** `RealEpisodicMemory.search_episodes` valida `limit` na entrada: aceita `int` não booleano, rejeita tipos inválidos com `TypeError`, negativos com `ValueError` e retorna `[]` para zero antes de lock ou SQL. Limites positivos preservam o comportamento existente; nenhum teto positivo novo é imposto.
- **Contexto:** o método encaminhava o valor diretamente para SQLite. Um `LIMIT` negativo significa ausência de limite e pode retornar todos os episódios do status consultado; `bool` também é subclasse de `int` e era aceito como 0/1.
- **Motivo:** impedir que um parâmetro chamado `limit` amplie inadvertidamente a consulta e alinhar a fronteira episódica ao contrato de `RealHierarchicalMemory.retrieve` (D-023).
- **Evidência:** a reprodução com dois episódios fechados retornou os dois para `limit=-1` e um para `limit=True`. `test_episodic_memory_search_limits_contract.py` cobre os dois caminhos de query, validação antes de SQL, zero sem consulta e limite positivo; três mutantes direcionados protegem tipo, negativo e retorno rápido para zero. Validação completa: unittest 275/275; integração 30 aprovados + 1 skip opcional; cobertura 95/94/76/68%; mutation 58/58 mortas, sem sobreviventes/não aplicadas; Bandit 131 LOW e nenhum MEDIUM/HIGH. Ruff, compileall, actionlint, zizmor, auditorias AST/realidade/gaps e benchmarks oficiais passaram. POMDP v2: 128/128 dentro de `1e-12` (erro máximo `2.7755575615628914e-16`) e 4/4 controles inválidos rejeitados.
- **Fonte:** [SQLite Language Reference — SELECT, seção LIMIT](https://www.sqlite.org/lang_select.html), que especifica que valor negativo não impõe teto.
- **Limitação:** limites positivos altos continuam permitidos e podem selecionar todos os episódios existentes; esta decisão não cria teto de memória nem altera query/status.
- **Reversibilidade:** simples; remover a validação restaura o comportamento anterior, inclusive `LIMIT < 0` sem teto.

## D-027 — Fatos temporais como livro-razão de eventos encadeados (proposta, branch isolada)

- **Status:** proposta em branch `feat/temporal-fact-ledger-2026-10-17`; não mesclada. Mesclar somente com autorização específica.
- **Decisão:** módulo novo `core/temporal_facts.py` (`TemporalFactStore`), somente biblioteca padrão, em vez de estender `RealKnowledgeGraph`. Fatos têm janela de validade `[valid_from, valid_to)` e `recorded_at`; invalidar é anexar um evento `close` a um log encadeado por SHA-256, nunca editar ou apagar. O estado atual é uma projeção do log e `known_at` reconstrói a crença passada.
- **Contexto:** `resolve_conflicts` nunca encontra versões múltiplas porque `add_entity` sobrescreve a entidade; o grafo não tem noção de validade nem de proveniência.
- **Motivo:** separar tempo do fato de tempo de registro permite correção retroativa sem reescrever o passado. Apenas predicados declarados `functional` substituem o valor anterior; os demais acumulam, para não destruir fatos multivalorados.
- **Evidência:** `test_temporal_facts_contract.py` cobre janelas semiabertas, `known_at`, inserção retroativa, retração, corroboração idempotente, round-trip de snapshot, adulteração/remoção/reordenação/truncamento, invariantes aleatórios e regressões para metadados JSON inválidos, watermarks após timestamp inválido e referências `superseded_by` incoerentes. A suíte completa mediu 303 testes aprovados; a integração mediu 30 aprovados e 1 skip opcional; cobertura combinada do módulo: 93% (piso 90%). Mutation testing específico para este módulo não foi medido nesta rodada.
- **Referência:** modelo bi-temporal do Graphiti/Zep (`valid_at`/`invalid_at` e `created_at`/`expired_at`), com a diferença de que aqui a detecção de conflito é determinística, por predicado funcional, e não feita por LLM.
- **Limitações:** o mais recente vence, sem arbitragem por confiança; truncamento final só é detectado com `expected_head` guardado fora do arquivo; consultas com `known_at` reconstroem a projeção (O(eventos)); ainda não há persistência SQLite nem integração com `RealKnowledgeGraph`.
- **Reversibilidade:** simples; o módulo é isolado e nenhum código existente o importa.

## D-028 — Aresta NetworkX acumula os tipos de relação do mesmo par de nós

- **Status:** proposta no mesmo branch.
- **Decisão:** `RealKnowledgeGraph.add_relation` acumula os tipos em `relations` na aresta do `DiGraph` (`relation` continua sendo a última, por compatibilidade); atributos estruturais prevalecem sobre propriedades homônimas do chamador.
- **Contexto:** `DiGraph` guarda uma aresta por par de nós; adicionar `mora_em` e depois `trabalha_em` entre os mesmos nós deixava 1 aresta com o atributo da última, enquanto `relation_index` registrava 2. O caminho de fallback (sem NetworkX) não tinha o problema.
- **Evidência:** `test_knowledge_graph_contract.py` cobre relações múltiplas e propriedades chamadas `relations`; os testes do caminho NetworkX passaram com NetworkX 3.6.1. A dependência está agora no lock de CI com hashes, de modo que o caminho `DiGraph` é exercitado no gate.
- **Limitação:** o grafo ainda não distingue relações repetidas por tempo; isso é escopo de D-027.
- **Reversibilidade:** simples; remover o acúmulo restaura o comportamento anterior.

## D-029 — Contador de testes derivado do manifesto (`tests: CURRENT_TESTS`)

- **Status:** proposta na branch `fix/handoff-counter-decoupling` (hipótese operacional até revisão humana).
- **Decisão:** os blocos `NEXUS-CURRENT-STATE` declaram `tests: CURRENT_TESTS`; `tools/state_consistency_audit.py` resolve a sentinela para `verification.tests.executed` do manifesto da mesma execução, como já fazia com `commit: HEAD` e `ci_run: CURRENT_RUN`. Cada sentinela passa a valer só no próprio campo; números literais continuam exigindo igualdade exata.
- **Contexto:** todo PR que adicionava teste editava `tests: N` em quatro documentos; PRs independentes conflitavam entre si (ex.: #52 × #53), embora cada um estivesse limpo contra `main`.
- **Alternativas excluídas:** tolerância `>=` (aceita contagem desatualizada e deixa de detectar remoção de testes); fonte única em arquivo versionado (desloca o conflito para esse arquivo); fragmentos `handoff/` sozinhos (não removem o contador exigido pelo gate). Fragmentos podem ser adotados depois, de forma independente.
- **Evidência:** `test_state_counter_sentinel_contract.py` (6 testes) falhou com 12 falhas antes da correção e passa depois; suíte completa 309/309 e `state_consistency_audit.py` aprovado localmente em 21/10/2026 (Python 3.13.5; CI hospedado não verificado).
- **Limitação:** o número de testes deixa de ser legível nos documentos; a fonte passa a ser o artefato `nexus-state-manifest` do CI. A garantia do gate (claims atuais coerentes com a execução, execução aprovada) é preservada; ele continua não provando validade semântica dos testes. `bandit_low` continua literal (muda raramente).
- **Reversibilidade:** simples; reverter o commit restaura literais.
## D-029 — Exceção B101 somente para guarda direta do módulo

- **Decisão:** `tools/embedded_assert_audit.py` procura a comparação convencional `__name__ == '__main__'` somente em `ast.Module.body`, preservando o predicado existente e as faixas de linhas do corpo positivo. Guardas em função, função assíncrona, classe ou outro bloco não criam exceções. A busca de todos os `ast.Assert` continua recursiva e independente dos findings Bandit.
- **Contexto:** em `d435083bb4c893a58128cee6cfa00f65bba7fe02`, `ast.walk` aceitava uma guarda dentro de `helper()` como demonstração do módulo, deixando `assert False` sem finding passar com `ok=True`.
- **Evidência:** sete novos métodos de regressão, incluindo nove contextos aninhados com/sem B101 e com/sem demonstração legítima adicional; o mutante `embedded_assert_only_module_level_guards` restaura `ast.walk` e é morto. Comandos, resultados locais e limites estão em `CONTINUATION.md` e `NEXUS_EVIDENCE.md` na rodada de 2026-10-20.
- **Limites:** verificação sintática de localização, não análise de fluxo, segurança dos asserts ou validação do schema/filename do relatório Bandit. Conteúdo dentro da faixa de uma guarda legítima mantém a exceção anterior; `else` fica fora. Não se ampliam as formas de comparação aceitas, nem se altera a triagem separada de escopos de `tools/security_triage.py`.
- **Reversibilidade:** simples; uma troca de iteração, acompanhada das regressões e do mutante. Não muda workflow, núcleo, dependências ou API pública.


## D-030 — Capacidades de working memory usam o protocolo de índice

- **Decisão:** `PhonologicalLoop` e `VisuospatialSketchpad` rejeitam `bool` built-in e aceitam outros objetos apenas quando `operator.index(capacity)` dá índice positivo. O valor fica normalizado por `int(index(capacity))`, garantindo `int` built-in inclusive antes de Python 3.10. `TypeError` vale para objetos sem protocolo/retorno inválido; `ValueError` para zero/negativos. Exceções próprias de `__index__` propagam-se. Defaults 7/4 e eviction para capacidades positivas permanecem.
- **Motivo e compatibilidade:** na base `cb74cad7cfcd0ab536178ed984989ef2425c8211`, os construtores guardavam valores sem validação e as inserções os comparavam numericamente. Verificação anterior observou `True`, float integral, `IntEnum`, `Fraction`, `Decimal` e `numpy.int64` funcionando como limite 1; `float('inf')` não evictou os itens testados. Zero/negativos construíam, mas a primeira inserção falhava em `min` sobre buffer vazio. A nova regra rejeita bool e operandos que só são comparáveis numericamente; isso é quebra deliberada, sem evidência de consumo externo.
- **Pesquisa e escolha:** a documentação oficial de [operator.index](https://docs.python.org/3/library/operator.html#operator.index) diz que desde Python 3.10 o retorno é `int` exato; o wrapper explícito mantém essa garantia em runtimes anteriores. [Bool](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool) é subclasse de `int`, por isso tem exclusão separada. As notas oficiais do [NumPy 1.14](https://numpy.org/doc/1.14/release.html) documentaram a indexação de `np.bool_` como depreciada. Assim, `numpy.bool_` acompanha o protocolo da versão instalada: se `operator.index` falha, o teste exige `TypeError`; índice positivo é aceito e normalizado, enquanto índice zero/negativo exige `ValueError`. Isso difere de `bool` built-in, sempre rejeitado. Não foi adicionada dependência.
- **Evidência:** os blobs Git de implementação e teste são `43a306ff04c24fdc361791ab210ab9ddbe91f060` e `b367244f9e64382c47344f9473ced9754b8c99d6`, obtidos por `git hash-object`; o diff funcional+teste tem SHA-256 reprodutível `5c9b4060948c2173e73f07284727271acf8d8a8879435951d0d52c8193325e9a`. As suítes completas foram repetidas sobre esses blobs em Python 3.12.3 e 3.11.16: 637 testes em cada, 635 aprovados, 2 skips, exit 0. A classe focal passou 7/7; a suíte inclui `test_pre_registered_checks_match_the_frozen_fingerprint`, que chama `eval_working_memory()` e verifica WM1–WM6. Coverage branch focalizado reportou 43% combinado para o módulo inteiro (não cobertura alta); três mutantes dirigidos foram mortos pelos testes de `bool`, limite zero e normalização. Ruff 0.17.1, compileall e `git diff --check` passaram. A CLI separada `python tools/cognitivo_eval.py --capability working_memory` não foi reexecutada. O caso `numpy.int64` tem skip quando NumPy falta, mas esse ramo não foi exercitado; comandos e limites estão em `CONTINUATION.md` e `NEXUS_EVIDENCE.md`.
- **Limitação:** o CI declara Python 3.11 nos jobs e `pyproject.toml` define alvo Ruff 3.11, mas não `requires-python`; não se afirma um runtime mínimo suportado. Reatribuições continuam permitidas, com validação e normalização; reduzir capacidade não trunca o buffer imediatamente. `episodic_capacity` fica fora da mudança. Uso externo e sentinela zero são desconhecidos. Não foram executados CI hospedado, coverage completo do gate, mutação agregada, demo integrado, Bandit ou auditoria de realidade nesta rodada.
- **Reversibilidade:** simples; remover a normalização compartilhada dos dois construtores restaura a aceitação sem validação, sem alterar as políticas de eviction.

### Snapshot histórico de D-030 — reatribuição validada — 2026-10-26 (superado)

- **Decisão final:** manter `capacity` reatribuível nas instâncias de `PhonologicalLoop` e `VisuospatialSketchpad`, mas validar e normalizar cada atribuição pelo mesmo setter usado na construção. `bool` built-in e objetos que não satisfaçam `operator.index` geram `TypeError`; índices zero/negativos geram `ValueError`. A validação termina antes da escrita, portanto uma atribuição rejeitada conserva a capacidade anterior. Valores positivos continuam ajustáveis e são armazenados como `int` built-in.
- **Efeito em buffers existentes:** reduzir a capacidade não remove itens imediatamente. Inserções futuras continuam usando as políticas de eviction preexistentes; nenhuma mudança é feita em `episodic_capacity` ou em outras classes com atributo `capacity`.
- **Resposta ao parecer secundário:** a revisão documental do ChatGPT apontou a reatribuição não validada, mas não executou testes nem consultou o checkout. Essa observação motivou esta decisão; não é evidência de execução independente.
- **Evidência atual:** blobs `fb5bcab566183b7a283a37249d2bda1d0b4a79c2` (`working_memory.py`) e `33be5dbbdace47858fcb542fe6d3687e2a97a1aa` (`test_working_memory_contract.py`); SHA-256 reproduzível do diff funcional/testes `88585eb3b8d01fd8c36f888b9bf57644400d875837d369d67ccaf69304b97a7`. Testes focados do arquivo: 18/18 em Python 3.11.16 e 3.12.3; suíte completa: 641 executados, 639 aprovados, 2 skips, exit 0 em cada runtime. Ruff 0.17.1, compileall e `git diff --check` passaram; os detalhes estão em `NEXUS_EVIDENCE.md` e no pacote de revisão.
- **Teste complementar de protocolo:** `test_capacity_reassignment_propagates_index_error_atomically` cobre ambos os buffers preenchidos e confirma que a mesma instância de `CapacityError` lançada por `__index__` chega ao chamador, sem alterar a capacidade normalizada, a lista do buffer ou a identidade dos itens. Foi aprovado nos dois runtimes; nenhuma lógica de produção foi alterada nesta atualização. É cobertura guardião do contrato já implementado, não um RED alegado para o setter atual.
- **Histórico separado:** os 43% de coverage focalizado e os três mutantes dirigidos descritos anteriormente pertencem ao snapshot anterior; não foram reexecutados para o snapshot desta atualização e não devem ser interpretados como seus resultados.
- **Limites:** CI hospedada, coverage integral do gate, mutation gate agregado, CLI autônoma de WM1–WM6, demo integrado, Bandit e auditoria de realidade não foram executados. A compatibilidade de consumidores externos e sentinela zero permanece desconhecida. `numpy.bool_` segue o protocolo da versão NumPy; avisos de depreciação sob `-W error` e o caminho sem NumPy não foram testados neste ambiente.


### Snapshot histórico de D-030 — revisão DeepSeek e CLI — 2026-10-26 (superado)

- **Parecer independente:** o parecer do DeepSeek examinou somente o pacote documental, sem acesso ao checkout ou execução de comandos. Veredicto **não bloqueia**; nenhum achado bloqueante ou médio. As ressalvas baixas registradas são o possível `DeprecationWarning` de `numpy.bool_` sob `-W error`, riscos desconhecidos de consumidores externos com `vars(obj)['capacity']`, `del obj.capacity` ou pickle/copy preenchendo `_capacity`, e o custo de navegar snapshots históricos rotulados. Os 43% de coverage e três mutantes permanecem evidência apenas dos snapshots anteriores ao setter.
- **Python 3.10 e `zip(strict=True)`:** os dois usos novos nos testes exigem Python 3.10+ nesses callsites; a checagem de comprimento é redundante diante dos asserts de igualdade/comprimento e dos dois itens construídos no cenário. Não foram removidos, pois o escopo autorizado limita esta atualização a registros e pacote. A base já continha uma ocorrência de produção em `core/cognitivo/predictive_coding.py:242`; assim, não se atribui ao patch uma exigência exclusiva de Python 3.10. `requires-python` continua não declarado e Python 3.10 continua não testado; não se afirma suporte nem se muda a política mínima.
- **Execução atual:** após a revisão, no branch `fix/working-memory-capacity-validation-20261026`, HEAD/base `cb74cad7cfcd0ab536178ed984989ef2425c8211`, foi executado `PYTHONPATH=. NEXUS_AUDIT_DB=/tmp/nexus-wm-deepseek-cli-20261026.sqlite3 python tools/cognitivo_eval.py --capability working_memory`. Criteria v2: WM1–WM6 passaram, `failed: []`, exit 0. A saída avisou `psutil not available` e `FAISS/SentenceTransformers not available - using keyword fallback`; são dependências opcionais não necessárias para estes seis critérios. WM3 reporta relógio de parede, sem evidência de generalização estatística. Nenhum arquivo rastreado foi criado.
- **Histórico e alcance:** afirmações anteriores de que a CLI não foi executada referem-se aos snapshots/checkpoints anteriores; este registro documenta e substitui essa limitação para o snapshot corrente. Não foram executados outros gates nem alterados código/testes nesta atualização. Consumidores externos, comportamento sob outras versões de NumPy e compatibilidade Python 3.10 permanecem sem verificação.


### Snapshot histórico de D-030 — compatibilidade pickle/copy legado — 2026-10-26 (superado)

- **Decisão:** manter a validação de capacidade e migrar somente o estado de dicionário padrão das duas classes, quando contém `capacity` mas ainda não contém `_capacity`. Validar com `_positive_capacity` antes de gravar qualquer parte do estado; converter para `_capacity` e remover a chave antiga. Estado atual com `_capacity` não é revalidado. `__setstate__` compartilhado preserva os outros campos e as referências/aliases que pickle/copy normalmente preservam.
- **RED observado:** o teste novo, executado antes da implementação, reproduziu `AttributeError: ... has no attribute '_capacity'` em PhonologicalLoop e VisuospatialSketchpad ao desserializar estado legado.
- **Regressões e verificação:** as regressões cobrem pickle legado válido, pickle atual, `deepcopy`, preservação de buffer/atributos e alias por identidade, além de `TypeError` para bool/não indexável, `ValueError` para zero/negativos e ausência de atualização parcial. O arquivo focal passou 21/21 em Python 3.11.16 e 3.12.3. As suítes completas passaram 644 testes em cada runtime (642 aprovados, 2 skips). Ruff 0.17.1, compileall e `git diff --check` passaram; comandos completos e resultados estão na atualização corrente de `CONTINUATION.md` e `NEXUS_EVIDENCE.md`.
- **Identidade do snapshot:** blob de produção `83808703575a3e4a8f8662117c9872642ea2828d`, blob de teste `22da97bcb2ecea6efe2c9b3bde980d5a7eae006b` e SHA-256 do diff código+teste `b4a8c3e8c8198b8683a298e13710334c1f89a62ae9b21aa2759d9821187a1180`.
- **Limites:** não se promete restauração manual via `vars`, escrita direta de `_capacity`, deleção de `capacity`, serializadores/formatos customizados nem segurança de pickle com entrada não confiável. Consumidores externos são desconhecidos. Nenhum efeito em eviction foi introduzido. Não foram executados coverage/mutation agregados, Bandit ou CI hospedada.
- **Proveniência da revisão:** a revisão final solicitada é somente leitura por Nexus Segurança, UID canônico `DBss8wRFjMUsGk57YtaVom`. Não se infere parecer ou aprovação até que a resposta real seja recebida e registrada.


### Snapshot histórico de D-030 — tuple-state com `__slots__` — 2026-10-26 (superado)

- **Decisão:** além do `dict` exato já suportado, aceitar somente o formato de estado padrão `tuple` exata com `(dict_state, slot_state)`, em que cada parcela é `dict` exato ou `None`. Validar shape, chaves e nomes de slots antes de modificar o receptor. Migrar `capacity` legado na parcela de dicionário para `_capacity` validada e normalizada antes de qualquer atualização; remover a chave antiga. Aplicar valores de slots com `setattr`, preservando descritores e aliases entre as duas parcelas. Rejeitar estruturas não suportadas com `TypeError` claro; não prometer formatos arbitrários/customizados.
- **Motivo:** foi reproduzida a falha de pickle/deepcopy em subclasses de ambos os buffers com `__slots__`, cujo estado padrão é uma tupla de dicionário e slots. Invalid legacy capacity precisa falhar antes de aplicar tanto o dicionário quanto os slots. `MemberDescriptorType` restringe as chaves de slot a descritores reais de slots.
- **RED/GREEN:** quatro testes novos falharam antes da correção em Python 3.12: tuple-state atual/legacy falhava com `TypeError` nos dois buffers e os casos de capacidade zero/negativa no tuple-state não produziam `ValueError`. Depois da correção, o arquivo focal passou 25/25 em Python 3.11.16 e 3.12.3; suítes completas passaram 648 testes em cada runtime (646 aprovados, 2 skips). Ruff 0.17.1, compileall e `git diff --check` passaram. Comandos estão em `CONTINUATION.md` e `NEXUS_EVIDENCE.md`.
- **Identidade:** blob de produção `b3588ef478088e986e04a169142c955d579e479b`; blob de testes `fed9fbf0654eb5088a4d4371a859510e2ac772ba`; SHA-256 do diff funcional+testes `14222dc9a5b05ffa271a33e23b29c3df16d37dfde5a262ac9e2ac4e3249715db`.
- **Limites e revisão:** o suporte é limitado aos estados padrão `dict` e `(dict, dict)` / parcelas `None`; não cobre hooks de serialização customizados ou pickle não confiável. Coverage/mutation agregados, Bandit e CI hospedada não foram executados. Solicita-se exatamente uma revisão somente leitura de Nexus Segurança, UID canônico `DBss8wRFjMUsGk57YtaVom`, sobre o hash contextual final. A resposta ficará pendente até recebimento efetivo; nenhum parecer é presumido.


### Checkpoint intermediário de D-030 — capacidade em dict-state — 2026-10-26 (superado)

> Este checkpoint validou a capacidade quando armazenada no dicionário do estado, mas não quando uma subclasse declara `_capacity` como slot. A lacuna foi reproduzida e corrigida no adendo final abaixo; as contagens e hashes deste checkpoint não representam a revisão atual.

- **Decisão vigente neste snapshot:** durante `__setstate__`, revalidar e normalizar `_capacity` atual pelo mesmo helper usado na construção; `bool`/valor sem `operator.index` válido dão `TypeError`, zero/negativo dão `ValueError`. Se o estado trouxer simultaneamente `capacity` legado e `_capacity` atual, aceitar somente quando ambos se normalizam ao mesmo inteiro positivo; conflito resulta em `ValueError`. Validar antes de atualizar o dicionário do objeto ou slots, e remover a chave legada após a migração.
- **Racional:** confiar no campo atual deixava uma lacuna se o estado de pickle/copy estivesse malformado ou ambíguo. Rejeitar conflito em vez de atribuir precedência evita que a origem do valor restaurado dependa da ordem/forma dos campos.
- **Base normativa e escopo:** a [documentação Python de `pickle`](https://docs.python.org/3/library/pickle.html) define estados padrão em dict ou em tupla de dict+slots para a restauração comum e informa que `__setstate__` customizado pode receber qualquer objeto; ela não prescreve a validação de capacidade nem exige este formato estreito. A aceitação apenas dos estados padrão dict/tuple é uma decisão local, não regra universal do protocolo. Não se promete compatibilidade com hooks/formas arbitrários nem segurança para pickle não confiável.
- **Evidência:** as três regressões RED deram 32 falhas de subcasos antes do ajuste; depois, o foco passou 28/28 e a suíte completa 651/649/2 skips em Python 3.11.16 e 3.12.3. Ruff, compileall e `git diff --check` passaram. Identidade, hashes e comandos exatos estão no novo adendo de `CONTINUATION.md` e `NEXUS_EVIDENCE.md`.
- **Limites:** snapshots prévios e pareceres deles não cobrem este diff; a revisão read-only específica de Nexus Segurança continua pendente de encaminhamento/recebimento. Não foram executados coverage/mutation agregados, Bandit ou CI hospedada. Sem alteração de eviction, API de inserção, outras classes ou claims de capacidade cognitiva.


### Snapshot histórico de D-030 — capacidade dict/slot — 2026-10-26 (superado)

- **Decisão vigente:** validar todos os valores encontrados em `capacity` legado, `_capacity` no dicionário e `_capacity` em `slot_state`; aceitar somente se todos normalizarem ao mesmo inteiro positivo. Rejeitar valores inválidos (`TypeError`/`ValueError`) ou inconsistentes (`ValueError`) antes de atualizar o receptor. Se a classe declara o descritor de slot `_capacity`, armazenar nele o valor normalizado; em caso contrário, usar o dict.
- **Evidência da lacuna:** subclasses pickláveis das duas classes com `__slots__ = ("_capacity", "label")` geram estado padrão tuple em que `_capacity` está em `slot_state`. Uma reprodução observou que o checkpoint intermediário aceitava `True` e ainda aplicava o restante do dict/slots; também ignorava divergência entre o slot `_capacity` e `capacity` legado. Três regressões falharam em 18 subcasos antes da extensão.
- **GREEN:** foco 31/31 e suíte integral 654 executados, 652 aprovados e 2 skips em Python 3.11.16 e 3.12.3; Ruff 0.17.1, compileall e `git diff --check` passaram. Hashes finais constam na atualização final de `CONTINUATION.md` e no pacote Claude congelado.
- **Base normativa e limites:** [documentação oficial Python de `pickle`](https://docs.python.org/3/library/pickle.html) descreve o estado padrão dict/slots e deixa claro que `__setstate__` customizado pode receber objeto arbitrário. A restrição a essas formas, a validação e a escolha de slot como armazenamento canônico são política local. Não se promete suporte a formatos customizados ou segurança para dados pickle não confiáveis.
- **Revisão:** nenhum parecer anterior cobre o hash contextual final. A revisão read-only de Nexus Segurança canônico `DBss8wRFjMUsGk57YtaVom` permanece pendente de solicitação/retorno via coordenador. Sem stage, commit, push, PR, merge ou alteração de `main`.


### Snapshot histórico de D-030 — state restaurado exige capacidade — 2026-10-26 (superado)

- **Decisão vigente:** `_restore_working_memory_state` rejeita, antes de qualquer atualização do receptor, dict ou tuple-state que não contenha uma fonte de capacidade reconhecida: `capacity` legado no dict, `_capacity` no dict ou `_capacity` nos slots. A ausência gera `ValueError` explícito. Valores presentes continuam normalizados como inteiros positivos e todas as fontes simultâneas precisam concordar; migração para slot quando declarado e aliases permanecem preservados.
- **Motivo e base normativa:** o parecer mais recente de Nexus Segurança encontrou que um state sem capacidade era aplicado e só falhava como `AttributeError` na leitura posterior. A [documentação oficial de `pickle`](https://docs.python.org/3/library/pickle.html) permite estado arbitrário quando existe `__setstate__`; exigir capacidade é, portanto, uma invariante específica das classes Nexus, não uma regra universal do protocolo. A rejeição é preferida a aplicar um estado que deixe o objeto inválido.
- **Evidência:** o novo teste falhou antes da implementação em quatro subcasos (dois buffers em estado dict e dois subtipos com `_capacity` em slots e tuple-state) porque o erro explícito não ocorria. Depois: foco 32/32 e suíte integral 655 executados, 653 aprovados, 2 skips em Python 3.11.16 e 3.12.3; Ruff 0.17.1, compileall, `git diff --check` e CLI WM1–WM6 passaram. O gate cognitivo completo matou 185/185 mutantes, incluindo `wm_restore_requires_capacity`; comandos e resultados exatos estão no adendo atual de `CONTINUATION.md` e `NEXUS_EVIDENCE.md`.
- **Identidade:** blob de produção `d603545c2be31e80939246df4c40ebdb7026611a`; blob de testes `2ac89ce3b59d7946c18c9c2c4c569860c299433c`; blob de `tools/cognitivo_mutation.py` `5da09d709b0b38238a048ca09d8bc1062281ad45`; SHA-256 do diff código+testes `b3b2128458a7d143d4478d63821018fe9ac688ad0a0a6e25f59e28823b4ce37f`.
- **Limites e revisão:** os resultados locais são contratos de software, não prova de capacidade cognitiva ou segurança geral. Pickle de origem não confiável, formatos/hooks customizados e compatibilidade com consumidores externos não foram validados. CI hospedada, Bandit e coverage agregado não foram executados. A revisão anterior do hash contextual `3b9af345e7af00f6083bbb23c233676d9fc29e1d4951536f96eb7d343963c67f` não cobre esta guarda; uma revisão read-only nova de Nexus Segurança canônico `DBss8wRFjMUsGk57YtaVom` permanece pendente. Sem stage, commit, publicação ou alteração de `main`/claims/relatórios reservados.


### Atualização vigente de D-030 — validação de estado e limites de compatibilidade — 2026-10-26
- **Decisão:** manter o contrato atual do restaurador e não alterar sua lógica. Acrescentar regressões atômicas para chaves não-string no dict/slot state e para slots não declarados, além de mutantes direcionados para cada guarda. Rejeitar subclasses arbitrárias que sobreponham descriptors de slots não é uma política nova; agora o limite também aparece na docstring do módulo.
- **Evidência de teste:** foco 34/34 em Python 3.12.3 e 3.11.16 após a correção final da docstring. Suíte completa no snapshot final de código/testes: 657 executados, 655 aprovados e 2 skips em ambos os runtimes (exit 0). `ruff check` e compileall nos dois runtimes passaram. Diagnóstico de formatação: `ruff format --check` falha nos mesmos três arquivos tanto em `HEAD` quanto na worktree; o workflow versionado executa `ruff check`, não o formatter. Entre 14 ranges adicionados, 4 apresentam diff do formatter; os ranges recentes de docstring/espaçamento estão limpos. Sem reformat automático. Gate de mutação pós-docstring: 188/188 mortos, 0 sobreviventes, 0 não aplicados.
- **Identidade do código:** blob de produção `10d359a28661c08b3ea79de691c1c4323cd4d1e9` (docstring e linha em branco; lógica igual), teste `c77400ae493dd41fc518cd3e79cbc10c26d03f50`, catálogo `d3cbd8852739b91ddf6732f98cd398ae96d6fd43`; SHA-256 do diff produção+testes `7ff912705676089e5a2f1a5c4e96dd0d7d2b64e7f9d6d9e9d854e3d4be90f10b`; catálogo `444da76fc10b8a1d961291c86a18a76287d577154d312afe8d1d62501a28bdf9`.
- **Pickle/downgrade:** reproduções reais confirmam que pickle da base é lido pelo código atual, enquanto pickle atual carregado pelo leitor da base deixa as instâncias sem `.capacity` e lança `AttributeError` ao consultar. Sem mecanismo de migração inversa; não foram encontrados checkpoints internos versionados nem consumidores internos persistentes. Compatibilidade externa continua desconhecida.
- **Subclasses e limites:** reatribuir estado de subclasse com property lançando durante `setattr` pode causar atualização parcial quando o slot herdado é aceito pelo MRO. Override de slot `capacity` também pode sombrear a property validada. Ambos são limites de subclasses arbitrárias não suportadas e ficam documentados, sem política `__init_subclass__`/guard adicional. `float`, `Fraction`, `Decimal` e `inf` continuam limites de compatibilidade identificados; `requires-python` segue ausente e Python 3.10/outras versões não testados. Pickle não confiável permanece inseguro.
- **Proveniência de revisão:** o segundo parecer Claude (`6fc81d5b…`) encontrou o placeholder literal do hash, a formulação da contagem e a falta de nota sobre subclasses. O terceiro (`cd1135eb…`) tratou da linha em branco, precisão da docstring, rerun do gate de mutação e menção nos ledgers. O quarto (`3c3d629c…`) verificou hashes do texto do pacote, não executou comandos no repositório e apontou a comparação de formatação com `HEAD`, a atribuição imprecisa nos ledgers e a suíte final. Os ledgers vigentes agora corrigem a atribuição sem reescrever snapshots históricos; a comparação confirmou falha global já presente em `HEAD`, e a suíte completa foi repetida. `ruff format --check` é diagnóstico, não gate do workflow. Os pareceres foram estáticos e não bloqueantes; reproduções de pickle/property permanecem limites documentados, sem testes automatizados próprios.
