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

## D-025 — Um snapshot JSON limitado alinha a chave e a execução da ferramenta

- **Decisão:** para parâmetros `dict` de ferramentas cacheáveis, construir um snapshot limitado por valor somente de estruturas JSON estritas reconhecidas. O mesmo snapshot é usado para validação do schema, guards semânticos, cache key e executor. A ordem é snapshot → validação/guards → cache key → lookup → executor; por exemplo, o limite de tamanho de `math_calculate` deve rejeitar a entrada antes da serialização da chave.
- **Contexto:** formar a chave a partir de `params` mutável e depois validar/executar o objeto original permite que uma mutação entre essas fases associe resultado a uma chave que descreve outro valor.
- **Motivo:** tornar explícita a semântica por valor e remover a divergência temporal entre a representação cacheada e o valor efetivamente validado/executado, sem deixar que a otimização altere a execução de valores que não podem ser snapshotados com segurança.
- **Fallback:** ciclo, valor não JSON, subclass/container não reconhecido ou exceder os budgets faz bypass da cache e mantém validação/execução com o objeto original. Se a criação da chave falhar depois de um snapshot válido, somente lookup/store são ignorados; validação e execução continuam usando o snapshot.
- **Evidência:** `test_tool_cache_snapshot_contract.py` verifica snapshot idêntico nas três fases, mutação do original depois da preparação da chave, mutação in-place isolada, aliases compartilhados copiados como valores independentes e bypass fora do JSON estrito; `test_tool_cache_key_contract.py` cobre budgets/ciclos em `dict` e `list`. `test_math_calculate_resource_limits.py` protege a rejeição de expressão superdimensionada antes da cache key. Validação final local: suítes focadas 9/9; unittest 269/269; integração 30 aprovações e 1 skip opcional FAISS; cobertura 95/94/76/68%; mutation gate 53/53 mortas, 0 sobreviventes/não aplicadas; Bandit 131 LOW e 0 MEDIUM/HIGH, triagem verificada; Ruff, compileall, actionlint, zizmor, auditorias AST/realidade/gaps e benchmarks de capacidade/POMDP v2 passaram.
- **Limitações:** identidade/aliasing e mutações in-place do objeto original não são preservados. A travessia não sincroniza o produtor: substituição de valor com o mesmo tamanho durante a própria cópia pode produzir um snapshot misto. Depois de construído, esse mesmo snapshot é usado para chave, validação e executor; consumidores que precisem de um instante atomicamente consistente devem sincronizar a origem. O limite de cache não é teto de memória global.
- **Convivência de branches:** esta branch parte de `main` pós-PR #31. Na sua criação, a PR #32 permanecia aberta e propunha D-024; esta decisão recebe D-025 para evitar duplicação entre branches paralelas. Se a PR #32 for mesclada primeiro, reconciliar/rebasear esta branch e a sequência documental antes de qualquer merge.
- **Base técnica:** [documentação oficial Python 3.12 do módulo `json`](https://docs.python.org/3.12/library/json.html).
- **Reversibilidade:** fácil; entradas fora do snapshot limitado apenas deixam de obter cache hit/store e continuam no caminho normal de validação/execução.
