# Nexus — Ledger de evidências

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 312
bandit_low: 131
-->

Snapshot histórico: manifesto `nexus-state-manifest` do commit `5b8bed8125002e25eb066620af67d3a65d44e59f` (CI run `38907987512`), com 200/200 testes `unittest`, 31 verificações integradas inventariadas (30 aprovadas, 1 ignorada por FAISS opcional ausente e 0 falhas) e 131 achados Bandit LOW (0 MEDIUM/HIGH). Esse snapshot e os resultados locais do benchmark v1 não representam o estado atual.
A PR #24 partiu do `main` pós-PR #23 (`e3cc760597aa1782aef9879a8f88ee6f48032348`) e foi mesclada em `main` como `dd0a73c6e57dfcccf9b522b3d2dd3a4623cdc732`; os dois checks oficiais passaram. Seus resultados locais de 240/240 testes permanecem históricos; a branch seguinte registra a nova validação em seção própria.
Os checkpoints de 105 e 86 testes abaixo são históricos de marcos anteriores e não devem ser usados como estado atual.

| Área | Estado | Evidência verificável | Limitação |
|---|---|---|---|
| Fila | verificado no contrato | testes de prioridade, retry, capacidade, snapshot e atomicidade | sem carga distribuída ou exactly-once |
| Log imutável | verificado no contrato | triggers append-only, verificação da cadeia, detecção de adulteração/remoção, reabertura e escritores concorrentes | truncamento final ou rehash por quem controla o arquivo exige âncora externa |
| Detector de jailbreak | testado por regressão EN/PT | corpus benigno/malicioso, NFKC, caracteres invisíveis, homoglifos, leetspeak e entradas longas | heurística, não fronteira de segurança semântica nem substituto de controle de privilégio |
| Diagnósticos | verificado no contrato | schema versionado, validação, dashboard e fuzzing de payloads JSON | estrutura não prova correção cognitiva nem cobre entradas não-JSON arbitrárias |
| Memória SQL | parcial/verificado no fallback | testes adversariais e queries parametrizadas | busca semântica opcional não estava disponível |
| Memória episódica | verificado no contrato | estatísticas do SQLite, ciclo de vida concorrente, validação de `memory_id` quando há backend resolutor e recuperação por provedor falsey | sem backend, o ID é uma referência opaca não validada; sem locking distribuído ou exactly-once |
| CentralRouter | verificado no contrato | contadores/cache e enqueue de emergência mesmo com marcador reservado no contexto do caller | cenários internos; o teste simula a pressão VRAM |
| Swarm | verificado no contrato | consenso/diversidade derivados de deliberações | não demonstra inteligência coletiva geral |
| Multimodalidade | parcial | protocolos de backend e fallbacks testados | sem backend não há percepção/transcrição real |
| Atenção bottom-up | parcial/verificado no contrato | contraste e movimento opcionais, saliência limitada e fallbacks testados | sem sinais observados, usa heurística; não é percepção biológica |
| Integração/orquestração | parcial/verificado no contrato | média de sinergia calculada sobre integrações registradas | score individual ainda é heurístico por nomes; não demonstra inteligência emergente |
| Grafo de conhecimento | parcial/verificado no contrato | entidades únicas por ID, média de confiança e índice de tipo consistente | sem consistência distribuída ou validação externa do conhecimento |
| Ledger temporal | verificado nos contratos; integração proposta nesta branch | janelas semiabertas, `known_at`, correções retroativas, replay/hash-chain e mapeamento opt-in de relações com proveniência | ainda sem persistência SQLite ou ingestão automatizada das alegações documentais; sem transação distribuída com o grafo |
| Planejamento MCTS | parcial/verificado no contrato | `None` usa ações padrão e lista vazia permanece vazia | recompensas e rollouts continuam heurísticos; não é prova de planejamento geral |
| Telemetria | parcial/verificado | Vita, audit log SQLite serializado e guard de VRAM | dependente do ambiente; sem locking distribuído ou exactly-once |
| Segurança | parcial | Bandit 131 LOW, triagem, auditorias AST e testes negativos | não substitui threat model ou revisão manual completa |
| CI | verificado por commit | run associado ao `HEAD` e manifesto retido como artefato quando disponível | qualquer novo commit exige leitura do manifesto correspondente |
| Benchmark sintético | verificado localmente nos protocolos v1 e v2 | v1 preserva as métricas composicional/causal/incerteza; v2 mede 128 modelos explícitos, erro absoluto máximo `2,7756e-16`, 128/128 casos dentro de `1e-12` e 4/4 controles inválidos rejeitados | trilha POMDP do v1 continua não suportada; v2 não aprende modelos nem escolhe políticas; não valida teorias cognitivas |
| Autonomia geral | não demonstrada | nenhuma evidência válida no repositório | não fazer esse claim |
| AGI/ASI/consciência | não demonstradas | explicitamente negadas pela documentação | nomes e banners não contam como evidência |

## Integração temporal opt-in do grafo — validação local em 2026-10-17

- `RealKnowledgeGraph` aceita um `TemporalFactStore` opcional. Quando fornecido, `add_relation` mapeia origem/tipo/destino, confiança, fonte e `valid_from`; predicados funcionais não são inferidos e precisam estar declarados no store.
- O registro temporal ocorre antes de criar nós ou alterar índices, rejeições não deixam estado parcial no grafo, e `properties` não é serializado implicitamente como metadata. As projeções e `find_relations` legados continuam preservados.
- Evidência: 309/309 testes unittest; suíte integrada 30 aprovados + 1 skip opcional; contratos focados 11/11; cobertura 100% em linhas e branches de `RealKnowledgeGraph.__init__` e `add_relation`; 62/62 mutantes direcionados mortos; Ruff passou. Os checks oficiais do run `39015023676` passaram 2/2 no commit de implementação `865063d21de9e986f9a502a790bfac2b68f7f801`; consulte a PR #38 para o estado do HEAD mais recente.
- A fonte literal de cada fato ainda é fornecida pelo chamador; esta mudança não importa automaticamente medições do resumo de CI para o ledger. Persistência SQLite é a próxima etapa separada.

## Verificação local da branch de correção — 2026-10-16

- Os contratos recém-adicionados reproduziram antes do patch: resposta `deferred` sem ID/enqueue quando o contexto continha `_deferred_reprocess=True`; link a memória inexistente aceito e contado; provedor resolutor falsey tratado como ausente. Após o patch, os contratos focados passaram.
- Suíte `unittest`: 213 executados, 213 aprovados, 0 falhas, 0 skips e 0 sucessos inesperados. Suíte integrada: 31 IDs inventariados, 30 aprovados, 1 skip FAISS opcional, 0 falhas.
- Cobertura: fila 95%, VRAM 94%, Vita 76% e núcleo 67%; mutation testing: 28/28 mortos, 0 sobreviventes/não aplicados; Bandit: 131 LOW e 0 MEDIUM/HIGH.
- Ruff, actionlint, zizmor, compileall, auditorias AST e checks de requirements passaram localmente. `AUDIT_REALITY_REPORT.md` e `audit_reality.json` foram regenerados e refletem 26.946 linhas e 859 funções no núcleo.
- POMDP continua não suportado e sem score no protocolo v1; não foi inferido posterior nem alterado o resultado do benchmark.

## Autoauditoria de recursos do `RealToolUse` — 2026-10-16

- Antes do patch, `execute_tool("math_calculate", {"expression": '"x" * 1000000'})` retornou sucesso com 1.000.000 caracteres e reteve o valor no cache; `2 ** 10000` também retornou sucesso. Depois de marcar uma entrada expirada e fazer outra chamada, a entrada antiga continuou retida.
- O patch rejeita constantes não numéricas, expressões acima de 256 caracteres, AST acima de 64 nós, expoentes acima de 1024 e resultados fora dos limites numéricos; a expressão longa é rejeitada antes de serializar a chave de cache.
- O cache agora expurga TTL vencido, limita 128 entradas, 1 MiB de payload estimado e 64 KiB por resultado; armazena e devolve cópias profundas para impedir que o chamador mutile a entrada após a medição. Resultados grandes de outras ferramentas ainda são retornados ao chamador, mas não armazenados no cache.
- Contratos: 9 novos testes cobrem aritmética comum, string/boolean/NaN, comprimento/complexidade/expoente, rejeição antes da cache key, TTL, alias mutável, limite por resultado e capacidade total. Suíte: 222/222; integração: 30 aprovados/1 skip FAISS; cobertura: fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation: 28/28 mortos, 0 sobreviventes/não aplicadas; Bandit: 131 LOW, nenhum MEDIUM/HIGH.
- Limites remanescentes naquela rodada: `ToolDefinition.timeout_seconds` não interrompia executores; o cache limita payload estimado via `sys.getsizeof`, não a memória total do processo; o estado local aguardava CI oficial. A validação e telemetria advisory de overrun foram implementadas na rodada subsequente.

## Filtro de crença POMDP e benchmark v2 — 2026-10-16

- Reproduções na versão inicial do filtro: 129 estados aceitos, `OverflowError` ao somar probabilidades enormes inválidas e falsa evidência zero com likelihoods ou produtos prior–transição extremamente pequenos.
- Correção: validação de entradas em `[0,1]`, sequências compatíveis, prior normalizado, matriz quadrada estocástica, máximo de 128 estados e normalização posterior em log-space; inputs não são retidos nem mutados.
- Contratos locais: 10 testes do filtro e 8 do benchmark; suíte total 240/240, sem falhas/skips/XPASS; integrada 30 aprovações e 1 skip FAISS opcional. Cobertura: fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation 28/28 mortas, 0 sobreviventes/não aplicadas.
- Benchmark v2 local: 128 casos gerados pela seed `20261016`, 635 probabilidades, MAE `3.001891167837976e-17`, erro máximo `2.7755575615628914e-16`, 128/128 dentro da tolerância e quatro controles negativos rejeitados; taskset SHA-256 `232371d0e8e948f4b0a004261e3d271b1003aed8d0de8c931d3439f9d370d025`.
- A seed fixa do CI (`20261016`) passa explicitamente a precisão e os quatro controles negativos; teste injetando posterior incorreto verifica que a CLI grava o relatório e retorna código 1, cobrindo a falha fechada.
- O benchmark v1 continuou com status `unsupported_interface` para observação parcial e sua implementação/protocolo não foram alterados. As métricas detalhadas acima são locais; posteriormente, a PR #24 foi mesclada após ambos os checks oficiais passarem.

## Categorias de resultado

- **Fato:** número ou comportamento reproduzido por comando.
- **Evidência:** contrato ou execução que sustenta apenas o escopo descrito.
- **Inferência:** risco arquitetural derivado do tamanho e acoplamento do núcleo.
- **Hipótese:** benefício futuro de modularização ainda não medido.
- **Intenção:** itens do roadmap, não funcionalidades presentes.

## Autoauditoria de `ToolDefinition.timeout_seconds` — 2026-10-16

Base local: `main` pós-PR #24, commit `dd0a73c6e57dfcccf9b522b3d2dd3a4623cdc732`; branch isolada `feat/post-pr24-contract-hardening-2026-10-16`.

Antes da correção, o registro aceitava timeout zero, negativo, booleano, float e `NaN`; um valor string podia lançar `TypeError`. A execução síncrona não consultava o orçamento: uma revisão independente reproduziu um executor de cerca de 30 ms com budget de 1 ms retornando sucesso sem sinalizar o excesso.

O novo contrato exige inteiro não booleano em `[1, 60]`, no registro e antes de cada execução, inclusive para detectar mutação da definição depois do registro. O executor é cronometrado com `time.monotonic()`; ao terminar além do budget, por retorno ou exceção, emite `TOOL_EXECUTION_OVERRUN`. Resultados bem-sucedidos são preservados, exceções continuam falhas estruturadas, e o evento informa `execution_completed=true` e `cancelled=false`.

Os seis testes em `test_tool_definition_contract.py` cobrem limites inclusivos, valores inválidos, mutação pós-registro, execução dentro do budget, overrun bem-sucedido e overrun com exceção; cinco mutantes específicos foram adicionados ao mutation gate. Validação local Python 3.12.3: 246/246 testes; suíte integrada 30 aprovações e 1 skip FAISS opcional; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation 33/33 mortas, 0 sobreviventes/não aplicadas; Ruff, actionlint, zizmor, compileall e auditorias AST passaram; Bandit 131 LOW, nenhum MEDIUM/HIGH.

O evento é diagnóstico pós-execução, não deadline nem cancelamento. A documentação oficial de `Future.cancel()` informa que chamada em execução não é cancelada; `Process.terminate()` pode corromper pipes/queues ou tornar locks indisponíveis: https://docs.python.org/3.12/library/concurrent.futures.html e https://docs.python.org/3.12/library/multiprocessing.html. Interrupção segura segue exigindo um contrato cooperativo ou runner isolado.

Os números são locais; a CI do head candidato ainda deve confirmá-los.


## Autoauditoria da fronteira de `RealToolUse.execute_tool` — 2026-10-16

Base local: `main` pós-PR #25, commit `675f5aa95416c3a92998a34dd2f905f529b90980`; branch isolada `feat/post-pr25-hardening-2026-10-16`.

Antes da correção, a chamada `execute_tool("list_operations", None)` reproduziu `TypeError("argument of type 'NoneType' is not iterable")` dentro de `validate_parameters`, antes do `try` do executor; portanto a exceção escapava em vez de virar o `ToolExecutionResult` de falha previsto. A reprodução local também confirmou que um dicionário vazio continua produzindo diagnóstico de parâmetro obrigatório ausente e que a forma válida executa normalmente.

A guarda inicial agora exige `isinstance(params, dict)` e registra `TOOL_PARAM_TYPE_MISMATCH` (`parameter=params`, `expected=dict`, `actual=<tipo>`), fazendo a chamada retornar `success=false`, `error="Parameter validation failed"` e sem invocar o executor. Três novos testes verificam tipos externos inválidos, dicionário incompleto e dicionário válido; um mutante que desativa a guarda é detectado.

Validação local Python 3.12.3: 249/249 testes; integração 30 aprovações/1 skip FAISS; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation gate 34/34 mortas, nenhum sobrevivente/não aplicado; Ruff, actionlint, zizmor, compileall, auditorias AST, capability benchmark e POMDP v2 passaram; Bandit 131 LOW, nenhum MEDIUM/HIGH. Os dois checks oficiais da PR #26 passaram antes do merge em `main` no commit `508020d77d5705f99c76eaca4df05e8e265662f7`.

## JSON estrito e falha da chave de cache — 2026-10-16

Base local: `main` pós-PR #26, commit `508020d77d5705f99c76eaca4df05e8e265662f7`; branch `feat/post-pr26-tool-input-boundary-2026-10-16`.

Estado posterior: a alteração foi publicada como PR #27; os dois checks oficiais passaram e foi mesclada em `main` no commit `c4375d9f42703b11d2632b3a701303c5a69563a4`.

Antes da alteração, `format_json` retornava sucesso com texto contendo `NaN`, `Infinity` e `-Infinity`, pois a serialização Python usava o valor padrão permissivo de `allow_nan`. Também reproduzi `TypeError: Object of type object is not JSON serializable` escapando de `execute_tool` ao serializar a chave do cache antes do `try` do executor. O exemplo usou parâmetros válidos de `list_operations` e uma extensão extra ignorada pelo schema; uma ferramenta marcada como não cacheável também tentava construir uma chave.

O formatador agora define `allow_nan=False`; os valores não finitos se tornam falhas estruturadas pelo tratamento de exceções já existente. A chave é construída somente para ferramentas cacheáveis; `TypeError`, `ValueError`, `OverflowError` ou `RecursionError` no encoder emitem `TOOL_CACHE_KEY_UNAVAILABLE`, ignoram leitura e armazenamento no cache e deixam o executor seguir. Assim, a extensão extra continua ignorada como antes e uma operação válida continua retornando seu resultado sem cache. O caso de dado não serializável enviado ao próprio `format_json` falha de forma estruturada no executor, sem exceção escapar.

Os contratos `test_format_json_contract.py` e `test_tool_cache_key_contract.py` passaram após reproduzirem as falhas antes do patch. O gate direcionado adicionou um mutante para desfazer `allow_nan=False` e dois mutantes para remover o bypass/serialização condicional.

Validação local em Python 3.12: 254/254 testes; suíte integrada 30 aprovados, 1 skip FAISS opcional e 0 falhas; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation testing 37/37 mortas, 0 sobreviventes ou não aplicadas; Bandit 131 LOW, 0 MEDIUM/HIGH; Ruff, actionlint, zizmor, compileall, auditorias AST e CLIs dos benchmarks de capacidade e POMDP v2 passaram. Relatórios de realidade foram regenerados. Os dois checks oficiais da PR #27 passaram antes do merge registrado acima.

Fundamento: a [documentação Python 3.12 de `json`](https://docs.python.org/3.12/library/json.html) descreve o padrão permissivo e o efeito estrito de `allow_nan=False`; a [RFC 8259, seção 6](https://www.rfc-editor.org/rfc/rfc8259) não permite Infinity/NaN na gramática numérica JSON. Não houve normalização silenciosa para strings ou `null`.

Limites: não foi introduzida validação recursiva de schema nem rejeição de chaves desconhecidas. O bypass não limita custo de parâmetros grandes porém serializáveis, e o tamanho estimado do cache não é limite global de memória. O diagnóstico de falha de cache não é falha da ferramenta; exceções reais do executor continuam seguindo seu tratamento original.


## Orçamento de recursos da cache key — 2026-10-16

Base local: `main` pós-PR #27, merge commit `c4375d9f42703b11d2632b3a701303c5a69563a4`; branch `feat/post-pr27-cache-key-resource-bounds-2026-10-16`.
Estado posterior: a branch foi publicada como PR #28, recebeu aprovação dos dois checks oficiais e foi mesclada em `main` no commit `fafcd95e6b40c5d51b34d344f164a9045279d8bb`.

A chave era construída por `json.dumps(params, sort_keys=True)` antes do executor, sem limite de bytes ou profundidade; os budgets existentes só cobriam resultados retidos no cache. A alteração adiciona percurso iterativo prévio com limite conservador de 64 KiB incluindo prefixo, profundidade 64, 4.096 nós/ocorrências e 1.024 itens por contêiner, além de detecção de ciclos. O percurso contabiliza expansão de escapes e tamanho de inteiros sem serializar o objeto completo.

Testes com orçamento excedido (bytes, profundidade, cardinalidade total, itens por contêiner e ciclo) verificam que `json.dumps` não é chamado, que `TOOL_CACHE_KEY_UNAVAILABLE` inclui o motivo e que duas chamadas válidas continuam no executor, sem hit nem armazenamento. O contrato também confirma compatibilidade do hash para entradas pequenas, execução de subclass de `dict`, bypass de valor extra não serializável e ausência de serialização em ferramenta não cacheável. Cinco mutantes direcionados removendo cada guarda/detecção são mortos.

Resultados locais medidos: unittest 257 executados/aprovados, 0 falhas/skips/XPASS; suíte integrada 30 aprovados e 1 skip FAISS opcional; cobertura 95% fila, 94% guard VRAM, 76% Vita, 68% núcleo; mutation 42/42 mortas, 0 sobreviventes/não aplicadas; Bandit 131 LOW, 0 MEDIUM/HIGH. Ruff, actionlint, zizmor, compileall, auditoria de realidade, auditorias AST e CLIs dos benchmarks de capacidade e POMDP v2 na seed `20261016` passaram. A fonte normativa consultada foi https://docs.python.org/3.12/library/json.html.

Limitação: execução fora do cache continua sujeita ao comportamento e custo do executor; a mudança limita somente a pré-verificação/serialização para cache. As métricas acima são locais; os checks oficiais da PR #28 passaram antes do merge.


## Falha de cópia durante armazenamento de cache — 2026-10-16

Base: `main` pós-PR #28, merge commit `fafcd95e6b40c5d51b34d344f164a9045279d8bb`; branch `feat/post-pr28-cache-store-fallback-2026-10-16`.

Reprodução pré-patch: um executor built-in foi substituído por retorno de listas aninhadas com profundidade 600. `_estimate_cache_value_size` mediu 38.445 bytes, abaixo de `MAX_CACHE_ENTRY_BYTES`; a cópia subsequente com `copy.deepcopy` lançou `RecursionError`. Observado: `execute_tool` devolvia `success=False`, erro `maximum recursion depth exceeded` e nenhum payload, embora o executor já tivesse retornado com sucesso.

Após o patch, o mesmo cenário devolve o objeto e sucesso originais, registra `TOOL_CACHE_STORE_UNAVAILABLE` com `reason=RecursionError` e `cache_bypassed=true`, mantém uma contagem de sucesso, não incrementa falhas e deixa o cache vazio. `test_tool_cache_store_contract.py` protege deterministicamente esse contrato por injeção da exceção. Um mutante que restaura a propagação da exceção é morto.

Resultados locais medidos: unittest 258/258; suíte integrada 30 aprovados, 1 skip opcional FAISS; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation 43/43 mortas, 0 sobreviventes e 0 não aplicadas; Bandit 131 LOW, 0 MEDIUM/HIGH; Ruff, actionlint, zizmor, compileall, auditoria de realidade, auditorias AST e benchmarks de capacidade/POMDP v2 na seed `20261016` passaram.

Limite: a captura é específica à cópia profunda antes do armazenamento; não contém exceções arbitrárias do executor, não muda sua saída e não oferece garantia de limite global de memória. Estado posterior: a PR #29 foi mesclada em `main` pelo merge commit `e06511396d407a3440ee1969e0af8d7e055203dc` após os dois checks oficiais passarem.


## Compatibilidade de subclasses em parâmetros tipados — 2026-10-16

Base: `main` pós-PR #29, merge commit `e06511396d407a3440ee1969e0af8d7e055203dc`; branch isolada `fix/tool-validator-container-subclasses-2026-10-16`.

Reprodução pré-patch: `ToolValidator.validate_parameters` comparava o nome da classe observado com o schema, rejeitando uma subclasse `Items(list)` para `list_operations.items`; o executor teria aceitado a lista em `sorted`, mas não era alcançado. O mesmo critério também aceitava indevidamente uma classe não relacionada se seu `__name__` imitasse um nome built-in, como `list`.

Contrato implementado: nomes built-in conhecidos são resolvidos para classes reais e a relação de subclasse é testada pelo tipo concreto. Uma subclasse de `list` executa e retorna `[1, 2]`; como o preflight de cache não serializa containers customizados, a otimização é bypassada e o sucesso do executor permanece. `int` aceita subclasses verdadeiras, mas `bool` continua rejeitado; um tipo falso com nome `list` não satisfaz o schema. `any` e `object` mantêm semântica wildcard; schemas desconhecidos preservam o matching legado por nome.

Resultados locais medidos: unittest 261/261, sem falhas ou skips; suíte integrada 30 aprovações e 1 skip opcional FAISS; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation gate 45/45 mortas, 0 sobreviventes e 0 não aplicadas; Bandit 131 LOW e 0 MEDIUM/HIGH; Ruff, actionlint, zizmor, compileall, auditorias AST/realidade e benchmarks oficiais passaram. No POMDP v2, 128/128 casos ficaram dentro de `1e-12` (erro máximo `2.7755575615628914e-16`) e 4/4 controles inválidos foram rejeitados.

Base técnica: [built-ins `isinstance`/`issubclass`, Python 3.12](https://docs.python.org/3.12/library/functions.html#isinstance) e [tipo booleano, Python 3.12](https://docs.python.org/3.12/library/stdtypes.html#boolean-type-bool). Limite: o contrato valida apenas o tipo do valor declarado, sem schema recursivo para elementos aninhados; a PR #30 foi mesclada em `main` no commit `e83ce9eb422e4cb9d2fb5ec818cb795351e57a13` após 2/2 checks oficiais aprovados.


## Limite negativo na recuperação SQLite/FAISS — 2026-10-16

Base local: `main` pós-PR #30, merge commit `e83ce9eb422e4cb9d2fb5ec818cb795351e57a13`; branch isolada `fix/memory-retrieve-nonnegative-limit-2026-10-16`.

Reprodução pré-patch com quatro memórias: busca vazia em `limit=0` retornou 0 linhas, `limit=1` retornou 1 e `limit=-1` retornou as 4. A busca por palavra também não rejeitava negativo; no modo semântico, o `k` negativo podia gerar erro e cair nesse mesmo fallback.

A correção valida antes de escolher o backend: `bool` e valores não `int` geram `TypeError`, negativos geram `ValueError` e zero retorna lista vazia sem chamar encoder/FAISS/SQLite. As consultas SQL permanecem parametrizadas. Regressões cobrem busca vazia, keyword, FAISS simulado, tipos inválidos e zero.

Validação local: unittest 265/265; integração 30 aprovados e 1 skip FAISS opcional; cobertura 95% fila, 94% VRAM, 76% Vita, 68% núcleo; mutation gate 47/47 mortos sem sobreviventes/não aplicados; Bandit 131 LOW e 0 MEDIUM/HIGH; Ruff, actionlint, zizmor, compileall, auditorias AST/realidade, CLIs de capacidade e POMDP v2 passaram. A fonte normativa é a [documentação oficial SQLite de SELECT/LIMIT](https://www.sqlite.org/lang_select.html).

Limite: não há teto para valores positivos altos nem garantia sobre memória global. Finding adjacente ainda não implementado: os metadados `requires_permission`/`permission_level` não têm enforcement; todos os built-ins atuais os deixam desativados, e a definição de um modelo de autorização foi adiada por depender de escolha explícita de API.

## Falha de cópia ao recuperar um cache hit — 2026-10-16

Base da atualização: `main` pós-PR #32, merge commit `b07254307a23cdaef68f7c2b0d23010ac0827580`, que já contém a PR #33 (merge commit `62f6af14a2ad1d3659a4a5940e713c588513bf9b`). A PR #32 foi mesclada e D-024 passou a integrar `main`.

Reprodução pré-patch: uma chamada válida a `list_operations` preenchia o cache; uma falha injetada (`RecursionError`) na cópia do hit seguinte escapava de `execute_tool`, pois ocorria fora do `try` que envolve o executor. A segunda chamada não executava a ferramenta; estado observado: `successful_executions=1`, `failed_executions=0`, `cached_executions=0` e a entrada defeituosa permanecia armazenada.

Correção: capturar a exceção apenas ao copiar o resultado do hit, remover a entrada e seu tamanho contabilizado, emitir `TOOL_CACHE_HIT_UNAVAILABLE` com tipo de exceção e `cache_bypassed=true`, e continuar pelo executor. A promoção LRU e os contadores de hit/sucesso só acontecem após uma cópia completa. Se a execução seguinte também não puder ser armazenada, o fallback pré-existente emite `TOOL_CACHE_STORE_UNAVAILABLE` e mantém o sucesso retornado pelo executor.

`test_tool_cache_hit_contract.py` protege a execução real após falha do hit, contadores sem hit espúrio, e evicção quando a cópia da tentativa de re-cache também falha. Dois mutantes direcionados cobrem propagação da exceção e remoção da entrada; o workflow executa este contrato em etapa dedicada.

Validação histórica pré-merge da branch sincronizada: suíte focada de cache 9/9; unittest 271/271 (0 falhas, skips ou XPASS); integração 30 aprovados + 1 skip FAISS opcional; pisos de cobertura 95/94/76/68%; mutation gate 55/55 mortas, sem sobreviventes/não aplicadas; Bandit 131 LOW e nenhum MEDIUM/HIGH, triagem consistente. Ruff, compileall, actionlint, zizmor, auditorias AST/realidade/gaps e CLIs dos benchmarks de capacidade/POMDP v2 passaram. POMDP v2: 128/128 casos dentro de `1e-12` (erro máximo `2.7755575615628914e-16`) e 4/4 controles inválidos rejeitados.

Status de publicação: a [PR #32](https://github.com/pedromiorini/Nexus/pull/32) foi mesclada em `main` no commit `b07254307a23cdaef68f7c2b0d23010ac0827580`; D-024 e D-025 estão juntos na base atual.

Fundamentação: a [documentação oficial `copy` do Python 3.12](https://docs.python.org/3.12/library/copy.html) descreve `deepcopy` como recursiva e observa problemas possíveis com objetos recursivos; a falha específica desta fronteira foi reproduzida por injeção determinística. Limite: esta alteração cobre cópia ao recuperar hit, não falhas do logger. A corrida de mutabilidade de parâmetros foi resolvida pelo snapshot documentado na seção seguinte (PR #33).

## Snapshot JSON limitado para parâmetros de ferramentas — PR #33 mesclada
Base da correção: a PR #33 foi desenvolvida sobre `main` pós-PR #31 e mesclada em `main` no commit `62f6af14a2ad1d3659a4a5940e713c588513bf9b`; a PR #32 foi mesclada depois, no commit `b07254307a23cdaef68f7c2b0d23010ac0827580`. Ambas as decisões estão atualmente em `main`.

A lacuna era temporal: `execute_tool` podia formar uma cache key a partir do valor mutável original e, antes de validar/executar, o chamador alterar esse objeto. A chave então descrevia um valor enquanto o executor observava outro. O contrato `test_tool_cache_snapshot_contract.py` reproduz a divergência e exige identidade do mesmo snapshot nas fases subsequentes; um caso adicional confirma que referências compartilhadas tornam-se cópias independentes por valor.

A correção constrói um snapshot iterativo e limitado para parâmetros cacheáveis estritamente JSON, com os budgets existentes de 64 KiB, profundidade 64, 4.096 ocorrências e 1.024 itens por contêiner. A ordem final é snapshot → validação/schema e guardas semânticas → cache key → lookup → executor. A primeira suíte completa detectou uma regressão de ordenação: expressões de `math_calculate` acima do limite eram serializadas antes da rejeição. A chave foi então movida para depois da validação e da guarda de tamanho, mantendo o mesmo snapshot nas três fases; `test_math_calculate_resource_limits.py` protege esse requisito.

Se o preflight do snapshot falhar (ciclo, tipo não JSON, subclass/container não reconhecido ou budget excedido), a cache é bypassada e a execução segue com o objeto original. Se apenas a geração da chave falhar depois de um snapshot válido, lookup/store são ignorados sem trocar o snapshot usado pela validação e pelo executor.

Uma revisão independente não encontrou bloqueadores, mas identificou uma limitação P2: substituição de valor com o mesmo tamanho enquanto a travessia está em curso pode formar um snapshot misto, pois a cópia não sincroniza o produtor. Depois de construído, o mesmo snapshot é usado por chave, validação e executor; aplicações que exijam um instante atomicamente consistente precisam sincronizar as escritas na origem. Identidade, aliasing e mutações in-place do original não fazem parte da semântica preservada.

Validação local do head original da PR #33 (pré-sincronização da PR #32), Python 3.12: suítes focadas de cache 9/9; suíte completa `unittest` 269/269, sem falhas/skips/XPASS; suíte integrada 30 aprovados e 1 skip FAISS opcional; cobertura fila 95%, VRAM 94%, Vita 76% e núcleo 68%; mutation gate 53/53 mortas, 0 sobreviventes e 0 não aplicadas. Bandit encontrou 131 LOW e nenhum MEDIUM/HIGH; a triagem verificou-se consistente. Ruff, compileall, actionlint, zizmor, auditorias AST/realidade/gaps e CLIs dos benchmarks de capacidade/POMDP v2 na seed `20261016` passaram. O v2 manteve 128/128 casos dentro de `1e-12` (erro máximo `2.7755575615628914e-16`) e rejeitou 4/4 controles inválidos.

## Limite não negativo na busca de memória episódica — 2026-10-16

Base original da alteração: branch `fix/episodic-search-nonnegative-limit-2026-10-16`, criada sobre `main` pós-PR #32 no commit `b07254307a23cdaef68f7c2b0d23010ac0827580` (que contém também a PR #33). A PR #34 foi mesclada em `main` no commit `51373dc77179e4edd094df14290f91aa9c8d8060`.

Reprodução pré-patch: com dois episódios fechados, `search_episodes(status="closed", limit=-1)` retornou os dois; `limit=True` também foi aceito como limite 1. A implementação repassava diretamente o valor para a cláusula SQL `LIMIT`.

Correção: `search_episodes` rejeita bool e tipos não inteiros com `TypeError`, negativos com `ValueError` e devolve `[]` para zero antes de adquirir lock ou executar SQL. Limites positivos e os filtros existentes de query/status permanecem inalterados; nenhum máximo positivo novo foi introduzido.

`test_episodic_memory_search_limits_contract.py` cobre limites negativos nos dois ramos de query e prova ausência de SQL, rejeição de bool/float/string/None, zero sem consulta e preservação de limite positivo. Validação final: 11/11 contratos episódicos focados e unittest 275/275; integração 30 aprovados + 1 skip FAISS opcional; cobertura 95/94/76/68%; mutation 58/58 mortas, sem sobreviventes/não aplicadas; Bandit 131 LOW e nenhum MEDIUM/HIGH. Ruff, compileall, actionlint, zizmor, auditorias AST/realidade/gaps e benchmarks oficiais passaram. O POMDP v2 obteve 128/128 casos dentro de `1e-12` (erro máximo `2.7755575615628914e-16`) e rejeitou 4/4 controles inválidos. O relatório de realidade foi regenerado para refletir as sete linhas acrescentadas ao núcleo.

Fonte normativa: [SQLite Language Reference — SELECT, seção LIMIT](https://www.sqlite.org/lang_select.html), que especifica que um LIMIT negativo não estabelece teto para as linhas retornadas.

## Hardening das fronteiras de snapshot da fila — validação local, 2026-10-17

Base: branch `fix/queue-snapshot-hardening-2026-10-17`, derivada de `origin/main` em `812ebd02b0c82ecc3169d620e36d25592b1d6727`. A versão `nexus.deferred_task_queue.v1` e as assinaturas públicas foram mantidas.

A inspeção e as regressões no checkout limpo confirmaram que isolamento de contexto aninhado, rejeição de timestamps NaN/±Infinity e rejeição de `bool` nos campos inteiros já funcionavam; esses defeitos alegados não foram tratados como atuais. Foram reproduzidas as lacunas reais: estatísticas com chaves extras podiam contaminar os contadores internos; `attempts == max_attempts` era aceito como pendência; a validação via `deepcopy` podia executar `__deepcopy__` de um objeto arbitrário; e uma falha injetada no segundo `heappush` deixava restore parcialmente aplicado. Revisões posteriores reproduziram TOCTOU de `task_id`, subclass `int` de `priority` executando `__neg__`, e `__eq__` de chaves não-string colidentes executado por lookup/membership em raiz/tarefas antes da rejeição (um hook em raiz; dois em tarefa). As regressões agora confirmam zero hooks e estado intacto nos dois modos. A decisão final preserva extras de raiz/tarefa com chaves built-in `str`, ignorando seus valores (incluindo object/inf) sem `deepcopy` ou propagação interna; estatísticas continuam com conjunto exato de quatro chaves. JSON textual com duplicatas foi aceito antes da correção; agora `object_pairs_hook` rejeita duplicatas em todos os níveis, inclusive dentro de `context`. `enqueue()` ainda usa `deepcopy`, assunto fora do escopo desta mudança.

A correção valida e copia árvores JSON nativas iterativamente, rejeita constantes JSON não padronizadas e chaves JSON duplicadas, exige chaves built-in `str` antes de qualquer lookup/membership, mantém extras de raiz/tarefa string-key sem inspecionar seus valores, exige estatísticas com exatamente quatro contadores inteiros built-in não negativos, e usa tipos exatos para campos escalares. Heap, índice, IDs em voo e contadores são montados temporariamente; extras ignorados não são propagados a tarefas internas e a troca de referências acontece só ao fim. `replace=False` preserva os contadores do destino; `replace=True` importa os do snapshot. Não foi adicionado teto de bytes ou profundidade.

Validação factual: `PYTHONPATH=. coverage run --branch -m unittest discover -p 'test_*.py'` executou 318 testes, com 316 aprovados, 2 skips por NetworkX opcional ausente e nenhuma falha; cobertura branch de `core/deferred_task_queue.py`: 91%. Os contratos de fila e snapshot passaram 51/51 tanto em Python 3.12.3 como em Python 3.11.16; nesta última versão, NumPy 2.4.6 foi usado somente em `/tmp` para permitir importar o roteador. A suíte integrada embutida passou 30 checks, ignorou 1 check opcional de FAISS e não teve faltantes. `compileall` passou em Python 3.12.3 e nos três arquivos alterados em Python 3.11.16. Ruff 0.16.10, actionlint 1.7.12 e zizmor 1.31.1 passaram. O mutation gate matou 80/80 mutantes aplicáveis, com 0 sobreviventes e 0 não aplicados. Bandit encontrou 131 LOW, 0 MEDIUM/HIGH; a triagem versionada passou em `--check` (o comando Bandit retorna 1 pelos LOW, comportamento previsto no workflow). `tools/reality_audit.py` foi executado: 27.549 linhas, 269 classes e 875 funções; os relatórios versionados permaneceram sem diff. `git diff --check` passou.

Limites e proveniência: o parser/encoder JSON padrão pode rejeitar aninhamentos extremos; não há orçamento explícito de payload. O produtor deve sincronizar mutações concorrentes do snapshot se exigir uma visão coerente. `enqueue()` continua aceitando seu contrato de contexto anterior e pode executar hooks de `deepcopy`; os métodos da fila ainda expõem objetos mutáveis fora das fronteiras de snapshot. A busca de callsites encontrou `restore_snapshot` somente na implementação e nos testes, sem uso de produção no checkout. Não houve push/merge; por isso não há resultado de CI oficial associado a esta branch.
