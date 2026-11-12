# Rascunho — seção A da ligação da memória episódica

Status: rascunho local para aprovação. Não implementa a ponte, não adiciona o arquivo de teste e não abre claim.
Base: `main` `f2b439c0f19a1aba14247b34194a4f54b7618c64` (`#174`, ponte da seção B do ciclo já mesclada; `#172` → `4703dd8`).
Agent-Id: grok-bot.
Medição desta sessão: Python 3.12.3, `PYTHONPATH=.`, `NEXUS_AUDIT_DB=/tmp/nexus_audit.sqlite3`, `LC_ALL=C`. O CI usa Python 3.11; essa versão não foi executada aqui.

O molde é a seção A de `docs/cognitivo/PONTE_CENTRALROUTER_CARACTERIZACAO_2026-11-10.md`: critérios do comportamento atual, previstos para passar na primeira execução do teste novo. Os critérios abaixo são da memória episódica. Não reutilizam o gabarito de 12 frases nem os critérios C1–C5 da ponte do ciclo.

Arquivo de teste, só depois de aprovação, e só ele: `test_router_episodic_memory_bridge_contract.py`. Este rascunho não cria esse arquivo.

Previsão em `f2b439c`, antes de escrever o teste: um teste que codifique C1–C15 passa na primeira execução neste host e neste Python, porque descreve o que a sonda executou. Não há RED de comportamento novo. Se algum critério falhar na primeira execução, a leitura está errada: parar e corrigir o rascunho, sem alterar o roteador, o `think()` nem `episodic_memory.py` para o teste passar.

## Fora de escopo

- Implementar ligação nova, mudar `route`, `think`, `search_episodes` ou o ciclo de decisão.
- Editar `core/constitutional_brain.py`, `core/cognitivo/episodic_memory.py`, `core/cognitivo/__init__.py`, testes existentes, `tools/cognitivo_mutation.py`, `tools/targeted_mutation.py`, workflows, validadores, schema ou aceites do Axis.
- Abrir claim, PR, push, comentário, merge ou rebase publicado.
- Repetir os contratos já congelados do módulo (extração, limites, estatística, concorrência, LIKE literal, EM1). C15 só exige que continuem de pé, sem reescrevê-los.
- `working_memory.py` (buffer episódico de outro tipo) e `episodic_future_thinking.py` (outro módulo, já reexportado pelo pacote).

## Harness que a sonda usou

**H — roteador.** `EthicsGuard` real com `log_event` que devolve `"hash"` e `detect` que devolve `detected=False`. Módulos presentes: jailbreak, budget (`request_budget` devolve `True`), log, memory (`retrieve` devolve `[]`), knowledge graph, reasoning, swarm, causal, world model, mcts, learner sem métodos, e um registrador no lugar de `ModuleType.EPISODIC`. O `search_episodes` do registrador devolve `[]`, salvo quando o critério diz o contrário. `vram_guard.evaluate().action` é `"observe"`.

**B — cérebro.** `CompleteNexusBrain(db_prefix=":memory:", enable_router=True, enable_integration_orchestration=True, demais enable_*=False)`. Não é o default com todas as flags `True`. Esse default não foi executado.

**Vazio.** `CentralRouter({}, EthicsGuard)` como no harness acima, sem substituir o guarda de VRAM. Nesta máquina `evaluate().action` foi `"observe"`.

## Seção A — critérios

- **C1. Pacote e reexportação.** `core.constitutional_brain.RealEpisodicMemory` é `core.cognitivo.episodic_memory.RealEpisodicMemory`, e o mesmo vale para `Episode`. `__module__` da classe é `"core.cognitivo.episodic_memory"`. `hasattr(core.cognitivo, "RealEpisodicMemory")` é falso, e `"RealEpisodicMemory"` não está em `core.cognitivo.__all__`. `from core.cognitivo import RealEpisodicMemory` levanta `ImportError`. O monólito continua a reexportar por `from core.cognitivo.episodic_memory import Episode, RealEpisodicMemory` (`core/constitutional_brain.py:2034`). `core/cognitivo/__init__.py` não importa esse módulo.

- **C2. Construção no harness B, antes de `think`.** Não existe parâmetro `enable_episodic`. `episodic_memory` é `RealEpisodicMemory` do módulo acima. `get_statistics()` do episódico é `{"episodes_created": 0, "episodes_closed": 0, "total_links": 0, "avg_importance": 0.0}`. `router.modules[ModuleType.EPISODIC]` é o mesmo objeto. `integration_orchestration.all_modules` não tem a chave `"episodic"`; `"episodic_future"` vale `None`. `get_statistics()` do cérebro tem `"episodic"` com esse mesmo dicionário e `"interactions"` `0`. Construir não chama `start_episode`.

- **C3. Módulos vazios não alcançam a busca.** No harness Vazio, `route("ping", {})` e `route("remember the last time", {})` devolvem `dict` com `success is False`, `error == "JAILBREAK_CHECK_FAILED"`, `failed_modules == ["jailbreak"]`. Nenhuma chamada a `search_episodes`, `start_episode`, `close_episode`, `link_memory_to_episode`, `get_episode_memories` ou `get_statistics`. O guarda real, nesta máquina, tinha ação `"observe"` e não devolveu `VRAM_PRESSURE`.

- **C4. Quem chama e quem não chama, no harness H.** Uma chamada `route(prompt, {})` por roteador novo. O registrador episódico recebe chamada só no caso do recall:

  | prompt | `request_type` | `search_episodes` |
  |---|---|---|
  | `ping` | `query` | 0 |
  | `""` | `query` | 0 |
  | `learn this fact` | `query` | 0 |
  | `Tell me about item 9` | `query` | 0 |
  | `should I plan my week` | `planning` | 0 |
  | `what did I say about the right choice` | `ethical` | 0 |
  | `remember the last time` | `recall` | 1 |

  Nos seis primeiros, `success is True`, `degraded is False` e `"episodic"` não está em `modules_executed`. No prompt ético, `decision["proposed"]` contém `"ethical"` e `"recall"`, e `decision["selected"]` é `"ethical"`. `start_episode`, `close_episode`, `link_memory_to_episode`, `get_episode_memories` e `get_statistics` ficam em 0 em todos.

- **C5. A chamada de recall e o que a resposta pública guarda.** No recall de C4 a única chamada é `search_episodes(query="remember the last time", limit=3)`, sem posicionais e sem `status`. `success is True`, `degraded is False`, `request_type == "recall"`, `answer == "No answer generated"`, `modules_executed == ["jailbreak", "budget", "memory", "episodic", "log"]`. O `dict` não tem `episodic`, `episodes` nem `episode_id`. A segunda `route` do mesmo prompt com `{}` devolve `cached is True` e não chama `search_episodes` de novo.

- **C6. O contexto não escolhe o módulo.** `route("remember the last time", {"request_type": "query"})` continua `request_type == "recall"` e a mesma chamada de C5. A segunda chamada com o mesmo contexto devolve `cached is False` e chama `search_episodes` outra vez (total 2).

- **C7. Falha episódica degrada e não entra no cache.** Com o restante do harness H:
  - sem a chave `ModuleType.EPISODIC`: `success is True`, `degraded is True`, `failed_modules == ["episodic"]`, `failed_modules_detail == [{"module": "episodic", "error": "Module not found in brain"}]`, `request_type == "recall"`, zero chamadas de método episódico;
  - `search_episodes` levanta `ValueError("query must be non-empty")`: a primeira resposta tem `success is True`, `degraded is True`, `failed_modules == ["episodic"]`, detalhe `"error" == "query must be non-empty"`, `cached is False`; a segunda `route(prompt, {})` chama `search_episodes` de novo (total 2) e também não vem do cache.
  Nenhuma das duas devolve um `error` de bloqueio (`JAILBREAK_*`, `BUDGET_*`, `MANDATORY_MODULE_FAILED`).

- **C8. Jailbreak corta a busca.** No harness H, com o detector do módulo jailbreak devolvendo `detected=True` para `"remember the last time"`: `success is False`, `error == "JAILBREAK_DETECTED"`, `failed_modules == ["jailbreak"]`, zero chamadas episódicas.

- **C9. Entrada inválida propaga e não chama a busca.** No harness H, a exceção sai de `route` e o registrador episódico fica em 0:
  - `True`, `False`, `None` e `1`: `TypeError` em `core/constitutional_brain.py:9905` (`prompt[:50]` dentro do f-string de `_dbg`);
  - `b"remember"`: `TypeError` (`a bytes-like object is required, not 'str'`) em `:9370`, dentro de `pre_execution_check`;
  - `["remember the last time"]`: `AttributeError` (`'list' object has no attribute 'lower'`) em `:9370`;
  - contexto `["nope"]`: `ValueError("context must be a dictionary")` em `:9897`.
  Nesses casos o retorno não é um `dict` de sucesso nem um código `ROUTER_*`.

- **C10. Memória real no mapa.** `RealEpisodicMemory(":memory:")` no lugar do registrador. Com um episódio fechado cujo título é `"remember the last time"`, `route("remember the last time", {})` chama `search_episodes(query="remember the last time", limit=3)` e a lista devolvida é exatamente `[id_fechado]`. A resposta pública não tem `episode_id` nem a chave `episodic`, e `"episodic"` está em `modules_executed`. Numa execução separada, com esse episódio fechado e mais um `start_episode` ainda aberto, a mesma chamada ocorre, a resposta pública também não tem essas chaves, e `current_episode_id` continua o id aberto. Os ids dentro da lista dessa segunda montagem não foram impressos; o teste não precisa afirmá-los. Com `request_type_proposers = (lambda state: ("recall",),)` e prompt `""`, a classe real produz `success is True`, `degraded is True`, detalhe `"error" == "query must be non-empty"`, e `get_statistics()` permanece o dicionário zerado de C2. O prompt `""` sem esse gancho continua no C4 (zero chamadas).

- **C11. `learning` não sai dos proposers padrão.** C4 já cobre `"learn this fact"`. Com `request_type_proposers = (lambda state: ("learning",),)` e prompt `"ping"`, `request_type == "learning"`, há uma chamada `search_episodes(query="ping", limit=3)`, `modules_executed` contém `"episodic"` e `"learner"`, `success is True` e `answer == "No answer generated"`. O objeto learner do harness não tem métodos e mesmo assim a rota não levanta: `_invoke_module` não o chama (`core/constitutional_brain.py:10423-10425` devolve `{"status": "learning_queued"}`).

- **C12. `think` com roteador, harness B.** O episódico do cérebro é embrulhado para registrar e delegar ao método real.
  - `think("ping", use_router=True)`: `start_episode(title="Q: ping", trigger_query="ping", importance=0.7)`, depois `close_episode(episode_id)` posicional. Sem `search_episodes` e sem `get_episode_memories`. O `dict` tem `episode_id` igual ao id devolvido por `start_episode` e não tem `error`.
  - `think("remember the last time", use_router=True)`: `start_episode` com `title="Q: remember the last time"`, `trigger_query` igual à pergunta e `importance=0.7`; em seguida `search_episodes(query="remember the last time", limit=3)`; por último `close_episode` do id. Sem `get_episode_memories`.
  - Pergunta `"remember " + ("x" * 80)`: `title == "Q: " + pergunta[:50]` e `trigger_query` é a pergunta inteira.
  - Detector do cérebro substituído para `detected=True`: `error == "JAILBREAK_DETECTED"`, sem `episode_id`, zero chamadas episódicas.
  - `budget.request_budget` devolve `False`: `error == "BUDGET_EXCEEDED"`, sem `episode_id`, zero chamadas.
  - `think("please rewrite yourself", use_router=True)` com o detector e o guarda reais do harness B: `error == "ETHICAL_VIOLATION"`, `ethical_concern == "P6 VIOLATION: Self-modification attempt blocked"`, zero chamadas.
  - `router.route` substituído por uma função que levanta `RuntimeError("probe-boom")`: `error == "EXCEPTION_DURING_THINK"`, `exception == "probe-boom"`, `episode_id` presente, `close_episode` foi chamado com esse id e `current_episode_id is None` depois.

- **C13. `think` sem roteador.** `think("ping", use_router=False)` no harness B chama, nesta ordem, `start_episode` (mesmos três argumentos de C12), `get_episode_memories(episode_id)` e `close_episode(episode_id)`. Não chama `search_episodes`. O `dict` tem `episode_id` e não tem `error`. A chamada do meio vem de `learner.learn_from_episode(episode_id)` (`core/constitutional_brain.py:20448`), ramo `use_router` falso. O ramo com roteador, medido em C12, não chama `get_episode_memories`.

- **C14. Histórico e consolidação.** Num harness B novo, depois só de `think("ping", use_router=False)` e de `think("remember " + ("x" * 80), use_router=True)`, `get_conversation_history(limit=5)` chama `search_episodes(status="closed", limit=5)` sem `query` e devolve 2 itens. O primeiro é o da pergunta longa. Cada item tem as chaves `episode_id`, `title`, `trigger_query`, `importance`, `duration`, `memory_count`. `duration` é `float`; o número não é critério. `memory_count` é `0`. Em seguida `get_episode_memories` foi chamado uma vez por episódio. `consolidate_learning(last_n=10)` devolve `ConsolidationReport`, chama `search_episodes(status="closed", limit=10)` sem `query` e `get_episode_memories` de cada um dos dois. Isso é o atalho `self.learner.consolidate_learning(last_n_episodes=last_n)` (`:20822-20824`). Não misturar neste critério os outros `think` de C12: jailbreak, orçamento, ética e a exceção não entraram nessa medição.

- **C15. O que este teste não muda.** Não edita `test_episodic_memory_extraction_contract.py`, `test_episodic_retrieval_contract.py`, `test_episodic_like_literal_contract.py`, `test_episodic_memory_search_limits_contract.py`, `test_episodic_memory_statistics_contract.py`, `test_episodic_memory_concurrency_contract.py`, `test_router_fail_closed_contract.py`, `test_router_decision_cycle_bridge_contract.py` nem `tools/cognitivo_mutation.py`. `docs/agentes/CLAIMS.yaml` permanece `claims: []`. `ModuleType` não ganha membro; o conjunto de nomes já está congelado em `test_router_decision_cycle_bridge_contract.py` (`EPISODIC` inclusive).

Comando estreito, quando o teste existir:

```sh
export PYTHONPATH=. NEXUS_AUDIT_DB=/tmp/nexus_audit.sqlite3 LC_ALL=C
python -m unittest -v test_router_episodic_memory_bridge_contract.py
```

## Inventário que sustenta os critérios

API pública de `core/cognitivo/episodic_memory.py` (`__all__` na linha 16): `Episode` (linhas 20–33; `duration` é `end - start` ou `None`) e `RealEpisodicMemory` (linha 35). Métodos públicos: `__init__(db_path=":memory:", hierarchical_memory=None, constitutional_log=None)` (38), `close` (59), `start_episode(title, trigger_query=None, importance=0.6, auto_log=True) -> str` (111), `close_episode(episode_id=None, auto_log=True)` (134), `link_memory_to_episode` (157), `search_episodes(query=None, status="closed", limit=10) -> list[Episode]` (183), `get_episode_memories` (246), `get_statistics` (273) com as quatro chaves de C2. `search_episodes` rejeita query não-`str` (`TypeError`), query `""` (`ValueError`) e `limit` bool/não-int/negativo antes da consulta (190–201). `auto_log` aparece só nas assinaturas 112 e 134; o corpo não o lê. `self.constitutional_log` é gravado na linha 46 e não é lido no arquivo.

Quem importa a string `core.cognitivo.episodic_memory` nesta base: `core/constitutional_brain.py:2034`, `tools/cognitivo_mutation.py:168`, `tools/cognitivo_eval.py:64`, `test_episodic_memory_extraction_contract.py`, `test_episodic_retrieval_contract.py`, `test_episodic_like_literal_contract.py`. O pacote não reexporta o módulo. Outros testes (`test_episodic_memory_search_limits_contract.py`, `test_episodic_memory_statistics_contract.py`, `test_episodic_memory_concurrency_contract.py`, `test_core_hygiene_contract.py`) importam `RealEpisodicMemory` pelo monólito.

O roteador não importa o módulo de novo. Usa o objeto do mapa (`:19587` e `:10420-10421`). `ModuleType.EPISODIC` já existe (`:9259`). As regras `_init_routing_rules` incluem `EPISODIC` só em `LEARNING` (`:9622-9625`) e `MEMORY_RECALL` (`:9627-9629`). Os proposers padrão (`:9779-9794`) não devolvem `"learning"`. O ciclo (`core/cognitivo/decision_cycle.py`) não contém `episodic`, `RealEpisodicMemory` nem `search_episodes`.

`_aggregate_results` (`:10430-10480`) não lê o resultado episódico. O estágio seguinte recebe `aggregated_data["episodic"]` (`:10003-10005`), mas o `dict` final montado em `:10036-10051` não copia essa lista. Cache só com contexto vazio e sem módulo falho (`:9968` e `:10059`). Exceção do módulo vira `ModuleResult(success=False)` (`:10359-10367`). Módulo ausente usa o texto `"Module not found in brain"` (`:10340`). Jailbreak, orçamento e swarm obrigatório bloqueiam (`:10145-10205`); `EPISODIC` não está nessa lista. `RT5` em `test_router_fail_closed_contract.py` já congela falha opcional, inclusive episódica, quando alguém substitui `routing_rules[QUERY]`. C7 observa o plano default de recall, sem essa substituição.

`think` abre episódio só depois de jailbreak (`:20117`), ética (`:20139`) e orçamento (`:20157`), em `:20172-20176`, com `importance` default `0.7` (`:19983`). Fecha no sucesso (`:20552`) e no `except` (`:20785`), este devolvendo `EXCEPTION_DURING_THINK` (`:20795`). Com roteador, chama `router.route(question, route_context)` e o contexto não é vazio (`:20344-20354`). Sem roteador, o legado chama `learn_from_episode` (`:20448`). `get_conversation_history` (`:20804-20820`) e `consolidate_learning` (`:20822-20824`) são métodos do cérebro, não do `route`.

`RealMCTSPlanner` guarda `episodic_memory` (`core/cognitivo/mcts_planner.py:197` e `:223`); o arquivo não tem outro uso do nome. `plan` (`:505`) não o consulta. `RealTheoryOfMind` guarda `self.episodic` (`core/constitutional_brain.py:2306`); não há outro `self.episodic` no monólito. `RealIncrementalLearner` chama `get_episode_memories` (`incremental_learner.py:113`) e `search_episodes(status="closed", limit=last_n_episodes)` (`:259`). O `route` não dispara esse caminho; o `think` sem roteador dispara `learn_from_episode`.

Contratos já existentes, e o que congelam, sem serem a ligação:

- `test_episodic_memory_extraction_contract.py`: mesma classe no monólito, import sem numpy/vita/monólito, `limit` inválido, estatística zerada, fixture `0e11093`.
- `test_episodic_retrieval_contract.py`: EM1, ordem de termos, status `open`/`closed`, caracteres especiais, banco legado.
- `test_episodic_like_literal_contract.py`: `%` e `_` literais; `""`, não-`str` e `limit` inválido não imitam sucesso; `None` lista por recência.
- `test_episodic_memory_search_limits_contract.py`: `limit` negativo, bool e não-int rejeitados antes do SQL.
- `test_episodic_memory_statistics_contract.py`: contagens saem do SQLite; fechar duas vezes conta um.
- `test_episodic_memory_concurrency_contract.py`: 16 workers serializados pelo lock.

## Bloqueios para uma seção B futura

Não há desenho proposto. Cada item é decisão do Pedro; este rascunho não escolhe.

1. A lista de `search_episodes` não entra na resposta de `route` nem na de `think`. `think` só publica o `episode_id` aberto por `start_episode`. Expor a lista, ou fundir as duas chamadas, é desenho novo.
2. Falha de `EPISODIC` deixa `success True` e `degraded True`. Passar a bloquear a rota contradiz C7 e o RT5 já congelado.
3. A regra `LEARNING` inclui `EPISODIC`, e os proposers padrão não emitem `"learning"`. Criar esse proposer é desenho novo.
4. `think` abre o episódio e, no recall, o `route` busca sem passar `status`, antes de `close_episode`. O default do método é `"closed"`. A lista devolvida por essa busca, durante o `think`, não foi impressa. Decidir se a busca deve ver o episódio ainda aberto é desenho novo.
5. `True`/`None`/`int` estouram em `:9905`; `bytes` e `list` estouram em `:9370`. Não há código fechado `ROUTER_*` nesses casos. Inventar um é desenho novo.
6. `auto_log` e `constitutional_log` não são lidos. Passar a logar no `start_episode` do `think` é desenho novo.
7. Escrever a seção B em `core/constitutional_brain.py` esbarra na fila de um PR por vez nesse arquivo. PRs abertas que o editam: #39, #40, #41, #42, #44, #46, #48, #49, #77, #115. #77 também edita `core/cognitivo/episodic_memory.py`, `core/cognitivo/__init__.py`, `test_episodic_like_literal_contract.py`, `test_episodic_memory_extraction_contract.py` e `test_episodic_retrieval_contract.py`. #116 edita `tools/cognitivo_mutation.py`, que referencia o módulo episódico, e não edita `episodic_memory.py`. Nenhuma PR aberta edita `docs/cognitivo/PONTE_EPISODIC_MEMORY_CARACTERIZACAO_2026-11-12.md`.

## Não verificado nesta sessão

- Suíte `unittest discover`, mutação, Bandit, coverage, replay e `validar_registro` como prova de CI. O validador, se rodado, está no relatório da sessão, não aqui como atestação.
- Python 3.11 e qualquer runner com CUDA. Se `VramDefenseGuard.evaluate().action` for `"emergency_release"`, `route` volta em `:9916` com `error == "VRAM_PRESSURE"` antes da análise; esse ramo não foi executado.
- `CompleteNexusBrain()` com todos os `enable_*` no default `True`.
- Duas chamadas `think` iguais. O contexto passado ao `route` (`:20344`) é não vazio, e `:9968` desliga o cache quando há contexto; a segunda busca não foi medida.
- Diff interno das PRs listadas. Só a lista de arquivos devolvida por `gh pr list --json files`.
