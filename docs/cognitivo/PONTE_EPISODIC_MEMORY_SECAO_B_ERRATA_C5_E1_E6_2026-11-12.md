# Pré-registro — errata de C5 e critérios E1–E6

Status: texto local. Ainda não aprovado. Não implementa B1. Não altera o teste vigente nem o doc de caracterização na main. E1–E6 não são contrato até o Pedro aceitar este texto.
Base: `main` `e21d25cfc0b2e43b19d82af05cdefc1ddb5bd9be` (pai `8e9feac5447c1c759fe1f0c3abb70e14a5587653`; squash da #178).
Desenho de origem, local e não publicado: `docs/cognitivo/PONTE_EPISODIC_MEMORY_SECAO_B_DESENHO_2026-11-12.md`, SHA-256 `01bba0df1ab5b894a5a1a11d68eaafb069e5d73d355a59326695430906a7faee`. B1 foi aceito em 2026-11-12 só como desenho: chave `episodes` no dict de `route`, errata de C5, `think` sem a chave nesta fatia. Implementar e publicar continuam não autorizados.
Agent-Id: grok-bot.

Fora desta fatia: B2, B3, B4, B5, S2a, a errata de `counterfactual.py:180`, a opção 1c da ponte do roteador, reexportação no pacote e membro novo de `ModuleType`.

O teste novo (RED e GREEN) não entra nesta tarefa. Até a errata ser aprovada, C5 é o contrato vigente. E1 contradiz esse C5.

## Fato

SHA-256 recalculados nesta sessão, iguais na worktree e em `e21d25c`:

- `docs/cognitivo/PONTE_EPISODIC_MEMORY_CARACTERIZACAO_2026-11-12.md`: `3ee447bfa9ed576df64c56eb4ea8fa15421d06b1f68ab47d6bbba9cb848a5b03`
- `test_router_episodic_memory_bridge_contract.py`: `0a48ee9589fd08d7000590c45119f745ebaa761d0776c259ff438a8ccb84991a`

Frase vigente de C5 no doc, linha 52:

> No recall de C4 a única chamada é `search_episodes(query="remember the last time", limit=3)`, sem posicionais e sem `status`. `success is True`, `degraded is False`, `request_type == "recall"`, `answer == "No answer generated"`, `modules_executed == ["jailbreak", "budget", "memory", "episodic", "log"]`. O `dict` não tem `episodic`, `episodes` nem `episode_id`. A segunda `route` do mesmo prompt com `{}` devolve `cached is True` e não chama `search_episodes` de novo.

O teste vigente, `C5RecallCallAndPublicResult`, linhas 364–366, exige a ausência das três chaves no primeiro dict:

```python
for key in ("episodic", "episodes", "episode_id"):
    self.assertNotIn(key, first)
```

O registrador do harness H devolve `[]` em `search_episodes`. Esse C5 é, portanto, uma busca que teve sucesso e achou zero episódios. A frase «o dict não tem `episodes`» cobre esse sucesso vazio.

O desenho B1 aceito diz: no sucesso, a chave `episodes` existe e tem a projeção, possivelmente vazia; a chave fica ausente quando o módulo não rodou ou não teve sucesso; `think` não ganha a chave; `episode_id` não entra no dict de `route`.

## Errata de C5

Texto apenas. O doc de caracterização e o teste na main não são editados nesta tarefa.

Substitui somente a frase «O `dict` não tem `episodic`, `episodes` nem `episode_id`.»

Frase proposta no lugar dela:

> Quando a busca episódica teve sucesso, a chave `episodes` existe e tem a projeção: uma lista de dicts, possivelmente vazia. Cada dict tem exatamente as chaves `episode_id`, `title`, `trigger_query`, `importance` e `status`, iguais aos campos do episódio devolvido. A chave `episodic` continua ausente. A chave `episode_id` continua ausente no dict de `route`. Nos outros pontos de C5 a rota não muda: a chamada única `search_episodes(query="remember the last time", limit=3)`, sem posicionais e sem `status`; `success is True`; `degraded is False`; `request_type == "recall"`; `answer == "No answer generated"`; `modules_executed == ["jailbreak", "budget", "memory", "episodic", "log"]`; a segunda `route` do mesmo prompt com `{}` devolve `cached is True` e não chama `search_episodes` de novo.

O resto de C1–C15 permanece. C4, C6, C7, C8, C10, C11, C12, C13, C14 e C15 não ganham a chave `episodes` por esta errata. C12 continua a publicar `episode_id` no dict de `think`, que é outro dict.

No harness H, o sucesso com lista vazia passa a ser `episodes == []`. A asserção vigente `assertNotIn("episodes", first)` deixa de valer só depois que o Pedro aceitar esta errata e uma tarefa futura editar o teste. Até lá ela permanece e E1 não pode ser exigido contra a main.

## Pré-registro E1–E6

Ainda não aprovado. Não há arquivo de teste. O harness é o da caracterização: H para o roteador com registrador, B para o cérebro com `enable_router=True`, `enable_integration_orchestration=True` e os demais `enable_*` em `False`, Vazio para o mapa sem módulos.

Projeção: dict com exatamente `episode_id`, `title`, `trigger_query`, `importance` e `status`. Sem `duration`. Sem o objeto `Episode`.

- **E1.** Com `RealEpisodicMemory(":memory:")` no lugar do registrador e um episódio fechado de título `"remember the last time"`, `route("remember the last time", {})` devolve `episodes` igual a uma lista de um dict. `status` é `"closed"`. `episode_id` dentro desse dict é o id desse episódio. `title` é `"remember the last time"`. `trigger_query` e `importance` são os gravados nesse episódio: se `start_episode` foi chamado só com o título, `trigger_query` é `None` e `importance` é `0.6`, o default do método. O dict da rota não tem a chave `episode_id` nem a chave `episodic`. A segunda `route` do mesmo prompt com `{}` devolve `cached is True`, não chama `search_episodes` de novo e repete a mesma lista `episodes`.

- **E2.** Busca com sucesso e zero episódios fechados: `episodes == []`, `success is True`, `degraded is False`. O caso do harness H em `route("remember the last time", {})`, cujo registrador devolve `[]`, é esse caso. Uma `RealEpisodicMemory` sem episódio fechado, na mesma rota, também devolve `episodes == []`.

- **E3.** No harness H, `ping`, `""`, `"learn this fact"`, `"Tell me about item 9"`, `"should I plan my week"` e `"what did I say about the right choice"` continuam sem a chave `episodes`.

- **E4.** Sem a chave `episodes`: o mapa do harness H sem `ModuleType.EPISODIC`; `search_episodes` que levanta `ValueError("query must be non-empty")`; o detector do módulo jailbreak com `detected=True` em `"remember the last time"`; o harness Vazio em `route("ping", {})` e em `route("remember the last time", {})`. C7 e C8 permanecem, inclusive degradação sem bloqueio e a ausência de cache depois da exceção.

- **E5.** No harness B, `think("remember the last time", use_router=True)` não ganha a chave `episodes`. C12 permanece, inclusive `episode_id` no dict de `think`.

- **E6.** Com `RealEpisodicMemory(":memory:")`, um episódio fechado de título `"ping"` e `request_type_proposers = (lambda state: ("learning",),)`, `route("ping", {})` devolve `episodes` com esse episódio, na mesma projeção de E1 (`status` `"closed"`, `title` `"ping"`, `episode_id` igual ao id fechado). Sem esse proposer, `"learn this fact"` continua em E3.

E1 contradiz o C5 vigente: o teste em `test_router_episodic_memory_bridge_contract.py:364-366` exige que `episodes` não esteja no dict, e o doc vigente diz o mesmo. Enquanto esta errata não for aceita, essa exigência é o contrato. O RED que mostraria a ausência da chave, e o GREEN que a preencheria, ficam para uma tarefa posterior.

## Inferência

- A segunda `route` com `{}` repete `episodes` porque o cache copia o dict inteiro. E1 não cria uma segunda busca.
- E2 é a leitura de B1 aplicada ao C5 atual: o registrador tem sucesso e devolve `[]`, então a projeção é a lista vazia, não a ausência da chave.
- E5 segue da lista fechada que `think` copia do roteador (`core/constitutional_brain.py:20646-20658`), que não inclui `episodes`. Não é preciso mudar `think` para E5 permanecer verde depois de um GREEN só em `route`.

## Hipótese

- Cinco chaves exatas, sem `duration`, bastam para o chamador reconhecer o episódio. Isso não foi pedido por um consumidor externo; é o recorte de B1.
- `importance` `0.6` e `trigger_query` `None` são o default de `start_episode` quando só o título é passado (`core/cognitivo/episodic_memory.py:111-112`). Se o teste futuro passar outro `importance` ou outro `trigger_query`, a projeção deve copiar esses valores. E1 não congela um segundo par de números além desse default.
