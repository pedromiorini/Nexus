# Medição — lista da busca no think em recall

Status: critérios gravados antes da execução. Caracterização do comportamento atual. Não implementa B2. Não muda a busca. Não é RED nem GREEN de semântica nova.
Base: `origin/main` `6c608f921f93a002c58604f80e1fc305245b67e5` (pai `e21d25cfc0b2e43b19d82af05cdefc1ddb5bd9be`; squash da #179).
Desenho citado, inalterado: `docs/cognitivo/PONTE_EPISODIC_MEMORY_SECAO_B_DESENHO_2026-11-12.md`, SHA-256 `01bba0df1ab5b894a5a1a11d68eaafb069e5d73d355a59326695430906a7faee`.
Agent-Id: grok-bot.
Python previsto: 3.11, o que estiver em `python3.11 --version` no momento da corrida. A versão exata entra só na seção Medição, depois da corrida.

Fora de escopo: B2, B3, B4, B5, opção 1c, S2a, errata de `counterfactual.py:180`, produção, testes C1–C15 e E1–E6 já na main, workflows, Axis.

## Fato já no código, antes da lista

Em `6c608f9`, `search_episodes` tem `status: str = "closed"` (`core/cognitivo/episodic_memory.py:183-184`). O roteador chama `module.search_episodes(query=prompt, limit=3)` e não passa `status` (`core/constitutional_brain.py:10436-10437`). `think` abre o episódio em `:20188-20191` (`title="Q: " + question[:50]`, `trigger_query=question`, `importance` default `0.7`) e, com roteador, chama `route(question, route_context)` em `:20370`. `close_episode(episode_id)` está em `:20568`, depois dessa chamada. A lista que `search_episodes` devolve nesse intervalo não foi impressa na caracterização. Este documento não afirma, antes da corrida, se o id aberto entra nela.

## Harness da medição

`CompleteNexusBrain(db_prefix=":memory:", enable_router=True, enable_integration_orchestration=True, demais enable_*=False)`. O episódico do cérebro é embrulhado: `search_episodes` delega ao método real e grava args, kwargs e cada item devolvido (`episode_id`, `title`, `trigger_query`, `importance`, `status`). `start_episode` também delega e grava o id devolvido. Nada disso altera a busca.

## Critérios pré-registrados

Não serão reescritos depois do resultado. Cada um diz o que a corrida tem de registrar e o que o falsifica. A corrida não escolhe um vencedor no texto abaixo.

- **M1. Chamada.** Em `think("remember the last time", use_router=True)` num cérebro sem episódio prévio, registrar o número de chamadas a `search_episodes`, os args e os kwargs da primeira. Falsifica «status foi passado» se a chave `status` estiver nos kwargs. Falsifica «status não foi passado» se a chave `status` estiver ausente. Falsifica «a chamada é query + limit=3» se os kwargs não forem exatamente `query="remember the last time"` e `limit=3`, ou se houver positional.
- **M2. Episódio desta chamada.** Registrar o id devolvido por `start_episode` nessa `think` e a lista completa devolvida pela busca (os cinco campos de cada item, na ordem devolvida). Falsifica «o episódio recém-aberto entra na lista» se esse id não estiver na lista. Falsifica «o episódio recém-aberto fica de fora» se esse id estiver na lista. Se estiver, registrar o `status` desse item; o valor observado não será trocado por outro no texto do critério.
- **M3. Momento.** No instante em que a busca devolve, registrar `current_episode_id`. Falsifica «a busca ocorre antes de fechar» se `current_episode_id` já for `None` ou for outro id. Falsifica «a busca ocorre depois de fechar» se `current_episode_id` ainda for o id aberto nesta chamada.
- **M4. Caso negativo, fechado que casa.** Antes da `think`, fechar um episódio cujo título é exatamente `"remember the last time"`. Registrar se esse id está na lista e com qual `status`. Falsifica «um fechado que casa no título entra» se o id não estiver. Falsifica «um fechado que casa no título fica de fora» se o id estiver.
- **M5. Caso negativo, fechado que não casa.** Antes da `think`, fechar um episódio de título `"unrelated note"`. Registrar se esse id está na lista. Falsifica «a query restringe» se esse id estiver. Falsifica «a query não restringe» se esse id não estiver e, ao mesmo tempo, a lista não for vazia por outro motivo já coberto por M4.
- **M6. Caso negativo, aberto anterior.** Antes da `think`, deixar aberto um episódio de título `"remember the last time prior"`, sem `close_episode`. Registrar se esse id está na lista e, se estiver, o `status`. Falsifica «todo aberto fica de fora» se esse id estiver. Falsifica «todo aberto entra» se esse id não estiver.
- **M7. Prompt que não é recall.** Num cérebro novo, no mesmo harness, `think("ping", use_router=True)` registra o número de chamadas a `search_episodes`. Falsifica «ping não busca» se o número for maior que zero. Falsifica «ping busca» se o número for zero.

M4, M5 e M6 rodam no mesmo cérebro, nessa ordem, e depois uma única `think("remember the last time", use_router=True)`. M1–M3 rodam num cérebro sem episódio prévio, separado. M7 é um terceiro cérebro. A lista impressa é a lista devolvida pelo método, não um resumo que omita item.

## O que esta medição não decide

B2 continuaria a ser mudança de semântica. O resultado não autoriza passar `status`, incluir o aberto, nem editar produção.

## Medição

Os critérios acima não foram reescritos. Esta seção só registra a corrida.

Comando, na raiz, Python 3.11.18:

```sh
PYTHONPATH=. NEXUS_AUDIT_DB=/tmp/nexus_audit_think_recall_list.sqlite3 LC_ALL=C python3.11
```

O script imprimiu os banners do cérebro e, no fim, o JSON. Arquivo integral: `/tmp/think_recall_list_measure.json`, copiado para `/opt/cursor/artifacts/think_recall_list_measure.json`. SHA-256 `e1ef9c5814bfa79b61d6bf394cebf4b4c6427ac6b1af3b84780f633849ec9cda`. Saída do processo: 0.

### M1–M3, cérebro sem episódio prévio

`think("remember the last time", use_router=True)`.

`start_episode` devolveu `a414c783-fb46-4476-b196-e6fd2a5946bc`, com `title="Q: remember the last time"`, `trigger_query="remember the last time"`, `importance=0.7`.

Uma chamada. `args` vazio. `kwargs` exatamente `{"query": "remember the last time", "limit": 3}`. A chave `status` não está nos kwargs.

No instante da devolução, `current_episode_id` era `a414c783-fb46-4476-b196-e6fd2a5946bc`.

Lista devolvida, integral:

```json
[]
```

O id aberto nesta chamada não está na lista.

O dict de `think` não tem a chave `episodes`. O log do roteador, impresso antes da cópia para `think`, mostra a chave `episodes` no resultado de `route`.

### M4–M6, o mesmo cérebro, nessa ordem, depois uma `think`

- Fechado que casa: `b4c0036c-28b2-4ffb-9ded-9b73c613bc60`, título `"remember the last time"`.
- Fechado que não casa: `88995f24-1074-4b6b-93bc-6151537179ae`, título `"unrelated note"`.
- Aberto anterior, sem `close_episode`: `4da1649f-96df-450b-bed5-a198ef7b838e`, título `"remember the last time prior"`. `current_episode_id` antes da `think` era esse id.

`start_episode` da `think` devolveu `e75d2d68-355c-44bc-aa78-eb3213041d63`. No instante da devolução da busca, `current_episode_id` era esse id novo. Depois da `think`, `current_episode_id` era `null`.

Uma chamada. `args` vazio. `kwargs` exatamente `{"query": "remember the last time", "limit": 3}`. Sem chave `status`.

Lista devolvida, integral:

```json
[
  {
    "episode_id": "b4c0036c-28b2-4ffb-9ded-9b73c613bc60",
    "title": "remember the last time",
    "trigger_query": null,
    "importance": 0.6,
    "status": "closed"
  }
]
```

O fechado que casa está na lista, com `status` `"closed"`. O fechado que não casa não está. O aberto anterior não está. O id aberto nesta `think` não está.

### M7

`think("ping", use_router=True)` num cérebro novo. Chamadas a `search_episodes`: 0. Lista: não houve devolução. `start_episode` devolveu `7e567b36-b942-497c-8b80-424b3a3224d0`.

### Leitura pela regra dos critérios, sem reescrevê-los

- M1: «status foi passado» fica falsificado. «status não foi passado» e «a chamada é query + limit=3» não ficam falsificados.
- M2, no cérebro vazio: «o episódio recém-aberto entra na lista» fica falsificado. «fica de fora» não fica.
- M3: «a busca ocorre depois de fechar» fica falsificado. «antes de fechar» não fica.
- M4: «um fechado que casa no título fica de fora» fica falsificado. «entra» não fica. O `status` observado é `"closed"`.
- M5: o id `"unrelated note"` não está, e a lista de M4–M6 não está vazia. «a query não restringe» fica falsificado.
- M6: o aberto anterior não está. «todo aberto entra» fica falsificado. «todo aberto fica de fora» não fica, neste par de abertos.
- M7: «ping busca» fica falsificado. «ping não busca» não fica.

B2 não foi implementado.
