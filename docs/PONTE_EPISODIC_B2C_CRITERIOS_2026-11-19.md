# Pré-registro — B2c: a rota vê o episódio aberto corrente

Status: critérios antes de qualquer edição de produção ou de teste. Não implementa B2c. Não escolhe a alternativa. Não é GREEN.

Base: `origin/main` `8fd87982e248a9aa12997d864683edbe633fa98b` (`B2b: registrar M1 e M2 no gate de mutação direcionada (#200)`).
Python da sonda: 3.12.3. `python3.11` não está neste host.

A B2b já entregou `include_current_open: bool = False`, keyword-only, em `search_episodes`. Sem o flag, o resultado é o de hoje. Com `True`, só o episódio aberto corrente entra, no fim da lista fechada, se casar no predicado. O pré-registro da B2b deixou C4–C7 de fora e exigiu que a B2c declarasse de antemão quais asserções de kwargs ela substitui. Este texto faz essa declaração e para na decisão do owner.

## Fato — a ponte hoje

A única busca do roteador está em `core/constitutional_brain.py:10437`:

```python
return module.search_episodes(query=prompt, limit=3)
```

Não passa `include_current_open` nem `status`. O default é `False` e `"closed"`.

O histórico chama `search_episodes(status="closed", limit=limit)` sem query (`:20821`). O learner chama `search_episodes(status="closed", limit=last_n_episodes)` sem query (`core/cognitivo/incremental_learner.py:259-261`).

O catálogo direcionado tem 87 `Mutation`. O cognitivo tem 1137, 379 por shard.

## Fato — o que quebra se a ponte passar o flag

Nesta VM, a linha `:10437` foi trocada, sem commit, por `return module.search_episodes(query=prompt, limit=3, include_current_open=True)`. A suíte rodou e o arquivo voltou ao original. `git diff` depois ficou vazio.

```sh
export PYTHONPATH=. NEXUS_AUDIT_DB=/tmp/nexus_audit_b2c_probe.sqlite3 LC_ALL=C
python3 -m unittest discover -p 'test_*.py'
```

Saída 1. `Ran 1663 tests in 78.147s`. `FAILED (failures=10, expected failures=5)`. Falharam só estes:

Contrato da ponte, `test_router_episodic_memory_bridge_contract.py`:

- `C5RecallCallAndPublicResult.test_search_kwargs_are_dropped_from_the_public_dict_and_cached`
- `C6ContextDoesNotRetarget.test_request_type_in_context_still_searches_and_skips_cache`
- `C10RealMemoryInTheMap.test_closed_episode_list_is_not_copied_into_the_public_result`
- `C10RealMemoryInTheMap.test_open_episode_stays_open_and_the_public_result_omits_the_list`
- `C11LearningIsNotADefaultProposer.test_forced_learning_searches_and_does_not_call_the_learner`
- `C12ThinkWithRouter.test_recall_searches_between_start_and_close`

Errata do mesmo contrato, `test_router_episodic_memory_bridge_e1_e6_red_contract.py`:

- `ErrataC5AndE2EmptySuccess.test_successful_empty_search_projects_an_empty_list_and_caches_it`
- `E1ClosedEpisodeProjection.test_closed_episode_is_projected_and_the_second_route_repeats_it`
- `E5ThinkDoesNotGainEpisodes.test_recall_think_keeps_c12_and_has_no_episodes`

B2b, `test_episodic_b2b_open_current_contract.py`:

- `EpisodicB2bOpenCurrentContractTests.test_n8_bridge_kwargs_stay_query_and_limit`

Esses testes exigem args `()` e kwargs exatamente `{"query": ..., "limit": 3}`. N8 também exige que a linha `:10437` seja o retorno sem o flag. Nenhum outro teste da suíte falhou.

## Hipótese

**H1.** A rota do `think` e o `route` direto passam a ver o episódio aberto corrente, porque a ponte pede `include_current_open=True` na linha `:10437`. O histórico e o learner não pedem o flag. O default do parâmetro continua `False`.

**H0.** Não há mudança. A linha `:10437` permanece sem o flag. C4–C7 continuam vermelhos na rota. A suíte desta base continua verde. H0 é o estado medido: a suíte sem a troca sai verde nos gates já registrados da B2b, e a troca medida acima produz as 10 falhas.

Falsifica H1 se o aberto corrente entrar sem a ponte pedir o flag, se o histórico ou o learner passarem a pedi-lo, ou se a suíte só voltar a sair 0 com edição de teste que o owner não tenha autorizado pelo nome.

## C4–C7 — comportamento observável na rota

Memória real. Harness do cérebro: `brain_kwargs` de `test_router_episodic_memory_bridge_contract.py:197-214`. O episódio de C1 da B2b é `start_episode("Q: remember the last time", trigger_query="remember the last time", importance=0.7)`, sem fechar.

Nesta `main` a linha não pede o flag. **Inferência**, a partir dessa linha e de N3 da B2b: a busca da rota não anexa o aberto corrente. Esta tarefa não reexecutou o `think` sem a troca.

- **C4.** `think("remember the last time", use_router=True)`. A única `search_episodes` fica entre `start_episode` e `close_episode`. A lista dessa chamada tem um item: o `episode_id` do dict de `think`, título `"Q: remember the last time"`, `trigger_query` `"remember the last time"`, `importance == 0.7`, `status` `"open"`. Na devolução da busca, `current_episode_id` é esse id. O dict de `think` não tem a chave `episodes`.

- **C5.** `question = "remember " + ("x" * 80)`. `think(question, use_router=True)`. A `query` é a pergunta inteira. A lista tem um item: título `"Q: " + question[:50]`, `trigger_query` a pergunta inteira, `importance == 0.7`, `status` `"open"`, id igual ao `episode_id`. `request_type` é `"recall"`. O dict de `think` não tem `episodes`.

- **C6.** No mesmo cérebro: fechar `start_episode("remember the last time")` (default `importance` 0,6 e `trigger_query` `None`); deixar aberto `start_episode("remember the last time prior")`; `think("remember the last time", use_router=True)`. A lista é o fechado e, depois, o episódio desta `think` com os campos de C4. O id anterior não está. Na devolução, `current_episode_id` é o id desta `think`.

- **C7.** Memória real no mapa. Fechar `start_episode("remember the last time")`. Abrir o episódio de C1. `route("remember the last time", {})` devolve `success is True`, `degraded is False` e `episodes` com dois dicts, fechado primeiro e aberto depois. Cada dict tem exatamente `episode_id`, `title`, `trigger_query`, `importance`, `status`. O dict da rota não tem `episode_id` nem `episodic`. `current_episode_id` continua o aberto: `route` não fecha.

## Não-regressão

- O histórico (`:20821`) e o learner (`incremental_learner.py:259-261`) continuam sem `include_current_open` e sem query.
- Toda chamada que não é a linha `:10437` permanece com os mesmos argumentos.
- O default `include_current_open=False` permanece. `test_episodic_like_literal_contract.py` continua verde sem edição.
- A suíte `python -m unittest discover -p 'test_*.py'` sai 0, com os cinco `expectedFailure` já existentes.
- `tools/cognitivo_mutation.py` permanece com 1137 mutantes, 379 por shard.
- `tools/targeted_mutation.py` permanece com 87 mutantes, os de M1 e M2 incluídos, mortos pelos testes atuais.
- `core/constitutional_brain.py` não muda fora da linha `:10437`, e só muda essa linha se o owner escolher a alternativa (a).

## RED e GREEN previstos

RED, nesta `main`, sem editar teste: um teste futuro de C4–C7 falha porque a rota não pede o flag e a lista não traz o aberto corrente. A suíte atual passa. A troca da linha, medida acima, não é RED de C4–C7: é a lista das 10 falhas.

GREEN depende da alternativa que o owner escolher. Não está autorizado por este texto.

- Se for (a): a linha `:10437` passa `include_current_open=True`; os testes nomeados no fato passam a esperar esse kwarg, por autorização explícita e prévia; C4–C7 ficam verdes; a suíte sai 0; os catálogos continuam 1137 e 87.
- Se for (b): os kwargs observados continuam exatamente `{"query", "limit": 3}` e, mesmo assim, C4–C7 ficam verdes. Este texto não descreve o mecanismo. Se não houver caminho legítimo, (b) não tem implementação.
- Se for (c): nada muda. C4–C7 permanecem vermelhos. A suíte permanece verde.

## Decisão do owner — alternativas

Os testes existentes não podem ser editados sem autorização específica. Esta seção não escolhe.

**(a) Emendar o contrato da #178, a errata e o N8, por autorização explícita.**

A ponte pede o flag na linha `:10437`. Os testes que congelam os kwargs passam a esperar `include_current_open=True`, sem mudar o resto da chamada: continua uma busca, `query` e `limit=3`, entre `start_episode` e `close_episode` no recall. O dict de `think` continua sem `episodes`.

Prós. O pedido é visível na única chamada que o roteador faz. Reusa o parâmetro que a B2b já entregou. C4–C7 descrevem o efeito dessa chamada. O histórico e o learner ficam de fora porque não são essa linha.

Contras. Edita testes que a #178 e a B2b congelaram. A autorização tem de nomear, antes do GREEN, cada método da lista de 10. Sem isso, a suíte fica vermelha, como a sonda mostrou.

**(b) Um caminho que não mude os kwargs da ponte.**

A linha `:10437` continua `search_episodes(query=prompt, limit=3)`. Os dez testes permanecem como estão. C4–C7 ainda ficariam verdes por outro meio.

Prós. Não abre o contrato de kwargs. A suíte atual não precisa de emenda.

Contras. O pedido deixa de aparecer nos kwargs que os contratos registram. Um embrulho só na instância do cérebro, ou outro desvio, esconde a chamada que a B2b tornou explícita. Este texto não encontrou um caminho que seja ao mesmo tempo explícito na ponte e invisível para esses `assertEqual` de kwargs. **Inferência:** (b), se existir, não é a linha `:10437` com o flag.

**(c) Não fazer a B2c.**

Prós. A suíte, os contratos e o default permanecem. Não há edição do monólito.

Contras. O `think` e o `route` continuam sem ver o episódio aberto corrente. Quem quiser o aberto chama `search_episodes` direto, com o flag.

## O que invalida

- Passar o flag na linha `:10437` e tratar a suíte como verde enquanto qualquer um dos 10 testes ainda espera os kwargs antigos.
- Editar um desses testes, ou qualquer outro `test_*.py`, sem autorização do owner que nomeie o método antes do GREEN.
- Mudar o default para `True`, ou fazer o histórico ou o learner passarem o flag.
- Incluir todo `status` `"open"`, ou alargar o SQL para fechado e aberto dentro do `LIMIT`.
- Publicar `episodes` no dict de `think`.
- Mudar `core/constitutional_brain.py` fora da linha `:10437`.
- Alterar o tamanho dos catálogos de mutação para fazer a B2c caber.
- Reescrever este texto para um GREEN futuro passar.

## Escopo proibido

Este documento não edita produção, testes, `tools/targeted_mutation.py`, `tools/cognitivo_mutation.py`, workflows, timeouts, pisos nem `docs/agentes/CLAIMS.yaml`. Não faz push nem PR. Não escolhe entre (a), (b) e (c).
