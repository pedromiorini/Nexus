# Pré-registro — B2b: o episódio aberto corrente só entra se a chamada pedir

Status: critérios ainda não aceitos. Não implementa B2b. Não é GREEN. A B2 pré-registrada está rejeitada em `docs/PONTE_EPISODIC_B2_RESULTADO_NEGATIVO_2026-11-18.md`. Este texto não reabre aquele critério e não o ajusta.

Revisão: a B2b fica só na camada de memória, C1–C3 e N1–N10. `core/constitutional_brain.py` inteiro é arquivo intocado, inclusive a linha `:10437`. N8 vale integralmente.

Base desta leitura e da suíte: `31f0aa0dff487cd7990e6fb29e63c093e30fc6a6`.
Python: 3.12.3 (`python3`). `python3.11` não está neste host. O CI usa 3.11. Os critérios abaixo não dependem de contagem de cobertura nem de um float que não seja o literal já gravado no fonte (`0.7` e `0.6`).

## Fato — suíte de partida, neste commit, antes destes documentos

```sh
export PYTHONPATH=. NEXUS_AUDIT_DB=/tmp/nexus_audit_b2b_base.sqlite3 LC_ALL=C PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -p 'test_*.py'
```

Saída 0. `Ran 1650 tests in 71.847s`. `OK (expected failures=5)`. Os cinco `expectedFailure` já estão no fonte (três em `test_semantic_networks_followup_contract.py`, dois em `test_red_pr156_veto_names.py`). A suíte não reportou sucesso inesperado. `test_episodic_b2_open_current_contract.py` não existe neste commit; os 1670 testes do GREEN da B2 incluíam os 20 daquele arquivo.

## Fato — o caminho que a ponte usa hoje

A única busca que o roteador dispara está em `core/constitutional_brain.py:10437`:

```python
return module.search_episodes(query=prompt, limit=3)
```

Não passa `status`. O default da assinatura é `"closed"` (`core/cognitivo/episodic_memory.py:183-184`). A função devolve `_search_validated` (`:194`). Não existe parâmetro que peça o episódio aberto.

Nesta tarefa, nesta base, a chamada direta abaixo levantou `TypeError` com a mensagem `RealEpisodicMemory.search_episodes() got an unexpected keyword argument 'include_current_open'`.

Outros chamadores, sem query:

- `get_conversation_history` chama `search_episodes(status="closed", limit=limit)` (`core/constitutional_brain.py:20821`).
- `RealIncrementalLearner.consolidate_learning` chama `search_episodes(status="closed", limit=last_n_episodes)` (`core/cognitivo/incremental_learner.py:259-261`).

Contratos vigentes congelam a chamada do roteador como args `()` e kwargs exatamente `{"query": ..., "limit": 3}`, sem outra chave:

- `C5RecallCallAndPublicResult.test_search_kwargs_are_dropped_from_the_public_dict_and_cached` (`test_router_episodic_memory_bridge_contract.py:352-354`).
- `C6ContextDoesNotRetarget.test_request_type_in_context_still_searches_and_skips_cache` (`:379`).
- `C10RealMemoryInTheMap.test_closed_episode_list_is_not_copied_into_the_public_result` (`:495`), e também a lista interna `[id_fechado]` quando o corrente é `None`.
- `C10RealMemoryInTheMap.test_open_episode_stays_open_and_the_public_result_omits_the_list` (`:517`) congela os kwargs e não lê a lista.
- `C11LearningIsNotADefaultProposer.test_forced_learning_searches_and_does_not_call_the_learner` (`:545-547`).
- `C12ThinkWithRouter.test_recall_searches_between_start_and_close` (`:589-590`) congela os mesmos kwargs entre `start_episode` e `close_episode`.

`think` abre o episódio, chama `router.route` e só depois fecha. Isso foi medido na tarefa B2 neste mesmo commit, antes do GREEN: no recall, a lista default é `[]` enquanto `current_episode_id` ainda é o id da chamada. Esta tarefa não reexecutou o `think`.

## Hipótese

**H1, hipótese de trabalho.** A chamada `search_episodes(query, limit)` e toda chamada que não pede o caminho novo continuam com o mesmo resultado que esta `main`. O episódio aberto corrente entra só quando o chamador passa um parâmetro keyword-only com default que preserva o resultado atual. A ponte não passa esse parâmetro. `core/constitutional_brain.py` permanece o arquivo desta base, linha `:10437` incluída.

**H0, alternativa nula.** Não há mudança. A B2 fica rejeitada. O `think` continua sem ver o episódio que acabou de abrir. A suíte desta base continua verde. H0 é o estado medido acima.

Falsifica H1 se uma chamada sem o parâmetro novo mudar de resultado, se o parâmetro default incluir o aberto, ou se a implementação alterar `core/constitutional_brain.py`.

## Desenho mínimo proposto

O mais simples dos dois caminhos explícitos nomeados para esta fatia é um parâmetro keyword-only, com default que não inclui o aberto:

```python
def search_episodes(self, query: Optional[str] = None,
                   status: str = "closed", limit: int = 10, *,
                   include_current_open: bool = False) -> List[Episode]:
```

O `*` impede que o flag ocupe um lugar posicional. `query`, `status` e `limit` continuam nos mesmos lugares. `search_episodes(query, limit=3)` e `search_episodes(query=prompt, limit=3)` continuam válidos.

Justificativa. Uma função nova, separada, também seria explícita, e a ponte teria de deixar de chamar `search_episodes`. Os contratos da seção anterior exigem esse nome e esses kwargs. O parâmetro mantém o nome e faz a omissão do argumento ser o comportamento já medido. Com `include_current_open` falso ou ausente, o corpo devolve o que `_search_validated` já devolve e não lê o corrente.

Com `include_current_open is True`, e só então, a lista é a que `_search_validated` devolveria, seguida de no máximo um `Episode`, quando todos os itens abaixo valem:

- `query` é `str` não vazia;
- `status == "closed"`;
- `limit` é `int` maior que zero, e não é `bool`;
- `self.current_episode_id` não é `None`;
- a linha desse id existe, `status` é `"open"`, e o predicado já usado por `_ranked_search` (FTS com os termos `\w+`, ou LIKE com o mesmo escape de `\`, `%` e `_`) a selecionaria;
- o id ainda não está na lista.

Esse `Episode` entra por último. Nenhuma linha fechada sai. A ordem do prefixo fechado não muda. O `LIMIT` do SQL fechado não corta essa linha extra. `limit == 0` continua `[]` antes desse acréscimo. Query vazia, query que não é `str` e `limit` inválido continuam a levantar no caminho atual. `status` que não é a string `"closed"` não recebe a linha extra: `"open"`, `bool`, `None` e o resto ficam como hoje, sem normalização nova. Não há segunda chamada pública a `search_episodes`. A leitura da linha é SQL interno ou função privada no mesmo arquivo, com SQL literal e placeholders. Sem `f-string` e sem concatenação na string SQL.

A linha `:10437` continua `return module.search_episodes(query=prompt, limit=3)`. O histórico (`:20821`) e o learner (`incremental_learner.py:259-261`) também não passam o flag. Quem quiser o aberto chama a memória direto, com `include_current_open=True`.

### Arquivos que uma implementação futura pode tocar

Ainda não autorizada por este texto:

- `core/cognitivo/episodic_memory.py`, no parâmetro e no acréscimo descrito acima.
- Um arquivo de teste novo, na raiz, ainda não nomeado e não criado aqui. Esse teste cobre C1–C3 e N1–N10 na memória direta.

### Arquivos intocados

- `core/constitutional_brain.py` inteiro, inclusive a linha `:10437`.
- `test_episodic_like_literal_contract.py`.
- `test_router_episodic_memory_bridge_contract.py` e os outros contratos vigentes.
- `tools/targeted_mutation.py` e `tools/cognitivo_mutation.py`.
- Workflows, timeouts e pisos de cobertura.
- `docs/agentes/CLAIMS.yaml`, `docs/agentes/REGISTRO_AGENTES.yaml`, `CONTINUATION.md` e os outros documentos de estado.
- `get_conversation_history`, `consolidate_learning` e qualquer busca sem query.

## Critérios que falham na main

Um teste futuro, ainda não escrito, deve falhar nesta `31f0aa0` e só passaria depois da mudança mínima do desenho. Memória direta: `RealEpisodicMemory(":memory:")`. A B2b não usa o cérebro.

- **C1.** `include_current_open=True` devolve o corrente que casa. `start_episode("Q: remember the last time", trigger_query="remember the last time", importance=0.7)`, sem fechar. `search_episodes("remember the last time", limit=3, include_current_open=True)` devolve um `Episode`: esse id, título `"Q: remember the last time"`, `trigger_query` `"remember the last time"`, `importance == 0.7`, `status == "open"`. `current_episode_id` continua esse id. Na main, medido nesta tarefa: `TypeError` com `unexpected keyword argument 'include_current_open'`.

- **C2.** O fechado permanece e o aberto corrente entra depois, só com o flag. Primeiro `start_episode("remember the last time")` e `close_episode` desse id (default `importance` 0,6 e `trigger_query` `None`). Depois o `start_episode` de C1, sem fechar. A chamada de C1 devolve dois itens, nesta ordem: o fechado e o aberto. Na main: o mesmo `TypeError`.

- **C3.** O `LIMIT` fechado não corta o extra quando o flag está ligado. Três fechados, títulos `"remember the last time 0"`, `"remember the last time 1"` e `"remember the last time 2"`, cada um com `trigger_query="remember the last time"`. Com o corrente `None`, `search_episodes("remember the last time", limit=3)` devolve um prefixo de três ids. Abrir o episódio de C1 e chamar `search_episodes("remember the last time", limit=3, include_current_open=True)`. A lista é o prefixo seguido do id aberto, uma vez. Na main: o mesmo `TypeError`. A chamada sem o flag, nesta main, devolve só o prefixo. Isso é não-regressão, N3.

## Fora de escopo: B2c futura

C4–C7 do rascunho anterior saem desta fatia. A B2c exigirá pré-registro próprio, que declare de antemão quais asserções de kwargs do contrato da ponte ela substitui. Este texto não define os critérios da B2c.

## Critérios de não-regressão

Já valem nesta `31f0aa0`. Uma implementação futura só é válida se continuarem valendo, sem editar os testes existentes.

- **N1.** `test_episodic_like_literal_contract.py` permanece o arquivo desta main e passa sem edição. A suíte de partida saiu 0 e inclui esse arquivo.

- **N2.** As três asserções que rejeitaram a B2 continuam verdes para a chamada que não passa o flag. `search_episodes("%")` devolve títulos equivalentes a `"100%"` e `"note"`, sem `"100% open"`. `search_episodes("%", status="open")` devolve `"100% open"`. `search_episodes("%", limit=1)` tem comprimento 1. O `setUp` desse arquivo deixa `"100% open"` aberto e corrente. Foi isso que a B2 anexou.

- **N3.** A chamada default não ganha o corrente. O cenário de C1, com `search_episodes("remember the last time", limit=3)` sem o flag, devolve `[]` e `current_episode_id` continua o id aberto. O cenário de C3, sem o flag, devolve só o prefixo de três ids. Isto é o oposto do C1 e do C3 da B2 rejeitada, e é o que esta fatia preserva.

- **N4.** Um aberto que não é o corrente continua de fora, com ou sem o flag. `test_status_filter_preserved` (`test_episodic_retrieval_contract.py:44-49`): o aberto `"kernel panic"` não é o corrente na hora da busca; `search_episodes("kernel")` devolve só o fechado; `search_episodes("kernel", status="open")` devolve só o aberto. Com `include_current_open=True`, o resultado default dessa montagem não ganha o `"kernel panic"`, porque ele não é `current_episode_id`.

- **N5.** Busca sem query não ganha o corrente. `test_fixed_fixture_matches_main_baseline` (`test_episodic_memory_extraction_contract.py:79-96`): `"beta"` fica aberto e corrente, `"alpha"` fecha, `search_episodes(limit=10)` devolve só `"alpha"`, e `search_episodes("alp", limit=5)` devolve só `"alpha"`. `"beta"` não casa com `"alp"`. O flag verdadeiro também não anexa `"beta"` nessa query.

- **N6.** `limit == 0` continua `[]`, também com o flag verdadeiro, no episódio de C1. Query `""` e query que não é `str` continuam a levantar antes do SQL. `limit` negativo e `limit` que não é `int` continuam a levantar. `bool` não vira inteiro.

- **N7.** `status="open"` não duplica. Dois abertos que casam (`"remember the last time prior"` e o episódio de C1, este corrente). `search_episodes("remember the last time", status="open", limit=10)` devolve os dois ids, cada um uma vez. A mesma chamada com `include_current_open=True` continua com comprimento 2. O flag não soma uma segunda cópia, porque `status` não é `"closed"`.

- **N8.** Os kwargs observados da ponte continuam exatamente `{"query": ..., "limit": 3}`, sem `include_current_open` e sem `status`, nas seis asserções listadas no fato da ponte. A forma da chamada do recall continua uma vez, entre `start_episode` e `close_episode`. O dict de `think` não ganha a chave `episodes`.

- **N9.** A suíte `python -m unittest discover -p 'test_*.py'` passa sem editar testes existentes, salvo a exceção estreita autorizada pelo owner em 2026-11-20: alterar somente as asserções de claims em `test_agent_pr79_composition_contract.py` e `test_router_episodic_memory_bridge_contract.py` para validar claims atuais; nenhuma asserção funcional B2B é alterada. `test_episodic_b2b_open_current_contract.py` mantém a pinagem SHA-256 dos testes: somente esses dois digests são atualizados para os bytes autorizados, e qualquer outra edição invalida a fatia. Nesta base histórica: 1650 testes, saída 0, cinco `expectedFailure`. Um arquivo de teste novo, se uma tarefa posterior o criar, não conta como edição dos existentes; qualquer outra edição existente para voltar ao verde também invalida a fatia.

- **N10.** O baseline do gate cognitivo chega a correr. `python tools/cognitivo_mutation.py --shard K/3`, para K em 1, 2 e 3, não pode sair com `Baseline failed before mutation (test_episodic_like_literal_contract.py)`. Na B2, os três shards saíram assim e nenhum mutante correu. Nesta base a suíte que contém esse arquivo saiu 0. Esta tarefa não reexecutou o gate. **Inferência:** o arquivo de curinga não está vermelho aqui, então o aborto da B2 não se repete por aquelas três falhas. O critério futuro é o processo do gate passar da linha de baseline, não a inferência.

## Mutante futuro, sem edição agora

Item separado. Exige aprovação própria para tocar `tools/targeted_mutation.py`. Este texto não autoriza essa edição, nem a de `tools/cognitivo_mutation.py`.

Dois mutantes previstos, para essa tarefa futura:

- O default `include_current_open: bool = False` vira `True`, ou o guarda do flag vira sempre verdadeiro. Quem mata é N2: `"100% open"` voltaria à busca de `"%"` e `limit=1` deixaria de ter comprimento 1. É o furo da B2.
- O ramo `include_current_open is True` desaparece. Quem mata é C1: a chamada explícita deixaria de devolver o aberto corrente.

Enquanto o arquivo não for autorizado, a lacuna fica declarada. Ausência de mutante não é evidência de que o ramo está coberto.

## O que invalida a B2b

A implementação para e o critério não é reescrito se qualquer item abaixo acontecer.

- Uma chamada sem `include_current_open=True` muda de resultado. Em particular, N2 fica vermelho. Foi a rejeição da B2.
- O default do parâmetro inclui o aberto. Isso é a B2 com outro nome.
- `core/constitutional_brain.py` muda, em qualquer linha, inclusive `:10437`. N8 continua a exigir kwargs exatamente `{"query": ..., "limit": 3}`, sem `include_current_open` e sem `status`. Passar o flag na ponte, ou qualquer outra edição do monólito, invalida a fatia.
- O histórico ou o learner passam a pedir o flag.
- Todo `status` `"open"` que casa entra na busca, ou o SQL passa a aceitar `"closed"` e `"open"` dentro do `LIMIT`.
- Há segunda chamada pública a `search_episodes`, ou o dict de `think` ganha `episodes`.
- `limit == 0`, query inválida ou `limit` inválido passam a devolver o aberto em vez de `[]` ou do erro atual.
- O SQL do caminho novo é montado com `f-string` ou concatenação.
- Este texto é editado para um teste futuro passar.

## O que esta tarefa não fez

Não criou teste. Não editou `core/cognitivo/episodic_memory.py`, `core/constitutional_brain.py`, `test_episodic_like_literal_contract.py`, `tools/targeted_mutation.py`, workflows, timeouts, pisos nem `CLAIMS.yaml`. Não fez push, não abriu PR e não criou branch remota. A branch da B2, `cursor/episodic-b2-open-current-ada1`, não foi movida.
