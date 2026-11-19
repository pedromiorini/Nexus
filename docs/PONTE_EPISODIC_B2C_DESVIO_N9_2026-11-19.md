# Desvio — N9 da B2b depois da emenda de kwargs da B2c

Status: registro do que a corrida local mediu e da 11ª emenda que o owner autorizou depois. Não reabre o pré-registro. Não escolhe outra alternativa. Não é uma mudança de produção.

Branch: `cursor/episodic-b2c-criterios-local`.
Base de produção: `8fd87982e248a9aa12997d864683edbe633fa98b`.
Pré-registro: `docs/PONTE_EPISODIC_B2C_CRITERIOS_2026-11-19.md`, SHA-256 `55d884a6e3e6940c3ad165a00e5c812838f765543327fa05d889195588cc25c5`, commit `5de49616aeec71358fbc40e57567d1a37ff10243`. Esse arquivo não foi editado.

## Fato — o que o pré-registro nomeou

O texto `55d884a6…` lista 10 testes cuja asserção de kwargs a alternativa (a) substitui:

Contrato da ponte, `test_router_episodic_memory_bridge_contract.py`:

- `C5RecallCallAndPublicResult.test_search_kwargs_are_dropped_from_the_public_dict_and_cached`
- `C6ContextDoesNotRetarget.test_request_type_in_context_still_searches_and_skips_cache`
- `C10RealMemoryInTheMap.test_closed_episode_list_is_not_copied_into_the_public_result`
- `C10RealMemoryInTheMap.test_open_episode_stays_open_and_the_public_result_omits_the_list`
- `C11LearningIsNotADefaultProposer.test_forced_learning_searches_and_does_not_call_the_learner`
- `C12ThinkWithRouter.test_recall_searches_between_start_and_close`

Errata, `test_router_episodic_memory_bridge_e1_e6_red_contract.py`:

- `ErrataC5AndE2EmptySuccess.test_successful_empty_search_projects_an_empty_list_and_caches_it`
- `E1ClosedEpisodeProjection.test_closed_episode_is_projected_and_the_second_route_repeats_it`
- `E5ThinkDoesNotGainEpisodes.test_recall_think_keeps_c12_and_has_no_episodes`

B2b, `test_episodic_b2b_open_current_contract.py`:

- `EpisodicB2bOpenCurrentContractTests.test_n8_bridge_kwargs_stay_query_and_limit`

A sonda do pré-registro, com a linha `:10437` trocada e depois revertida, teve 10 falhas, só esses métodos. `Ran 1663 tests in 78.147s`. `FAILED (failures=10, expected failures=5)`. O N9 não está nessa lista.

## Fato — o que o N9 fixa

`test_n9_existing_tests_are_not_edited` percorre `EXISTING_TEST_SHA256` e compara `hashlib.sha256` dos bytes de cada `test_*.py` que existia em `31f0aa0`. Os dois arquivos da ponte estão nesse mapa. Os pins medidos com `git show 31f0aa0:<arquivo>` são:

| Arquivo | SHA-256 em `31f0aa0` |
|---|---|
| `test_router_episodic_memory_bridge_contract.py` | `3d3fabe0ff069ebe9f8143b8b17320156f154b8b44861dc0cc2afa286d5a3893` |
| `test_router_episodic_memory_bridge_e1_e6_red_contract.py` | `2df0499cd9febc8bde6ed7063b0aafd4781627d728534e8223be2da0cd29832e` |

O teste para na primeira divergência. O método N9 em si não estava entre os 10. O pré-registro não diz que a emenda desses arquivos troca o pin.

## Fato — commits da corrida, até o GREEN

| Papel | SHA |
|---|---|
| Pré-registro | `5de49616aeec71358fbc40e57567d1a37ff10243` |
| RED de C4–C7, produção ainda sem o flag | `794aaeed140d8fe400a3863f00ad75112a53bcc9` |
| Emenda dos 10 métodos | `cdfabc4e681c754cee46dcc106f27a1d396e8a37` |
| GREEN, só a linha `:10437` | `51747d5eae8e26e5f2858ce32f8fd7217486a689` |

No RED, `:10437` ainda era `return module.search_episodes(query=prompt, limit=3)`. C4–C7 falharam porque a lista não trazia o aberto. A suíte nesse ponto: 1667 testes, 78,786 s, `FAILED (failures=4, expected failures=5)`, só C4–C7.

No GREEN a linha passou a `return module.search_episodes(query=prompt, limit=3, include_current_open=True)`. C4–C7: 4 testes, 0,077 s, `OK`. Histórico `:20821` e learner `:259–261` continuam sem o flag.

## Fato — a falha depois do GREEN

`python3 -m unittest discover -p 'test_*.py'` em `51747d5`: 1667 testes, 223,687 s, saída 1, `FAILED (failures=1, expected failures=5)`. A única falha:

```
FAIL: test_n9_existing_tests_are_not_edited
AssertionError: '42b9df0f2b6fd7b71fae43a1511ab4192cea16f3582ac687a80d90f500d77767' != '3d3fabe0ff069ebe9f8143b8b17320156f154b8b44861dc0cc2afa286d5a3893'
 : test_router_episodic_memory_bridge_contract.py
```

`sha256sum` nesse HEAD, antes de qualquer edição do mapa:

```
42b9df0f2b6fd7b71fae43a1511ab4192cea16f3582ac687a80d90f500d77767  test_router_episodic_memory_bridge_contract.py
0a5324cb32e88199310f32c963803127fea9456b035308f08f98eaa133c968c1  test_router_episodic_memory_bridge_e1_e6_red_contract.py
```

O segundo arquivo também diverge do pin de `31f0aa0`. O teste não chegou a relatá-lo porque para no primeiro `assertEqual`.

`python3 tools/targeted_mutation.py` no mesmo HEAD saiu 1 antes dos mutantes:

```
Baseline failed before mutation (test_episodic_b2b_open_current_contract.py, test_episodic_like_literal_contract.py):
FAIL: test_n9_existing_tests_are_not_edited
```

O placar 87/87 não existe nessa corrida. Os shards cognitivos 1/3, 2/3 e 3/3 saíram 0: 379 mortos cada, catálogo 1137. Ruff 0.17.1 saiu 0. Cobertura ficou acima dos pisos. Replay 17/17. Bandit 1.9.4: 139 LOW dos dois lados do diff contra `8fd8798`, sem diferença de achado. Drift saiu 0.

Nada além do autorizado foi editado para fazer esses gates passarem. O N9 e o mapa ficaram intactos até esta autorização.

## Fato — a 11ª emenda

O owner autorizou, só neste mapa e só estes dois pins, os SHA-256 medidos em `51747d5`:

| Arquivo | Pin antigo (`31f0aa0`) | Pin novo (`51747d5`) |
|---|---|---|
| `test_router_episodic_memory_bridge_contract.py` | `3d3fabe0ff069ebe9f8143b8b17320156f154b8b44861dc0cc2afa286d5a3893` | `42b9df0f2b6fd7b71fae43a1511ab4192cea16f3582ac687a80d90f500d77767` |
| `test_router_episodic_memory_bridge_e1_e6_red_contract.py` | `2df0499cd9febc8bde6ed7063b0aafd4781627d728534e8223be2da0cd29832e` | `0a5324cb32e88199310f32c963803127fea9456b035308f08f98eaa133c968c1` |

Nenhum outro pin muda. Nenhuma outra linha de `test_episodic_b2b_open_current_contract.py` muda. Produção, C4–C7, os 10 métodos já emendados, catálogos, workflows e `CLAIMS.yaml` ficam fora deste commit. Este arquivo e essa troca de pins entram no mesmo commit.

**Inferência.** Os 10 métodos autorizados no pré-registro alteram os bytes que o N9 congela. Atualizar o pin registra o arquivo já emendado. Não devolve os bytes de `31f0aa0`.

**Hipótese.** Nenhuma.
