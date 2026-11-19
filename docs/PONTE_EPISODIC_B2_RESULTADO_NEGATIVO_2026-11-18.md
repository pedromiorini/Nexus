# Resultado negativo — B2 da ponte episódica

Status: registro do que a tarefa local mediu. Não reabre o critério. Não é uma implementação. Não autoriza GREEN.

Base do código de produção em que o RED foi medido: `31f0aa0dff487cd7990e6fb29e63c093e30fc6a6`.
Critérios: `docs/PONTE_EPISODIC_B2_CRITERIOS_2026-11-13.md` na branch local `cursor/episodic-b2-open-current-ada1`. SHA-256 do arquivo: `e660589f2b20a4f948a0c8f0d28c1a0d7a8e88ff0c710bf7ab0bbbb045d9a53b`.
Essa branch permanece intacta em `d4154532e7e57cf17047b16c5f3c05c9208da20d`. Este documento nasce noutra branch, a partir de `31f0aa0`, sem o diff de produção da B2.

## Fato — commits locais da tentativa

| Papel | SHA |
|---|---|
| Cópia byte a byte dos critérios | `e527f4c3c97d291234591723fd2d19d0cb5efdfc` |
| Teste C1–C20, antes de produção | `a602dd443b33e7ddd465a51e2472cd8a5cc394c9` |
| GREEN em `search_episodes` | `d2d3fd6229663e5dcf91bac120f80c6cb4b8974b` |
| Relatórios de realidade, depois do drift 7 | `d4154532e7e57cf17047b16c5f3c05c9208da20d` |

Python da medição: 3.12.3. `python3.11` não estava no host. Pacotes: `requirements-ci.lock` com `--require-hashes`. Não houve push, PR, branch remota, comentário, nem edição de `CLAIMS.yaml`.

## Fato — RED, produção ainda em `31f0aa0`

`python -m unittest -v test_episodic_b2_open_current_contract.py`: 20 testes, 7 falhas, saída 1, 0,430 s. C1–C7 falharam. C8–C20 passaram. O padrão era o que os critérios previam. Nenhum critério foi reescrito.

- **C1.** `search_episodes("remember the last time", limit=3)`, com o episódio do `think` ainda aberto, devolveu `[]`.
- **C2.** A lista tinha só o fechado (`trigger_query` `None`, `importance` 0,6, `status` `"closed"`). O aberto não entrou.
- **C3.** Com três fechados e `limit=3`, o prefixo tinha três ids. Depois de abrir o corrente, a segunda lista era esse prefixo.
- **C4.** A chamada do recall era `()` e `{"query": "remember the last time", "limit": 3}`, entre `start_episode` e `close_episode`, e `current_episode_id` na devolução já era o id do `think`. A lista era `[]`.
- **C5.** A pergunta longa era `request_type` `"recall"`, a `query` era a pergunta inteira, e a lista era `[]`.
- **C6.** O corrente na devolução era o id desta `think`. A lista era só o fechado. O aberto anterior não estava nela.
- **C7.** `route` devolvia `success` verdadeiro, `degraded` falso, sem as chaves `episode_id` e `episodic`. `episodes` tinha só o dict fechado.

## Fato — GREEN dos vinte critérios, e a suíte

Depois de `d2d3fd6`, o mesmo arquivo deu 20 testes, `OK`, saída 0, 0,395 s.

A suíte `python -m unittest discover -p 'test_*.py'` nesse GREEN: 1670 testes, 213,855 s, saída 1, `FAILED (failures=3, expected failures=5)`. As três falhas são só estas, em `test_episodic_like_literal_contract.py`:

- `test_percent_matches_only_literal_percent`. `assertCountEqual` de `search_episodes("%")` com `["100%", "note"]`. A lista real também tinha `"100% open"`.
- `test_open_episode_stays_out_of_the_default_status`. `assertNotIn("100% open", search_episodes("%"))`. `"100% open"` estava em `["note", "100%", "100% open"]`.
- `test_limit_still_bounds_literal_matches`. `len(search_episodes("%", limit=1))` era 2. O teste exige 1.

O arquivo não foi editado. O GREEN não foi alterado para excluir esse caso.

## Fato — mecanismo do choque

O `setUp` daquele contrato fecha várias linhas e depois faz `start_episode("100% open")` sem `close_episode`. Essa linha fica `open` e é `current_episode_id`.

`search_episodes("%")` usa o default `status="closed"`. A query é string não vazia. `limit` default é 10, e no terceiro teste é 1, ambos maiores que zero.

O GREEN da B2 anexa o corrente quando a query é string não vazia, `status == "closed"`, `limit` é `int` maior que zero e o predicado de `_ranked_search` selecionaria a linha. Em `"%"` não há termo `\w+`. O FTS não corre. O LIKE com escape de `%` casa o caractere literal. `"100% open"` contém `%`, está `open` e é o corrente. A linha entra no fim da lista fechada. O `LIMIT` do SQL fechado não a corta: com `limit=1` a lista fechada tem um item e a lista devolvida tem dois.

**Inferência.** É o mesmo acréscimo que deixou C2 e C3 verdes. Os três testes exigem o contrário para a chamada default: o aberto corrente que casa com `"%"` fica de fora, e `limit=1` limita o resultado inteiro.

## Fato — baseline do gate cognitivo

`python tools/cognitivo_mutation.py --jobs 2 --shard 1/3`, `2/3` e `3/3` saíram 1. Os três pararam com `Baseline failed before mutation (test_episodic_like_literal_contract.py)` e as mesmas três falhas. Nenhum mutante chegou a correr. Não há placar de mortos.

`python tools/targeted_mutation.py`, no mesmo GREEN, saiu 0: 85 declarados, 85 aplicáveis, 85 mortos. Esse catálogo não inclui o arquivo de curinga. O arquivo não foi editado.

## Conclusão

A B2 como pré-registrada é rejeitada. O comportamento novo que ela pedia na chamada default `search_episodes(query, limit)` torna vermelhos três testes que os próprios critérios deixaram fora do diff. Não houve ajuste de critério, nem edição de `test_episodic_like_literal_contract.py`, para fazer esse GREEN caber na suíte.

**Hipótese, não seguida.** Uma implementação futura pode pedir o episódio aberto por um caminho explícito, com a chamada default intacta. Isso não é este documento. O pré-registro separado está em `docs/PONTE_EPISODIC_B2B_CRITERIOS_2026-11-18.md`.
