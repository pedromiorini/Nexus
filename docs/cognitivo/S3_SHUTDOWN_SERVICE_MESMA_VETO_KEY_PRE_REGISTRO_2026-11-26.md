# Pré-registro S3 — `shutdown.service`, mesma `veto_key`, simulado sem veto

Status: critérios antes de qualquer RED, de qualquer edição de produção e de qualquer edição de `test_*.py`. Não implementa S3. A seção 5 recomenda a alternativa 1 e congela os critérios dessa alternativa. Se o Pedro escolher a 2 ou a 3, os critérios S3-C são reescritos antes de qualquer RED.

Autorização deste texto: Pedro Miorini, 2026-11-26 20:35 America/Sao_Paulo — preparar só localmente, sobre a main `d7ac1fd`, o pré-registro da lacuna `shutdown.service` (mesma `veto_key`, simulado sem veto), sem produção, sem push e sem PR.

Base: `d7ac1fdd53d0e2149770b8b06d71235ba36548f3` (`fix(counterfactual): fallback de tokens só com a mesma veto_key (S2 Pearl) (#227)`).
Python da sonda: 3.12.3. `python3.11` não está neste host. O CI usa 3.11; esta sonda não o executou.
`hypothesis` não está instalado neste interpretador (`ModuleNotFoundError`). O gate de mutação não foi executado. A troca da agulha abaixo foi feita numa cópia do método, em memória, fora do arquivo de produção.
Instruções aceitas nesta sessão: `docs/agentes/versoes/3.4.2/8f03762d2a2d128eea955597f1b55210e387c08385bdc5544514ea2f39701b5d.md`, SHA-256 `8f03762d2a2d128eea955597f1b55210e387c08385bdc5544514ea2f39701b5d`.

Consulta de PRs abertos nesta sessão (`gh pr list -R pedromiorini/Nexus --state open` e a API de pulls) devolveu lista vazia. `docs/agentes/CLAIMS.yaml` nesta base tem `claims: []`. Este commit não abre claim e não edita o registro de agentes.

Pré-registro S2 aceito: `docs/cognitivo/S2_FALLBACK_TOKENS_VETO_KEY_PRE_REGISTRO_2026-11-26.md`. Errata CF9: `docs/cognitivo/S2_ERRATA_CF9_2026-11-26.md`. A seção 6 do S2 deixou esta lacuna fora de escopo.

## 1. Problema

Depois do S2, o fallback de `_map_intervention` só devolve o nome da tabela quando o conjunto de tokens `[a-z0-9]+` coincide e `veto_key(intervenção) == veto_key(nome)`. `shutdown.service` passa nesse filtro: a `veto_key` dos dois lados é `SHUTDOWN_SERVICE`. `generate_counterfactual_scenario` então simula `SHUTDOWN_SERVICE` e grava `system_status=offline`. Esse caminho não chama `veto_lookup`. `rank_counterfactuals` não usa o mapa: a grafia não é chave de `ACTION_EFFECTS`, cai em `unmodeled` com `ranking` e `vetoed` vazios, e o `veto_lookup` que marcaria a grafia não chega a rodar.

`veto_lookup("shutdown.service", DEFAULT_VETOES)` sozinho acerta, com o motivo de `SHUTDOWN_SERVICE`. `compare_allowed_futures` também veta essa grafia. O escape é o par generate/rank do contrafactual, não a função de veto.

## 2. Caminho medido

Fiação da sonda, a mesma do contrato S2: `RecordingWorld` (subclasse de `RealWorldModelSimulator` que grava `simulate`), `RealCausalReasoning` que já extraiu `noop causes nothing relevant`, situação `"system vulnerable"`. `RANK` usa a observação do contrato S2 (`system_status=compromised`, `alert_level=critical`, `security_patch_applied=False`, `logging_level=normal`) e factual `["IGNORE_ALERT"]`.

`_normalize_action_key` (`counterfactual.py:19-22`) só troca espaço e hífen ASCII por `_`. O ponto, a barra, os dois-pontos, a vírgula e o `()` ficam. `veto_key` (`action_effects.py:57-76`) trata esses separadores como `_` e tira a sintaxe de chamada final.

### Generate — `shutdown.service`

O traço (`sys.settrace`, só linha executada) de `generate_counterfactual_scenario("system vulnerable", "shutdown.service")` passou por:

1. `_map_intervention` (`counterfactual.py:256`). É `str` não vazia (`:258-262`).
2. `_normalize_action_key` devolve `SHUTDOWN.SERVICE` (`:263`, `:21-22`). `:265` (`normalized in ACTION_EFFECTS`) é falsa: a linha `:266` não executou. Passo 1 não é este caso.
3. Adjacência (`:267`, `:269`, `:271-274`). Com a aresta irrelevante da sonda, nenhum `return` de `:268`, `:270`, `:273` ou `:275` executou.
4. Tokens (`:276-277`). O conjunto não é vazio.
5. `veto_key` importada em `:279` e chamada no filtro `:280-285`. `veto_key("shutdown.service")` e `veto_key("SHUTDOWN_SERVICE")` são `SHUTDOWN_SERVICE`.
6. `:286-287`: `len(matches) == 1`, devolve `SHUTDOWN_SERVICE`. Esta é a agulha do mutante `cf_token_set_match_dropped`.
7. `_outcome_via_do` (`:311-319`) chama `simulate(["SHUTDOWN_SERVICE"])`. `predict_outcome` do mundo não existe nesta fiação, então `:350-352` ficam com o retorno do `simulate`.
8. `RealWorldModelSimulator.simulate` (`causal_world_model.py:226`) faz `ACTION_EFFECTS.get` em `:243` e, com entrada presente, aplica `entry["world"]` em `:245-246`. O mundo de `SHUTDOWN_SERVICE` (`action_effects.py:36-37`) escreve `alert_level=low` e `system_status=offline`. O estado inicial (`causal_world_model.py:190-193`) traz `security_patch_applied=False`, que essa ação não sobrescreve.

`veto_lookup` não está no fonte de `counterfactual.py` (contagem 0). O traço do generate registrou `veto_key` e não registrou `veto_lookup`.

O cenário fica com `calls == [["SHUTDOWN_SERVICE"]]`, outcome `state:alert_level=low,security_patch_applied=False,system_status=offline`, `actual_outcome == "outcome_without_intervention"`, `intervention == "shutdown.service"`, cadeia `["system vulnerable causes shutdown.service"]`.

### Rank — a mesma grafia

`rank_counterfactuals` (`counterfactual.py:579-581`) só delega a `scm_rank_alternatives`. O traço não entrou em `_map_intervention`.

1. `rank_alternatives` (`scm_counterfactual.py:206-209`) monta `veto_index(DEFAULT_VETOES)`. O `veto_key` desse trecho roda sobre as duas chaves do mapa (`IGNORE_ALERT`, `SHUTDOWN_SERVICE`), não sobre a grafia.
2. `_mechanism` (`:125-141`) faz `ACTION_EFFECTS.get("shutdown.service")`. A chave não existe e não há declaração. `:139` devolve `None`.
3. `_assignments` (`:144-147`) levanta `SCM_UNMODELED`.
4. O `except` (`:226-230`) coloca a grafia em `unmodeled` e dá `continue`. O `veto_lookup` de `:244-245` não executou. `RANK_SAW_VETO_LOOKUP` foi falso.

Saída medida, grafia sozinha: `ranking == []`, `vetoed == []`, `unmodeled == [["shutdown.service"]]`, `best is None`.

O nome exato no par `[["SHUTDOWN_SERVICE"], ["ISOLATE_SYSTEM"]]` continua modelado e vetado: `best["actions"] == ["ISOLATE_SYSTEM"]`, `ranking` tem as duas sequências, `unmodeled == []`, e `vetoed` traz o motivo `DEFAULT_VETOES["SHUTDOWN_SERVICE"]`.

### Onde o veto já pega a grafia

`veto_lookup("shutdown.service", DEFAULT_VETOES)` devolve `(True, "desligar o serviço é uma ação destrutiva e exige decisão humana explícita")` (`action_effects.py:92-106`: nome exato falha, a chave canônica acerta o índice).

`compare_allowed_futures` (`mcts_planner.py:78-92`) com `[["shutdown.service"], ["MONITOR"]]` devolve `allowed_indices == [1]`, `vetoed_futures == [0]`, `best_future_index == 1` e `vetoed_actions == [{"action": "shutdown.service", "reason": <motivo de SHUTDOWN_SERVICE>}]`. Esse caminho não é o generate.

## 3. Evidência desta base

Comando da sonda: interpretador 3.12.3, `PYTHONPATH` na raiz do commit `d7ac1fd`, script descartável fora do repositório (não commitado). A tabela junta essa saída. Seis linhas não estavam no rank do primeiro script (`Service, Shutdown!`, `service_shutdown()`, `\ufb01INCREASE_LOGGING`, `apply_patch`, `A\u200bPPLY_PATCH`, `isolate the system`): o generate delas foi medido num segundo processo, com a mesma fiação, e o rank num terceiro (as cinco primeiras) e num quarto (`isolate the system`), mesma observação e o mesmo factual. `heur` é `outcome_with_` + espaços trocados por `_` quando a string é não vazia; não é uma segunda execução do motor.

| grafia | passo 1 | `veto_key` | `veto_lookup` no padrão | mapa | `calls` | outcome | rank sozinha |
|---|---|---|---|---|---|---|---|
| `shutdown.service` | não | `SHUTDOWN_SERVICE` | acerta | `SHUTDOWN_SERVICE` | `[["SHUTDOWN_SERVICE"]]` | `offline` (abaixo) | `unmodeled` |
| `shutdown_service()` | não | `SHUTDOWN_SERVICE` | acerta | `SHUTDOWN_SERVICE` | `[["SHUTDOWN_SERVICE"]]` | `offline` | `unmodeled` |
| `shutdown/service` | não | `SHUTDOWN_SERVICE` | acerta | `SHUTDOWN_SERVICE` | `[["SHUTDOWN_SERVICE"]]` | `offline` | `unmodeled` |
| `shutdown:service` | não | `SHUTDOWN_SERVICE` | acerta | `SHUTDOWN_SERVICE` | `[["SHUTDOWN_SERVICE"]]` | `offline` | `unmodeled` |
| `Shutdown, Service` | não | `SHUTDOWN_SERVICE` | acerta | `SHUTDOWN_SERVICE` | `[["SHUTDOWN_SERVICE"]]` | `offline` | `unmodeled` |
| `ignore.alert` | não | `IGNORE_ALERT` | acerta | `IGNORE_ALERT` | `[["IGNORE_ALERT"]]` | `compromised` (abaixo) | `unmodeled` |
| `alert.ignore` | não | `ALERT_IGNORE` | falha | `None` | `[]` | `outcome_with_alert.ignore` | `unmodeled` |
| `increase.logging` | não | `INCREASE_LOGGING` | falha | `INCREASE_LOGGING` | `[["INCREASE_LOGGING"]]` | verbose (abaixo) | `unmodeled` |
| `apply.patch` | não | `APPLY_PATCH` | falha | `APPLY_PATCH` | `[["APPLY_PATCH"]]` | `secure` | `unmodeled` |
| `deploy.update` | não | `DEPLOY_UPDATE` | falha | `DEPLOY_UPDATE` | `[["DEPLOY_UPDATE"]]` | `secure` | `unmodeled` |
| `restore.backup` | não | `RESTORE_BACKUP` | falha | `RESTORE_BACKUP` | `[["RESTORE_BACKUP"]]` | `stable` | `unmodeled` |
| `isolate.system` | não | `ISOLATE_SYSTEM` | falha | `ISOLATE_SYSTEM` | `[["ISOLATE_SYSTEM"]]` | `isolated` | `unmodeled` |
| `service shutdown` | não | `SERVICE_SHUTDOWN` | falha | `None` | `[]` | `outcome_with_service_shutdown` | `unmodeled` |
| `Service, Shutdown!` | não | `SERVICE_SHUTDOWN` | falha | `None` | `[]` | `outcome_with_Service,_Shutdown!` | `unmodeled` |
| `service_shutdown()` | não | `SERVICE_SHUTDOWN` | falha | `None` | `[]` | `outcome_with_service_shutdown()` | `unmodeled` |
| `patch apply` | não | `PATCH_APPLY` | falha | `None` | `[]` | `outcome_with_patch_apply` | `unmodeled` |
| `alert ignore` | não | `ALERT_IGNORE` | falha | `None` | `[]` | `outcome_with_alert_ignore` | `unmodeled` |
| `\ufb01INCREASE_LOGGING` | não | `FIINCREASE_LOGGING` | falha | `None` | `[]` | `outcome_with_` + a string | `unmodeled` |
| `shutdown service` | sim | `SHUTDOWN_SERVICE` | acerta | `SHUTDOWN_SERVICE` | `[["SHUTDOWN_SERVICE"]]` | `offline` | `unmodeled` |
| `SHUTDOWN_SERVICE` | sim | `SHUTDOWN_SERVICE` | acerta | `SHUTDOWN_SERVICE` | `[["SHUTDOWN_SERVICE"]]` | `offline` | modelado e vetado; `best is None` se for a única alternativa |
| `apply patch` | sim | `APPLY_PATCH` | falha | `APPLY_PATCH` | `[["APPLY_PATCH"]]` | `secure` | `unmodeled` |
| `apply_patch` | sim | `APPLY_PATCH` | falha | `APPLY_PATCH` | `[["APPLY_PATCH"]]` | `secure` | `unmodeled` |
| `APPLY_PATCH` | sim | `APPLY_PATCH` | falha | `APPLY_PATCH` | `[["APPLY_PATCH"]]` | `secure` | modelado, `vetoed` vazio, `best` é a própria sequência |
| `A\u200bPPLY_PATCH` | não | `APPLY_PATCH` | falha | `None` | `[]` | `outcome_with_` + a string | `unmodeled` |
| `hello` | não | `HELLO` | falha | `None` | `[]` | `outcome_with_hello` | `unmodeled` |
| `patch` | não | `PATCH` | falha | `None` | `[]` | `outcome_with_patch` | `unmodeled` |
| `isolate the system` | não | `ISOLATE_THE_SYSTEM` | falha | `None` | `[]` | `outcome_with_isolate_the_system` | `unmodeled` |

Outcomes nomeados na tabela:

- `offline`: `state:alert_level=low,security_patch_applied=False,system_status=offline`
- `secure`: `state:alert_level=low,security_patch_applied=True,system_status=secure`
- `compromised`: `state:alert_level=critical,security_patch_applied=False,system_status=compromised`
- verbose: `state:alert_level=high,logging_level=verbose,security_patch_applied=False,system_status=vulnerable`
- `stable`: `state:alert_level=medium,security_patch_applied=False,system_status=stable`
- `isolated`: `state:alert_level=medium,security_patch_applied=False,system_status=isolated`

Cadeia, nas grafias não vazias com extrator que devolveu relação: um passo, `"{situation} causes {intervention}"`. `""` e `"   "` (três espaços): `calls == []`; `""` tem outcome `outcome_without_intervention` e cadeia de 5 passos; `"   "` tem outcome `outcome_with____` e cadeia de 5 passos. `False`: `calls == []`, outcome `outcome_without_intervention`, cadeia `["system vulnerable causes False"]`.

O par do rank exato, medido de novo nesta base: `best["actions"] == ["ISOLATE_SYSTEM"]`; `SHUTDOWN_SERVICE` está em `ranking` e em `vetoed` com o motivo de `DEFAULT_VETOES`; `unmodeled == []`.

### Inferência

As seis primeiras linhas da tabela são o mesmo mecanismo: passo 1 falha, tokens batem com um nome de `DEFAULT_VETOES`, `veto_key` coincide, o fallback devolve esse nome, o `simulate` aplica o efeito, o rank não modela. `shutdown.service` é o exemplo nomeado; não é a única string. `ignore.alert` é o outro nome do veto padrão.

`shutdown service` (espaço) tem a mesma `veto_key` e o mesmo efeito simulado, e o rank também a deixa `unmodeled`. O mapa dela é o passo 1 (`:265-266`), anterior ao fallback. A alternativa 1 não mexe nessa linha. A alternativa 2 mexe.

`apply.patch` e `increase.logging` também saem do fallback com `veto_key` igual, e o nome não está em `DEFAULT_VETOES`. São o controle que ainda devolve um nome quando a agulha `:286-287` permanece.

`A\u200bPPLY_PATCH` tem `veto_key` `APPLY_PATCH` e mapa `None`. Um lookup só pela `veto_key` a faria mapear. Nenhuma alternativa desta página faz isso.

## 4. Hipóteses

| id | hipótese | o que a falsifica | o que a sonda fez |
|---|---|---|---|
| H1 | `shutdown.service` mapeia no fallback porque tokens e `veto_key` coincidem com `SHUTDOWN_SERVICE` | passo 1 verdadeiro, ou mapa diferente, ou `:286` não executada | sobrevive: passo 1 falso, mapa `SHUTDOWN_SERVICE`, `:286-287` no traço |
| H2 | o generate consulta `veto_lookup` antes de simular | o traço do generate entra em `veto_lookup` | falsificada: contagem 0 no arquivo e traço sem `veto_lookup` |
| H3 | o rank veta `shutdown.service` | `vetoed` não vazio, ou o traço entra em `veto_lookup` para essa grafia | falsificada: `unmodeled`, `vetoed` vazio, `continue` em `:230` |
| H4 | `veto_lookup` da grafia contra `DEFAULT_VETOES` falha | o par `(True, motivo)` | falsificada: acerta, com o motivo de `SHUTDOWN_SERVICE` |
| H5 | a lacuna é só a string literal `shutdown.service` | outra grafia de passo 1 falso, mesma `veto_key` de um nome vetado, também simula | falsificada por `shutdown_service()`, `shutdown/service`, `shutdown:service`, `Shutdown, Service` e `ignore.alert` |
| H6 | recusar no fallback o nome que está em `DEFAULT_VETOES` também impede o passo 1 | `shutdown service` ou `SHUTDOWN_SERVICE` deixarem de mapear nesse recorte | falsificada pelo recorte da alternativa 1 (seção 5): os dois continuam `offline` |

Hipótese de trabalho, só se o Pedro aprovar a alternativa 1: no fallback, o único nome cujo token e cuja `veto_key` coincidem deixa de ser devolvido quando esse nome é chave de `DEFAULT_VETOES`. O generate passa a tratar a grafia como não mapeada. O rank permanece o de hoje. Isto não grava um motivo de veto no cenário nem no `vetoed` do rank.

## 5. Três alternativas

Nenhuma é implementada aqui. Os critérios da seção 6 são os da alternativa 1. A escolha continua com o Pedro.

Comportamento de «não mapeou», para string não vazia que não é só espaço, na fiação da sonda: o de `hello`. `_map_intervention` devolve `None`. `simulate` não é chamado. `counterfactual_outcome` é `outcome_with_` + `intervention.replace(" ", "_")` (`counterfactual.py:354-356`). `actual_outcome` é `outcome_without_intervention`. `intervention` do cenário continua a string do chamador. A cadeia é um passo, `"{situation} causes {intervention}"`. `""`, `False` e `"   "` ficam como a sonda mediu.

### Alternativa 1 — o fallback não devolve nome de `DEFAULT_VETOES`

Manter tokens e `veto_key` iguais. Se o único nome estiver em `DEFAULT_VETOES`, não devolvê-lo (`None`). O passo 1 e a adjacência ficam como estão. O lugar que preserva a agulha `:286-287` é o filtro da lista `matches`, antes do `if`, no mesmo desenho que o S2 usou para a `veto_key`.

Efeito medido num invólucro que chama o método real e, só quando o passo 1 é falso e o nome devolvido está em `DEFAULT_VETOES`, substitui por `None`. Não é um diff aplicado ao arquivo.

Deixam de mapear: `shutdown.service`, `shutdown_service()`, `shutdown/service`, `shutdown:service`, `Shutdown, Service`, `ignore.alert`. Cada uma passa ao outcome heurístico da seção 3.

Continuam mapeando: `increase.logging`, `apply.patch`, `deploy.update`, `restore.backup`, `isolate.system`, e o passo 1 (`shutdown service`, `SHUTDOWN_SERVICE`, `apply patch`, `apply_patch`, `APPLY_PATCH`). `service shutdown`, `patch apply`, `alert ignore`, `alert.ignore`, `\ufb01INCREASE_LOGGING` e `A\u200bPPLY_PATCH` continuam `None`.

O rank dessas grafias, no invólucro, ficou igual ao de hoje: `shutdown.service` segue `unmodeled` com `vetoed` vazio; o nome exato sozinho segue modelado e vetado. O golden de `test_counterfactual_extraction_contract.py` (`golden_sha`) saiu `64488affedd89003faa5bc18de333bff4c52cb1e126a379e6f376b00cdd2c509` com o invólucro e sem ele, igual a `GOLDEN_SHA256` desse arquivo.

Prós: fecha a simulação das grafias que o S2 deixou passar com a mesma `veto_key` de um veto padrão; não reabre `patch apply`; não mexe em S2-C3 nem em S2-C6. Contras: não grava veto no generate nem no rank (os dois passam a dizer «não modelado», e o motivo continua só em `compare_allowed_futures` e no nome exato do rank); `shutdown service` (passo 1) continua simulado e `unmodeled`; o filtro fica acoplado ao conteúdo atual de `DEFAULT_VETOES`.

### Alternativa 2 — nenhum mapa cujo nome está em `DEFAULT_VETOES` é simulado

Depois de qualquer retorno de `_map_intervention`, se o nome está em `DEFAULT_VETOES`, não chamar `simulate` e seguir a heurística de não mapeou. Pega o fallback e também o passo 1: `shutdown service` e `SHUTDOWN_SERVICE` deixam de sair `offline` no generate.

Prós: uma grafia e o nome exato deixam de produzir o efeito destrutivo no generate. Contras: quebra S2-C3, que exige `calls == [["SHUTDOWN_SERVICE"]]` e o outcome `offline` para `SHUTDOWN_SERVICE` e `shutdown service`. O rank do nome exato continuaria modelando e vetando, e o generate esconderia o efeito: os dois caminhos divergiriam no sentido oposto ao de hoje. O what-if do nome exato deixaria de mostrar `offline`. Não recomendo.

### Alternativa 3 — o rank modela a grafia e o `veto_lookup` já existente a marca

Ensinar `rank_alternatives` (ou `_mechanism`) a tratar a grafia como o único nome de `ACTION_EFFECTS` com a mesma `veto_key`, e deixar `:244-245` marcar o veto. O generate continuaria simulando `offline`.

Prós: `shutdown.service` passaria a aparecer em `ranking` e em `vetoed`, com o motivo já medido, e o efeito simulado deixaria de ser invisível para o ranking. Contras: revoga S2-C6 para essa grafia (`unmodeled` e `vetoed` vazio). O generate continuaria sem consultar `veto_lookup`. `A\u200bPPLY_PATCH` entraria na zona que o S2 separou (lookup pela chave alarga o vocabulário). Os critérios S3-C abaixo não servem para esta alternativa.

### Recomendação

Recomendo a **alternativa 1**. Ela fecha o escape que a sonda mostrou no fallback (simular o efeito de um nome de `DEFAULT_VETOES` sem passar por `veto_lookup`) e conserva o que o S2 congelou no passo 1, no rank e nas grafias de `veto_key` diferente. A alternativa 2 desfaz S2-C3. A alternativa 3 desfaz S2-C6. As duas continuam disponíveis se o Pedro quiser outro corte; nesse caso este texto é reescrito antes do RED.

Uma exceção literal só da string `shutdown.service` fica de fora: a sonda mostrou mais cinco grafias do mesmo fallback, incluindo `ignore.alert`.

## 6. Critérios da alternativa 1 (RED esperado nesta base)

Arquivo novo, único: `test_counterfactual_s3_shutdown_service_veto_contract.py`. Fiação de C1–C3: a da sonda. Não executei esse arquivo: ele não existe. A previsão compara a saída medida com a asserção. Previsão de falha em `d7ac1fd` está marcada.

### S3-C1 — fallback com veto padrão não simula (previsão: RED)

Teste: `test_s3_c1_same_veto_key_default_veto_does_not_simulate`.

Para cada grafia: `calls == []`, outcome é o heurístico medido, `actual_outcome == "outcome_without_intervention"`, `intervention` é a própria grafia, cadeia é o passo lexical medido.

| grafia | outcome exigido |
|---|---|
| `shutdown.service` | `outcome_with_shutdown.service` |
| `shutdown_service()` | `outcome_with_shutdown_service()` |
| `shutdown/service` | `outcome_with_shutdown/service` |
| `shutdown:service` | `outcome_with_shutdown:service` |
| `Shutdown, Service` | `outcome_with_Shutdown,_Service` |
| `ignore.alert` | `outcome_with_ignore.alert` |

Hoje as seis chamam `simulate` e produzem o estado da tabela. Por isso o teste falha nesta base.

### S3-C2 — propriedade do fallback (previsão: RED nos exemplos de C1)

Teste: `test_s3_c2_fallback_simulation_is_not_a_default_veto`.

Para a string da estratégia: calcular o passo 1 como a sonda (`_normalize_action_key` no resultado e esse resultado em `ACTION_EFFECTS`). Se o passo 1 é verdadeiro, o teste não exige nada desta propriedade. Se `simulate` não é chamado, passa. Se `simulate` é chamado, a única ação chamada não é chave de `DEFAULT_VETOES`.

Exemplos fixos que têm de falhar nesta base: os seis de C1. Exemplos fixos que passam hoje: `apply.patch`, `increase.logging`, `hello`, `patch apply`, `shutdown service`, `SHUTDOWN_SERVICE`, `A\u200bPPLY_PATCH`. Estratégia: letras, dígitos, espaço, `_`, `-`, `.`, `/`, `:`, `,`, `!`, `()` e U+200B, mais os exemplos. Sem `deadline`.

Se este teste falhar numa string que o passo 1 já devolve, parar e relatar. A amostra não achou esse caso, porque o passo 1 está fora da propriedade. A correção não entra no passo 1 para «salvar» o teste.

### S3-C3 — o rank desta lacuna não muda (previsão: passa hoje)

Teste: `test_s3_c3_rank_of_the_gap_stays`.

Mesma observação e factual da sonda.

- `[["shutdown.service"]]`, `[["shutdown_service()"]]`, `[["ignore.alert"]]`, `[["shutdown service"]]`, `[["apply.patch"]]`: `ranking == []`, `vetoed == []`, `unmodeled` é a lista da grafia, `best is None`.
- `[["SHUTDOWN_SERVICE"], ["ISOLATE_SYSTEM"]]`: `best["actions"] == ["ISOLATE_SYSTEM"]`; `SHUTDOWN_SERVICE` está em `ranking` e em `vetoed` com o motivo `DEFAULT_VETOES["SHUTDOWN_SERVICE"]`; `unmodeled == []`.

Isto repete o fato que S2-C6 já exige para `shutdown.service` e para o nome exato. Está aqui para a implementação não «corrigir» a lacuna mudando o rank. Se a alternativa 3 for a escolhida, S3-C3 está errado e o texto é reescrito.

## 7. Não-regressão

Não há teste novo para N, fora o que S3-C3 já observa do rank. O restante é o diff contra `d7ac1fdd53d0e2149770b8b06d71235ba36548f3` mais os contratos que já existem.

- **S3-N1.** S2-C1 permanece. Medido de novo nesta base, com a fiação da sonda: `service shutdown`, `Service, Shutdown!`, `service_shutdown()`, `patch apply`, `alert ignore` e `\ufb01INCREASE_LOGGING` ficam com `calls == []`, o outcome heurístico da tabela e a cadeia lexical. `alert.ignore` também fica `None`.
- **S3-N2.** S2-C3 permanece. `apply patch`, `apply_patch` e `APPLY_PATCH` seguem `calls == [["APPLY_PATCH"]]` e outcome `secure`. `SHUTDOWN_SERVICE` e `shutdown service` seguem `calls == [["SHUTDOWN_SERVICE"]]` e outcome `offline`. O nome exato continua no passo 1.
- **S3-N3.** S2-C4 permanece. `hello`, `patch` e `isolate the system`: `calls == []`, outcome heurístico, cadeia lexical. `""` e `False`: `calls == []` e `outcome_without_intervention`. `"   "`: `calls == []`, `outcome_with____`, cadeia de 5 passos.
- **S3-N4.** S2-C5 permanece no que já exige: ação simulada tem a mesma `veto_key` da string, ou `simulate` não é chamado. A alternativa 1 não cria mapa novo. `A\u200bPPLY_PATCH` continua `None`.
- **S3-N5.** S2-C6 permanece, inclusive `[["service shutdown"]]`, `[["patch apply"]]` e `[["shutdown service"]]` em `unmodeled` com `vetoed` vazio, e o par exato `SHUTDOWN_SERVICE` + `ISOLATE_SYSTEM` como a sonda mediu.
- **S3-N6.** Fallback de nome que não está no veto padrão continua. `apply.patch` e `increase.logging`: `calls` com esse nome e o outcome medido na tabela. São estas as que ainda distinguem a agulha `:286-287` ligada da agulha desligada, depois que C1 deixar de exigir mapa.
- **S3-N7.** `compare_allowed_futures([["shutdown.service"], ["MONITOR"]])` continua vetando a primeira sequência com o motivo de `SHUTDOWN_SERVICE` e escolhendo o índice 1. O teste que já exige isso é `test_mcts_p1c_veto_key_errata_contract.py` (`test_main_canonical_variants_are_vetoed`). Não é editado.
- **S3-N8.** Produção fora do filtro `matches` de `_map_intervention` intocada. Inclui o passo 1, a adjacência, `:286-287` se o filtro entrar na lista, `action_effects.py`, `scm_counterfactual.py`, `mcts_planner.py`, `causal_world_model.py` e `core/constitutional_brain.py`.
- **S3-N9.** `tools/targeted_mutation.py` intocado. Esse arquivo não cita `_map_intervention` nem `counterfactual.py`. Não peço autorização para editá-lo.
- **S3-N10.** `tools/cognitivo_mutation.py` intocado neste pré-registro. Na implementação, a autorização da seção 9 é a tupla de `cf_token_set_match_dropped`. Sem ela, a previsão do gate está na seção 9.
- **S3-N11.** O digest `GOLDEN_SHA256` de `test_counterfactual_extraction_contract.py` permanece `64488affedd89003faa5bc18de333bff4c52cb1e126a379e6f376b00cdd2c509`. O invólucro da alternativa 1 não o moveu. Os insumos desse golden são `apply patch` e `patch`, os dois fora do recorte.
- **S3-N12.** `CONTINUATION.md`, `docs/agentes/CLAIMS.yaml`, `docs/agentes/REGISTRO_AGENTES.yaml`, workflows, timeouts, pisos de cobertura e o contrato B2b intocados.
- **S3-N13.** `test_*.py` já existentes intocados, salvo a errata de S2-C2 descrita na seção 8, e só se o Pedro autorizar essa errata junto com a alternativa 1.

## 8. Testes existentes e o pin N9

`test_counterfactual_s2_fallback_veto_key_contract.py`, `test_s2_c2_same_veto_key_fallback_still_maps` (`:83-90`), exige hoje `calls == [["SHUTDOWN_SERVICE"]]` e o outcome `offline` para `shutdown.service` e `shutdown_service()`. Isso é o oposto de S3-C1. Com a alternativa 1, esse método falha até ser emendado: as duas grafias passam a exigir `calls == []` e o outcome heurístico de C1. O `@example("shutdown.service")` de S2-C5 (`:135`) não precisa mudar: sem `simulate`, a propriedade de S2-C5 passa. S2-C6 (`:155-179`) não precisa mudar.

Essa emenda é errata do critério S2-C2, que foi escrito para conservar o mapa. Não reescreve o pré-registro S2; um bloco datado, no estilo de `S2_ERRATA_CF9_2026-11-26.md`, registra que S2-C2 deixou de exigir o mapa dessas duas grafias.

`test_n9_existing_tests_are_not_edited` (`test_episodic_b2b_open_current_contract.py:486-491`) compara SHA-256 só das chaves de `EXISTING_TEST_SHA256`. `test_counterfactual_s2_fallback_veto_key_contract.py` não está nesse dicionário. Emendar o C2 não altera digest que o N9 confere. Não peço autorização para atualizar o pin por causa dessa errata.

Arquivos que estão no pin e citam a grafia, e que a alternativa 1 não edita:

- `test_mcts_p1c_veto_key_errata_contract.py` — `compare_allowed_futures` veta `shutdown.service` (S3-N7).
- `test_decision_cycle_review_contract.py` — a string entra nas variantes do ciclo de decisão, não em `_map_intervention`.
- `test_red_pr156_veto_names.py` — `select_operator(["shutdown.service"], ...)` rejeita por vocabulário.
- `test_veto_key_precedence_contract.py` — `shutdown_service()` como entrada de `veto_key`.
- `test_counterfactual_dox_contract.py` — o oráculo `:70-80` repete o fallback atual (tokens e `veto_key`). Os exemplos fixos `:143-163` são passo 1 ou `patch apply` (`calls == []`). O alfabeto da propriedade `:404` é `abcdefghijklmnopqrstuvwxyz _-`, sem `.`, `/`, `:`, `,` nem `()`.
- `test_counterfactual_extraction_contract.py` — o golden medido na seção 5.

**Inferência:** a alternativa 1 não pede edição desses arquivos pinados, e portanto não pede autorização para recalcular o N9. Não rodei `python -m unittest` deles: `test_counterfactual_dox_contract.py` e o contrato S2 importam `hypothesis`, ausente neste interpretador. A conclusão sobre o alfabeto do dox é leitura do fonte mais a tabela da seção 3, não uma suíte executada.

Se a implementação editar qualquer chave de `EXISTING_TEST_SHA256` para ficar verde, o N9 falha e a atualização do digest precisa de autorização do Pedro. A alternativa 1, como está escrita, não usa esse caminho. A alternativa 2 precisaria rever S2-C3; esse arquivo também não está no pin, mas o corte já não é o desta página.

Um arquivo de teste novo não entra no pin. O N9, no texto de `docs/PONTE_EPISODIC_B2B_CRITERIOS_2026-11-18.md`, diz que arquivo novo não conta como edição dos existentes.

## 9. Mutante `cf_token_set_match_dropped`

`tools/cognitivo_mutation.py:570-576`. A agulha é:

```text
if len(matches) == 1:
    return matches[0]
```

No fonte de `d7ac1fd` isso é `counterfactual.py:286-287`, uma ocorrência no arquivo (contagem 1). A tupla `tests` é `test_counterfactual_dox_contract.py` e `test_counterfactual_s2_fallback_veto_key_contract.py`. `tools/targeted_mutation.py` não cita esse método.

A sonda compilou uma cópia dedentada do método com a agulha trocada por `if False and len(matches) == 1`, sem escrever no arquivo. Efeito medido:

| grafia | mapa com a agulha desligada |
|---|---|
| `shutdown.service`, `shutdown_service()`, `ignore.alert`, `apply.patch`, `increase.logging` | `None`, `calls == []`, outcome heurístico |
| `shutdown service`, `SHUTDOWN_SERVICE` | continuam `SHUTDOWN_SERVICE` / `offline` |
| `apply patch` | continua `APPLY_PATCH` / `secure` |
| `patch apply`, `service shutdown` | continuam `None` (já eram) |

**Inferência** (o catálogo não rodou; `hypothesis` ausente): hoje o mutante morre em S2-C2, porque esse teste exige mapa positivo de `shutdown.service` e `shutdown_service()`, e o dox, depois da errata CF9, não exige retorno positivo do fallback (`patch apply` espera `calls == []`; `apply patch` é passo 1).

Com a alternativa 1 e a errata de S2-C2, essas duas grafias ficam `None` nos dois lados (código emendado e agulha desligada). S2-C2 deixa de matar o mutante. O dox continua sem um assert positivo do fallback. A previsão, se a tupla não ganhar o arquivo novo, é `survived` e o gate diferente de 0.

O que ainda separa os dois lados, na tabela, é o fallback de nome fora do veto padrão: `apply.patch` e `increase.logging` mapeiam com a agulha ligada e deixam de mapear com ela desligada. S3-N6 é o assert que mata, desde que o arquivo novo esteja na tupla.

Autorização da implementação, além da alternativa 1, da errata de S2-C2 e do teste novo: em `tools/cognitivo_mutation.py`, acrescentar `test_counterfactual_s3_shutdown_service_veto_contract.py` à tupla `tests` de `cf_token_set_match_dropped`. Não reescrever a agulha se `:286-287` permanecerem, o que pede o filtro dentro da lista `matches`. Se o `if` deixar de ter o texto da agulha, a previsão passa a ser `not_applied`, e a autorização teria de incluir a agulha nova. Sem a tupla, não implemento.

`tools/targeted_mutation.py` continua fora. Não há mutante desse arquivo para este ramo.

## 10. Fora de escopo

- O passo 1 `shutdown service`, que a sonda mediu: simula `offline` e no rank cai em `unmodeled`. O hífen entra no mesmo normalizador (`counterfactual.py:19-22`); esta sonda não executou `Shutdown-Service`. É a outra metade da frase que o S2 deixou fora. A alternativa 2 a incluiria e quebraria S2-C3.
- Gravar motivo de veto dentro de `CounterfactualScenario`. A alternativa 1 não cria campo.
- Ensinar o rank a modelar a grafia (alternativa 3) e qualquer lookup só por `veto_key` (`A\u200bPPLY_PATCH` continuaria `None`).
- `compare_allowed_futures`, o ciclo de decisão, `DEFAULT_VETOES`, `ACTION_EFFECTS`, a fórmula de plausibilidade e o planner.
- `core/constitutional_brain.py`, workflows, `CONTINUATION.md`, claims e o pin N9.

## O que este commit é

Um arquivo novo. Não há RED, não há produção alterada, não há push e não há PR.
