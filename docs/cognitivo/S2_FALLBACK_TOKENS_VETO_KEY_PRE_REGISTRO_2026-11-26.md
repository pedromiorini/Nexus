# Pré-registro S2 — fallback de tokens só com a mesma `veto_key`

Status: critérios antes de qualquer RED, de qualquer edição de produção e de qualquer edição de `test_*.py`. Não implementa S2. Não escolhe entre as duas alternativas de código da seção 2. A seção 4 recomenda uma opção para o teste da linha 143 e diz por quê; a escolha continua com o Pedro.

Autorização deste texto: Pedro Miorini, 2026-11-26 17:35 America/Sao_Paulo — preparar só localmente o pré-registro, sem produção, sem push e sem PR.

Base: `01d98868db455cf8f426a52fc5cdf95054c4392b` (`docs: corrigir citação da errata :180 para mcts_planner.py (#226)`).
Python da sonda: 3.12.3. `python3.11` não está neste host. O CI usa 3.11; esta sonda não o executou.
`hypothesis` não está instalado neste interpretador. O gate de mutação não foi executado. A falha de import abaixo não é um mutante morto.

Consulta de PRs abertos nesta sessão devolveu lista vazia. `docs/agentes/CLAIMS.yaml` nesta base tem `claims: []`. Este commit não abre claim e não edita o registro de agentes.

## 1. Problema

O ramo final de `_map_intervention` trata como a mesma ação duas strings que têm o mesmo conjunto de tokens `[a-z0-9]+`, sem olhar a ordem e sem chamar `veto_key`. `generate_counterfactual_scenario` então simula o efeito da ação da tabela. `rank_counterfactuals` não usa esse mapeamento: a grafia que não é chave exata cai em `unmodeled`, com `ranking` e `vetoed` vazios quando ela é a única alternativa.

### Fato — código na `01d9886`

`core/cognitivo/counterfactual.py` importa `ACTION_EFFECTS` e não contém a string `veto_key` (contagem 0 no arquivo). `_map_intervention` está na linha 256. A ordem é:

1. Não-`str`, vazio depois de `strip`, devolve `None` (`:258-262`).
2. `_normalize_action_key` (`:19-22`: `strip`, espaços e hífens ASCII viram `_`, `upper`). Se o resultado é chave de `ACTION_EFFECTS`, devolve essa chave (`:263-266`). `apply patch`, `apply_patch` e `APPLY_PATCH` caem aqui. `patch apply` vira `PATCH_APPLY`, que não é chave.
3. Adjacência do grafo, nas formas crua, normalizada e dobrada (`:267-275`). Não é o ramo dos casos abaixo: o valor devolvido é um nome de `ACTION_EFFECTS`, não uma chave do grafo da sonda.
4. Fallback de tokens (`:276-286`), introduzido no GREEN `0184a728`. Conjunto `set(re.findall(r"[a-z0-9]+", intervention.lower()))` igual ao de exatamente um nome de `ACTION_EFFECTS` devolve esse nome. Os dez nomes da tabela têm conjuntos distintos. Zero ou dois ou mais devolve `None` (`:286`).

`veto_key` está em `core/cognitivo/action_effects.py:57`. O comentário `:46-48` diz que a chave canônica serve ao veto e que variante não declarada segue sem modelo. `DEFAULT_VETOES` (`core/cognitivo/mcts_planner.py:24-27`) tem `IGNORE_ALERT` e `SHUTDOWN_SERVICE`.

`rank_counterfactuals` (`counterfactual.py:577`) delega a `scm_rank_alternatives`. O SCM exige o nome exato (`scm_counterfactual.py:109-122` e `_mechanism` em `:125-141`). Ação sem efeito verificado vira `SCM_UNMODELED` e entra em `unmodeled` (`:223-230`), antes de `veto_lookup`. O nome exato `SHUTDOWN_SERVICE` é modelado e vetado.

Origem do fallback: pré-registro CF9 no commit `c187c12548af1e5a84d21431286b2ae8c22ead40`, seção «CF gap do(X)», `docs/cognitivo/AUDITORIA_2026-10-22.md`. Naquele commit e ainda na `01d9886`, a linha 512 (passo 5) diz que `apply patch` e `patch apply` mapeiam para `APPLY_PATCH`. A linha 554 (CF9-outcome) repete as duas grafias. A linha 562 põe as duas no conjunto D. O contrato `test_counterfactual_dox_contract.py` fixa isso: oráculo `:50-80` (o trecho de tokens é `:70-79`), `test_real_simulator_maps_apply_patch_token_orders` `:142-153` (o laço está na `:143`), e `@example("patch apply")` na `:397` dentro de `test_mapping_matches_preregistered_oracle` (`:402`).

`tools/cognitivo_mutation.py:570-575`, mutante `cf_token_set_match_dropped`: a agulha é o bloco `if len(matches) == 1:` / `return matches[0]`, uma ocorrência no arquivo de produção (contagem 1 nesta base). A tupla `tests` é só `test_counterfactual_dox_contract.py`. `tools/targeted_mutation.py` não cita `_map_intervention`, `len(matches) == 1` nem `counterfactual.py`.

### Fato — reprodução nesta base

Comando (só leitura; não altera o módulo). `PYTHONPATH=.`, Python 3.12.3, commit `01d98868db455cf8f426a52fc5cdf95054c4392b`. `heur_line_354` é a expressão de `counterfactual.py:354` (`outcome_with_` + espaços trocados por `_`) calculada na sonda. Não é uma segunda execução do motor e não é o comportamento depois da mudança.

`RecordingWorld` grava `simulate`. O causal é um `RealCausalReasoning` que já extraiu `noop causes nothing relevant`, a mesma fiação de aresta irrelevante do contrato CF9. `RANK` usa a observação do contrato de veto por nome (`system_status=compromised`, `alert_level=critical`, `security_patch_applied=False`, `logging_level=normal`) e factual `["IGNORE_ALERT"]`.

```text
'service shutdown'
  normalize SERVICE_SHUTDOWN step1 False
  veto_key SERVICE_SHUTDOWN veto_key(mapped) SHUTDOWN_SERVICE
  map SHUTDOWN_SERVICE calls [['SHUTDOWN_SERVICE']]
  outcome state:alert_level=low,security_patch_applied=False,system_status=offline
  heur_line_354 outcome_with_service_shutdown
  chain ['system vulnerable causes service shutdown']
'Service, Shutdown!'
  normalize SERVICE,_SHUTDOWN! step1 False
  veto_key SERVICE_SHUTDOWN veto_key(mapped) SHUTDOWN_SERVICE
  map SHUTDOWN_SERVICE calls [['SHUTDOWN_SERVICE']]
  outcome state:alert_level=low,security_patch_applied=False,system_status=offline
  heur_line_354 outcome_with_Service,_Shutdown!
  chain ['system vulnerable causes Service, Shutdown!']
'service_shutdown()'
  normalize SERVICE_SHUTDOWN() step1 False
  veto_key SERVICE_SHUTDOWN veto_key(mapped) SHUTDOWN_SERVICE
  map SHUTDOWN_SERVICE calls [['SHUTDOWN_SERVICE']]
  outcome state:alert_level=low,security_patch_applied=False,system_status=offline
  heur_line_354 outcome_with_service_shutdown()
  chain ['system vulnerable causes service_shutdown()']
'shutdown.service'
  normalize SHUTDOWN.SERVICE step1 False
  veto_key SHUTDOWN_SERVICE veto_key(mapped) SHUTDOWN_SERVICE
  map SHUTDOWN_SERVICE calls [['SHUTDOWN_SERVICE']]
  outcome state:alert_level=low,security_patch_applied=False,system_status=offline
  heur_line_354 outcome_with_shutdown.service
  chain ['system vulnerable causes shutdown.service']
'patch apply'
  normalize PATCH_APPLY step1 False
  veto_key PATCH_APPLY veto_key(mapped) APPLY_PATCH
  map APPLY_PATCH calls [['APPLY_PATCH']]
  outcome state:alert_level=low,security_patch_applied=True,system_status=secure
  heur_line_354 outcome_with_patch_apply
  chain ['system vulnerable causes patch apply']
'apply patch'
  normalize APPLY_PATCH step1 True
  veto_key APPLY_PATCH veto_key(mapped) APPLY_PATCH
  map APPLY_PATCH calls [['APPLY_PATCH']]
  outcome state:alert_level=low,security_patch_applied=True,system_status=secure
  heur_line_354 outcome_with_apply_patch
  chain ['system vulnerable causes apply patch']
'apply_patch'
  normalize APPLY_PATCH step1 True
  veto_key APPLY_PATCH veto_key(mapped) APPLY_PATCH
  map APPLY_PATCH calls [['APPLY_PATCH']]
  outcome state:alert_level=low,security_patch_applied=True,system_status=secure
  heur_line_354 outcome_with_apply_patch
  chain ['system vulnerable causes apply_patch']
'APPLY_PATCH'
  normalize APPLY_PATCH step1 True
  veto_key APPLY_PATCH veto_key(mapped) APPLY_PATCH
  map APPLY_PATCH calls [['APPLY_PATCH']]
  outcome state:alert_level=low,security_patch_applied=True,system_status=secure
  heur_line_354 outcome_with_APPLY_PATCH
  chain ['system vulnerable causes APPLY_PATCH']
'SHUTDOWN_SERVICE'
  normalize SHUTDOWN_SERVICE step1 True
  veto_key SHUTDOWN_SERVICE veto_key(mapped) SHUTDOWN_SERVICE
  map SHUTDOWN_SERVICE calls [['SHUTDOWN_SERVICE']]
  outcome state:alert_level=low,security_patch_applied=False,system_status=offline
  heur_line_354 outcome_with_SHUTDOWN_SERVICE
  chain ['system vulnerable causes SHUTDOWN_SERVICE']
'shutdown service'
  normalize SHUTDOWN_SERVICE step1 True
  veto_key SHUTDOWN_SERVICE veto_key(mapped) SHUTDOWN_SERVICE
  map SHUTDOWN_SERVICE calls [['SHUTDOWN_SERVICE']]
  outcome state:alert_level=low,security_patch_applied=False,system_status=offline
  heur_line_354 outcome_with_shutdown_service
  chain ['system vulnerable causes shutdown service']
'shutdown_service()'
  normalize SHUTDOWN_SERVICE() step1 False
  veto_key SHUTDOWN_SERVICE veto_key(mapped) SHUTDOWN_SERVICE
  map SHUTDOWN_SERVICE calls [['SHUTDOWN_SERVICE']]
  outcome state:alert_level=low,security_patch_applied=False,system_status=offline
  heur_line_354 outcome_with_shutdown_service()
  chain ['system vulnerable causes shutdown_service()']
'alert ignore'
  normalize ALERT_IGNORE step1 False
  veto_key ALERT_IGNORE veto_key(mapped) IGNORE_ALERT
  map IGNORE_ALERT calls [['IGNORE_ALERT']]
  outcome state:alert_level=critical,security_patch_applied=False,system_status=compromised
  heur_line_354 outcome_with_alert_ignore
  chain ['system vulnerable causes alert ignore']
'patch'
  normalize PATCH step1 False
  veto_key PATCH veto_key(mapped) None
  map None calls []
  outcome outcome_with_patch
  heur_line_354 outcome_with_patch
  chain ['system vulnerable causes patch']
'hello'
  normalize HELLO step1 False
  veto_key HELLO veto_key(mapped) None
  map None calls []
  outcome outcome_with_hello
  heur_line_354 outcome_with_hello
  chain ['system vulnerable causes hello']
RANK [['service shutdown']]
  ranking_actions []
  vetoed []
  unmodeled [['service shutdown']]
  best None
RANK [['shutdown.service']]
  ranking_actions []
  vetoed []
  unmodeled [['shutdown.service']]
  best None
RANK [['patch apply']]
  ranking_actions []
  vetoed []
  unmodeled [['patch apply']]
  best None
RANK [['SHUTDOWN_SERVICE'], ['ISOLATE_SYSTEM']]
  ranking_actions [['ISOLATE_SYSTEM'], ['SHUTDOWN_SERVICE']]
  vetoed [{'actions': ['SHUTDOWN_SERVICE'], 'vetoes': [{'action': 'SHUTDOWN_SERVICE', 'reason': 'desligar o serviço é uma ação destrutiva e exige decisão humana explícita'}]}]
  unmodeled []
  best ['ISOLATE_SYSTEM']
RANK [['shutdown service']]
  ranking_actions []
  vetoed []
  unmodeled [['shutdown service']]
  best None
HELLO calls []
HELLO outcome outcome_with_hello
HELLO actual outcome_without_intervention
HELLO intervention hello
HELLO chain ['system vulnerable causes hello']
HELLO_NO_CAUSAL calls []
HELLO_NO_CAUSAL outcome outcome_with_hello
HELLO_NO_CAUSAL chain ['1. system vulnerable', '2. Intervention: hello', '3. Immediate effect', '4. Secondary effect', '5. Final outcome']
```

Na mesma sessão, com causal, `"   "` (três espaços) mapeou `None`, `calls []`, outcome `outcome_with____`, cadeia de 5 passos. `""` e `False` deram `outcome_without_intervention` e `calls []`. `False` teve cadeia lexical `system vulnerable causes False`. Essas três não são o modelo de «não mapeou» das grafias não vazias da tabela.

Três chamadas a mais do motor real, mesma base:

- `"A\u200bPPLY_PATCH"` (U+200B entre A e P): `map None`, `veto_key` `APPLY_PATCH`.
- `"A\u00adPPLY_PATCH"` (U+00AD): `map None`, `veto_key` `APPLY_PATCH`.
- `"\ufb01INCREASE_LOGGING"` (U+FB01): `map INCREASE_LOGGING`, `veto_key` `FIINCREASE_LOGGING`.

Sonda à parte, 8242 strings (nomes da tabela, separadores e invisíveis inseridos, 8000 sorteios `random.Random(0)` num alfabeto curto, deduplicadas). Em 561 o mapa atual tinha `veto_key` diferente da ação devolvida, todas fora do passo 1. Nenhuma string do passo 1 nessa amostra teve `veto_key` diferente. Em 40, um lookup só por `veto_key` devolveria um nome e o filtro de tokens não. As 40 são do tipo do U+200B acima.

### Inferência

Há dois caminhos até `APPLY_PATCH` dentro de `:143`. `apply patch`, `apply_patch` e `APPLY_PATCH` param no passo 1. Só `patch apply` usa o fallback. Uma regra só no fallback não muda os três primeiros.

`shutdown service` (mesma ordem) também para no passo 1 e já é simulada como `offline`. No rank ela vai para `unmodeled`, não para `vetoed`. A mudança pedida, limitada ao fallback, não mexe nesse par. `shutdown.service` usa o fallback e a `veto_key` coincide: a restrição pedida continua mapeando.

O pedido listou `service shutdown`, `Service, Shutdown!`, `service_shutdown()`, `shutdown.service`, `patch apply` e `apply patch` como grafias que «continuam mapeando». A medição mostra que todas mapeiam hoje. A regra «só devolve se `veto_key` coincidir» separa o grupo: as quatro primeiras dessa lista com chave diferente deixam de mapear; `shutdown.service` e `apply patch` continuam, por motivos diferentes (fallback com chave igual; passo 1).

### Hipóteses

| id | hipótese | o que a falsifica | o que a sonda fez |
|---|---|---|---|
| H1 | o fallback devolve um nome cuja `veto_key` difere da intervenção | as grafias medidas tiverem a mesma `veto_key` dos dois lados | sobrevive para `service shutdown`, `patch apply`, `alert ignore`, U+FB01 |
| H2 | o rank já veta essas grafias | `vetoed` não vazio para a grafia sozinha | falsificada: `unmodeled`, `ranking` e `vetoed` vazios |
| H3 | `apply patch` e `patch apply` são o mesmo ramo | `step1` igual para as duas | falsificada: só `patch apply` tem `step1 False` |
| H4 | toda grafia da lista do pedido deixaria de mapear | alguma com `veto_key` igual continuaria | falsificada por `shutdown.service` |
| H5 | filtrar o fallback por `veto_key` é o mesmo que procurar o nome pela `veto_key` | uma string que hoje não mapeia passaria a mapear no lookup | falsificada por `A\u200bPPLY_PATCH` |

Hipótese de trabalho, só se o Pedro aprovar a alternativa 1: no fallback, devolver o único nome se `veto_key(intervention) == veto_key(nome)`; senão, `None`. Isto não alinha o `generate` com o rank para grafias de mesma chave que não são a chave exata.

## 2. Duas alternativas, só em `_map_intervention`

Nenhuma é implementada aqui. Os critérios da seção 3 são os da alternativa 1, que é a regra pedida. A alternativa 2 ficou ao lado porque a sonda a separou. Não escolho entre elas.

Comportamento de «não mapeou», para string não vazia que não é só espaço, na fiação da sonda (mundo + extrator): o de `hello`. `_map_intervention` devolve `None`. `simulate` não é chamado. `counterfactual_outcome` é `outcome_with_` + `intervention.replace(" ", "_")` (`:354`). `actual_outcome` é `outcome_without_intervention`. `intervention` do cenário continua a string do chamador. A cadeia é um passo, `"{situation} causes {intervention}"`. Sem extrator, a cadeia são os 5 passos heurísticos e o outcome heurístico permanece. `""`, `False` e `"   "` ficam como estão hoje.

### Alternativa 1 — restringir o fallback

Manter o conjunto de tokens. Se houver exatamente um nome e `veto_key(intervention) == veto_key(nome)`, devolver esse nome. Senão, `None`. Não criar mapeamento novo. O passo 1 e a adjacência ficam como estão.

Efeito previsto nas grafias medidas: deixam de mapear `service shutdown`, `Service, Shutdown!`, `service_shutdown()`, `patch apply`, `alert ignore` e `\ufb01INCREASE_LOGGING`. Continuam `shutdown.service`, `shutdown_service()`, e tudo o que já para no passo 1 (`apply patch`, `APPLY_PATCH`, `shutdown service`, `SHUTDOWN_SERVICE`). `A\u200bPPLY_PATCH` continua `None`.

As quatro linhas da agulha do mutante podem permanecer se o filtro entrar na lista `matches`, antes do `if`. Isso é desenho, não diff.

### Alternativa 2 — lookup pela `veto_key`

Trocar o teste de tokens por: entre `ACTION_EFFECTS`, o único nome com a mesma `veto_key` da intervenção. Na amostra, as 40 strings que hoje não mapeiam passariam a mapear. Exemplo medido: `A\u200bPPLY_PATCH` passaria a `APPLY_PATCH`. Isso alarga o vocabulário simulado. As quatro linhas da agulha deixam de ocorrer; o mutante fica `not_applied` se o catálogo não for reancorado.

Se a alternativa 2 for a escolhida, o S2-C1 abaixo está errado para `A\u200bPPLY_PATCH` e o texto tem de ser reescrito antes de qualquer RED.

## 3. Critérios

Arquivo novo, único: `test_counterfactual_s2_fallback_veto_key_contract.py`. Fiação de C1–C4: a da sonda (`RecordingWorld` + extrator com a aresta irrelevante; situação `"system vulnerable"`). Previsão de falha na `01d9886` está marcada. Não executei esses testes: o arquivo não existe. A previsão compara a saída medida com a asserção.

O grupo «continua mapeando» do pedido não é um critério único. Segue a alternativa 1.

### S2-C1 — chave diferente não mapeia (previsão: RED)

Teste: `test_s2_c1_different_veto_key_does_not_map`.

Para cada grafia, `calls == []`, `counterfactual_outcome` é o `heur_line_354` medido, `actual_outcome == "outcome_without_intervention"`, `intervention` é a própria grafia, cadeia é o passo lexical medido.

| grafia | outcome exigido |
|---|---|
| `service shutdown` | `outcome_with_service_shutdown` |
| `Service, Shutdown!` | `outcome_with_Service,_Shutdown!` |
| `service_shutdown()` | `outcome_with_service_shutdown()` |
| `patch apply` | `outcome_with_patch_apply` |
| `alert ignore` | `outcome_with_alert_ignore` |
| `\ufb01INCREASE_LOGGING` | `outcome_with_` + essa string (não há espaço) |

`alert ignore` entra porque `IGNORE_ALERT` é o outro nome de `DEFAULT_VETOES` e a sonda mostrou o mesmo desvio de ordem. U+FB01 entra como caso de tokens iguais e `veto_key` diferente que não é troca de ordem.

### S2-C2 — fallback com a mesma chave continua (previsão: passa hoje)

Teste: `test_s2_c2_same_veto_key_fallback_still_maps`.

`shutdown.service` e `shutdown_service()`: `calls == [["SHUTDOWN_SERVICE"]]`, outcome `state:alert_level=low,security_patch_applied=False,system_status=offline`, cadeia lexical medida, `intervention` intacta. Estas duas são as que ainda matam `cf_token_set_match_dropped` se o catálogo passar a rodar este arquivo.

### S2-C3 — passo 1 intacto (previsão: passa hoje)

Teste: `test_s2_c3_step1_names_still_map`.

| grafia | `calls` | outcome |
|---|---|---|
| `apply patch`, `apply_patch`, `APPLY_PATCH` | `[["APPLY_PATCH"]]` | `state:alert_level=low,security_patch_applied=True,system_status=secure` |
| `SHUTDOWN_SERVICE`, `shutdown service` | `[["SHUTDOWN_SERVICE"]]` | `state:alert_level=low,security_patch_applied=False,system_status=offline` |

Nome exato continua mapeando por este critério, não pelo fallback.

### S2-C4 — não identificada permanece (previsão: passa hoje)

Teste: `test_s2_c4_unidentified_strings_stay`.

`hello`, `patch` e `isolate the system`: `calls == []`, outcome `outcome_with_` + espaços em `_`, cadeia lexical, `intervention` intacta. `""` e `False`: `calls == []` e `outcome_without_intervention`. `"   "`: `calls == []` e `outcome_with____`.

### S2-C5 — propriedade (previsão: RED nos exemplos de C1)

Teste: `test_s2_c5_simulated_action_is_none_or_same_veto_key`.

Para a string da estratégia: ou `simulate` não é chamado, ou a única ação chamada `A` satisfaz `veto_key(texto) == veto_key(A)`. Exemplos fixos: os de C1 (têm de falhar na base), os de C2 e C3, `hello`, `patch`, `A\u200bPPLY_PATCH`. Estratégia: texto com letras, dígitos, espaço, `_`, `-`, `.`, `!`, `()` e U+200B, mais os exemplos. Sem `deadline`.

Se este teste falhar numa string que o passo 1 já devolve, parar e relatar. A amostra não achou esse caso. A correção não pode sair de `_map_intervention` nem entrar no passo 1 para «salvar» a propriedade.

### S2-C6 — rank não muda (previsão: passa hoje)

Teste: `test_s2_c6_rank_noncanonical_stays_unmodeled_and_exact_stays_vetoed`.

Mesma observação e factual da sonda.

- `[["service shutdown"]]`, `[["shutdown.service"]]`, `[["patch apply"]]`, `[["shutdown service"]]`: `ranking == []`, `vetoed == []`, `unmodeled` é a lista da grafia, `best is None`.
- `[["SHUTDOWN_SERVICE"], ["ISOLATE_SYSTEM"]]`: `best["actions"] == ["ISOLATE_SYSTEM"]`; `SHUTDOWN_SERVICE` está em `ranking` e em `vetoed`, com o motivo `DEFAULT_VETOES["SHUTDOWN_SERVICE"]` (a string medida na sonda). `unmodeled == []`.

O rank não chama `_map_intervention`. Este critério pega uma regressão acidental, não o fechamento do vão «mesma chave, nome não exato».

### S2-N — não-mudança

Não há teste novo para N. O verificador é o diff contra `01d98868db455cf8f426a52fc5cdf95054c4392b`.

- **S2-N1.** Produção fora de `_map_intervention` intocada. Inclui o passo 1, a adjacência, `action_effects.py`, `scm_counterfactual.py`, `mcts_planner.py`, `causal_world_model.py` e `core/constitutional_brain.py`.
- **S2-N2.** `tools/targeted_mutation.py` intocado. Nenhum mutante desse arquivo ancora o fallback. Não peço autorização para editá-lo.
- **S2-N3.** `tools/cognitivo_mutation.py` intocado neste pré-registro. Na implementação, a autorização precisa está na seção 4, conforme a opção. Sem ela, a previsão é gate vermelho (`survived` ou `not_applied`).
- **S2-N4.** `CONTINUATION.md` e `docs/agentes/CLAIMS.yaml` intocados.
- **S2-N5.** Workflows, timeouts, pisos de cobertura, validadores, contratos e aceites do Axis intocados.
- **S2-N6.** `test_*.py` já existentes intocados, salvo a autorização da opção (a), limitada aos três pontos nomeados de `test_counterfactual_dox_contract.py`.

## 4. Opções para `:143`, o oráculo `:70-79` e o `@example` `:397`

As quatro strings de `:143` não são um mecanismo só. Três passam no passo 1. Só `patch apply` usa o fallback e é a única que a alternativa 1 deixa de mapear. `@example("apply patch")` (`:396`) não precisa mudar. `@example("patch apply")` (`:397`) precisa, se a alternativa 1 for aprovada.

### (a) Errata de CF9 e inverter `patch apply`

Errata datada, sem reescrever o parágrafo antigo do CF9: o passo 5 e o CF9-outcome deixam de exigir `patch apply` → `APPLY_PATCH`. `apply patch` continua no passo 1.

Editar só, em `test_counterfactual_dox_contract.py`:

- `:143`: `patch apply` passa a exigir `calls == []` e `outcome_with_patch_apply`. `apply patch`, `apply_patch` e `APPLY_PATCH` continuam exigindo `[["APPLY_PATCH"]]` e o estado `secure`.
- oráculo `:70-79`: o mesmo filtro da alternativa 1 (`veto_key` igual). O resto de `spec_action` fica.
- `:397`: o exemplo deixa de esperar mapeamento. O teste de propriedade antigo passa a concordar com o oráculo novo.

Consequência: o contrato CF9 deixa de mentir sobre `patch apply`. O mutante `cf_token_set_match_dropped`, cuja tupla só roda esse arquivo, deixa de ser morto por ele. **Inferência** (o mutante não rodou de verdade: `hypothesis` ausente; a agulha e a tupla foram lidas): com o `if` desligado, `patch apply` também não mapeia, e nenhum outro assert desse arquivo exige um retorno positivo do fallback.

Autorização precisa, além da errata e dos três pontos do teste: em `tools/cognitivo_mutation.py`, acrescentar `test_counterfactual_s2_fallback_veto_key_contract.py` à tupla `tests` de `cf_token_set_match_dropped`. Não reescrever a agulha se as quatro linhas permanecerem. S2-C2 (`shutdown.service`) é o que mata o mutante. Sem esse acréscimo, a previsão é `survived` e o gate sai diferente de 0.

Arquivos tocados na implementação, se (a) e alternativa 1: `core/cognitivo/counterfactual.py` (só o fallback), o teste novo, os três pontos do dox, a tupla desse mutante, e a errata no doc do CF9 ou num bloco datado. Nada mais.

### (b) Manter `:143` e limitar a S2 ao veto destrutivo

O fallback só recusa o nome quando ele está em `DEFAULT_VETOES` (`SHUTDOWN_SERVICE` e `IGNORE_ALERT`) e a `veto_key` difere. `patch apply` continua `APPLY_PATCH`. `:143`, o oráculo e `:397` ficam. S2-C1 perde a linha `patch apply` e a linha U+FB01 se o nome não for veto. S2-C5 ganha uma exceção escrita: ordem trocada de nome que não está no veto padrão pode simular.

Coerência: serve para não quebrar o texto do CF9 e para recusar `service shutdown` e `alert ignore`. Não serve como regra de identidade. O mesmo mecanismo que o CF9 pediu para `patch apply` continua para `APPLY_PATCH`, `DEPLOY_UPDATE`, `RESTORE_BACKUP` e para `apply!patch` (`veto_key` `None`, mapeado hoje a `APPLY_PATCH`). O simulador fica acoplado ao conteúdo atual de `DEFAULT_VETOES`: pôr `APPLY_PATCH` no veto depois não corrige o fallback. `shutdown.service` continua simulado nos dois lados, porque a chave coincide; (b) não fecha esse caso.

O mutante existente continua morto por `patch apply` no dox, desde que a agulha permaneça. Não peço edição de `tools/cognitivo_mutation.py` nesta opção.

### (c) Exceção literal só da string `patch apply`

A restrição da alternativa 1 vale para toda string, exceto a igualdade exata `intervention == "patch apply"`, que continua devolvendo `APPLY_PATCH`. `:143`, o oráculo e `:397` ficam byte a byte. `update deploy`, `alert ignore`, `service shutdown` e U+FB01 deixam de mapear. S2-C5 não pode ser o invariante puro: a exceção é uma linha do teste.

Coerência: mantém o único subcaso de `:143` que o fallback sustenta, e recusa o resto. É uma constante no meio da função, não a `veto_key`. Qualquer outra ordem trocada de `APPLY_PATCH` (`apply, patch` com vírgula, por exemplo) deixa de mapear, ao contrário de (b). O mutante existente continua morto por `patch apply`, se a agulha e o `return` dessa string permanecerem no caminho que o dox exercita.

### Recomendação

Recomendo **(a)** com a alternativa 1. `patch apply` é a única string de `:143` que a regra pedida rejeita, e o CF9 a nomeou no passo 5. Manter a asserção é manter um mapa cuja `veto_key` medida é `PATCH_APPLY` contra `APPLY_PATCH`. (b) preserva esse mapa de propósito e ainda preserva os outros nomes que não estão no veto. (c) preserva uma string. A autorização extra de (a) é uma tupla de um mutante, nomeada acima. Se essa tupla não puder ser editada, (a) deixa o gate de mutação vermelho e (b) é a opção que não pede o arquivo.

## 5. Gates previstos da implementação

Não rodaram neste commit. Números de suíte, cobertura e Bandit não são copiados de documento antigo.

Com `LC_ALL=C`, `PYTHONPATH=.` e `NEXUS_AUDIT_DB` fora do repositório:

- `python -m unittest discover -p 'test_*.py'`
- `python -m unittest test_counterfactual_dox_contract.py test_counterfactual_s2_fallback_veto_key_contract.py` e os outros `test_counterfactual*.py` / `test_scm_counterfactual*.py`
- `python tools/cognitivo_mutation.py` — declarado = morto, `survived` 0, `not_applied` 0
- `python tools/targeted_mutation.py`
- `bash tools/agente/replay/run_replay.sh`
- `ruff check` nos caminhos do workflow `.github/workflows/nexus-contract-gate.yml`
- Bandit pelo caminho desse workflow (`security_triage.py --check`): 0 achados novos. O piso de cobertura `core/cognitivo/*` `--fail-under=90` permanece o do workflow; este item não o altera
- Se `reality_audit.py` reescrever `AUDIT_REALITY_REPORT.md` ou `audit_reality.json`, o commit da implementação leva os relatórios. Este pré-registro não os toca

Python do CI (3.11) e o CI hospedado ficam não verificados até essa implementação.

## 6. Fora de escopo

- S2a (`expected_reward == 0.0` para ação inventada) e a errata da linha 180 de `mcts_planner.py`.
- `compare_futures` e veto por efeito.
- A issue #77.
- Fechar o vão que a sonda mediu e esta mudança não cobre: `shutdown service` (passo 1) e `shutdown.service` (fallback com a mesma `veto_key`) simulam `system_status=offline` e, no rank, caem em `unmodeled` com `vetoed` vazio. Alinhar generate e rank para mesma chave exigiria outro pré-registro.
- Passo 1, adjacência, planner, fórmula de plausibilidade, `ACTION_EFFECTS` e `DEFAULT_VETOES`.

## O que este commit é

Um arquivo novo. Não há RED, não há produção alterada, não há push e não há PR.
