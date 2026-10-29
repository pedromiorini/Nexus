# Guarda de margem do timeout do CI — pré-registro e resultados

Origem: proposta de retro `2026-10-28-margem-timeout-ci` (lição L-009). O job `Contract and regression gate` (timeout 12 min) foi cancelado por tempo em #121 e #122, apesar de sucessos já em 689 s e 700 s; o #123 subiu o timeout para 20 min. Pedro aprovou implementar a guarda `timeout_margin.sh`. Regra do Pedro: regra nova só é adotada se provada contra o histórico real do Nexus, com critérios pré-registrados.

## Pré-registro (commit anterior a qualquer coleta ou execução contra o histórico)

**O que foi visto antes de escrever isto (declarado):** a proposta (script, fixtures, casos de replay, resultados dela), a lista de execuções do workflow só para contagem (300 execuções, 16/10 a 29/10; sem durações), a duração dos 3 incidentes e do job do #129 que a proposta/relatos já citam, e o fato de que o timeout mudou de 12 para 20 no #123. **Nenhuma** série de durações do histórico foi lida, nenhum job foi classificado.

### Guarda (comportamento exato)
`tools/agente/timeout_margin.sh <timeout_min> <duracoes.txt>`; `duracoes.txt` tem uma duração em segundos por linha (inteiros positivos).
- **0**: `max(duração) ≤ ⌊timeout_min·60·85/100⌋`. **6**: acima disso (bloqueio). **3**: entrada inválida (timeout ausente/vazio/não inteiro/zero/negativo/decimal/com mais de 5 dígitos; arquivo ausente, diretório, ilegível ou vazio; só linhas em branco; qualquer linha que não seja inteiro positivo com no máximo 9 dígitos, incluindo `0`, negativos, decimais, notação científica, espaços e CR). 3 é distinto de 0 e de 6.
- Decisão só por código de saída (`set -uo pipefail`), sem pipe no caminho de decisão. A proposta usa `^[1-9][0-9]*$` sem limite de dígitos; **endurecimento declarado:** limite de dígitos (a proposta deixa um inteiro de 25 dígitos estourar a aritmética do bash). Previsão: o script da proposta, sem esse endurecimento, falha no caso de linha enorme (RED).

### Protocolo da retroprojeção (determinístico; sem seeds, sem aleatoriedade)
Dados: todas as execuções concluídas do workflow `nexus-contract-gate.yml` que `gh run list --limit 1000` devolver; para cada uma, todos os jobs de todas as tentativas (`actions/runs/{id}/jobs?filter=all`), só o job `Contract and regression gate`. Duração = `completed_at − started_at` em segundos. Timeout `T` do job = `timeout-minutes` do job `contract-gate` no `nexus-contract-gate.yml` do `headSha` da execução (API de conteúdo). Os dados são congelados em `docs/agentes/timeout_margin_historico.json` (instantâneo) e o avaliador (`tools/agente/timeout_margin_backtest.py`) roda **offline** sobre ele e chama o script real da guarda para cada decisão.
- **Sucesso**: conclusão `success`. **Incidente de timeout**: conclusão `cancelled` com duração ≥ `T·60` (cancelamento por concorrência dura menos e não conta). **Fora de escopo** (contados, não avaliados): `failure`, `cancelled` curto, jobs sem `T` obtido (se > 10 % dos jobs → `scenario_invalid`, saída 3).
- **Janela** `N = 5` (fixo, não será ajustado): as 5 últimas durações de sucesso com `completed_at` ≤ instante de avaliação (menos se houver menos; zero → o ponto é descartado e contado). Para um sucesso `J` o instante é o fim de `J`; para um incidente é o início dele. A guarda é chamada com o `T` do job avaliado.
- **Ponto de alarme**: janela de um sucesso `J` cuja guarda sai 6.

### Critérios (todos precisam passar; saída do avaliador 0; 1 se algum falha; 3 se dado inválido/`scenario_invalid`)
- **TM1 incidentes**: os 3 incidentes conhecidos (run `40765682957` tentativa 1; run `40783977991` tentativas 1 e 2) estão no conjunto de incidentes detectado (senão `scenario_invalid`) e **todo** incidente de timeout do histórico (conhecido ou novo) recebe 6 com a janela anterior ao seu início. Alvo: recall 100 %.
- **TM2 sinal antecipado**: o 1º alarme (qualquer sucesso com 6 sob `T = 12`) ocorre **antes** do início do 1º incidente. Alvo: existe, com antecedência > 0 s.
- **TM3 execução recente**: o job do #129 (run `40923631957`, 18 min 20 s com `T = 20`) tem sua janela (incluindo ele) com saída 6.
- **TM4 controle saudável**: todo ponto cuja janela tem máximo ≤ 70 % de `T·60` sai 0, e há **≥ 30** desses pontos (senão `scenario_invalid`). É o controle de implementação/coleta; não é prova estatística.
- **TM5 falsos alarmes**: entre os pontos de alarme classificáveis (os 5 jobs seguintes, de mesmo `T`, só sucesso ou incidente de timeout, existem), **falso alarme** = nenhum dos 5 seguintes passa de 70 % de `T·60` e nenhum é incidente. Alvo: taxa ≤ 20 % com **≥ 5** alarmes classificáveis; com menos de 5 → critério **inconclusivo = falha** (parar e relatar opções, sem trocar limites).
- **TM6 entradas inválidas**: casos negativos da guarda (tabela abaixo) e do avaliador (instantâneo ausente, vazio, JSON inválido, campo faltando, lista de jobs vazia) saem com o código de entrada inválida (3 em ambos), nunca 0 nem 6.
- **TM7 sem regressão**: replay existente (11 casos) + casos novos `tmo` verdes; suíte completa verde em 3.11 e 3.12; CI hospedado verde.
- **TM8 modo de operação** (regra decidida antes de ver dados): a guarda **não** é ligada como passo bloqueante do CI. O CI executa só os casos determinísticos do replay (que não dependem do tempo real). O uso vivo é **relatório** (`python tools/agente/timeout_margin_backtest.py atual`, sai 0 mesmo com 6, mostrando a decisão) antes de marcar pronto um PR que cresce a suíte. Reporta-se também se a guarda, sobre as 5 últimas durações de sucesso, bloquearia o `main` de hoje; se sim, isso é relatado com opções, **sem** subir o timeout.
- **TM9 tendência**: relatório (informativo, sem alvo) da mediana, do máximo e da razão ao limite das durações de sucesso por dia.

### Casos negativos e de fronteira da guarda (esperados)
`12 fixtures/tmo_origem.txt`→6; `12 tmo_anterior`→0; `20 tmo_atual`(938 955 841 872)→0 e `20` com 1100→6; fronteira `12` com `612`→0, com `613`→6; `1` com `51`→0, `52`→6; timeout `abc`, `""`, `0`, `-5`, `12.5`, `012`, `1e1`, `999999` (6 dígitos)→3; sem argumentos→3; arquivo ausente, diretório, vazio, só linhas em branco→3; linhas `abc`, `0`, `-3`, `3.5`, `1e3`, ` 5`, `5 `, `5\r`, `1234567890` (10 dígitos), `99999999999999999999999`→3; arquivo sem newline final `500`→0; linhas em branco entre números são ignoradas.

### Previsões (registradas, não vinculantes)
Guarda da proposta: casos da proposta passam; o caso de 10+ dígitos falha no RED. Retroprojeção: TM1–TM4 passam; TM3 deve dar 6 (1100 s > 1020 s); a guarda hoje provavelmente daria 6 sobre as últimas durações do `main` (então o modo é relatório), mas isso é só previsão.

### Disciplina
Se um critério ou protocolo falhar: parar e relatar opções; **sem errata, sem ajuste** de `N`, limiares, definições ou dados depois de ver o resultado. Reconhecida a limitação: a guarda é um limiar determinístico; a retroprojeção mede recall nos incidentes e o ruído dos alarmes no histórico real, não prova que 85 % seja ótimo. Não toca testes/contrato do Axis, `tools/targeted_mutation.py` nem `versao_aceita`.

## Resultados (registrados depois da pré-registro; critérios e dados não foram alterados)

### RED (commit `a4de80b`)
Testes de contrato rodados contra a guarda da proposta, sem endurecimento: 11 falhas + 2 erros (avaliador/replay ainda inexistentes e guarda sem limite de dígitos). Ressalvas honestas:
- A previsão "o caso de 10+ dígitos falha" confirmou-se, mas a redação "estoura a aritmética do bash" para o inteiro de 25 dígitos não foi exata: a guarda da proposta devolveu 6 (não houve *wrap*). Mantida a redação pré-registrada, sem errata; o endurecimento (limite de dígitos) continua necessário porque a decisão por 6 num lixo de 23 dígitos não é uma entrada válida.
- A falha do timeout de 6 dígitos (`999999` aceito pela guarda da proposta) **não** estava na previsão registrada.

### Dados (commit `48f15e2`)
Instantâneo congelado de 396 jobs `Contract and regression gate` em 393 execuções (347 sucessos, 4 incidentes de timeout, 45 falhas/cancelamentos curtos fora de escopo, 0 sem timeout conhecido). Coleta feita uma vez; nenhum dado editado.

### Critérios (`python tools/agente/timeout_margin_backtest.py avaliar` → exit 0, "resultado: PASSOU")
| Critério | Resultado | Evidência |
|---|---|---|
| TM1 incidentes | PASSOU | 4/4 sinalizados (os 3 conhecidos + 1 **novo**: run `40911472730`, push em `main` `d15b50e`, `T=20`, cancelado por timeout aos 1210 s em 29/10 16:25 UTC) |
| TM2 sinal antecipado | PASSOU | 1º alarme sob `T=12` em 2026-10-28T17:34Z, 1º incidente 20:42Z (antecedência 11287 s ≈ 3 h 08) |
| TM3 execução recente | PASSOU | run `40923631957`: 1100 s, `T=20`, limite 1020 s → 6 |
| TM4 controle saudável | PASSOU | 323 pontos saudáveis (≥ 30), 0 sinalizados indevidos |
| TM5 falsos alarmes | PASSOU | 18 alarmes, 12 classificáveis, 0 falsos (taxa 0,0 % ≤ 20 %) |
| TM6 inválidos | PASSOU | casos negativos da guarda, do avaliador (snapshot ausente/vazio/JSON inválido/campo faltando/lista vazia) e `gh` falso com não-JSON → 3 |
| TM9 tendência | informativo | tabela abaixo |

### TM8 — modo de operação e o `main` de hoje
`python tools/agente/timeout_margin_backtest.py atual`: últimas 5 durações de sucesso do `main` = 938, 1096, 917, 1188, 1186 s → máximo 1188 s > 1020 s (85 % de 20 min) → a guarda **daria 6 (bloquearia) o `main` de hoje**. Por isso a guarda **não** é passo bloqueante: o CI só roda os casos `tmo` do replay e o uso vivo é o relatório `atual` (sai 0). O timeout não foi alterado. Opções (decisão do dono): (a) manter relatório; (b) reduzir o custo do job (a suíte leva ~30 s local e ~18 min no CI hospedado); (c) subir o timeout (fora desta entrega); (d) tornar a guarda bloqueante só depois de (b) ou (c).

### TM9 — tendência (sucessos por dia UTC: n, mediana s, máx s, máx/timeout)
| Dia | n | mediana | máx | razão |
|---|---|---|---|---|
| 2026-10-01..06 | 64 | 29–38 | 37–43 | 0,06–0,07 |
| 2026-10-16 | 58 | 89 | 108 | 0,15 |
| 2026-10-17 | 34 | 112 | 127 | 0,18 |
| 2026-10-20 | 11 | 117 | 155 | 0,22 |
| 2026-10-21 | 42 | 126 | 211 | 0,29 |
| 2026-10-22 | 65 | 212 | 401 | 0,56 |
| 2026-10-23 | 35 | 394 | 467 | 0,65 |
| 2026-10-28 | 16 | 518 | 700 | 0,97 (T=12 até o #123) |
| 2026-10-29 | 14 | 1003 | 1188 | 0,99 (T=20) |

A duração mediana subiu de ~30 s (início de outubro) para ~1003 s hoje; com `T=20` a margem restante do pior sucesso é 12 s e um job de `main` já foi cancelado por timeout (1210 s).

### Limitações
A guarda é um limiar determinístico; a retroprojeção mede recall nos incidentes reais e o ruído de alarmes (0/12 classificáveis), não prova que 85 % seja ótimo. Há dependência de durações de runners hospedados (variam ±10 %).

### Efeito colateral nos contadores de estado
O avaliador chama `subprocess` (`gh` e a guarda), o que acrescenta 7 achados LOW ao Bandit (B404 ×1, B603 ×3, B607 ×3): 131 → 138. Os achados não foram suprimidos (`# nosec`); a triagem foi regenerada (`SECURITY_TRIAGE.md`, `--check` ok) e o contador `bandit_low` foi sincronizado em `NEXUS_KNOWLEDGE.md`, `NEXUS_EVIDENCE.md`, `CONTINUATION.md`, `SECURITY_TRIAGE.md` e na fixture de `test_state_stray_marker_contract.py` (131 → 138), senão o gate de consistência de estado do CI hospedado falharia.

### Falha do CI hospedado na 1ª tentativa (erro de teste meu, não de critério)
Na 1ª execução hospedada (run 40935534453) o `contract-gate` falhou: `test_default_table_includes_timeout_cases_and_passes` rodava a tabela padrão inteira do replay, cujos casos antigos (`A1-origem` etc.) exigem histórico git completo, e o checkout do `contract-gate` é raso. O job `Agent guard replay` (`fetch-depth: 0`) passou. Correção: o teste agora executa só as 6 linhas `tmo` (verificado também em clone `--depth 1`). Nenhum critério, limiar ou dado foi alterado.
