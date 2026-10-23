# Auditoria funcional — capacidades extraídas em core/cognitivo (22/10/2026)

Base: main `f370513`. Critérios escritos ANTES de qualquer execução. Seeds fixas.
Regra do Pedro: só adotar se provado melhor. "Passa" = todos os critérios do item.

## 1. Memória episódica (`episodic_memory.py`)
**Propósito declarado:** "POST5: Episodic Memory". Registrar episódios (título, gatilho, importância)
e recuperá-los depois (`search_episodes`); persistir em SQLite; usado pelo learner
(`search_episodes(status="closed", limit=n)`) e pelo router (`search_episodes(query=prompt, limit=3)`).

| Id | Critério falseável | Limite | Baseline |
|---|---|---|---|
| EM1 | Recuperação: 20 tópicos × 10 episódios fechados (títulos = 3 palavras-chave do tópico em ordem aleatória + 2 de enchimento, seed 7); 40 consultas com 2 palavras-chave do tópico em ordem trocada. precision@5 média | ≥ 0,80 **e** ≥ baseline + 0,30 | recência (`search_episodes(None, limit=5)`) |
| EM2 | Persistência: arquivo SQLite, 50 episódios, fechar, reabrir; mesmos ids e estatísticas | igualdade exata | — |
| EM3 | Escala: 10 000 episódios em `:memory:`; p95 de `search_episodes(query, limit=10)` em 200 consultas | ≤ 50 ms | — |
| EM4 (info) | `importance` influencia a ordenação? | informativo | — |

## 2. Codificação preditiva (`predictive_coding.py`)
**Propósito declarado:** "REAL Bayesian Prediction Machine": gera predições, compara com observações,
"atualiza predições para minimizar erro", ponderação por precisão; `update_belief_state` = filtro bayesiano POMDP.

| Id | Critério falseável | Limite | Baseline |
|---|---|---|---|
| PC1 | Fonte estacionária categórica {a,b,c} p=(0,7;0,2;0,1), 500 passos via `hierarchical_prediction_cycle({"sensory": x})`, seed 11. Taxa de erro de nível 0 (`error_magnitude`) nos últimos 200 | ≤ 0,35 **e** ≤ persistência − 0,05 | sem aprendizado (comportamento atual) e persistência (prever a última observação) |
| PC2 | Mudança de regime: 300 passos com p=(0,7;0,2;0,1), depois 300 com p=(0,1;0,2;0,7). Taxa de erro nos passos 100–300 após a troca | ≤ 0,40 | persistência |
| PC3 | Calibração: ECE (10 bins) de `confidence` do nível 0 contra acerto, na série do PC1 | ≤ 0,10 | — |
| PC4 | Filtro POMDP: 100 POMDPs aleatórios (2–6 estados, seed 13) contra posterior em forma fechada | erro máx ≤ 1e-9 | — |

## 3. Planejador MCTS (`mcts_planner.py`)
**Propósito declarado:** "Monte Carlo Tree Search Planner": achar a melhor sequência de ações para um goal,
respeitando `available_actions`; docstring diz "200 = ~50ms".

| Id | Critério falseável | Limite | Baseline |
|---|---|---|---|
| M1 | Ótimo conhecido: domínio padrão, goals {security, stability, cost, genérico}, max_depth ∈ {2,3}; valor = `_evaluate_state` após aplicar a sequência; ótimo por força bruta (mesmas regras de ações). 2000 iterações, 10 seeds | valor ≥ 0,98 × ótimo em ≥ 90% | — |
| M2 | Valor médio do MCTS (2000 it.) | ≥ guloso de 1 passo **e** > aleatório + 0,02 | guloso, aleatório (média de 200 sequências) |
| M3 | Melhora com iterações: regret médio | regret(2000) < regret(50) | — |
| M4 | `available_actions`: toda ação do plano ∈ lista; lista do chamador não é alterada; `_action_effects` não cresce entre chamadas | 100% | — |
| M5 | Tempo: `plan(max_iterations=200)` p95 em 20 execuções | ≤ 0,10 s | — |

## Harness
`tools/cognitivo_eval.py --capability {episodic,predictive,mcts,all}`: exit 0 se tudo passa, 1 se algum critério falha,
2 para entrada ausente/vazia/desconhecida. Saída JSON com métricas.

## Regra de adoção
Melhoria só é promovida se: critério pré-registrado passa, sem regressão (suíte, demo, replay 11/11,
pisos de cobertura, os dois gates de mutação, bandit/triage), API pública compatível, sem dependência nova pesada.

---

## v2 — emenda de critérios (22/10/2026 15:29, aprovada por Pedro; escrita ANTES de qualquer avaliação v2)
A v1 acima permanece inalterada e visível. A v2 substitui apenas PC1, PC3 e M3; os demais critérios seguem a v1.
O harness passa a aceitar `--criteria {v1,v2}` (padrão v2); v1 continua reproduzível.

**Motivos da emenda**
- **PC1/PC3, variância amostral:** com uma única seed (11) e janela de 200 passos, o desvio-padrão da taxa de erro
  (~0,03) é da ordem da folga até o limite; o resultado de uma seed não distingue método de sorte.
- **M3, efeito teto:** com max_depth ∈ {2,3}, um planejador correto atinge o ótimo já com 50 iterações; os dois regrets
  valem 0 e o `<` estrito fica inatingível, embora o critério pretenda medir melhora com iterações.

| Id | Critério v2 | Limite |
|---|---|---|
| PC1 v2 | Mesmo protocolo da v1 (fonte p=(0,7;0,2;0,1), 500 passos, últimos 200), repetido nas **20 seeds fixas 100,101,…,119**. Métrica = média das 20 taxas de erro | média ≤ 0,35 **e** média ≤ média da persistência − 0,05. Taxa de aprovação por seed reportada (informativa) |
| PC3 v2 | ECE (10 bins) da v1 em cada uma das mesmas 20 seeds. Métrica = média dos 20 ECE | média ≤ 0,10. Taxa de aprovação por seed reportada (informativa) |
| M3 v2 | Cenário endurecido: domínio padrão, os 4 goals da v1, **max_depth = 8**, seeds 0–9 (`random.seed(s)` antes de cada `plan`). Valor = `_evaluate_state` após a sequência. Regret = (melhor valor observado no cenário, entre todos os planos avaliados) − valor; equivalente a comparar médias | regret médio(2000) **<** regret médio(50), `<` estrito mantido |

Regra: se um PR falhar a v2, os critérios não serão ajustados de novo; a falha é reportada.

### Registro de resultados v2 (22/10/2026); critérios acima não alterados
- #89 (episódica): EM1 1,00, passa.
- #90 (preditiva): PC1 v2 0,3195 (persistência 0,479; 17/20 seeds), PC2 0,25, PC3 v2 0,0798 (17/20 seeds): passa.
- #91 (MCTS): M1 1,00, M2 0,8808, M4 e M5 passam. **M3 v2 FALHA**: regret(2000) = regret(50) = 0,0 com max_depth=8
  (o teto persiste). Mesclado com a falha registrada, por aprovação de Pedro em 22/10 16:52; sem nova emenda.

---

## v3: emenda do M3 (22/10/2026 17:38, aprovada por Pedro; escrita ANTES de qualquer avaliação v3)
As seções v1 e v2 acima permanecem inalteradas. A v3 acrescenta apenas `M3 v3` e o pré-teste `M3P`; os demais critérios seguem a v2.
O harness passa a aceitar `--criteria v3`. Os padrões anteriores não mudam.

**Motivo:** o M3 v2 falhou em main (`feef2f3`) com regret 0,0 em 50 e em 2000 iterações. O diagnóstico é efeito teto:
no domínio padrão o ótimo tem uma ação (`APPLY_PATCH`) e a raiz o expande nas primeiras ~10 iterações. Além disso, o
regret era medido contra o "melhor observado", não contra um ótimo exato.
Referências: docs `PESQUISA_MCTS_2026-10-22.md`; Schadd et al. 2008 (SP-MCTS); Cazenave 2009 (NMCS); Rosin 2011 (NRPA); testes de valor exato do OpenSpiel.

**Cenário (domínio sintético injetado no planejador; o domínio padrão continua coberto por M1/M2/M4):**
- 20 instâncias, `i = 0..19`, cada uma uma árvore determinística gerada por `random.Random(1000 + i)`:
  - ramificação 6 (ações `A0..A5`, todas disponíveis em todo nó), profundidade 6 (46 656 folhas); `max_depth = 6`;
  - cada aresta recebe um peso U[0,1), sorteado em ordem DFS;
  - valor de folha = média dos 6 pesos do caminho; nós internos valem 0.
- **Ramo enganoso:** um primeiro passo `D` sorteado tem todas as folhas multiplicadas por 0,5.
- **Agulha:** um caminho sorteado dentro de `D` vale 1,0. É o ótimo único, porque todas as outras folhas valem menos que 1.
- **Ótimo exato:** calculado por força bruta sobre as 46 656 folhas.
- **Execuções:** seeds de execução 0..9, com `random.seed(s)` antes de cada `plan`, e orçamentos {50, 200, 800, 2000}.
- **Regret** de uma execução = ótimo exato − valor da sequência devolvida (uma sequência que não termina em folha vale 0).

| Id | Critério | Limite |
|---|---|---|
| M3P (pré-teste de validade, independente da qualidade do planejador) | Verifica, por instância: (a) a agulha é o ótimo único e está em `D`; (b) a média das folhas de `D` é menor que a de cada outro primeiro passo; (c) o número de folhas distintas avaliadas com orçamento 50 é ≤ 50, contado por instrumentação. Para qualquer algoritmo com ≤ 50 avaliações de folha, a agulha é uniforme entre as 7 776 folhas de `D` e os demais valores são independentes da sua posição, então P(ótimo em 50) ≤ 50/7 776 ≈ 0,64%. A cota implica orçamento 50 subótimo esperado em ≥ 99% das execuções | (a), (b) e (c) em 20/20 instâncias **e** cota ≥ 50%. Se M3P falhar, o cenário é inválido e M3 v3 não é julgado |
| M3 v3 | Regret médio nas 200 execuções (20 × 10) por orçamento | média(2000) **<** média(50), estrito, **e** mediana não crescente: med(50) ≥ med(200) ≥ med(800) ≥ med(2000) |
| M3I (informativo) | Fração de execuções no ótimo com 2000; avaliações de folha até o ótimo, nas execuções que o atingem | — |

Regra: se o M3 v3 falhar, os critérios não serão reajustados; a falha é reportada. Mudanças no planejador ficam fora deste PR.

---

## M6: qualidade de busca (22/10/2026 18:37, aprovada por Pedro; escrita ANTES de qualquer execução de candidato)
As seções v1, v2 e v3 acima permanecem inalteradas. O M6 é um critério novo. Esta seção foi pré-registrada no commit `5b9ad36` (branch `cursor/mcts-search-m6`, PR #93) antes de qualquer execução.

**Candidato pré-registrado (um só):** Nested Monte-Carlo Search (Cazenave, IJCAI 2009, https://www.ijcai.org/Proceedings/09/Papers/083.pdf),
nível 2, com reinícios até esgotar o orçamento.
- A melhor sequência encontrada é memorizada, no espírito do top score do SP-MCTS (Schadd et al. 2008).
- **Ativação:** opt-in por `RealMCTSPlanner(strategy="nmcs")`. O padrão `strategy="uct"` mantém o comportamento atual.
- **Orçamento equivalente:** `max_iterations` = número máximo de avaliações de folha (rollouts completos). A UCT faz uma por iteração.
- **Fora deste PR:** NRPA (Rosin 2011), o termo de variância do SP-MCTS e progressive widening. Ficam para outra rodada, para manter um só candidato.

**Cenário:** idêntico ao M3 v3. São as 20 instâncias `SyntheticTree(0..19)`, seeds de execução 0..9 com `random.seed(s)` antes de cada `plan`, e orçamento 2000.
Ótimo exato por força bruta.

**Baseline:** o planejador atual (`uct`) em main `76bce11`, medido no #92: fração no ótimo @2000 = 0,015 e regret médio @2000 = 0,1377.

**Cota analítica (informativa, calculada antes):**
- Com 2000 avaliações, nenhum algoritmo passa de P(ótimo) ≤ 2000/7 776 ≈ 0,257, porque a agulha é uniforme nas 7 776 folhas do ramo enganoso e independente dos demais valores.
- A busca uniforme dá ≈ 2000/46 656 ≈ 0,043.

| Id | Critério | Limite |
|---|---|---|
| M6a | Fração das 200 execuções que atingem o ótimo exato @2000 com `strategy="nmcs"` | **≥ 0,10** e > 0,015 |
| M6b | Regret médio @2000 com `strategy="nmcs"` | **≤ 0,10** e < 0,1377 |
| M6c (não regressão do candidato) | Com `strategy="nmcs"`: M1, M2, M4 e M5 (p95 de `plan(200)` ≤ 0,10 s) da v2, e M3 v3 | todos passam |
| M6d (não regressão do padrão) | Com `strategy` omitido: plano do domínio padrão com seed fixa idêntico ao de main (golden SHA da extração), suíte, M1–M5 v2 e M3 v3 inalterados | idêntico |
| M6e (API) | Assinaturas públicas compatíveis: `strategy` é opcional e só por palavra-chave; um valor desconhecido gera `ValueError` | — |

**Sucesso** exige M6a e M6b e M6c e M6d e M6e. Se o candidato falhar, os limites não serão ajustados e a falha é reportada.
Se o candidato passar, tornar `nmcs` o padrão fica como proposta ao Pedro, não neste PR.

### Registro de resultados M6
Medido em 22/10/2026 com `python tools/cognitivo_eval.py --capability mcts --criteria m6`. Os limites pré-registrados no commit `5b9ad36` não foram alterados.

| Id | Valor (`nmcs`) | Baseline (`uct`) | Resultado |
|---|---|---|---|
| M6a | fração no ótimo @2000 = 0,010 | 0,015 | **FALHA** |
| M6b | regret médio @2000 = 0,1283 | 0,1377 | **FALHA** (melhora, mas acima de 0,10) |
| M6c | M1 = 0,75 (FALHA); M2 = 0,8708; M3 v3 passa; M4 passa; M5 p95 = 0,005 s | — | **FALHA** |
| M6d | padrão inalterado: M1–M5 e M3 v3 passam, plano com seed fixa idêntico | — | passa |
| M6e | API compatível | — | passa |

**Conclusão:** o candidato NMCS nível 2 **não passa** (falha em M6a, M6b e M6c). O padrão `uct` não muda.
- **Destino do código:** por decisão do Pedro (22/10 19:23), o código NMCS **não** entra na main.
- **Onde recuperá-lo:** PR #93 (fechado sem merge), head `ea470fb`, branch `cursor/mcts-search-m6` mantida. Lá estão a estratégia opt-in, o harness `--criteria m6`, os testes e os mutantes.
- **Limite de qualquer algoritmo neste cenário:** com 2000 avaliações, P(ótimo) ≤ 2000/7 776 ≈ 0,257. A agulha do ramo enganoso não tem sinal nos irmãos.
- **Desvio de processo:** os testes de contrato foram escritos logo após o código, mas antes de qualquer push do código. O pré-registro dos critérios foi respeitado.
- **Hipóteses ainda NÃO testadas** (cada uma exige pré-registro novo): NRPA (Rosin, IJCAI 2011), o termo de variância do SP-MCTS (Schadd et al. 2008) e progressive widening/bias.

---

## Auditoria funcional C — raciocínio causal + modelo de mundo (pré-registro)

Registrado **antes** de qualquer execução, em commit próprio, sobre `core/cognitivo/causal_world_model.py` (extraído no #95, move-only). Propósito declarado no código: POST7 "Deep Causal Reasoning" (extrair relações causa→efeito de texto e manter um grafo causal) e POST13 "World Model & Simulation" (prever as consequências de sequências de ações e escolher o melhor futuro). Avaliador: `tools/cognitivo_eval.py --capability causal --criteria v1`. Conjuntos congelados abaixo; qualquer mudança neles invalida a rodada.

**Conjunto L (extração, 30 frases, rótulo = par (causa, efeito) normalizado em minúsculas, ou ∅):**
- 18 positivas, 6 formas (3 cada): `X causes Y`, `X cause Y` (plural), `Y because X`/`Y because of X`, `X leads to Y`, `X results in Y`, `Y due to X`.
- 12 negativas: 4 negações (`X does not cause Y`, `X never causes Y`, `X doesn't cause Y`, `there is no evidence that X causes Y`), 4 correlações (`X is correlated with Y`, `X and Y co-occur`, `X is associated with Y`, `X often appears with Y`), 4 sem relação contendo a substring "cause" (`because`, `causeway`, `the cause is unknown`, `causal models are hard`).
- Acerto = par extraído igual ao rótulo (após `strip().lower()` e remoção de pontuação final); par extra numa negativa = FP.

**Critérios (falsificáveis, com baseline definido a priori):**

| id | propósito | métrica | limite | baseline de referência |
|---|---|---|---|---|
| C1a | extrair causalidade expressa em linguagem natural | F1 em L | ≥ 0.70 | regex só `causes?` (o próprio desenho atual) — documenta-se, não é alvo |
| C1b | não afirmar causa negada/correlacional | taxa de FP nas 12 negativas | ≤ 0.10 | extrair sempre que houver "cause" |
| C2a | prever consequências das ações que o próprio sistema oferece | fração de ações do vocabulário de consumidores (`RealMCTSPlanner` default + benchmark) com efeito ≠ no-op **ou** aviso explícito | ≥ 0.80 | — |
| C2b | não mascarar ação desconhecida | `simulate(["UNKNOWN_X"])` produz `warnings` não vazio | verdadeiro | — |
| C3 | escolher o melhor futuro | acurácia top-1 em 6 conjuntos de futuros com ordem verdadeira estrita (inclui 3 conjuntos em que todos falham mas um é menos ruim: `vulnerable` > `compromised`) | ≥ 5/6 | escolher índice 0 |
| C4 | integrar o grafo causal ao modelo de mundo | após `extract_causal_relations("DEPLOY_FIX causes system_status secure")`, `simulate(["DEPLOY_FIX"])` reflete o efeito | verdadeiro | — |

Regra: nenhum ajuste de código nesta rodada (apenas medir); lacunas e opções de correção com base em literatura são registradas abaixo após a execução.

### Resultados C (v1, executado após o pré-registro 285851b)

`python tools/cognitivo_eval.py --capability causal --criteria v1` — determinístico (sem semente).

| id | valor | limite | passou | referência |
|---|---|---|---|---|
| C1a | F1 0.4138 (P 0.5455, R 0.3333) | ≥ 0.70 | não | recall por forma: causes 3/3, cause 3/3, because/leads to/results in/due to 0/3 |
| C1b | FP 0.4167 (5/12) | ≤ 0.10 | não | baseline "tem 'cause'": 0.5833 — as 4 negações explícitas viram relações positivas |
| C2a | 0.20 (2/10) | ≥ 0.80 | não | só APPLY_PATCH e IGNORE_ALERT; as outras 8 ações do RealMCTSPlanner são no-op silenciosos |
| C2b | falso | verdadeiro | não | `warnings` é sempre `[]` |
| C3 | 3/6 | ≥ 5/6 | não | baseline índice 0: 1/6; erra os 3 conjuntos em que todos falham (sucesso booleano → empate → índice 0) |
| C4 | falso | verdadeiro | não | `causal_reasoning` é guardado mas nunca lido pelo simulador |

**Conclusão honesta:** o módulo cumpre só o caso literal "X causes Y" e as duas transições codificadas; como raciocínio causal e modelo de mundo, falha nos 6 critérios. Nada foi corrigido nesta rodada.

### Lacunas e opções de correção (com base em literatura; nenhuma aplicada)

1. **Extração (C1a)** — banco de padrões de marcadores causais (Khoo et al., 2000; Girju, 2003: *because (of)*, *lead(s) to*, *result(s) in*, *due to*, *therefore*) com ordem causa/efeito por padrão; custo baixo, só stdlib. Alternativa forte: extração por árvore de dependências ou modelos supervisionados (UniCausal, Tan et al., 2023) — dependência pesada, exige aprovação.
2. **Negação/correlação (C1b)** — escopo de negação estilo NegEx (Chapman et al., 2001) e lista de *hedges*/correlacionais que bloqueiam a relação ou a marcam com `pattern_type="negated"`; usar `\bcauses?\b` com limites de palavra.
3. **Cobertura de ações (C2a/C2b)** — tabela declarativa de efeitos STRIPS/PDDL (Fikes & Nilsson, 1971) como fonte única compartilhada com `RealMCTSPlanner._action_effects`; ação desconhecida → `warnings`. C2b é quase trivial, mas muda o golden e foi deixado para a rodada de correção.
4. **Ranking de futuros (C3)** — utilidade ordinal/esperada por estado (secure > stable > vulnerable > compromised) em vez de sucesso booleano; desempate explícito.
5. **Integração causal (C4)** — compilar arestas extraídas em um modelo causal estrutural e simular intervenções `do(X)` (Pearl, 2009); a longo prazo, modelos de transição aprendidos (Ha & Schmidhuber, 2018; Dreamer, Hafner et al., 2020) — pesado.

Ordem sugerida (custo/benefício): 3 → 4 → 2 → 1 (padrões) → 5 (SCM mínimo em stdlib).

---

## Correção R1 — tabela única de efeitos de ação + aviso de ação desconhecida (pré-registro)

Registrado antes de implementar/executar. Critérios C1–C4 inalterados (v1, `--capability causal --criteria v1`).

| id | critério de sucesso R1 | antes (main b1f8fdc) |
|---|---|---|
| C2a | ≥ 0.80 | 0.20 |
| C2b | verdadeiro | falso |
| não regressão MCTS v2 | M1 = 1.0, M2 ≥ 0.8808, M4 verdadeiro, M5 ≤ 0.10 s | 1.0 / 0.8808 / verdadeiro / 0.0066 |
| não regressão MCTS v3 | M3P, M3, M3I passam (M3 ≥ valor de main 0.1377 não exigido; só passar) | 0.9936 / 0.1377 / 0.015, todos passam |
| golden de extração | `feb0815a…` inalterado (se mudar, re-pin só com justificativa no PR) | — |

Escopo: tabela declarativa única em `core/cognitivo/action_effects.py`, consumida pelo `RealMCTSPlanner` (deltas numéricos) e pelo `RealWorldModelSimulator` (efeitos simbólicos); ação fora da tabela → `warnings` e estado inalterado. C1, C3 e C4 não são alvo (resultado relatado mesmo assim).

---

## Correção R2 — ranking ordinal de futuros (pré-registro)

Registrado antes de implementar/executar. Empilhado em R1 (#97) só para evitar conflitos nos arquivos append-only (AUDITORIA, CLAIMS, lista de mutantes). Critérios C inalterados.

| id | critério de sucesso R2 | antes (R1) |
|---|---|---|
| C3 | ≥ 5/6 top-1 nos 6 conjuntos congelados | 3/6 |
| compatibilidade | `compare_futures` mantém as chaves `results`, `best_future_index`, `best_future` e o `ValueError` para lista vazia; quando o sucesso difere, o futuro com sucesso continua vencendo | — |
| não regressão | C2a ≥ 0.80, C2b verdadeiro; golden de extração inalterado (re-pin só com justificativa) | 0.80 / verdadeiro |

Escopo: utilidade ordinal por `system_status` (secure > stable > isolated = offline = vulnerable > compromised), chave `(success, utilidade)`, empate → menor índice.

---

## Correção R3 — escopo de negação (NegEx) + "cause" como palavra inteira + banco de marcadores causais (pré-registro)

Registrado antes de implementar/executar, junto com o conjunto **H** (escrito antes de qualquer código R3, nunca usado para ajustar padrões). Empilhado em R2 (#98) pelos mesmos arquivos append-only. Critérios C inalterados.

| id | critério de sucesso R3 | antes (R2) |
|---|---|---|
| C1a (L) | F1 ≥ 0.70 | 0.4138 |
| C1b (L) | FP ≤ 0.10 | 0.4167 |
| H-F1 (guarda) | F1 em H ≥ 0.60 **e** F1(L) − F1(H) ≤ 0.20 | medido na execução |
| H-FP (guarda) | FP nas negativas de H ≤ 0.20 | medido na execução |
| não regressão | C2a/C2b/C3 mantidos; extração "X causes Y" do golden inalterada (re-pin só com justificativa) | 0.80 / verdadeiro / 6/6 |

Mesma regra de acerto de L (par exato após `strip().lower()` e pontuação final removida).

**Conjunto H (24 frases; 14 positivas, 10 negativas):**

| # | frase | rótulo (causa, efeito) |
|---|---|---|
| 1 | Poor sanitation causes cholera | (poor sanitation, cholera) |
| 2 | Frequent deploys cause regressions | (frequent deploys, regressions) |
| 3 | The outage happened because the router failed | (the router failed, the outage happened) |
| 4 | Prices rose because of the drought | (the drought, prices rose) |
| 5 | Overfishing leads to population collapse | (overfishing, population collapse) |
| 6 | Misuse of antibiotics leads to resistance | (misuse of antibiotics, resistance) |
| 7 | The leak results in pressure loss | (the leak, pressure loss) |
| 8 | Cancellations were due to the strike | (the strike, cancellations were) |
| 9 | The fire was caused by faulty wiring | (faulty wiring, the fire) |
| 10 | Heavy load triggers throttling | (heavy load, throttling) |
| 11 | The disk filled up, so the job failed | (the disk filled up, the job failed) |
| 12 | Inflation was driven by energy costs | (energy costs, inflation) |
| 13 | As a result of the patch, crashes stopped | (the patch, crashes stopped) |
| 14 | Stress contributes to heart disease | (stress, heart disease) |
| 15 | Video games do not cause violence | ∅ |
| 16 | It is false that vaccines cause autism | ∅ |
| 17 | Coffee is not known to cause ulcers | ∅ |
| 18 | Nothing suggests that 5G causes illness | ∅ |
| 19 | Shark attacks rise with ice cream sales | ∅ |
| 20 | Height is correlated with income | ∅ |
| 21 | Umbrellas and rain appear together | ∅ |
| 22 | Because is a conjunction | ∅ |
| 23 | He gave no cause for alarm | ∅ |
| 24 | Causality is a philosophical topic | ∅ |

## Correção C4 — SCM mínimo com intervenções do(X) (pré-registro)

Registrado **antes** de implementar/executar, em commit próprio. Critérios C1–C3, H-F1/H-FP e MCTS (M1/M2/M4/M5, M3 v3) são guarda de não-regressão; o alvo é só C4 (v1 inalterado).

| id | critério de sucesso C4 | antes (pós-R3) |
|---|---|---|
| C4 | após `extract_causal_relations("DEPLOY_FIX causes system_status secure")`, `simulate(["DEPLOY_FIX"]).final_state.state["system_status"] == "secure"` | falso (`causal_reasoning` nunca lido pelo simulador) |
| não regressão | C1a/C1b/C2a/C2b/C3/H-F1/H-FP e MCTS M1/M2/M4/M5 + M3 v3 | 1.0 / 0.0 / 0.80 / T / 6/6 / 0.963 / 0.0 |

Escopo: compilar arestas do grafo causal já extraído num SCM mínimo (stdlib) e, em `simulate`, aplicar intervenção `do(ação)` (Pearl, 2009) — efeitos simbólicos das arestas causa→efeito, sem substituir a tabela `ACTION_EFFECTS`. API retrocompatível. Sem dependências pesadas, sem alterar workflows/Axis/congelados.

## Auditoria funcional CF — raciocínio contrafactual (pré-registro)

Registrado **antes** de qualquer execução, em commit próprio, sobre `core/cognitivo/counterfactual.py` (extraído no PR de move-only). Propósito declarado: POST37 "What-if reasoning" (Pearl 2009; Roese & Olson 1995). Avaliador: `tools/cognitivo_eval.py --capability counterfactual --criteria v1`. Conjuntos congelados abaixo; qualquer mudança neles invalida a rodada.

### Conjunto W (parse "what if", 5 itens com marcador + 1 negativo informativo)

| # | questão | intervenção esperada |
|---|---|---|
| 1 | What if we apply the patch? | we apply the patch |
| 2 | what if the server reboots? | the server reboots |
| 3 | WHAT IF isolation is enabled? | isolation is enabled |
| 4 | What if? | *(vazio)* |
| 5 | And what if DNS fails | dns fails |
| — | Suppose we rollback | *(sem marcador; não conta no denominador de CF4)* |

### Pares S (situação, intervenção) para CF6

`("system vulnerable", "apply patch")`, `("alarm", "patch")`, `("outage", "rollback")`, `("host", "isolate")`.

| id | capacidade | métrica | alvo | baseline documentado (não é alvo) |
|---|---|---|---|---|
| CF1 | honrar protocolo `world_model.predict_outcome` | com WM stub que prefixa `wm:`, `_predict_outcome("sit","act") == "wm:sit:act"` | verdadeiro | heurística `outcome_with_*` |
| CF2 | honrar protocolo `causal.extract_causal_relations` | `_build_causal_chain("alarm","patch") == ["alarm causes patch"]` com `RealCausalReasoning` | verdadeiro | chain heurística de 5 passos |
| CF3 | usar o `planner` injetado | `self.planner` aparece no corpo de algum método público/privado além de `__init__` (AST) | verdadeiro | — |
| CF4 | parse de "what if" em inglês | acertos em W (5 com marcador) | ≥ 5/5 | — |
| CF5 | plausibilidade monótona no comprimento da chain | para comprimentos (1,3,5,10), cada passo é estritamente menor | verdadeiro | — |
| CF6 | intervenção altera o outcome declarado | nos 4 pares S, `counterfactual_outcome != actual_outcome` | 4/4 | — |

Comando: `python tools/cognitivo_eval.py --capability counterfactual --criteria v1` — determinístico (sem semente; uuid irrelevante aos critérios).

### Resultados CF (v1, executado após o pré-registro b3e7c04)

| id | valor | alvo | passou | nota |
|---|---|---|---|---|
| CF1 | verdadeiro | verdadeiro | sim | protocolo world_model |
| CF2 | verdadeiro | verdadeiro | sim | protocolo causal |
| CF3 | falso | verdadeiro | **não** | `self.planner` nunca lido fora de `__init__` |
| CF4 | 5/5 | ≥ 5/5 | sim | só inglês "what if" |
| CF5 | verdadeiro | verdadeiro | sim | cai até o piso 0.1 |
| CF6 | 4/4 | 4/4 | sim | heurística `outcome_with_*` vs `outcome_without_intervention` |

**Conclusão honesta:** o módulo cumpre os protocolos duck-typed já cobertos por `test_counterfactual_contract` e as heurísticas internas são coerentes consigo mesmas (CF4–CF6), mas **não integra o planner** (CF3) e o "raciocínio contrafactual" permanece lexical (diferença por zip de caracteres; situação original descartada no parse; template `{situation} causes {intervention}` não é intervenção `do(X)` de Pearl). Nada foi "corrigido" nesta rodada além de medir.

### Lacunas e opções de correção (literatura; nenhuma aplicada)

1. **Integrar o planner (CF3)** — em `explore_what_if` / `generate_counterfactual_scenario`, se `self.planner` expõe `plan`, usá-lo para propor intervenções ou avaliar sequências (Pearl 2009 + planejamento; cf. "counterfactual planning"). Custo médio; exige contrato do planner.
2. **Intervenção `do(X)` real** — em vez de `"situation causes intervention"`, mapear a intervenção para `do(variável=valor)` sobre o SCM/world model já usado em C4 (Pearl, *Causality*, 2009; Woodward, 2003). Custo médio-alto.
3. **Preservar a situação no parse** — "What if we had patched *during the outage*?" deveria carregar contexto; hoje vira sempre `current_situation`. Opção: spaCy/regex leve de adjunctos; ou API explícita `(situation, intervention)`. Custo baixo se API explícita.
4. **Diferença semântica de outcomes** — substituir zip de caracteres por overlap de tokens / embedding leve (stdlib: token Jaccard). Evita `difference≈1` para strings longas quase iguais no prefixo. Custo baixo.
5. **Plausibilidade informada** — hoje `0.7 - 0.05*|chain|`; calibrar com frequência de co-ocorrência ou comprimento mínimo de cadeia causal real (Roese & Olson, 1995). Custo médio.

Ordem sugerida (custo/benefício): 3 (API explícita) → 4 (Jaccard) → 1 (planner) → 2 (do(X)) → 5.
