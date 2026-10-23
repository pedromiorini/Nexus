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

## CF gap 3 — API explícita (situation, intervention) — critérios pré-registrados

**Objetivo:** a situação não é descartada quando o chamador a informa explicitamente.
**Não escopo:** CF3 (planner), Pearl `do(X)`, calibração de plausibilidade, Jaccard (gap 4).

### Critério CF7 (v1)

| id | afirmação | protocolo falsificável | alvo |
|---|---|---|---|
| CF7 | API explícita preserva `situation` | `explore_what_if(situation=S, intervention=I)` (sem `question`) produz `scenario.original_situation == S` e `scenario.intervention == I` para o conjunto E abaixo; a API antiga `explore_what_if(question)` permanece compatível (mesmos parses de W / CF4) | 4/4 no conjunto E **e** CF4 ainda ≥ 5/5 |

### Conjunto E (situation, intervention)

| situation | intervention |
|---|---|
| during the outage | we had patched |
| on host db-1 | isolate the host |
| system vulnerable | apply patch |
| alarm firing | rollback |

Pré-registro commitado **antes** da implementação e da execução do eval.

### Resultados CF7 (após pré-registro 624ce27)

| id | valor | alvo | passou | nota |
|---|---|---|---|---|
| CF7 | 4/4 | 4/4 + CF4≥5/5 | **sim** | `explore_what_if(situation=, intervention=)` preserva S; string API intacta (CF4 5/5) |
| CF1–CF2, CF4–CF6 | (regressão) | pass | **sim** | sem regressão |
| CF3 | falso | — | não | fora de escopo (não corrigido) |

**Antes:** `explore_what_if` só aceitava `question: str`; parse forçava `situation="current_situation"`.
**Depois:** kwargs explícitos; chamador controla a situação.

## CF gap 4 — diferença Jaccard de tokens — critérios pré-registrados

**Objetivo:** substituir diff por `zip` de caracteres por `1 − Jaccard` sobre tokens alfanuméricos (stdlib `re`), para outcomes lexicalmente aparentados.
**Não escopo:** CF3, Pearl `do(X)`, calibração de plausibilidade.

### Critério CF8 (v1)

| id | afirmação | protocolo falsificável | alvo |
|---|---|---|---|
| CF8 | diferença = 1 − Jaccard(tokens alnum) | (a) `""`/`""` e iguais → 0.0; (b) no conjunto J, valor exato = `1 - \|A∩B\|/\|A∪B\|` com tokens `re.findall(r"[a-z0-9]+", s.lower())`; (c) para o par de outcomes padrão `outcome_without_intervention` vs `outcome_with_apply_patch`, diferença **estritamente menor que 1.0** (compartilham o token `outcome`) | 3/3 subchecagens |

### Conjunto J (pares → diferença Jaccard esperada)

| actual | counterfactual | esperado |
|---|---|---|
| the cat sat on the mat | the cat sat on a mat | 1 − 5/6 ≈ 0.1667 |
| outcome without intervention | outcome with apply patch | 1 − 1/6 ≈ 0.8333 |
| alpha beta | gamma delta | 1.0 |

Pré-registro commitado **antes** da implementação e da execução do eval.

### Resultados CF8 (após pré-registro b538702)

| id | valor | alvo | passou | nota |
|---|---|---|---|---|
| CF8 | 3/3 | 3/3 | **sim** | `1 − Jaccard` tokens alnum; outcomes padrão compartilham `outcome` (diff≈0.833 < 1) |
| CF1/2/4/5/6/7 | (regressão) | pass | **sim** | sem regressão |
| CF3 | falso | — | não | fora de escopo |

**Antes (zip):** `outcome_without_intervention` vs `outcome_with_apply_patch` → ≈0.571 (prefixo comum de chars; ignora tokens).
**Depois (Jaccard alnum):** ≈0.833; para "the cat sat on the mat" vs "… a mat" → ≈0.167 (antes zip ≈0.318).
Golden SHA re-medido: `fbfb32e9466bf161155a7546e2db07e2f73434230d581217a44fd4b324eb646e`.

## CF gap CF3 / planner — critérios pré-registrados

**Objetivo:** o `planner` injetado (em especial `RealMCTSPlanner`) passa a ser usado em `generate_counterfactual_scenario`. `explore_what_if` entra no mesmo caminho porque delega a esse método.
**Não escopo:** Pearl `do(X)` completo, calibração de plausibilidade, alterar a fórmula de plausibilidade, Jaccard, a API explícita CF7, state docs, escopo Axis, `core/constitutional_brain.py`.

### Observação herdada (rodada CF v1; este commit não reexecuta o eval)

Após o pré-registro `b3e7c04`, CF3 ficou falso: `self.planner` só é atribuído em `__init__`. Os critérios abaixo foram escritos **antes** de nova execução e **antes** de implementar.

### Hipóteses concorrentes (Strong Inference)

| id | hipótese | o que a falsifica | teste barato |
|---|---|---|---|
| H1 | CF3 falha porque o AST não vê `self.planner` fora de `__init__`; o planner já seria chamado por via dinâmica | no base `fe57c82`, um stub com `plan` registra 0 chamadas em `generate_counterfactual_scenario` / `explore_what_if` | contrato CF3b no código atual |
| H2 | qualquer leitura de `self.planner` fora de `__init__` cumpre o objetivo | CF3 (AST) passa e o stub ainda registra 0 chamadas a `plan` | CF3 versus CF3b no mesmo commit |
| H3 | integrar `plan` muda a heurística quando `planner is None` | CF1, CF2, CF4, CF5, CF6, CF7, CF8 ou o golden `fbfb32e9466bf161155a7546e2db07e2f73434230d581217a44fd4b324eb646e` mudam com `planner=None` | `cognitivo_eval` + extração |
| H4 | `RealMCTSPlanner.plan(goal, available_actions=None, max_iterations=200)` é incompatível com o what-if | `RealMCTSPlanner(max_depth=1)` injetado não incrementa `total_plans`, ou levanta, para `situation`/`intervention` `str` | CF3b com o planner real |
| H5 | desenho rejeitado antes de codar: trocar `intervention` pela primeira ação do plano | quebraria a preservação de `intervention` (CF7) assim que um planner fosse injetado | não implementar; CF3b exige `intervention` intacta |

Hipótese de trabalho (não vinculante; só depois deste pré-registro): se `self.planner` expõe `plan` chamável, `generate_counterfactual_scenario` chama `plan(situation, available_actions=[intervention], max_iterations=8)`. Se `best_action_sequence` for `list` ou `tuple` de `str`, anexa `planner: {ação}` ao fim de `causal_chain`. `planner is None`, `plan` ausente ou `plan` não chamável (`bool`, `int` negativo, `NaN`, `±inf`) mantêm a cadeia atual. A fórmula de plausibilidade não muda; o comprimento da cadeia só cresce quando há passos `str` do planner. Isto não é `do(X)`.

### Critério CF3 (já existente, v1)

| id | afirmação | protocolo falsificável | alvo |
|---|---|---|---|
| CF3 | `self.planner` referenciado fora de `__init__` | AST do avaliador em `tools/cognitivo_eval.py` (`Attribute` `planner` sobre `self` em método ≠ `__init__`) | verdadeiro |

### Critério CF3b (novo, comportamental)

Caminho documentado **P**: `generate_counterfactual_scenario(situation, intervention)`. `explore_what_if` (string ou kwargs) entra em P por delegação.

| id | afirmação | protocolo falsificável | alvo |
|---|---|---|---|
| CF3b-call | stub ou `RealMCTSPlanner` com `plan` chamável é consultado em P | 1 chamada por `generate_counterfactual_scenario` ou por `explore_what_if`; `goal==situation` (ou `current_situation` no parse); `available_actions==[intervention]`; `max_iterations==8`. No planner real, `total_plans` sobe 1 e não há exceção para pares `str` | 1 chamada |
| CF3b-use | a sequência devolvida entra na cadeia | cada item `str` de `best_action_sequence` (`list`/`tuple`, ou dict com essa chave) aparece como `planner: {ação}`; uma `str` solta não é fatiada em caracteres; item não-`str` é ignorado; sequência vazia ou ausente não acrescenta passo | só `str` |
| CF3b-keep | sem planner útil, o comportamento atual fica | `planner is None` ou `plan` não chamável: cadeia heurística de 5 passos sem extrator causal; outcomes heurísticos; `original_situation` e `intervention` preservados; CF1/2/4/5/6/7/8 passam | sem regressão |
| CF3b-scope | o planner não vaza para o resto da API | `model_causal_intervention`, `compare_outcomes` (cenário já pronto) e `_predict_outcome` não chamam `plan` | 0 chamadas |
| CF3b-invalid | entrada inválida do contrato de `plan` não é normalizada | `situation is None` com `RealMCTSPlanner` propaga `ValueError`; `plan` igual a `True`, `-1`, `float("nan")`, `float("inf")` ou `float("-inf")` não é chamado | exceção ou fallback |
| CF3b-empty | string vazia não vira situação default | `situation=""` e `intervention=""` ainda chamam `plan("", available_actions=[""], max_iterations=8)`; `planner is None` não chama | chamada com vazio |

Pares de exemplo (não substituem W, S, E, J): `("system vulnerable", "isolate")`, `("during the outage", "we had patched")`, `"What if we isolate the host?"` → `("current_situation", "we isolate the host")`.

O inteiro `8` é orçamento de sonda, distinto do default `200` de `RealMCTSPlanner.plan` e distinto de calibração de plausibilidade.

### Regressão exigida

`python tools/cognitivo_eval.py --capability counterfactual --criteria v1` — CF1, CF2, CF4, CF5, CF6, CF7 e CF8 continuam passando; CF3 passa a passar. Golden de extração permanece `fbfb32e9466bf161155a7546e2db07e2f73434230d581217a44fd4b324eb646e` enquanto `planner is None` (H3).

Pré-registro commitado **antes** da implementação e **antes** de rodar o eval nesta rodada: `a4f591e122ccb34646e7dbb344c1f1dea3591cdc`. Resultados abaixo são de execução posterior.

### Resultados CF3 / CF3b (após o pré-registro)

Ambiente local: Python 3.12.3, venv com `requirements-ci.lock` (ruff 0.17.1, hypothesis 6.170.5, coverage 7.16.2, bandit 1.9.4). O CI usa Python 3.11; esta máquina não o executou.

| id | valor | alvo | passou | nota |
|---|---|---|---|---|
| CF3 | verdadeiro | verdadeiro | **sim** | `self.planner` lido em `_planner_steps` |
| CF3b | 13/13 no contrato | chamada + uso + fallback | **sim** | ver comando abaixo |
| CF1, CF2, CF4, CF5, CF6, CF7, CF8 | pass | pass | **sim** | `cognitivo_eval` rc 0, `failed: []` |

**RED (antes da correção, commit `85c0280`):** `python -m unittest test_counterfactual_planner_contract.py` → 11 falhas, rc 1. Dois controles que já descreviam o comportamento sem planner passaram (`test_absent_or_noncallable_plan_keeps_heuristic`, `test_other_methods_do_not_call_plan`).

**GREEN:** o mesmo unittest, mais os contratos CF de extração/gap3/gap4/eval, passou (28 testes no conjunto focado). `python tools/cognitivo_eval.py --capability counterfactual --criteria v1` → rc 0. Golden de extração permaneceu `fbfb32e9466bf161155a7546e2db07e2f73434230d581217a44fd4b324eb646e` (o teste de extração passou com `planner is None`).

**Implementação:** `generate_counterfactual_scenario` chama `_planner_steps`. Se `plan` é chamável, a chamada é `plan(situation, available_actions=[intervention], max_iterations=8)`. Itens `str` de `best_action_sequence` entram no fim da cadeia como `planner: {ação}`. A intervenção do chamador não é trocada. Isto não é `do(X)`.

**Hipóteses:** H1 (planner já era chamado por via dinâmica) ficou excluída pelo RED (0 chamadas). H3 (heurística muda com `planner is None`) ficou excluída pelo eval e pelo golden. H4 (contrato de `RealMCTSPlanner.plan` incompatível com pares `str`) ficou excluída pelo teste que vê `total_plans == 1`. H5 não foi implementada. H2 (basta uma leitura morta de `self.planner`) não é a correção entregue: o mutante `cf_planner_never_consulted` morre no contrato.

**Mutação:** `python tools/cognitivo_mutation.py` → `{"declared": 92, "killed": 92, "survived": 0, "not_applied": 0}`, rc 0. `python tools/targeted_mutation.py` → 69/69 mortos, rc 0.

**Suíte:** `python -m unittest discover -p 'test_*.py'` → 530 testes, rc 0. Cobertura `core/cognitivo/*` 92% (piso 90). `ruff check` nos alvos do workflow, rc 0. `reality_audit.py` sem diff depois do commit de recontagem (counterfactual.py 409 linhas, 15 funções). Bandit: 131 LOW, 0 MEDIUM/HIGH; rc 1 do Bandit é o exit conhecido por LOW; `security_triage.py --check` rc 0. `validar_registro.py --expirados aviso` rc 0. `predeleg.sh HEAD origin/main` → `OK: sem conflito`.

**Fora do que este comando mede:** orçamento 8 não avalia qualidade do MCTS; a fórmula de plausibilidade não mudou, mas uma sequência não vazia alonga a cadeia e portanto reduz o número pela fórmula já existente.

PR draft: https://github.com/pedromiorini/Nexus/pull/105


### Varredura §7.2 (2026-10-23, `gh pr list --state open`)

Nenhum PR aberto altera `core/cognitivo/counterfactual.py`, `tools/cognitivo_mutation.py` ou este arquivo. O #79 (não editado) mexe no comentário de `docs/agentes/CLAIMS.yaml` e esvazia `claims` na branch dele. O claim obsoleto do #104, já mesclado, é substituído pelo claim desta tarefa a partir da `main` `fe57c82`, sem usar o #79 como base. State docs, Axis, #38–#50 e #60 ficam de fora.

## CF gap do(X) — critérios pré-registrados (CF9)

**Objetivo:** quando `world_model` ou o grafo causal expõem intervenção, `generate_counterfactual_scenario` deixa de responder só com o prefixo `outcome_with_` e com o template lexical `"{situation} causes {intervention}"`. A intervenção mapeada consulta `simulate` / `do` / `_do` (Pearl, *Causality*, 2009; o mesmo recorte simbólico já usado em C4). Sem esse protocolo, a heurística de CF1/CF2/CF4–CF8 e o planner de CF3 permanecem.
**Não escopo:** calibração de plausibilidade (a fórmula `0.7 - 0.05*|chain|` não muda), Axis, state docs congelados (`CONTINUATION.md`, `NEXUS_EVIDENCE.md`, `NEXUS_KNOWLEDGE.md`, `SECURITY_TRIAGE.md`), `core/constitutional_brain.py`, `tools/targeted_mutation.py`, PRs #38–#50 / #60 / #79, abdução, ruído exógeno e twin network completo. O factual (`intervention is None`) continua `outcome_without_intervention` quando não há `predict_outcome` chamável — limitação desta rodada, não um Y e um Y_x ambos lidos do SCM.
**Eval:** `tools/cognitivo_eval.py` não ganha id novo. O contrato `test_counterfactual_eval_contract.py` fixa CF1–CF8. CF9 é medido pelo contrato novo. Regressão exigida: `python tools/cognitivo_eval.py --capability counterfactual --criteria v1` com CF1–CF8 passando, inclusive CF3.

### Observação (base `629541f`, antes de código e antes de reexecutar o eval)

Comando (só leitura), `PYTHONPATH=.` e Python 3.12.3:

`RealCausalReasoning.extract_causal_relations("DEPLOY_FIX causes system_status secure")` grava a aresta `DEPLOY_FIX → {system_status secure}`. `RealWorldModelSimulator._do("DEPLOY_FIX")` devolve `{system_status: secure}` e `simulate(["DEPLOY_FIX"]).final_state.state["system_status"]` fica `secure`, sem warning. O mesmo par, passado a `RealCounterfactualReasoningEngine(world_model=..., causal_reasoning=...)`, produz `counterfactual_outcome == "outcome_with_DEPLOY_FIX"`, `actual_outcome == "outcome_without_intervention"` e `causal_chain == ["host exposed causes DEPLOY_FIX"]`. Um spy com `simulate` e sem `predict_outcome`, intervenção `APPLY_PATCH`, registra `calls == []` e outcome `outcome_with_APPLY_PATCH`. `apply patch` no motor com extrator cai em `outcome_with_apply_patch` e na cadeia lexical `system vulnerable causes apply patch`. Nenhum dos dois caminhos chama `_do`.

### Protocolo duck-typed (o que CF9 chama de capaz de do(X))

Precedência em `_predict_outcome`: se `world_model.predict_outcome` é chamável, ele ganha (CF1) e `simulate` / `do` / `_do` não são chamados.

Senão, uma intervenção `str` não vazia depois de `strip` mapeia para no máximo uma ação, nesta ordem, ou para nenhuma:

1. Normalizar: `strip`, trocar espaços e hífens por `_`, colapsar `_` repetido, `upper`, tirar `_` das pontas. Se o resultado é chave de `ACTION_EFFECTS`, a ação é essa chave.
2. Senão, se a string crua (`strip`) é chave de adjacência, a ação é essa chave.
3. Senão, se o normalizado é chave de adjacência, a ação é essa chave.
4. Senão, se exatamente uma chave de adjacência, em forma `upper` depois do mesmo normalizador, iguala o normalizado, a ação é a chave original (não a forma `upper`). Duas chaves ou mais: não mapeia.
5. Senão, tokens `[a-z0-9]+` em `intervention.lower()`. Se exatamente uma chave de `ACTION_EFFECTS` tem o mesmo conjunto de tokens, a ação é essa chave (`apply patch` e `patch apply` → `APPLY_PATCH`). Zero ou duas ou mais: não mapeia.

`patch` sozinho não mapeia (`{patch}` ≠ `{apply, patch}`). `isolate the system` não mapeia. Não-`str` (`True`, `False`, `int`, `float` inclusive NaN e ±inf) não mapeia e não é coerção. `False` e `""` continuam no ramo falsy já existente (`outcome_without_intervention`). `True`, `-1`, NaN e ±inf continuam a levantar `AttributeError` no `.replace` da heurística, sem chamada a `simulate`/`do`/`_do`.

Adjacência lida, sem copiar aresta nova: `causal.causal_graph["adjacency"]` e, se for outro objeto, `world_model.causal_reasoning.causal_graph["adjacency"]`. A leitura dos efeitos da cadeia ocorre **antes** de `extract_causal_relations("{situation} causes {intervention}")`, para a aresta criada pelo template não contar como SCM prévio. O template continua a ser chamado quando o extrator existe (CF2).

Efeito de aresta: o mesmo corte de C4 (`str(effect).strip().split(None, 1)` com dois pedaços). Pares `(chave, valor)` ordenados. Passo `do({ação}): {chave}={valor}`.

Outcome contrafactual, só se houve ação mapeada e não houve `predict_outcome` chamável:

1. Se `simulate` é chamável: uma chamada `simulate([ação])`. Se `final_state.state` é `dict`, o outcome é `state:` + pares `k=v` com chaves ordenadas, unidos por vírgula sem espaço. Ex.: estado C4 pós-`DEPLOY_FIX` → `state:alert_level=high,security_patch_applied=False,system_status=secure`.
2. Senão, se `do` é chamável: uma chamada `do(ação)`.
3. Senão, se `_do` é chamável: uma chamada `_do(ação)`.
4. Nos ganchos `do`/`_do`, dict de efeitos vazio ou não-dict → heurística `outcome_with_`. Dict não vazio é mesclado por cima de `current_state.state` quando esse estado é dict (cópia; o estado do modelo não é obrigatoriamente mutado aqui).
5. `simulate` e `do`/`_do` não são chamados juntos. Exceção de `simulate`/`do`/`_do` propaga; o cenário não é gravado.
6. Sem ação, sem gancho chamável, ou `simulate` sem `dict`: heurística atual. Gancho igual a `True` ou outro não-chamável não é chamado.

Cadeia (`_build_causal_chain`):

- `M_do`: um passo por par de efeito da ação mapeada, no formato acima, antes dos passos lexicais.
- Se o extrator devolve relações, elas continuam `{cause} causes {effect}` **depois** dos passos `do(`.
- Se não há efeitos e não há relações: os 5 passos heurísticos, e nenhum passo começa com `do(`.
- Métrica falsificável: se `|E| > 0` efeitos prévios da ação, a cadeia contém cada `do({ação}): {chave}={valor}` e **não** é igual a `["{situation} causes {intervention}"]`. Se `|E| == 0`, nenhum passo começa com `do(`.

`explore_what_if` entra no mesmo caminho porque delega a `generate_counterfactual_scenario`. A string `intervention` do chamador não é trocada pela ação mapeada (CF7). Passos `planner: …` (CF3), quando existem, continuam no fim da cadeia. `model_causal_intervention`, `compare_outcomes` e `_predict_outcome(situation, None)` não chamam `simulate`/`do`/`_do`.

### Hipóteses concorrentes (Strong Inference)

| id | hipótese | o que a falsifica | teste barato |
|---|---|---|---|
| H1 | o template lexical já reflete a aresta SCM; só o prefixo `outcome_with_` está desligado de `_do` | na base `629541f`, com a aresta `DEPLOY_FIX → system_status secure` já gravada, a cadeia é só `host exposed causes DEPLOY_FIX` | sonda desta seção (já observada) e CF9-chain |
| H2 | toda string, inclusive vazia e não mapeada, deve chamar `simulate` | `""`, `patch`, `hello` ou `isolate the system` disparam `simulate` | CF9-fallback |
| H3 | `predict_outcome` pode ser substituído por `do(X)` quando os dois existem | CF1 deixa de valer, ou um spy vê `simulate` chamado | CF9-precedence |
| H4 | substring basta (`patch` → `APPLY_PATCH`) | `patch` com world model capaz chama `simulate(["APPLY_PATCH"])` | CF9-fallback |
| H5 | ligar `do(X)` altera a heurística sem modelo e derruba CF1–CF8 ou CF3 | `cognitivo_eval --criteria v1` falha, ou sem modelo o outcome deixa de ser `outcome_with_*` / a cadeia deixa de ter 5 passos | CF9-regression |

Hipótese de trabalho (não vinculante; só depois deste pré-registro): implementar o protocolo acima em `core/cognitivo/counterfactual.py`, sem editar `causal_world_model.py`. O factual permanece heurístico. Isto não é abdução nem identifiability.

### Critérios CF9 (v1)

| id | afirmação | protocolo falsificável | alvo |
|---|---|---|---|
| CF9-outcome | intervenção mapeada muda o outcome contrafactual via o modelo | spy com `simulate` e sem `predict_outcome`: `generate_counterfactual_scenario("system vulnerable", "APPLY_PATCH")` chama `simulate` uma vez com `["APPLY_PATCH"]`; `counterfactual_outcome == "state:marker=from-model,system_status=secure"`; não começa com `outcome_with_`; `actual_outcome == "outcome_without_intervention"`; `intervention` permanece `APPLY_PATCH`. `apply patch` e `patch apply` chamam `simulate(["APPLY_PATCH"])` no `RealWorldModelSimulator` e o outcome é `state:alert_level=low,security_patch_applied=True,system_status=secure`. `DEPLOY_FIX` com a aresta C4 já extraída: uma `simulate(["DEPLOY_FIX"])` no simulador real e outcome `state:alert_level=high,security_patch_applied=False,system_status=secure` (o valor `secure` não está em `ACTION_EFFECTS` para essa chave; some se `_do` for ignorado). Gancho só `do` ou só `_do` (sem `simulate`) é chamado uma vez e o dict de efeitos entra na serialização | 1 chamada e serialização |
| CF9-chain | a cadeia deixa de ser só o template quando há efeitos prévios | com a aresta C4, `_build_causal_chain("host exposed", "DEPLOY_FIX")` e a cadeia do cenário são `["do(DEPLOY_FIX): system_status=secure", "host exposed causes DEPLOY_FIX"]`, com ou sem `world_model`. `|E|==0` (grafo vazio, intervenção `APPLY_PATCH`) não produz passo `do(`. Chave original `deploy_fix` com intervenção `DEPLOY_FIX` usa `do(deploy_fix): …` e `simulate(["deploy_fix"])` | igualdade da lista |
| CF9-fallback | sem protocolo útil, a heurística fica | sem `world_model` e sem aresta: `outcome_with_patch`, cadeia de 5 passos, CF2 `["alarm causes patch"]`. Com spy capaz: `patch`, `hello`, `isolate the system`, `"   "` e `""` não chamam `simulate`; `""` e `False` → `outcome_without_intervention`; string não vazia não mapeada → `outcome_with_` + espaços virados `_` | 0 chamadas |
| CF9-precedence | `predict_outcome` ganha de `simulate` | objeto com os dois: `_predict_outcome("sit","act") == "wm:sit:act"` e `generate` não chama `simulate` | 0 chamadas a simulate |
| CF9-invalid | não-str e gancho inválido não são normalizados | `True`, `-1`, `nan`, `inf`, `-inf` levantam `AttributeError` e não chamam `simulate`. Atributo `simulate=True` não é chamado; outcome heurístico. `simulate` que levanta `ValueError` propaga e não incrementa `total_scenarios_generated`. `model_causal_intervention` e `compare_outcomes` não chamam `simulate` | exceção ou 0 chamadas |
| CF9-keep | planner e plausibilidade não são redefinidos | com planner e spy, uma chamada `plan` (CF3b) e uma `simulate`; o passo `planner:` fica depois de `do(` / lexical. `plausibility_score == max(0.1, 0.7 - 0.05*len(chain))` | fórmula intacta |
| CF9-regression | CF1–CF8 não regridem | `python tools/cognitivo_eval.py --capability counterfactual --criteria v1` → rc 0 e `failed: []` | 8/8 |

Conjunto D (mapeia): `APPLY_PATCH`, `apply patch`, `patch apply`, `apply_patch`, `DEPLOY_FIX` com aresta. Conjunto N (não mapeia): `patch`, `isolate the system`, `hello`, `""`, `"   "`.

### Regressão exigida

`python tools/cognitivo_eval.py --capability counterfactual --criteria v1` — CF1, CF2, CF3, CF4, CF5, CF6, CF7 e CF8 continuam passando. Golden de extração com `planner is None` e sem world model permanece o da rodada CF3 enquanto a heurística sem modelo não mudar (H5).

Pré-registro escrito **antes** da implementação e **antes** de rodar o eval nesta rodada. O SHA deste commit entra no PR. Resultados, se houver, só numa seção posterior.

### Resultados CF9 (após o pré-registro `c187c12`)

Ambiente local: Python 3.12.3, venv com `requirements-ci.lock` (ruff 0.17.1, hypothesis 6.170.5, coverage 7.16.2, bandit 1.9.4). O CI usa Python 3.11; esta máquina não o executou. Números abaixo são desta execução, não um carimbo de CI hospedado.

| id | valor | alvo | passou | nota |
|---|---|---|---|---|
| CF9-outcome | serialização do spy e do simulador real | 1 chamada e `state:` | **sim** | `APPLY_PATCH` / `apply patch` / `patch apply`; `DEPLOY_FIX` fica `system_status=secure` via `simulate` (C4 `_do` por dentro) |
| CF9-chain | lista com `do(DEPLOY_FIX): system_status=secure` antes do template | igualdade | **sim** | também sem `world_model`, se a aresta já existe; `|E|==0` não inventa `do(` |
| CF9-fallback | 0 chamadas em N; heurística sem modelo | 0 chamadas | **sim** | `patch` não vira `APPLY_PATCH` |
| CF9-precedence | `wm:sit:act` e `simulate` parado | 0 chamadas | **sim** | CF1 intacto |
| CF9-invalid | `AttributeError` em não-str; `ValueError` de `simulate` propaga | exceção | **sim** | cenário não é gravado |
| CF9-keep | `plan` com a string original; fórmula `max(0.1, 0.7-0.05*len)` | fórmula | **sim** | não recalibrada |
| CF9-regression | CF1–CF8 | `failed: []` | **sim** | comando abaixo |

**RED** (commit `312c751`, código ainda o de `629541f`): `python -m unittest test_counterfactual_dox_contract.py` → 15 falhas, rc 1. Sete métodos de controle passaram.

**GREEN:** `0184a72` (código, mutantes e o caso extra de `simulate` sem `dict`, cláusula já escrita no pré-registro). `python -m unittest test_counterfactual_dox_contract.py` → 20 testes, rc 0. O RED tinha 19 métodos; o vigésimo não entra nas 15 falhas.

**Eval:** `python tools/cognitivo_eval.py --capability counterfactual --criteria v1` → rc 0, `failed: []`. O contrato de extração que fixa o golden `fbfb32e9466bf161155a7546e2db07e2f73434230d581217a44fd4b324eb646e` passou na suíte (não reexecutei o hash à parte).

**Implementação:** `_predict_outcome` continua a honrar `predict_outcome` chamável. Senão, se a intervenção mapeia, chama `simulate([ação])` ou, na falta dele, `do` e só então `_do`. A cadeia lê efeitos da adjacência antes do extrator. `causal_world_model.py` não foi editado. O factual sem `predict_outcome` continua `outcome_without_intervention`.

**Hipóteses:** H1 ficou excluída pela sonda anterior ao código (cadeia só lexical com a aresta já gravada). H2, H3 e H4 não se sustentam nos controles que passaram. H5 (regressão de CF1–CF8) não se sustentou neste eval. Nenhuma delas vira alegação de capacidade causal geral.

**Mutação:** `python tools/cognitivo_mutation.py` → `{"declared": 97, "killed": 97, "survived": 0, "not_applied": 0}`, rc 0. Inclui `cf_do_never_consulted`, `cf_scm_edges_ignored`, `cf_do_swallows_simulate_error`, `cf_token_set_match_dropped` e `cf_do_hook_prefers_private`. `python tools/targeted_mutation.py` → 69/69 mortos, rc 0. O catálogo do Axis não foi editado.

**Suíte:** `python -m unittest discover -p 'test_*.py'` → 550 testes, rc 0. Cobertura `core/cognitivo/*` 94% (piso 90); `counterfactual.py` 94% (8 statements sem execução). `core/constitutional_brain.py` 68% (piso 62), com `NEXUS_INTEGRATED_SUMMARY=/tmp/integrated.json`. `ruff check` nos alvos do workflow, rc 0. `reality_audit.py`: counterfactual.py 524 linhas, 21 funções; soma `core/cognitivo` 1976 linhas; núcleo permanece 26071 / 252 / 813. Bandit: 131 LOW, 0 MEDIUM/HIGH; `security_triage.py --check` rc 0. `validar_registro.py --expirados aviso` rc 0. `predeleg.sh HEAD origin/main` → `OK: sem conflito`.

**Fora do que estes comandos medem:** o factual não é lido do SCM; não há abdução nem twin network; a fórmula de plausibilidade não foi calibrada; uma cadeia mais longa só muda o número pela fórmula antiga. CI hospedado e Python 3.11 não rodaram. Conflito de merge com os PRs que também versionam o relatório de realidade não foi resolvido.

PR draft: https://github.com/pedromiorini/Nexus/pull/106

### Varredura §7.2 desta rodada (2026-10-23, `gh pr list --state open`, 16 PRs)

Nenhum PR aberto altera `core/cognitivo/counterfactual.py`, `core/cognitivo/causal_world_model.py`, `core/cognitivo/action_effects.py`, `tools/cognitivo_mutation.py`, `tools/cognitivo_eval.py` ou este arquivo. O #79 (draft, não usado como base) altera `docs/agentes/CLAIMS.yaml`. `AUDIT_REALITY_REPORT.md` / `audit_reality.json` aparecem em #38, #39, #40, #41, #42, #44, #46, #48 e #49; se o gate de realidade exigir recontagem depois da edição de `counterfactual.py`, o diff fica só nessa recontagem, sem o código desses PRs. State docs, Axis, `constitutional_brain` e #60 ficam de fora. O claim do #105 (já mesclado) é substituído pelo claim grok-bot desta tarefa, prazo `2026-10-26T03:00:00Z`.

## CF gap plausibilidade — critérios pré-registrados (CF10–CF13)

**Objetivo:** `_calculate_plausibility` deixa de ser só `max(0.1, 0.7 - 0.05*|chain|)`. O comprimento da chain continua na conta, e um termo de atestação no grafo causal injetado muda o número quando há aresta real. Heurística explícita, não probabilidade e não réplica de Roese e Olson (1995): o paper está citado na lacuna 5 como motivo para não usar só o comprimento; a operacionalização desta rodada é a fração de passos da cadeia que coincidem com uma aresta já gravada.
**Não escopo:** twin network, abdução de Pearl, Axis, state docs congelados (`CONTINUATION.md`, `NEXUS_EVIDENCE.md`, `NEXUS_KNOWLEDGE.md`, `SECURITY_TRIAGE.md`), `core/constitutional_brain.py`, `tools/targeted_mutation.py`, `tools/cognitivo_eval.py` (CF5 v1 permanece o critério do avaliador), PRs #38–#50 / #60 / #79. Nenhum script novo: o harness `tools/cognitivo_eval.py` já recusa capability vazia com rc 2, distinto de 0 e de 1 (`test_cognitivo_eval_cli_contract.py`).
**CF5:** o alvo v1 não muda. Monotonia estrita em comprimentos (1, 3, 5, 10), com a mesma semântica de comprimento (`len(causal_chain)`), no motor sem grafo que o avaliador já usa. Não há CF5v2. Se o eval falhar depois da implementação, o defeito é a fórmula vazar evidência no grafo vazio; o alvo não será reescrito para acomodar a falha.

### Observação (base `29a5711`, antes de código novo e antes de reexecutar o eval)

Comando (só leitura), `PYTHONPATH=.`, Python 3.12.3, `RealCounterfactualReasoningEngine` sem argumentos e com um grafo stub `alarm → {patch}` e `APPLY_PATCH → {system_status secure}`:

| chamada | score impresso |
|---|---|
| `_calculate_plausibility("x", ["a"]*n)` para n = 1, 3, 5, 10 | 0.65, 0.55, 0.45, 0.20 (monótono; resíduos binários de `0.05`) |
| cadeia `["noise"]*5`, texto `do(APPLY_PATCH): system_status=secure` ×5, ou `alarm causes patch` ×5, com ou sem o grafo | o mesmo 0.45 |
| intervenção `""` ou `"   "` com cadeia de 5 | 0.45 |
| `generate_counterfactual_scenario("alarm", "patch")` com `RealCausalReasoning` | cadeia `["alarm causes patch"]`, score 0.65 |
| `generate_counterfactual_scenario("system vulnerable", "apply patch")` sem causal | 5 passos heurísticos, score 0.45 |

O grafo não entra na conta. Duas cadeias de comprimento 5, uma com aresta e outra sem, recebem o mesmo número. Intervenção vazia não abaixa o score.

### Fórmula congelada (heurística; constantes de calibração, não evidência de capacidade)

`length_score(n) = max(0.1, 0.7 - 0.05*n)`.

1. `causal_chain` tem de ser `list` de `str`. Caso contrário `TypeError` (não iterar `str`/`tuple` como cadeia; não aceitar item `bool`/`int`/`float`).
2. Intervenção `False`, ou `str` cujo `strip` é vazio (`""`, `"   "`): retornar `0.1`, mesmo se a cadeia estaria atestada. `False` acompanha o sentinela já coberto por CF9 (`outcome_without_intervention`); não entra como magnitude. `True`, `int` (inclusive `-1`), `float` (`nan`, `±inf`), `None` e qualquer outro não-`str`: `TypeError`. A string `"-1"` ou `"nan"` é intervenção não vazia e segue a fórmula.
3. Passo atestado, só com o mesmo `_adjacency()` de CF9 (aresta prévia; o template lexical desta chamada só conta se o extrator já a gravou antes da plausibilidade, que é a ordem atual de `generate_counterfactual_scenario`):
   - `do(AÇÃO): chave=valor` casa o passo inteiro, e algum efeito de `adjacency[AÇÃO]` parte por `str(effect).strip().split(None, 1)` em exatamente `(chave, valor)`;
   - `CAUSA causes EFEITO` casa o passo inteiro, e `str(item) == EFEITO` para algum item de `adjacency[CAUSA]`.
   - Valor de adjacência que seja `str` ou `bytes` não é conjunto de efeitos (não iterar caracteres) e não conta como aresta.
4. `f = (passos atestados) / n` quando `n > 0`. Se há passo atestado: `min(1.0, length_score(n) + 0.2*f)`. Se não há passo atestado e o grafo tem ao menos uma aresta (container `set`/`list`/`tuple`/`frozenset` não vazio): `max(0.1, length_score(n) - 0.2)`. Senão: `length_score(n)`.
5. Comparações nos testes: `math.isclose(..., abs_tol=1e-9)`. Os literais abaixo são os valores decimais dessa conta.

Passos `planner: …`, `Immediate effect` e o resto do template de 5 passos não casam o formato. Sem grafo, não ganham bônus: o comprimento efetivo depois de filtrar esses passos **não** substitui `len(causal_chain)` (hipótese H2, rejeitada antes de codar).

### Hipóteses concorrentes (Strong Inference)

| id | hipótese | o que a falsifica | teste barato |
|---|---|---|---|
| H1 | a fórmula fixa já separa cadeia SCM de cadeia heurística; nada a fazer | na base `29a5711`, comprimento 5 com aresta e comprimento 5 sem aresta dão o mesmo 0.45 | sonda desta seção (já observada) e CF10 |
| H2 | o alvo certo é o comprimento depois de descartar "Immediate effect" / "Secondary effect" / "Final outcome", e CF5 vira CF5v2 | seria preciso mudar o alvo do eval, que mede `len(["a"]*n)` | não implementar; CF5 v1 permanece |
| H3 | basta o texto `do(` ou `causes`, sem aresta no grafo | cadeia `alarm causes patch` ×5 ou `do(APPLY_PATCH): system_status=secure` ×5, grafo vazio, score ≠ `length_score(5)` | CF10-text |
| H4 | intervenção vazia deve continuar em `length_score` | `""` / `"   "` com n=5 dão 0.45 | CF11 |
| H5 | o termo de atestação derruba CF1–CF8 ou a monotonia de CF5 no motor sem grafo | `cognitivo_eval --criteria v1` falha, em particular CF5 | CF5 no contrato e o eval |

Hipótese de trabalho (não vinculante para além do que a tabela de critérios já fixa): implementar os itens 1–5 em `_calculate_plausibility`, sem editar `causal_world_model.py` nem o avaliador.

### Critérios

| id | afirmação | protocolo falsificável | alvo |
|---|---|---|---|
| CF5 | monotonia v1 intacta, mesmos valores da fórmula de comprimento quando não há grafo | motor sem grafo, intervenção `"x"`, cadeias `["a"]*n` para n em (1, 3, 5, 10): scores `isclose` de 0.65, 0.55, 0.45, 0.20 e cada um estritamente menor que o anterior. n=2 continua `isclose` de 0.6 (o assert já existente em CF9) | monótono e esses literais |
| CF10-length | heurística de 5 passos sem grafo não muda | `generate_counterfactual_scenario("system vulnerable", "apply patch")` sem causal: cadeia de 5, score `isclose` de 0.45 | igual à fórmula antiga |
| CF10-scm | aresta SCM, mesmo comprimento, sobe | grafo `APPLY_PATCH → {system_status secure}`, intervenção `"APPLY_PATCH"`, cadeia de 5 cópias de `do(APPLY_PATCH): system_status=secure`: score `isclose` de 0.65, e não `isclose` de 0.45 | 0.65 |
| CF10-rel | relação extraída no grafo, mesmo comprimento, sobe | grafo `alarm → {patch}`, intervenção `"patch"`, `alarm causes patch` ×5: score `isclose` de 0.65 | 0.65 |
| CF10-freq | a fração entra na conta | o mesmo grafo, cadeia com 1 passo `alarm causes patch` e 4 `"noise"`: score `isclose` de 0.49 (0.45+0.2×0.2), não `isclose` de 0.65 nem de 0.45 | 0.49 |
| CF10-text | texto sem aresta não é evidência | sem grafo, as duas cadeias de CF10-scm e CF10-rel: score `isclose` de 0.45 | 0.45 |
| CF10-live | o cenário gravado usa a conta nova | com `RealCausalReasoning` e sem world model: `("alarm","patch")` → cadeia `["alarm causes patch"]` e score `isclose` de 0.85 (não 0.65). Com a aresta C4 já extraída (`DEPLOY_FIX causes system_status secure`): cadeia `["do(DEPLOY_FIX): system_status=secure", "host exposed causes DEPLOY_FIX"]` e score `isclose` de 0.8 (não 0.6) | 0.85 e 0.8 |
| CF11 | intervenção vazia ou sem texto → piso | `""`, `"   "` e `False`, cadeia de 5, com ou sem o grafo de CF10-rel: score `isclose` de 0.1, estritamente menor que 0.45. `generate(..., "")`, `generate(..., "   ")` e `generate(..., False)` gravam cenário com esse piso; o outcome de `""` e de `False` continua `outcome_without_intervention` | 0.1 |
| CF12 | grafo com aresta e cadeia sem passo atestado → abaixo da fórmula de comprimento | grafo de CF10-rel, intervenção `"patch"`, `["noise"]*5`: score `isclose` de 0.25. Cadeia `[]` com o mesmo grafo: `isclose` de 0.5. Sem grafo, cadeia `[]` e intervenção `"x"`: `isclose` de 0.7. Valor `str` em vez de conjunto (`{"alarm": "patch"}`) não é aresta: score de `["noise"]*5` continua `isclose` de 0.45 | 0.25 / 0.5 / 0.7 |
| CF13 | tipo inválido não vira número | `TypeError` em `_calculate_plausibility` para intervenção `True`, `1`, `-1`, `nan`, `inf`, `-inf`, `None`; cadeia `"abc"`, `("a",)`, `None`, `True`, `["a", 1]`, `["a", True]`, `["a", nan]`. A string `"-1"` com n=1 e sem grafo permanece `isclose` de 0.65 | exceção ou 0.65 |

Ordem exigida no contraste de comprimento 5, intervenção não vazia: score sem suporte (0.25) < score sem grafo (0.45) < score com os 5 passos atestados (0.65).

O golden de extração inclui `plausibility_score` de `generate("alarm","patch")` com `RealCausalReasoning`. Esse campo, e `stats2["avg_plausibility"]`, passam de 0.65 a 0.85. O SHA será medido de novo **depois** da implementação e registrado na seção de resultados; não é alvo deste pré-registro. Os demais campos do payload (cenário sem grafo, Jaccard, parse) não são alvo de mudança.

### Regressão exigida

`python tools/cognitivo_eval.py --capability counterfactual --criteria v1` — CF1, CF2, CF3, CF4, CF5, CF6, CF7 e CF8 continuam passando. CF5 conforme a linha CF5 acima (monotonia v1; o eval não exige os literais, o contrato novo exige). Os asserts de CF9 que comparam a fórmula de comprimento em motor **sem** aresta (`test_plausibility_formula_not_retuned`, planner+spy sem grafo) continuam válidos e não serão relaxados.

### Varredura §7.2 (2026-10-23, `gh pr list --state open`)

Nenhum PR aberto altera `core/cognitivo/counterfactual.py`, `test_counterfactual_extraction_contract.py`, `tools/cognitivo_mutation.py`, `tools/cognitivo_eval.py` ou este arquivo. O #79 (draft, não usado como base) altera `docs/agentes/CLAIMS.yaml`; o claim do #106, já mesclado, é substituído por este, nos arquivos desta tarefa, sem editar `versao_aceita` de outro agente. `AUDIT_REALITY_REPORT.md` / `audit_reality.json` aparecem em #38, #39, #40, #41, #42, #44, #46, #48 e #49; recontagem, se o gate exigir, fica só nos números deste módulo. State docs, Axis, `constitutional_brain` e #60 ficam de fora.

Pré-registro escrito **antes** da implementação, **antes** do contrato e **antes** de rodar o eval nesta rodada. O SHA deste commit entra no PR. Resultados só numa seção posterior.

### Resultados CF10 (após o pré-registro `c77e57f`)

Ambiente local: Python 3.12.3, venv com `requirements-ci.lock` (ruff 0.17.1, hypothesis 6.170.5, coverage 7.16.2, bandit 1.9.4). O CI usa Python 3.11; esta máquina não o executou. Números abaixo são desta execução, não um carimbo de CI hospedado.

| id | valor | alvo | passou | nota |
|---|---|---|---|---|
| CF5 | 0.65, 0.55, 0.45, 0.20 e monótono | v1, sem grafo | **sim** | eval `values` e o contrato; alvo não foi reescrito |
| CF10-length | 0.45 na heurística de 5 passos | igual à fórmula antiga | **sim** | sem grafo |
| CF10-scm / CF10-rel | 0.65 em cadeia de 5 atestada | acima de 0.45 | **sim** | mesmo comprimento, número diferente |
| CF10-freq | 0.49 com 1/5 passos | fração, não bônus cheio | **sim** | |
| CF10-text | 0.45 sem grafo | texto não é evidência | **sim** | `do(` e `causes` sozinhos |
| CF10-live | 0.85 e 0.8 | não 0.65 e não 0.6 | **sim** | relação extraída; aresta C4 + template |
| CF11 | 0.1 | piso | **sim** | `""`, `"   "`, `False`, com ou sem grafo |
| CF12 | 0.25 / 0.5 / 0.7 | abaixo / vazio / sem grafo | **sim** | `str` no lugar do conjunto não é aresta |
| CF13 | `TypeError` | não vira número | **sim** | string `"-1"` permanece 0.65 |
| CF1–CF4, CF6–CF8 | pass | pass | **sim** | `failed: []`, inclusive CF3 |

**RED** (commit `51b8e07`, código ainda o de `29a5711`): `python -m unittest test_counterfactual_plausibility_contract.py` → rc 1, 14 testes, 26 falhas de subteste. Quatro controles sem grafo passaram (CF5 nos literais, heurística de 5 passos, texto sem aresta, propriedade de grafo vazio).

**GREEN:** `eed5139` mais a recontagem de realidade deste commit. O mesmo unittest, rc 0. Golden re-medido `64488affedd89003faa5bc18de333bff4c52cb1e126a379e6f376b00cdd2c509`. O payload contra a fórmula antiga difere só em `sc2.plausibility_score` e `stats2.avg_plausibility` (0.65 → 0.85).

**Eval:** `python tools/cognitivo_eval.py --capability counterfactual --criteria v1` → rc 0, `failed: []`. CF5 `values`: 0.6499999999999999, 0.5499999999999999, 0.44999999999999996, 0.19999999999999996.

**Implementação:** comprimento `max(0.1, 0.7-0.05*n)` mais fração de passos atestados por `_adjacency()` (`+0.2*f`), penalidade `0.2` se o grafo tem aresta e nenhum passo casa, piso `0.1` se a intervenção é `False` ou `str` vazia depois de `strip`. `causal_world_model.py` e `tools/cognitivo_eval.py` não foram editados.

**Hipóteses:** H1 ficou excluída pela sonda anterior ao código (comprimento 5 com e sem aresta = 0.45) e pelo RED. H2 não foi implementada; não há CF5v2. H3 não se sustenta em CF10-text. H4 não se sustenta em CF11. H5 não se sustentou neste eval. Nenhuma delas vira alegação de capacidade causal geral.

**Efeito colateral observado** (não era critério): `compare_outcomes` acrescenta `"Scenario is highly plausible"` quando o score passa de 0.7. Sonda à parte, `PYTHONPATH=.`: cadeia `alarm causes patch` (0.85) inclui essa frase; a heurística de 5 passos (0.45) não. A suíte de cobertura não executou o statement 495.

**Mutação:** `python tools/cognitivo_mutation.py` → `{"declared": 105, "killed": 105, "survived": 0, "not_applied": 0}`, rc 0. Inclui os oito mutantes novos de plausibilidade (fórmula antiga sempre, vazio na fórmula de comprimento, texto `do(` ou `causes` sem grafo, bônus sem fração, sem penalidade, atestação zerada, aresta `str` iterada). `python tools/targeted_mutation.py` → 69/69 mortos, rc 0. O catálogo do Axis não foi editado.

**Suíte:** `coverage run --branch -m unittest discover -p 'test_*.py'` → 564 testes, rc 0. Cobertura `core/cognitivo/*` 95% (piso 90); `counterfactual.py` 95% (8 statements sem execução). `core/constitutional_brain.py` 68% (piso 62), com `NEXUS_INTEGRATED_SUMMARY=/tmp/integrated.json`, rc 0 do script. Pisos de fila 95%, VRAM 94%, temporal 93%, vita 76%. `ruff check` nos alvos do workflow, rc 0. `reality_audit.py`: counterfactual.py 581 linhas, 26 funções; soma `core/cognitivo` 2033 linhas; núcleo permanece 26071 / 252 / 813. Bandit: 131 LOW, 0 MEDIUM/HIGH; `security_triage.py --check` rc 0. `validar_registro.py --expirados aviso` rc 0. `predeleg.sh HEAD origin/main` → `OK: sem conflito`.

**Fora do que estes comandos medem:** a fração é atestação na cadeia, não frequência num corpus; não há abdução nem twin network; o factual não é lido do SCM; passo `planner:` não é atestado; CI hospedado e Python 3.11 não rodaram. Nenhum script novo: capability vazia do eval continua no contrato já existente (rc 2). Conflito de merge com os PRs que também versionam o relatório de realidade não foi resolvido contra as branches deles.

PR draft: https://github.com/pedromiorini/Nexus/pull/107

---

## Working Memory — critérios pré-registrados (WM1–WM6)

**Objetivo desta rodada:** extrair, em move-only, o bloco POST6 (`WorkingMemoryItem`, `RehearsalEvent`, `PhonologicalLoop`, `VisuospatialSketchpad`, `CentralExecutive`, `RealWorkingMemorySystem`) de `core/constitutional_brain.py` para `core/cognitivo/working_memory.py`, com reexport strangler no ponto original e em `core/cognitivo/__init__.py`. Medir WM1–WM6. Se todos passam, documentar e parar. Se algum falha no monólito atual, registrar o gap. Não corrigir comportamento depois de ver o número.
**Não escopo:** injetar relógio, Attention, IncrementalLearner, HierarchicalMemory/FAISS, Metacognition (#42), Axis, state docs (`CONTINUATION.md`, `NEXUS_EVIDENCE.md`, `NEXUS_KNOWLEDGE.md`, `SECURITY_TRIAGE.md`), `tools/targeted_mutation.py`, PRs congelados #38–#50 / #60 / #79, twin network. Bool/`NaN` no clamp ficam como estão: o golden os registra; esta rodada não os rejeita.
**Eval:** `python tools/cognitivo_eval.py --capability working_memory`. O protocolo abaixo é único; `--criteria` v1/v2/v3 não o altera. Seeds não se aplicam: `item_id` é `uuid.uuid4()` e os instantes vêm de `time.time()`. O golden não inclui ids nem timestamps crus.
**Base da observação:** `b6da175` (merge do #107). Sonda só de leitura, `PYTHONPATH=.`, Python 3.12.3, antes deste texto e antes de qualquer harness `working_memory`.

### Varredura §7.2 (2026-10-23, `gh pr list --state open`)

PRs abertos que editam `core/constitutional_brain.py`: #38, #39, #40, #41, #42, #44, #46, #48, #49. São os congelados do pedido; não são a base. A extração precisa do reexport no monólito (§16). Isso tensiona a fila única da §11.2: o conflito com esses PRs fica declarado, sem rebase neles.
Nenhum PR aberto altera `test_working_memory_contract.py`, `tools/cognitivo_eval.py`, `tools/cognitivo_mutation.py`, `core/cognitivo/__init__.py` ou este arquivo. O #79 (draft, não usado como base) altera `docs/agentes/CLAIMS.yaml`. `AUDIT_REALITY_REPORT.md` / `audit_reality.json` aparecem em #38, #39, #40, #41, #42, #44, #46, #48 e #49; se o gate exigir recontagem depois do move, o diff fica só nos números deste módulo. O claim do #107, já mesclado em `b6da175` e ainda listado, é substituído pelo claim grok-bot desta tarefa, prazo `2026-10-26T04:00:00Z`.

### Observação (código e sonda, antes do move e antes do eval)

**Fato, lido no bloco ~L5412–L5827:** não há parâmetro de relógio. `PhonologicalLoop.decay_items` e `rehearse` chamam `time.time()`. `store` remove `min(buffer, key=last_rehearsed)` quando `len >= capacity`. `store_visual` / `store_spatial` removem `min(buffer, key=activation)`. O filtro de decay é `activation > 0.1` (igual a 0,1 sai). `rehearse` faz `min(1.0, activation + 0.3)` e reescreve `last_rehearsed`. `retrieve` é `str(query).lower() in str(content).lower()`, primeiro hit, senão `None`. `allocate_attention` faz `max(0.0, min(1.0, float(strength)))`.

**Fato, sonda no monólito** (atribuir `last_rehearsed` / `activation` / `entered_at` já públicos; isso não é API nova):

| sonda | valor |
|---|---|
| 7 `store`, `last_rehearsed` = 1..7, 8º store | len 7; sai o de `last_rehearsed` 1 |
| empate de `last_rehearsed` | `min` estável: sai o primeiro do buffer |
| 4 `store_visual`, activations 0.9, 0.2, 0.8, 0.4, 5º store | len 4; sai o 0.2 |
| empate de activation 1.0 | sai o primeiro |
| activation 1.0, `last_rehearsed = time.time()-0.4`, mais um com `last_rehearsed=0`, mais um com activation 0.1 | depois de `decay_items` só o primeiro permanece, activation impressa 0.8 em 4 casas; os outros saem |
| `rehearse` com activation 0.4 | 0.7; `last_rehearsed` atualizado; id ausente → `False`; 0.9 → 1.0; 1.0 → 1.0 |
| gêmeo com `last_rehearsed=0` não ensaiado | `decay_items` o remove; o ensaiado fica |
| `retrieve("alpha"/"BETA"/"alp")` em `"Alpha Beta"` | o item; `"zzz"` → `None`; `""` → o primeiro (substring vazia) |
| clamp `float` | 2.0→1.0, −1→0.0, 0.5→0.5, `True`→1.0, `False`→0.0, `inf`→1.0, `-inf`→0.0, `nan`→1.0 (`min(1.0, nan)` fica 1.0 porque comparação com NaN é falsa) |
| `"abc"` / `None` / `[]` / `{}` | `ValueError` / `TypeError` / `TypeError` / `TypeError` |
| `switch_task` da tarefa corrente / outra | 0.0 / 0.2 |
| `prioritize_item` 0.5 com prioridade 1, depois 10 | 0.8, depois 1.0 |
| `refresh` 0.5 / 0.9 / id ausente | 0.7 / 1.0 / `False` |
| `maintain_items` com 4 itens activation 1.0 e `last_rehearsed=time.time()` | `rehearsal_count` 3; `rehearse_verbal` de id ausente não incrementa |
| `integrate_multimodal` cheio, vítima = menor `entered_at` (não a menor activation) | len 4; o mais antigo sai |
| `store_spatial` cheio | vítima = menor activation, não o menor `entered_at` |
| `transfer_to_ltm` com stub / sem ltm | uma chamada `store(content, "long_term", importance=activation)` e contador 1 / contador 0 |
| `consciousness.attend_to` | `("working_memory", "Stored in WM: " + content[:50] + "...", "cognitive", salience=0.5)` |
| `retrieve_from_wm` com activation 0.5 | 0.7; miss → `None` |
| `get_active_items` | ordem decrescente de activation |
| estatísticas do sistema vazio | contadores 0 e usos 0.0; o `__init__` imprime a linha `POST 6` |

O 0.8 da sonda de decay **não** é limite. É uma leitura de wall-clock. O critério WM3 não exige igualdade.

### Hipóteses concorrentes (Strong Inference)

| id | hipótese | o que a falsifica | teste barato |
|---|---|---|---|
| H1 | o estouro fonológico remove o menor `last_rehearsed` (empate: o primeiro do buffer), não `entered_at` | no protocolo WM1 o item de `last_rehearsed` mínimo permanece, ou o segundo do empate sai | WM1 |
| H2 | o estouro visual/espacial remove a menor activation (empate: o primeiro), não o mais antigo | no protocolo WM2 o item de activation 0.2 permanece | WM2 |
| H3 | já existe relógio injetável e a activation depois de Δt=0,4 s é exatamente 0,8 | a assinatura de `decay_items` / `__init__` não tem relógio, ou a activation observada não é `== 0.8` | leitura da assinatura (já feita) e WM3 |
| H4 | `rehearse` não altera a sobrevivência; os dois gêmeos caem juntos | o ensaiado sai, ou o não ensaiado fica, ou 0.4 não vai a 0.7, ou 0.9 passa de 1.0 | WM4 |
| H5 | `retrieve` é igualdade de token, ou distingue maiúsculas | `"alp"` ou `"BETA"` não acham `"Alpha Beta"`, ou `"zzz"` devolve item | WM5 |
| H6 | mover o bloco muda o clamp ou o restante do fingerprint | SHA do protocolo WM6 ≠ o valor abaixo, ou as classes reexportadas não são o mesmo objeto | WM6 |

Hipótese de trabalho (não vinculante para além da tabela): move-only, sem injetar relógio e sem alterar fórmulas. Se H3 no sentido "exatamente 0,8" falhar, isso é o gap de relógio já previsto, não um convite a mudar o código.

### Critérios

| id | afirmação | protocolo falsificável | alvo |
|---|---|---|---|
| WM1 | `PhonologicalLoop(capacity=7)`: 8 `store` distintos deixam `len(buffer)==7`; a vítima é o menor `last_rehearsed` | Depois de 7 `store`, atribuir `last_rehearsed` a 1..7 e dar o 8º `store`. O item de valor 1 sai; o 8º fica; len 7. Empate: os 7 com `last_rehearsed=5.0`; sai o primeiro; o segundo fica | len 7 e essa vítima |
| WM2 | `VisuospatialSketchpad(capacity=4)`: 5 `store_visual` deixam len 4; a vítima é a menor activation | 4 stores, activations 0.9, 0.2, 0.8, 0.4, 5º store: sai o 0.2. Empate com activation 1.0: sai o primeiro | len 4 e essa vítima |
| WM3 | sem relógio injetável, medir o que é determinístico: decay reduz activation; item com activation ≤ 0.1 sai | (1) `inspect.signature`: `decay_items` só tem `self` e `__init__` não tem parâmetro de relógio — registrar `clock: "wall"`, sem falhar o id por isso. (2) Três itens no mesmo loop: A activation 1.0 e `last_rehearsed=time.time()-0.4`; B activation 1.0 e `last_rehearsed=0.0`; C activation 0.1 e `last_rehearsed=time.time()`. Depois de `decay_items`: B ausente, C ausente, A presente, `A.activation < 1.0` e `A.activation > 0.1`. Não exigir `== 0.8`. Se o processo parar por mais de ~1 s entre a atribuição e o decay, A pode cair a ≤ 0.1: isso é o gap de wall-clock, registrado, e **não** autoriza injetar relógio nesta rodada | (2) como escrito; (1) só documenta |
| WM4 | `rehearse(id)` sobe activation com teto 1.0, atualiza `last_rehearsed`; o ensaiado sobrevive a um decay que remove o gêmeo | Dois itens, activation 0.4, `last_rehearsed=0`. `rehearse` do primeiro → `True`, activation 0.7, `last_rehearsed` ≥ instante lido imediatamente antes da chamada. Id ausente → `False`. Outro loop: 0.9 → 1.0; 1.0 → 1.0. `decay_items` no par: o ensaiado fica, o gêmeo sai | 0.7, teto 1.0, sobrevivência |
| WM5 | `retrieve(query)` é substring sem distinguir maiúsculas; miss → `None` | `store("Alpha Beta")` depois `store("other")`. `retrieve("alpha")`, `retrieve("BETA")` e `retrieve("alp")` devolvem o primeiro. `retrieve("zzz")` é `None` | esses quatro resultados |
| WM6 | move-only: reexports públicos são o mesmo objeto; fingerprint do comportamento atual, inclusive o clamp, é idêntico | Classes `WorkingMemoryItem`, `RehearsalEvent`, `PhonologicalLoop`, `VisuospatialSketchpad`, `CentralExecutive`, `RealWorkingMemorySystem`: `core.constitutional_brain` e `core.cognitivo` são o mesmo objeto que `core.cognitivo.working_memory`, e `__module__` é `core.cognitivo.working_memory`. O módulo importa sem puxar o monólito, numpy ou vita. O dict do protocolo abaixo, serializado com `json.dumps(..., sort_keys=True, ensure_ascii=False)`, tem SHA-256 `3b6b63cb69d6ffced16ffec0c1bbbbeb8fe92bf7cecd13e20f4989b0d17051ca`, igual pela classe do monólito e pela do módulo novo | SHA e identidade |

**Protocolo do fingerprint WM6** (o que entra no dict; stdout do banner desviado para não misturar com o hash, mas `banner_printed` é verdadeiro se o texto capturado contém `POST 6`):

- `clamp`: `repr(s) → strength` para s em `(2.0, -1.0, 0.5, 0.0, 1.0, 1, True, False, inf, -inf, nan)`, alvo `"planning"`.
- Em seguida, nesse mesmo executivo (tarefa corrente `"planning"`): `switch_same = switch_task("planning")`, `switch_other = switch_task("other")`.
- `plain`: `allocate_attention("planning", 2.0)` sem attention → `{target: planning, strength: 1.0, delegated: false}`.
- `delegated`: com `attention_system=object()`, `("monitoring", -1.0)` → strength 0.0 e `delegated: true`.
- `errors`: `"abc"` → `ValueError`; `None`, `[]`, `{}` → `TypeError`.
- `prioritize`: item activation 0.5, prioridade 1.0 → 0.8; o mesmo item, prioridade 10.0 → 1.0. Lista `[0.8, 1.0]`.
- `refresh`: visual 0.5 → 0.7; 0.9 → 1.0; id ausente → `false`. Lista `[0.7, 1.0, false]`.
- Esboço espacial capacity 4, pares (activation, entered_at) `(0.9, 1.0), (0.1, 4.0), (0.8, 2.0), (0.5, 3.0)`, depois mais um `store_spatial`: `spatial_victim_is_low_activation` verdadeiro e `spatial_oldest_remains` verdadeiro.
- `PhonologicalLoop`: `store("Alpha Beta")`, `store("other")`, `retrieve("")` → `"Alpha Beta"` (`retrieve_empty`). Isso não é WM5; é o comportamento atual da substring vazia, preso no golden para o move não "corrigir" em silêncio.
- Sistema vazio: `stats0` com as nove chaves atuais (`total_items_stored`, `rehearsal_count`, `items_transferred_to_ltm`, três `current_*`, três `*_capacity_usage`) em zero. Quatro `store_verbal`, activation 1.0, `last_rehearsed=time.time()` imediatamente antes de `maintain_items`: `maintain_count==3`. `rehearse_verbal("missing")` deixa o contador em 3.
- `store_spatial({"x": 1})` → `spatial_modality=="spatial"`.
- Quatro `integrate_multimodal`, depois `(entered_at, activation)` = `(1.0, 0.9), (2.0, 0.8), (3.0, 0.4), (4.0, 0.2)`, quinto integrate: len 4, o menor `entered_at` sai, o de activation 0.2 fica.
- LTM stub: `transfer_to_ltm` de `"memo"` → `[["memo", "long_term", 1.0]]` e contador 1. Sem LTM, contador 0.
- Consciência stub: um `attend_to` com args e `salience` 0.5, texto `Stored in WM: ` + os primeiros 50 caracteres + `...`.
- Outro sistema: verbal activation 0.2 conteúdo `"low"`, visual activation 0.9 conteúdo `"high"`: `get_active_items` → `["high", "low"]`.
- `retrieve_from_wm("boostme")` com activation 0.5 → 0.7; `"zzz"` → miss.
- `decay_rate` 0.5; `capacities` `[7, 4, 4]`.

`retrieve("")` e a coerção de `bool`/`nan` fazem parte do golden de identidade. Não são metas de qualidade e não serão "corrigidas" nesta rodada.

### Regressão exigida

O contrato que já existe (`allocate_attention` 2.0 → strength 1.0 e `delegated` false; −1.0 com attention externo → strength 0.0 e `delegated` true) continua válido, sem relaxar asserts. O eval dos outros capabilities não é alvo de mudança de critério.

Pré-registro escrito **antes** do move, **antes** do contrato novo de extração e **antes** de rodar `tools/cognitivo_eval.py --capability working_memory`. O SHA deste commit entra no PR. Resultados só numa seção posterior.

### Resultados WM (após o pré-registro `dcfa180`)

Ambiente local: Python 3.12.3, venv com `requirements-ci.lock` (ruff 0.17.1, coverage 7.16.2, bandit 1.9.4). O CI usa Python 3.11; esta máquina não o executou. Os números abaixo são desta execução, não um carimbo de CI hospedado.

**Commits.** Pré-registro `dcfa1804eb41f184503cfedfd6ff947b14d143b3` (só `CLAIMS.yaml` e esta auditoria). Move-only `bf6b6747e01a7d506e000b377b7305a98509fb9f`. Base `b6da175`.

**RED**, antes de existir `core/cognitivo/working_memory.py`: `python -m unittest test_working_memory_contract.py` → rc 1, 7 testes, 4 `ModuleNotFoundError`. Os dois testes antigos de clamp e o teste de tipo inválido passaram no monólito.

**GREEN**, mesmo comando depois do move: 7 testes, rc 0.

**Eval** `python tools/cognitivo_eval.py --capability working_memory` (o padrão `--criteria v2` não altera o protocolo): rc 0, `failed: []`.

| id | valor medido | alvo | passou |
|---|---|---|---|
| WM1 | len 7; vítima = menor `last_rehearsed`; empate sai o primeiro | len 7 e essa vítima | sim |
| WM2 | len 4; vítima = activation 0.2; empate sai o primeiro | len 4 e essa vítima | sim |
| WM3 | `clock: wall`; activation restante 0.7999968528747559; época e fronteira 0.1 saem | presente, `< 1` e `> 0.1`; sem exigir `== 0.8` | sim |
| WM4 | boost 0.7 lido antes de `decay_items`; teto 1.0; id ausente `false`; ensaiado fica; gêmeo sai | 0.7, teto, sobrevivência | sim |
| WM5 | `"alpha"` / `"BETA"` / `"alp"` → `"Alpha Beta"`; `"zzz"` → `null` | esses quatro | sim |
| WM6 | SHA do módulo = SHA do monólito = `3b6b63cb69d6ffced16ffec0c1bbbbeb8fe92bf7cecd13e20f4989b0d17051ca`; identidade verdadeira | SHA pré-registrado e mesmo objeto | sim |

O 0.7999968528747559 é a leitura desta reexecução. Na medição do commit de move o valor foi 0.7999963760375977. Os dois ficam em `(0.1, 1)`. Nenhum dos dois é limite.

**Relógio.** `decay_items` continua só com `self`. Não há parâmetro de relógio em `__init__`. O gap de wall-clock fica registrado: se o processo parar por ~1 s, o item A pode cair a ≤ 0.1 e WM3 falha sem que a fórmula tenha mudado. Relógio não foi injetado.

**Harness, não o módulo.** A primeira leitura de WM4 depois de `decay_items` deu ~0.699998 por causa do mesmo wall-clock. O critério pede 0.7 no instante do `rehearse`. O eval passou a gravar o boost antes do decay. O código de produção não mudou. `retrieve("")` continua devolvendo o primeiro item, e `True`/`False`/`nan` continuam 1.0/0.0/1.0 no golden. Não são correções.

**Standalone do WM6.** Uma varredura por substring no texto do módulo falhou porque o docstring cita `constitutional_brain`. O critério pré-registrado fala de import, não de menção. O eval passou a ler a árvore de `import` / `import from` (numpy, vita, `core.constitutional_brain`). O bloqueio em tempo de execução, com `sys.meta_path`, permanece em `test_working_memory_contract.py`. Bandit não varre `test_*.py`.

**Por que o eval não usa subprocesso.** No commit de move, `_wm_standalone` chamava `subprocess`. Bandit nessa árvore: 133 LOW (B404 e B603 em `tools/cognitivo_eval.py`); `security_triage.py --check` rc 1. `SECURITY_TRIAGE.md` é state doc e ficou fora de escopo. Depois da troca para AST: `bandit -r core vita tools` → 131 LOW, zero achados em `cognitivo_eval.py` e em `working_memory.py`; `--check` rc 0. `working_memory.py` não ganhou achado em nenhuma das duas contagens.

**Hipóteses.** H1, H2, H4, H5 e H6 não se sustentam como falha neste eval: o comportamento medido é o da tabela de observação, e o SHA do fingerprint é o que foi congelado no monólito antes do splice. H3, no sentido "já existe relógio injetável e a activation é exatamente 0,8", fica excluída pela assinatura e por não exigir igualdade. Nenhuma delas vira alegação de memória de trabalho humana nem de prontidão.

**Cobertura**, no commit de move (`coverage run --branch -m unittest discover -p 'test_*.py'`, depois o demo do núcleo): 569 testes, rc 0, 56.346 s. Demo `python core/constitutional_brain.py` rc 0. `working_memory.py` 161 statements, 30 ramos, 100%. `core/cognitivo/*` 1118 statements, 95% (piso 90). `constitutional_brain.py` 68% (piso 62). Fila 95%, VRAM 94%, temporal 93%, vita 76%. A edição posterior do eval não toca esses arquivos; os pisos não foram reexecutados depois dela.

**Mutação.** `python tools/targeted_mutation.py` no commit de move: 69/69 mortos, 100%, rc 0. O catálogo do Axis não foi editado. `python tools/cognitivo_mutation.py` nesse mesmo commit, e de novo depois da troca do standalone para AST: `{"declared": 129, "killed": 129, "survived": 0, "not_applied": 0}`, rc 0. Os 24 mutantes do módulo novo estão nesse total. `wm_module_pulls_monolith` insere `import core.constitutional_brain`; o AST do eval e o contrato em subprocesso o matam.

**Realidade** (recontagem exigida pelo gate, commitada com o move): núcleo 25664 linhas, 246 classes, 788 funções (antes 26071 / 252 / 813). `working_memory.py` 438 linhas, 6 classes, 25 funções. `core/cognitivo` 8 módulos, 2489 linhas. Literais dinâmicos 52. Frases fortes 51. Terminologia científica 233. Prefixo `Real` 63. Testes embutidos 31. `pass` puro 0. Retorno constante 2. `reality_audit` reexecutado depois do commit: diff vazio.

**Outros gates locais no commit de move, e o que foi repetido depois:** `ruff check` nos alvos do workflow, rc 0; repetido no eval editado, rc 0. `validar_registro.py --expirados aviso`, rc 0. `predeleg.sh HEAD origin/main`, `OK: sem conflito` (o fetch trouxe `origin/main` até `b6da175`, que já é a base).

**Fora do que estes comandos medem:** CI hospedado; Python 3.11; conflito de merge contra as branches dos PRs #38, #39, #40, #41, #42, #44, #46, #48, #49 e #79 (o `predeleg` só compara com `origin/main`); state docs não atualizados, de propósito; Attention, IncrementalLearner, HierarchicalMemory/FAISS, Metacognition, Axis e twin network não foram tocados. Não há interface web nesta extração.

PR draft: https://github.com/pedromiorini/Nexus/pull/108

## Attention — critérios pré-registrados (AT1–AT6)

**Objetivo desta rodada:** extrair, em move-only, o bloco POST25 (`StimulusItem`, `AttentionalFocus`, `AttentionShift`, `BottomUpAttention`, `TopDownAttention`, `AttentionalSpotlight`, `RealAttentionSystem`) de `core/constitutional_brain.py` para `core/cognitivo/attention.py`, com reexport strangler no ponto original e em `core/cognitivo/__init__.py`. Medir AT1–AT6. Se todos passam, documentar e parar. Se algum falha no monólito atual, registrar o gap. Não corrigir comportamento depois de ver o número.
**Não escopo:** injetar relógio; corrigir gaps de comportamento (substring de goal, coerção de `bool`/`NaN`, `attend` que não recomputa prioridade, igualdade de dataclass no spotlight, ramo morto de `relevances`, clamp de saliência que os pesos atuais não atingem); `AttentionalState`; GlobalWorkspace/Consciousness; IncrementalLearner; HierarchicalMemory/FAISS; Metacognition (#42); POST49 (`AttentionRequest` / `AttentionRoute` / `AttentionAllocation` / `RealAttentionRoutingSystem`); Axis e `tools/targeted_mutation.py`; state docs (`CONTINUATION.md`, `NEXUS_EVIDENCE.md`, `NEXUS_KNOWLEDGE.md`, `SECURITY_TRIAGE.md`); PRs congelados #38–#50, #60, #77, #79 (não rebasear, não editar as branches, não resolver os conflitos deles).
**Eval:** `python tools/cognitivo_eval.py --capability attention`. O protocolo abaixo é único; `--criteria` v1/v2/v3 não o altera. Seeds não se aplicam: `stimulus_id` de `register_stimulus` e ids de foco/shift são `uuid.uuid4()`, e os instantes vêm de `time.time()`. O golden não inclui esses ids nem timestamps crus. O contrato que já existe (`test_attention_signal_contract.py`: contrast 1.4→1.0, motion −0.2→0.0, context 0.25, fallbacks 0.5 e 0.6) continua válido, sem relaxar asserts.
**Base da observação:** `4f1613020de535d64bc6a69fe3046c523b14fd98` (merge do #108). Sonda só de leitura, `PYTHONPATH=.`, Python 3.12.3, antes deste texto, antes de criar `core/cognitivo/attention.py` e antes de qualquer harness `attention`.

### Varredura §7.2 (2026-10-23, `gh pr list --state open`)

PRs abertos que editam `core/constitutional_brain.py`: #38, #39, #40, #41, #42, #44, #46, #48, #49. São os congelados do pedido; não são a base. A extração precisa do reexport no monólito (§16). Isso tensiona a fila única da §11.2: o conflito com esses PRs fica declarado, sem rebase neles.
Nenhum PR aberto altera `core/cognitivo/attention.py` (o arquivo ainda não existe), `test_attention_contract.py`, `test_attention_signal_contract.py`, `tools/cognitivo_eval.py`, `tools/cognitivo_mutation.py`, `core/cognitivo/__init__.py` ou este arquivo. O #79 (draft, não usado como base) altera `docs/agentes/CLAIMS.yaml`. `AUDIT_REALITY_REPORT.md` / `audit_reality.json` aparecem em #38, #39, #40, #41, #42, #44, #46, #48 e #49; se o gate exigir recontagem depois do move, o diff fica só nos números deste módulo. O claim da Working Memory, já mesclado em `4f16130` e ainda listado, é substituído pelo claim grok-bot desta tarefa, prazo `2026-10-26T05:00:00Z`.

### Observação (código e sonda, antes do move e antes do eval)

**Fato, lido no bloco ~L4921–L5405:** `BottomUpAttention.salience_factors` é novelty 0.3, intensity 0.25, contrast 0.25, motion 0.2. `compute_salience` devolve `min(1.0, soma ponderada)`. Novidade: 0.9 se `stimulus_id` não está em `context["seen_stimuli"]` (ausente → lista vazia), senão 0.2. Intensidade: visual 0.7, auditory 0.6, qualquer outro tipo 0.4 (o nome "internal" não é especial; é o `else`). Contraste: se `stimulus.contrast is not None`, `min(1.0, max(0.0, valor))` sem `float()`; senão `context["contrast"]` passa por `float()` e o mesmo clamp; senão 0.5. `context["contrast"] is None` cai no fallback. Movimento: se `stimulus.motion is not None`, o mesmo clamp; senão 0.6. Não há movimento vindo do context.

`TopDownAttention.compute_goal_relevance`: sem `active_goals` → 0.3. Com goals, `max(relevance * weight)`. `set_goals` grava a lista e põe 0.5 só no goal que ainda não está em `goal_relevance`. `_assess_relevance_to_goal` faz `word in str(content).lower()` para cada palavra de `goal.lower().split()`; se `matches > 0`, `min(1.0, matches / len(words))`; senão 0.1. Goal vazio tem `split()` vazio, `matches == 0`, devolve 0.1. `update_goal_priority` é `min(1.0, max(0.0, priority))`.

`AttentionalSpotlight(capacity=3)`. `focus_on`: se `stimulus in self.spotlight` (igualdade de dataclass, campo a campo), retorna sem mudar força nem histórico. Se `len >= capacity`, remove `min(spotlight, key=salience)` — empate fica com o primeiro — e então acrescenta. `release_focus` filtra por `stimulus_id`; se o foco corrente é esse id, `duration = time.time() - start_time` e `current_focus = None`. `get_focused_items` devolve `spotlight.copy()`. `is_focused` procura o id.

`RealAttentionSystem.attend`: se `goals` é verdadeiro, só chama `set_goals`. Não recomputa `salience` nem `priority` dos estímulos já registrados. Lista vazia → `[]` e o contador de episódios não sobe. Senão `_select_for_attention` grava `attention_score = (salience + priority) / 2` e `sorted(..., reverse=True)` estável, corta em `spotlight.capacity`, e `focus_on` só nos que `is_focused` ainda recusa. `shift_attention` de id ausente devolve `False` antes de criar shift. Custo base 0.2; se há foco corrente e a modalidade difere, soma 0.3; `min(1.0, base)`. `reason == "voluntary"` incrementa `voluntary_shifts`; qualquer outra string incrementa `involuntary_shifts`. `clear_attended_stimuli` mantém `time.time() - timestamp < 60`. `__init__` imprime a linha `POST 25`. `consciousness.attend_to("attention", f"Attending to {n} stimuli", "attentional", salience=0.6)` só se `consciousness` não é `None`. Não há parâmetro de relógio em `__init__`, `clear_attended_stimuli` nem `focus_on`.

**Fato, sonda no monólito** (atribuir campos já públicos; isso não é API nova):

| sonda | valor |
|---|---|
| soma dos quatro pesos | 1.0 |
| novidade ausente / presente | 0.9 / 0.2 |
| intensidade visual / auditory / internal / tactile | 0.7 / 0.6 / 0.4 / 0.4 |
| visual, contrast 1, motion 1, id novo | saliência 0.895 |
| o mesmo id em `seen_stimuli` | 0.685 |
| contrast 1.4 / −0.2 / 0 / 1 no stimulus | 1.0 / 0.0 / 0.0 / 1.0 |
| contrast ausente; context 0.25 / `"0.25"` / `None` | 0.25 / 0.25 / 0.5 |
| stimulus 0.1 com context 0.9 | 0.1 (o stimulus vence) |
| motion ausente / 1.4 / −0.2 | 0.6 / 1.0 / 0.0 |
| contrast ou motion `str` no stimulus | `TypeError` |
| context contrast `"abc"` | `ValueError` |
| contrast/motion `True` / `False` / `nan` / `inf` / `-inf` | 1.0 / 0.0 / 0.0 / 1.0 / 0.0 |
| sem goals | 0.3 |
| goal `"find cat"` (peso 0.5), conteúdo com as duas palavras / só `"find"` / nenhuma / maiúsculas | 0.5 / 0.25 / 0.05 / 0.5 |
| goal `"cat"`, conteúdo `"category"` | 0.5 (substring, não token) |
| goal `""` | 0.05 (0.1 × peso 0.5); não é o 0.3 de lista vazia |
| goals `"alpha"` peso 1.0 e `"beta gamma"` peso 0.5 | conteúdo `"alpha"` → 1.0; `"beta only"` → 0.25 (máximo, não soma) |
| `set_goals` de novo depois de `update_goal_priority(..., 1.0)` | o 1.0 permanece |
| `update_goal_priority` 2 / −1 / 0.5 / `True` / `False` / `inf` / `-inf` / `nan` | 1.0 / 0.0 / 0.5 / 1.0 / 0.0 / 1.0 / 0.0 / 0.0 |
| `update_goal_priority` `"abc"` / `None` / `[]` / `{}` | `TypeError` |
| spotlight 0.9, 0.2, 0.8 e 4º com 0.5 | ficam o 0.9, o 0.8 e o 4º; sai o 0.2 |
| empate de saliência 0.4, 0.4, 0.9 e 4º com 0.7 | sai o primeiro 0.4 |
| `focus_on` repetido no mesmo objeto, ou noutro com os mesmos campos | histórico não cresce; strength permanece a da primeira chamada |
| `release_focus` do id de saliência baixa | esse id sai; o foco corrente continua o último |
| `release_focus` do foco corrente | `current_focus is None`, `duration is not None` e `>= 0`, `is_focused` falso; `get_focused_items()` não é o mesmo objeto da lista |
| `attend` com scores (0.2+0.2)/2, (0.9+0.9)/2, (0.4+0.8)/2, (1.0+0.0)/2 | ordem b, c, d; o a fica de fora; score de c é o float IEEE de 0.6 |
| quatro itens com score 0.5, 0.5, 0.5, 0.0 | os três primeiros na ordem original |
| `attend(goals=["zzz"])` num item já gravado com prioridade 0.1 | prioridade e saliência continuam 0.1; o goal fica gravado; episódios sobem 1 |
| `register_stimulus("visual", "a cat")` depois de `set_goals(["cat"])` | saliência é o float IEEE da fórmula com fallbacks 0.5 e 0.6 (JSON `0.6900000000000001`); prioridade 0.5 |
| `shift_attention` de id ausente | `False`; nenhum shift é acrescentado |
| primeiro shift, mesma modalidade, cross-modal | custos 0.2, 0.2, 0.5; o primeiro `from_focus` é `None`; o seguinte não é; voluntary 1, involuntary 2 no trio voluntary/goal/salience |
| shift sem foco corrente | custo 0.2 |
| `attend` sem estímulos | `[]`, episódios 0 |
| `clear` com timestamps 0, `time.time()-61` e `time.time()` | só o último permanece |
| consciência stub | uma chamada `("attention", "Attending to 1 stimuli", "attentional")`, `salience` 0.6 |
| estatísticas do sistema vazio | os sete contadores em zero e `focus_capacity` 3; o `__init__` imprime `POST 25` |
| assinatura | `clear_attended_stimuli` só tem `self`; `__init__` não tem `clock` nem `now` |

O 0.0 do `nan` **não** é um rejeite de tipo. É `max(0.0, nan)` neste CPython 3.12.3: comparação com NaN é falsa e `max` fica com o primeiro argumento; `min(1.0, 0.0)` é 0.0. `True`/`False` passam pelo mesmo clamp porque `bool` é `int`. Não serão "corrigidos" nesta rodada. Python 3.11 não foi executado.

A soma ponderada máxima com estes pesos e fatores já clampados é 0.895. `return min(1.0, salience)` não altera nenhum caso desta sonda. O `else` de `max(relevances) if relevances else 0.3` não é alcançado por uma lista normal: lista vazia já voltou 0.3 no `if not self.active_goals`. Os dois ficam registrados como estão. Não autorizam editar o bloco.

### Hipóteses concorrentes (Strong Inference)

| id | hipótese | o que a falsifica | teste barato |
|---|---|---|---|
| H1 | os pesos não somam 1, ou a novidade/intensidade não são as constantes lidas | AT1 mede outra soma, 0.9/0.2 trocados, ou tactile diferente de internal | AT1 |
| H2 | contrast/motion ignoram o clamp, o context ou os fallbacks 0.5/0.6 | AT2 ou o contrato já existente (`test_attention_signal_contract.py`) falha sem que o assert tenha sido relaxado | AT2 |
| H3 | sem goals a relevância não é 0.3, ou o match é token inteiro (logo `"cat"` não casa com `"category"`), ou `update_goal_priority` não clampa | AT3 mede outro número, `"category"` fica em 0.1×peso, ou 2.0 permanece 2.0 | AT3 |
| H4 | o estouro do spotlight remove a maior saliência, ou o empate remove o segundo | no protocolo AT4 o 0.2 permanece, ou o segundo 0.4 sai | AT4 |
| H5 | `attend(goals=)` recomputa prioridade, ou o custo cross-modal não é 0.2+0.3, ou id ausente devolve `True` | prioridade 0.1 muda ao passar goals, o custo cross-modal não é 0.5, ou o id ausente não é `False` | AT5 |
| H6 | mover o bloco muda o texto ou o fingerprint de comportamento | SHA de fonte ou SHA comportamental ≠ os valores abaixo, ou as classes reexportadas não são o mesmo objeto | AT6 |

Hipótese de trabalho (não vinculante para além da tabela): move-only, sem injetar relógio e sem alterar fórmulas. Se H5 no sentido "recomputa prioridade" falhar, isso é o gap já lido no código, não um convite a mudar `attend`.

### Critérios

| id | afirmação | protocolo falsificável | alvo |
|---|---|---|---|
| AT1 | pesos somam 1.0; novidade 0.9 se o id não está em `seen_stimuli`, 0.2 se está; intensidade visual/auditory/internal/tactile = 0.7/0.6/0.4/0.4 | Ler `salience_factors`. `compute_salience` de visual com contrast 1 e motion 1: id novo = `0.3*0.9+0.25*0.7+0.25*1+0.2*1`; o mesmo id em `seen_stimuli` troca 0.9 por 0.2. tactile usa o `else`, igual a internal | soma 1.0 e esses produtos |
| AT2 | contrast/motion no stimulus clampam [0, 1]; senão context (só contrast, via `float`); senão 0.5 / 0.6 | Stimulus 1.4→1.0 e −0.2→0.0 nos dois sinais. Context 0.25 e `"0.25"` → 0.25; context `None` → 0.5. Stimulus 0.1 vence context 0.9. Motion sem valor → 0.6. `str` no stimulus → `TypeError`; context `"abc"` → `ValueError`. Os três testes de `test_attention_signal_contract.py` permanecem. `bool`/`nan`/`inf` ficam no golden de AT6 e não são motivo para rejeitar o tipo nesta rodada | os números e as duas exceções |
| AT3 | sem goals → 0.3; match é substring sem maiúsculas, vezes o peso; `update_goal_priority` clampa [0, 1] | `"find cat"` peso 0.5: as duas palavras → 0.5, uma → 0.25, nenhuma → 0.05, `"FIND CAT"` → 0.5. `"cat"` em `"category"` → 0.5. Goal `""` → 0.05. Dois goals: máximo, não soma (`"alpha"` peso 1 → 1.0; `"beta only"` contra `"beta gamma"` peso 0.5 → 0.25). `set_goals` não apaga peso 1.0 já escrito. Clamp 2→1, −1→0, 0.5→0.5. `"abc"`, `None`, `[]`, `{}` → `TypeError` | esses números e `TypeError` |
| AT4 | capacity 3: o 4º `focus_on` remove o de menor saliência; empate = o primeiro; refoco e `release_focus` | Itens 0.9, 0.2, 0.8 e 4º 0.5: sai o 0.2. Empate 0.4/0.4/0.9 e 4º 0.7: sai o primeiro 0.4. Segundo `focus_on` do mesmo objeto, strength 0.9, não muda o 0.2 nem o tamanho do histórico. Objeto com os mesmos campos também não entra de novo. `release_focus` do menor id o tira e deixa o foco corrente. `release_focus` do corrente: `current_focus is None`, duração definida e `>= 0`, `is_focused` falso. `get_focused_items()` não é a lista interna | essa vítima, o empate e o refoco |
| AT5 | `attend` ordena por `(salience+priority)/2`, estável, top capacity; id ausente em `shift_attention` → `False`; cross-modal custa 0.5 | Quatro itens (0.2,0.2), (0.9,0.9), (0.4,0.8), (1.0,0.0): selecionados b, c, d. Empate de score 0.5: os três primeiros na ordem de registro. `attend(goals=)` não recomputa prioridade já gravada. Id ausente → `False` e a lista de shifts não cresce. Sem foco ou mesma modalidade → 0.2. Modalidade diferente → 0.5. `"voluntary"` conta em voluntary; outra razão, em involuntary. Sem estímulos, `attend` devolve `[]` e episódios continuam 0 | ordem, `False`, custos 0.2 e 0.5 |
| AT6 | move-only: o texto do bloco e o fingerprint comportamental são os mesmos; reexport é o mesmo objeto | O texto UTF-8 que começa na linha de `═` imediatamente antes de `# MÓDULO 11A3: ATTENTION & FOCUS SYSTEM (POST 25)` e termina no `}` de `get_statistics` (a chave `"focus_capacity"`) tem SHA-256 `96090684b99d2568ea61f040c5eed9bbf6de774f5bad9d781f60c2a566ea2378` no monólito, antes do splice, e o mesmo SHA no módulo novo. As sete classes: `core.constitutional_brain` e `core.cognitivo` são o mesmo objeto que `core.cognitivo.attention`, e `__module__` é `core.cognitivo.attention`. O módulo importa sem puxar o monólito, numpy ou vita (árvore de `import` / `import from`; menção no docstring não conta). O dict do protocolo abaixo, `json.dumps(..., sort_keys=True, ensure_ascii=False)`, tem SHA-256 `07877c7f811006b479729aa7faa7d6b82f824afc0d2bf2dc7514d059ce5f990b`, igual pela classe do monólito e pela do módulo novo | os dois SHA e a identidade |

**Protocolo do fingerprint comportamental de AT6** (o que entra no dict; stdout do banner desviado, mas `banner` é verdadeiro se o texto capturado contém `POST 25`):

- `weights` e `weights_sum`; `novelty_absent` 0.9 e `novelty_present` 0.2; `intensity` visual/auditory/internal/tactile.
- `salience_novel` / `salience_familiar` com contrast 1 e motion 1.
- `contrast`: fallback, stimulus 1.4, −0.2, `True`, `False`, `nan`, `inf`, `-inf`, context 0.25, `"0.25"`, `None`, `True`, `nan`, e stimulus 0.1 contra context 0.9.
- `motion`: fallback, 1.4, −0.2, `True`, `False`, `nan`, `inf`, `-inf`.
- `errors`: contrast `str` → `TypeError`; context `"abc"` → `ValueError`; motion `str` → `TypeError`.
- `no_goals` 0.3; `matches` full/one/none/case do goal `"find cat"`; `substring` de `"cat"` em `"category"`; `empty_goal` de `""`.
- `multi` beta/alpha; `priority_kept` 1.0 depois de `set_goals` de novo.
- `clamps` de 2, −1, 0.5, `True`, `False`, `inf`, `-inf`, `nan`. `priority_errors` de `"abc"`, `None`, `[]`, `{}`.
- `evicted` = `["id0", "id2", "id3"]`; `tie_spotlight` = `["id1", "id2", "id3"]`.
- `refocus`: len 1, history 1, strength 0.2, `twin_equal` verdadeiro.
- `released_low` = `["id0", "id2"]` e `current_after_release` = `"id2"`.
- `released_current`: current none, duration set, focused falso, copy verdadeiro. A duração numérica não entra (relógio de parede).
- `order` = `[["b", 0.9], ["c", 0.6000000000000001], ["d", 0.5]]`; `tie_order` = `["t0", "t1", "t2"]`.
- `no_recompute`: saliência 0.1, prioridade 0.1, score 0.1, goals `["zzz"]`, episódios 1.
- `registered`: saliência `0.6900000000000001`, prioridade 0.5.
- `empty_attend`: lista vazia, episódios 0.
- `shift`: missing falso, zero shifts depois do miss, três `True`, custos `[0.2, 0.2, 0.5]`, primeiro `from_focus` none, o seguinte preenchido, voluntary 1, involuntary 2.
- `no_focus_cost` 0.2.
- `consciousness`: args `["attention", "Attending to 1 stimuli", "attentional"]`, salience 0.6, uma chamada.
- `clear_kept` = `["live"]` para timestamps 0, `time.time()-61` e `time.time()` lido imediatamente antes. `clock` = `"wall"`.
- `stats0` com as sete chaves atuais em zero e `focus_capacity` 3. `banner` verdadeiro.

`bool`/`nan`, a substring `"category"` e o `attend` que não recomputa fazem parte do golden de identidade. Não são metas de qualidade e não serão "corrigidas" nesta rodada.

### Regressão exigida

`test_attention_signal_contract.py` continua importando `BottomUpAttention` e `StimulusItem` de `core.constitutional_brain` e os três asserts permanecem. O eval dos outros capabilities não é alvo de mudança de critério.

Pré-registro escrito **antes** do move, **antes** do contrato novo de extração e **antes** de rodar `tools/cognitivo_eval.py --capability attention`. O SHA deste commit entra no PR. Resultados só numa seção posterior.

### Resultados AT (após o pré-registro `a8323fa`)

Ambiente local: Python 3.12.3, venv com `requirements-ci.lock` (ruff 0.17.1, coverage 7.16.2, bandit 1.9.4, hypothesis 6.170.5, numpy 2.4.6). O CI usa Python 3.11; esta máquina não o executou. Os números abaixo são desta execução, não um carimbo de CI hospedado.

**Commits.** Pré-registro `a8323faf8b612503382106892db11ed6340b0a62` (só `CLAIMS.yaml` e esta auditoria). Move-only `27aad64e962a34d99d9bb9ccf4faa34b233f32be`. Base `4f1613020de535d64bc6a69fe3046c523b14fd98`.

**RED**, antes de existir `core/cognitivo/attention.py`, com o contrato de 6 testes e o eval já apontando para o módulo: `PYTHONPATH=. python -m unittest test_attention_contract.py` → rc 1, 6 testes, 3 `ModuleNotFoundError: No module named 'core.cognitivo.attention'`. Os três testes de tipo inválido (`str` em contrast/motion, context `"abc"`, prioridade `"abc"`/`None`/`[]`/`{}`) passaram no monólito.

**GREEN**, mesmo comando depois do move, já com o sétimo teste (segundo `attend`, ramo em que o item já está no spotlight): `test_attention_contract.py` + `test_attention_signal_contract.py` → 10 testes, rc 0. O contrato de sinais antigo não foi relaxado.

**Eval** `python tools/cognitivo_eval.py --capability attention --criteria v3` (v1/v2/v3 não alteram o protocolo; o padrão do argparse continua `v2`): rc 0, `failed: []`.

| id | valor medido | alvo | passou |
|---|---|---|---|
| AT1 | soma 1.0; novidade 0.9/0.2; intensidade 0.7/0.6/0.4/0.4; saliência nova 0.895; familiar 0.685 | esses produtos | sim |
| AT2 | clamp 1.4→1.0 e −0.2→0.0; context 0.25; fallbacks 0.5/0.6; `str` → `TypeError`; context `"abc"` → `ValueError` | os números e as duas exceções | sim |
| AT3 | sem goals 0.3; `"find cat"` 0.5/0.25/0.05/0.5; `"category"` 0.5; goal vazio 0.05; máximo 1.0 e 0.25; peso 1.0 preservado; clamp 2→1; tipos inválidos → `TypeError` | esses números e `TypeError` | sim |
| AT4 | sai o 0.2; empate sai o primeiro 0.4; refoco não cresce o histórico; strength permanece 0.2 | essa vítima e o refoco | sim |
| AT5 | ordem b, c, d; empate t0, t1, t2; id ausente `false` e zero shifts; custos 0.2, 0.2, 0.5; `attend(goals=)` não recomputa a prioridade 0.1; sem estímulos, episódios 0 | ordem, `False`, 0.2 e 0.5 | sim |
| AT6 | SHA de fonte do módulo = SHA medido no monólito antes do splice = `96090684b99d2568ea61f040c5eed9bbf6de774f5bad9d781f60c2a566ea2378`; SHA comportamental módulo = monólito = `07877c7f811006b479729aa7faa7d6b82f824afc0d2bf2dc7514d059ce5f990b`; identidade verdadeira; `clock: wall` | os dois SHA pré-registrados e o mesmo objeto | sim |

**Relógio.** `clear_attended_stimuli` continua só com `self`. `__init__` não tem `clock` nem `now`. Duração de foco e o corte de 60 s usam `time.time()`. Relógio não foi injetado. O golden de AT6 não grava a duração numérica.

**Gaps que o eval mediu e não corrigiu.** `attend(goals=)` só chama `set_goals`. `"cat"` casa com `"category"`. `bool` vira 1.0/0.0 no clamp; `nan` vira 0.0 neste CPython 3.12.3. Um dataclass com os mesmos campos é tratado como já focado. A soma ponderada máxima observada é 0.895, então `min(1.0, salience)` não muda estes casos. Nada disso foi editado no bloco.

**Harness, não o módulo.** O sétimo teste chama `attend` duas vezes para cobrir o ramo em que `is_focused` já é verdadeiro. O código de produção não mudou. Coverage 7.16.2, só nesses testes, reportou `attention.py` com 181 statements, 48 ramos, 100%.

**Cobertura da suíte**, no commit de move (`NEXUS_INTEGRATED_SUMMARY=/tmp/integrated.json coverage run --branch -m unittest discover -p 'test_*.py'`, depois o demo do núcleo): 576 testes, rc 0, 52.603 s. Demo `python core/constitutional_brain.py` rc 0. `attention.py` 100%. `core/cognitivo/*` 1300 statements, 96% (piso 90). `constitutional_brain.py` 68% (piso 62). Fila 95%, VRAM 94%, temporal 93%, vita 76%.

**Mutação.** `python tools/targeted_mutation.py`: `{"declared": 69, "applicable": 69, "killed": 69, "survived": 0, "not_applied": 0, "score_percent": 100.0}`, rc 0. O catálogo do Axis não foi editado. `python tools/cognitivo_mutation.py`: `{"declared": 156, "killed": 156, "survived": 0, "not_applied": 0}`, rc 0. Os 27 mutantes do módulo novo estão nesse total (saliência, capacity, vítima de menor saliência, relevância de goal, custo cross-modal, reexport e o import do monólito).

**Realidade** (recontagem exigida pelo gate, commitada com o move): núcleo 25192 linhas, 239 classes, 764 funções (antes 25664 / 246 / 788). `attention.py` 508 linhas, 7 classes, 24 funções. `core/cognitivo` 9 módulos, 3013 linhas. Literais dinâmicos 51. Frases fortes 51. Terminologia científica 233. Prefixo `Real` 63. Testes embutidos 31. `pass` puro 0. Retorno constante 2. `reality_audit` reexecutado depois do commit de move: diff vazio.

**Outros gates locais no commit de move:** `ruff check` nos alvos do workflow, rc 0. `bandit -q -r core vita tools` grava o JSON com rc 1 (há achados LOW; o workflow aceita o relatório não vazio). Contagem 131 LOW, zero MEDIUM/HIGH, nenhum achado em `attention.py` nem em `tools/cognitivo_eval.py`. `security_triage.py --check` rc 0. `SECURITY_TRIAGE.md` não foi editado. `validar_registro.py --expirados aviso`, rc 0. `bash tools/agente/predeleg.sh HEAD origin/main` no commit de move `27aad64` contra `origin/main` `4f16130`: `OK: sem conflito`, rc 0.

**Hipóteses.** H1–H6 não se sustentam como falha neste eval: os números medidos são os da tabela de observação, e os dois SHA são os que foram congelados no monólito antes do splice. Nenhuma delas vira alegação de atenção humana, de modelo de Posner verificado, nem de prontidão. O banner `POST 25` e a suíte embutida do núcleo continuam sendo demo, não evidência de capacidade.

**Fora do que estes comandos medem:** CI hospedado; Python 3.11 (o 0.0 do `nan` foi lido em 3.12.3); conflito de merge contra as branches dos PRs #38, #39, #40, #41, #42, #44, #46, #48, #49, #50, #60, #77 e #79 (o `predeleg` só compara com `origin/main`); state docs não atualizados, de propósito; `AttentionalState`, POST49, GlobalWorkspace/Consciousness, IncrementalLearner, HierarchicalMemory/FAISS, Metacognition, Axis e o catálogo `tools/targeted_mutation.py` não foram tocados. Não há interface web nesta extração.

PR draft: https://github.com/pedromiorini/Nexus/pull/109

## Incremental Learner — critérios pré-registrados (IL1–IL6)

**Objetivo desta rodada:** extrair, em move-only, o bloco POST12 (`LearningEvent`, `ConsolidationReport`, `RealIncrementalLearner`) de `core/constitutional_brain.py` para `core/cognitivo/incremental_learner.py`, com reexport strangler no ponto original e em `core/cognitivo/__init__.py`. Medir IL1–IL6. Se todos passam, documentar e parar. Se algum falha no monólito atual, registrar o gap. Não corrigir comportamento depois de ver o número.
**Não escopo:** injetar relógio; corrigir gaps de comportamento (o guard de padrão não reconhece o próprio sintético; `apply_decay` é heurística e não faz UPDATE; `bool`/`NaN`/`±inf` seguem a aritmética de `int`/`float`; `id` 0 não é ausência); ObservationalLearner, ImitationLearner, VicariousLearner; HierarchicalMemory/FAISS (POST1); Metacognition (#42 e o bloco POST24); Attention e Working Memory já extraídos; Lifelong Learning Engine; o atalho `consolidate_learning` do orquestrador (~L23047), que só delega a `self.learner` e permanece no monólito; Axis e `tools/targeted_mutation.py` (mutantes só em `tools/cognitivo_mutation.py`); state docs (`CONTINUATION.md`, `NEXUS_EVIDENCE.md`, `NEXUS_KNOWLEDGE.md`, `SECURITY_TRIAGE.md`); PRs congelados #38–#50, #60, #77, #79 (não rebasear, não editar as branches, não resolver os conflitos deles).
**Eval:** `python tools/cognitivo_eval.py --capability incremental`. O id curto é `incremental` (não `incremental_learner`). O protocolo abaixo é único; `--criteria` v1/v2/v3 não o altera. Seeds não se aplicam. Timestamps de `LearningEvent` e o número de `duration` vêm de `time.time()` e não entram no golden. O golden guarda só `timestamp_is_float` e `duration_ok` (float ≥ 0 já arredondado a 4 casas).
**Base da observação:** `ab3a22bf9697f2a5e45ebbb4d32a423b8150da74` (merge do #109). Sonda só de leitura, Python 3.12.3, antes deste texto, antes de criar `core/cognitivo/incremental_learner.py` e antes de qualquer harness `incremental`. O fingerprint foi calculado duas vezes: executando o texto fatiado do monólito e, com o venv do `requirements-ci.lock`, importando `core.constitutional_brain`. Os dois SHA comportamentais coincidiram. Fakes duck-typed (`get_episode_memories`, `search_episodes`, `store`, `retrieve`, `log_event`); nenhum SQLite.

### Varredura §7.2 (2026-10-23, `gh pr list --state open`)

PRs abertos que editam `core/constitutional_brain.py`: #38, #39, #40, #41, #42, #44, #46, #48, #49. São os congelados do pedido; não são a base. A extração precisa do reexport no monólito (§16). Isso tensiona a fila única da §11.2: o conflito com esses PRs fica declarado, sem rebase neles. O mesmo padrão foi o do #108 e do #109, já mesclados.
Nenhum PR aberto altera `core/cognitivo/incremental_learner.py` (o arquivo ainda não existe), `test_incremental_learner_contract.py`, `tools/cognitivo_eval.py`, `tools/cognitivo_mutation.py`, `core/cognitivo/__init__.py` ou este arquivo. O #79 (draft, não usado como base) altera `docs/agentes/CLAIMS.yaml`. `AUDIT_REALITY_REPORT.md` / `audit_reality.json` aparecem em #38, #39, #40, #41, #42, #44, #46, #48 e #49; se o gate exigir recontagem depois do move, o diff fica só nos números deste módulo. O claim da Attention, já mesclado em `ab3a22b` e ainda listado, é substituído pelo claim grok-bot desta tarefa, prazo `2026-10-26T05:00:00Z`.

### Observação (código e sonda, antes do move e antes do eval)

**Fato, lido no bloco ~L2008–L2322:** as constantes de classe são `PROMOTION_RELEVANCE_THRESHOLD=0.75`, `REINFORCEMENT_IMPORTANCE_BOOST=0.08`, `IMPORTANCE_CAP=0.99`, `DECAY_RATE=0.02`, `PATTERN_MIN_FREQ=2`, `PATTERN_IMPORTANCE=0.92`. `__init__(episodic_memory, hierarchical_memory, constitutional_log=None)` guarda os três por atributo e não recebe `clock` nem `now`. Episódio e plano são `Any`: o bloco não importa `Episode` nem `PlanResult`.

`learn_from_episode` ignora o item só quando `id is None` (id ausente também). `id` 0 promove. Relevância ausente vale 0.0; importância ausente vale 0.5; conteúdo ausente vale `""`. Promoção (`if`) exige relevância ≥ 0.75 e id fora de `promoted_ids`: `store(conteúdo inteiro, "long_term", min(0.99, old*1.15))`, entra em `promoted_ids`, `total_promotions += 1`, evento com `promotion=True` e `content[:80]`. Reforço é `elif` (não acumula com a promoção): relevância ≥ 0.5 e importância ≥ 0.7 → `store(..., min(0.99, old+0.08))`, `promotion=False`. O padrão é contado mesmo sem evento: chave `content[:60].lower().strip()`.

`detect_and_store_patterns`: frequência < 2 não chama `retrieve`. Frequência ≥ 2 busca `tag="[PATTERN] {chave}"` com `limit=1` e só grava se nenhum `content` devolvido contém `tag[:30]`. O que grava é `"[PATTERN freq={freq}] {chave}"` com importância 0.92. Essa string **não** contém `tag[:30]`. Com log, emite `PATTERN_DETECTED`.

`apply_decay`: sem log, ou com conjunto vazio, devolve 0 e não grava. Com log e conjunto não vazio, emite `DECAY_APPLIED` (`accessed_count`, `decay_rate=0.02`) e devolve `max(0, 5-len(ids))`. Não chama `store`. `consolidate_learning` descarta esse retorno.

`learn_from_plan`: `None` → `None`. `expected_reward` ausente cai em 0.0 via `getattr` e também devolve `None`. `reward < 0.7` → `None`. Senão grava a sequência, `importance=min(0.99, 0.6+reward*0.35)`, `episode_id="mcts_direct"`, `promotion=True`, `old_importance=0.0`, `memory_id` é o retorno de `store`, incrementa `total_promotions` e **não** põe o id em `promoted_ids`. O evento corta `content` em 80; o `store` e o log recebem o texto e o goal inteiros.

`consolidate_learning` soma `total_consolidations`, busca `search_episodes(status="closed", limit=last_n_episodes)` (padrão 10), aprende cada `episode.episode_id`, detecta padrão só se `run_pattern_detection`, chama `apply_decay` só se `run_decay`, e loga `LEARNING_CONSOLIDATION` se há log. `memories_reinforced` conta eventos com `promotion` falso. Os ids acessados são só os dos eventos, não os que só entraram no registro de padrão.

| caso | medido |
|---|---|
| relevância 0.75, importância 0.8, id 7 | promoção; `new=min(0.99, 0.8*1.15)` = `0.9199999999999999`; reason `relevance=0.75 >= 0.75`; segunda chamada no mesmo id não promove de novo e reforça para 0.88 |
| relevância 0.749, importância 0.4 | nenhum evento; a chave do padrão mesmo assim conta 1 |
| importância 0.9, relevância 0.9 | `new=0.99` |
| id `None` ou ausente | nenhum evento e registro de padrão vazio |
| id 0 | promove; `memory_id` 0 |
| conteúdo de 100 `L` | evento com 80; `store` com 100; chave do padrão com 60 `l` |
| importância ausente / relevância ausente / conteúdo ausente | old 0.5 e promove; nenhum evento mas chave registrada; evento com `content=""` |
| relevância 0.5 e importância 0.7 | reforço, `new=min(0.99, 0.7+0.08)` = `0.7799999999999999`; `promoted_ids` vazio |
| relevância 0.5 e importância 0.699999999999; relevância 0.499 e importância 0.9 | nenhum evento |
| relevância 0.6 e importância 0.95 | reforço com teto 0.99 |
| relevância 0.749 e importância 0.8 | reforço, não promoção; o reason formata `rel=0.75` por `:.2f` |
| relevância 0.8 e importância 0.8 | um único evento, o de promoção (`if`/`elif`) |
| id já em `promoted_ids`, relevância 0.9, importância 0.8 | reforço, sem segunda promoção |
| freq 1 | `detect` devolve 0, sem `retrieve` e sem `store` |
| `"Repeat Me"` e `"  REPEAT ME  "` | a mesma chave `repeat me`, freq 2 |
| `retrieve` devolve o sintético gravado | a chamada seguinte grava de novo `[PATTERN freq=2] repeat me` com 0.92 |
| `retrieve` devolve `[]` e há log | uma gravação e `PATTERN_DETECTED` `{content: repeat me, frequency: 2}` |
| `retrieve` devolve um item cujo content é a própria tag | não grava; `total_patterns` permanece 0 |
| `learn_from_plan(None)`, reward 0.699, reward ausente | `None`, sem `store` |
| reward 0.7, goal de 80 `G`, sequência `look`,`go` | importância 0.845; evento com 80 caracteres; `store` com 113; log com o goal inteiro; `promoted_ids` vazio; `total_promotions` 1 |
| reward 1.2 e sequência vazia | importância 0.99; o texto termina em `Sequence: ` |
| `apply_decay` sem log / conjunto vazio / 1, 4, 5, 6 ids | 0 / 0 sem log / 4, 1, 0, 0; nenhum `store` |
| consolidar 3 episódios (`solo` promove, `solo` reforça, `quiet` só no padrão) | processados 3, promovidas 1, reforçadas 1, padrões 1, eventos 2; `DECAY_APPLIED` com `accessed_count` 2 |
| flags `run_pattern_detection=False` e `run_decay=False`, `last_n=2` | padrões 0, sem `retrieve`, sem `DECAY_APPLIED`; o log de consolidação continua |
| sem log | o padrão ainda é gravado; a lista de log fica vazia |
| busca vazia, argumentos padrão | `limit` 10, zero episódios, `total_consolidations` 1, sem `DECAY_APPLIED` |
| `str` em relevância, importância ou reward | `TypeError` |
| plano sem `best_action_sequence` | `AttributeError` |
| `goal is None`; item de memória que não é dict | `TypeError`; `AttributeError` |

**Gaps medidos, fora de correção nesta rodada.** `bool` é `int`: relevância `True` promove com reason `relevance=1.00`; importância `True` no reforço vai ao teto 0.99; reward `True` vale 1.0 (importância 0.95) e reward `False` devolve `None`. Relevância `nan` não promove e ainda conta padrão. Importância `nan` promove e `new_importance` fica 0.99, porque `min(0.99, nan)` neste CPython 3.12.3 fica com o 0.99. Importância `-inf` é gravada como `-inf`. Reward `nan` não é `< 0.7`, então o plano é gravado com `reward=nan` e importância 0.99. O JSON do golden contém os tokens `NaN` e `-Infinity`. Python 3.11 não foi executado.

### Hipóteses concorrentes (Strong Inference)

| id | hipótese | o que a falsifica | teste barato |
|---|---|---|---|
| H1 | mover o bloco muda o texto ou o fingerprint de comportamento | SHA de fonte ou SHA comportamental ≠ os valores abaixo, ou as três classes reexportadas não são o mesmo objeto | IL6 |
| H2 | promoção e reforço disparam juntos quando relevância ≥ 0.75 e importância ≥ 0.7 | o protocolo mostra um único evento, o de promoção | IL2 |
| H3 | a segunda detecção é idempotente contra o sintético que a própria função grava | a chamada seguinte grava de novo quando `retrieve` devolve esse sintético | IL3 |
| H4 | `apply_decay` atualiza a importância na memória com `DECAY_RATE` | não há `store` no decay; o retorno é `max(0, 5-len)` e só existe log com conjunto não vazio | IL5 |
| H5 | `learn_from_plan` exige a classe `PlanResult` | um objeto com `expected_reward`, `goal` e `best_action_sequence` basta; `None` e reward < 0.7 devolvem `None` | IL4 |

A sonda já exclui H2, H3, H4 e H5 como descrição do bloco atual. O eval as repete para o move não as reintroduzir. H1 ainda não foi testada: o módulo novo não existe. Hipótese de trabalho, não vinculante para além da tabela: move-only, sem injetar relógio e sem alterar fórmulas. O gap do padrão e o decay heurístico não autorizam editar o bloco.

### Critérios

| id | afirmação | protocolo falsificável | alvo |
|---|---|---|---|
| IL1 | relevância ≥ 0.75 e id novo promovem com `min(0.99, old*1.15)`; o mesmo id não promove de novo; id `None` é ignorado | Id 7, `"alpha fact"`, 0.75, 0.8 → um `store` long_term com `0.9199999999999999`, `promotion=True`, reason `relevance=0.75 >= 0.75`. A segunda chamada reforça (0.88) e `total_promotions` continua 1. 0.749 com importância 0.4 não gera evento, mas a chave conta. Importância 0.9 → 0.99. Id `None` ou ausente: zero eventos e registro vazio. Id 0 promove. Conteúdo de 100 caracteres: evento 80, `store` 100, chave 60 já em minúsculas. Importância ausente usa 0.5 | esses stores, o teto e a não-repromoção |
| IL2 | reforço é o `elif`: +0.08 com teto 0.99, só sem promoção, relevância ≥ 0.5 e importância ≥ 0.7 | 0.5/0.7 → `0.7799999999999999` e `promotion=False`. 0.5 com importância logo abaixo de 0.7, e 0.499 com 0.9, não geram evento. 0.95 vai a 0.99. 0.749/0.8 reforça e o reason mostra `rel=0.75`. 0.8/0.8 gera um único evento de promoção. Id já promovido com 0.9/0.8 reforça e não entra de novo | um evento, o boost e o teto |
| IL3 | freq ≥ 2 grava `[PATTERN freq=…]` com importância 0.92 se `retrieve` não acha `tag[:30]`; a tag achada não duplica; o sintético da própria função não conta como achado | Freq 1: devolve 0, sem `retrieve`. `"Repeat Me"` e `"  REPEAT ME  "` viram `repeat me`. `retrieve` que ecoa o sintético: a chamada seguinte grava outra vez. `retrieve` vazio com log: uma gravação e `PATTERN_DETECTED`. `retrieve` que devolve a tag: zero gravações e `total_patterns` 0 | 0.92, a tag e a segunda gravação do sintético |
| IL4 | `learn_from_plan` devolve `None` se o plano é `None` ou reward < 0.7; reward ≥ 0.7 grava a sequência e marca promoção | `None`, 0.699 e reward ausente → `None`. Reward 0.7 → importância 0.845, `episode_id="mcts_direct"`, `memory_id` igual ao retorno de `store`, evento cortado em 80, `store` com o texto inteiro, log com goal de 80 e a sequência, `promoted_ids` vazio, `total_promotions` 1. Reward 1.2 → 0.99. `str` → `TypeError`. Sem sequência → `AttributeError`. `goal=None` → `TypeError`. Item de memória `str` → `AttributeError` | `None`, 0.845 e as exceções |
| IL5 | `consolidate_learning` conta episódios, promoções, reforços e padrões; as flags mandam no padrão e no decay; o decay não atualiza memória | Três episódios fechados: 3/1/1/1 e 2 eventos; busca `status=closed`, `limit=10`; log `PATTERN_DETECTED`, `DECAY_APPLIED` com `accessed_count` 2, `LEARNING_CONSOLIDATION`. Flags falsas e `last_n=2`: padrões 0, sem `DECAY_APPLIED`, consolidação ainda logada. Sem log: padrão gravado, log vazio. Busca vazia no padrão: `limit` 10 e `total_consolidations` 1. `apply_decay` devolve 0/0/4/1/0/0 nos casos sem log, vazio, 1, 4, 5 e 6 ids, e não chama `store`. `duration_ok` é verdadeiro sem fixar o número | as contagens, as flags e a heurística |
| IL6 | move-only: o texto do bloco e o fingerprint comportamental são os mesmos; reexport é o mesmo objeto | O texto UTF-8 que começa na linha de `═` imediatamente antes de `# MÓDULO 11: INCREMENTAL LEARNER (POST12)` e termina no `}` de `get_statistics` (a chave `"pattern_registry_size"`) tem SHA-256 `e09c2d75df964dd52915eaf0c0660dd45a6ac34cbf476804c786423910057542` no monólito, antes do splice, e o mesmo SHA no módulo novo. As três classes: `core.constitutional_brain` e `core.cognitivo` são o mesmo objeto que `core.cognitivo.incremental_learner`, e `__module__` é `core.cognitivo.incremental_learner`. O módulo importa sem puxar o monólito, numpy ou vita (árvore de `import` / `import from`; menção no docstring não conta). O dict do protocolo, `json.dumps(..., sort_keys=True, ensure_ascii=False)`, tem SHA-256 `636c95bfaebbd60ca55a5e14f12e9f5ff7b6f7868f5a476b31fa479528db7767`, igual pela classe do monólito e pela do módulo novo. `clock` é `wall` | os dois SHA e a identidade |

**Protocolo do fingerprint comportamental de IL6** (o que entra no dict; fakes em memória, sem SQLite):

- `hyper` com as seis constantes; `clock`; `init_params` = `self`, `episodic_memory`, `hierarchical_memory`, `constitutional_log`.
- `promotion`: o caso 0.75/0.8 antes da segunda chamada (`stores`, `promoted`, `stats`) e `again` / `again_stores` / `again_stats` depois.
- `below` 0.749/0.4; `cap` 0.9/0.9; `skip_none`; `long`; `zero_id`; `defaults` (importância, relevância ou conteúdo ausentes).
- `boundaries`: 0.5/0.7, 0.5/0.699999999999, 0.499/0.9, 0.6/0.95, 0.749/0.8, 0.8/0.8. `preset_promoted` com o id 9 já inserido em `promoted_ids`.
- `patterns`: freq 1; registro `repeat me`; duas gravações quando o `retrieve` ecoa o sintético; uma gravação com log e `retrieve` vazio; zero quando o `retrieve` devolve a tag.
- `plans`: `none`, `low` e `missing_reward` verdadeiros; evento e `store` do reward 0.7; reward 1.2; reward `True`, `False` e `nan` no golden, sem virar rejeição de tipo.
- `decay`: sem log, conjunto vazio, e contagens 1/4/5/6. `consolidate`: cheio, flags desligadas, sem log, e busca vazia no padrão. `duration` numérica de fora.
- `gaps`: relevância `True`/`False`, importância `True`, relevância `nan`, importância `nan`, relevância `inf`, importância `-inf`, relevância −1. `errors`: os `TypeError` e `AttributeError` da tabela.

`bool`/`NaN`/`-inf`, o reason que arredonda 0.749 para `0.75`, a segunda gravação do sintético e o decay que não atualiza memória fazem parte do golden de identidade. Não são metas de qualidade e não serão corrigidos nesta rodada.

### Regressão exigida

Não há contrato anterior destes três símbolos. O eval dos outros capabilities não é alvo de mudança de critério. O atalho do orquestrador continua chamando `self.learner.consolidate_learning`.

Pré-registro escrito **antes** do move, **antes** do contrato novo de extração e **antes** de rodar `tools/cognitivo_eval.py --capability incremental`. O SHA deste commit entra no PR. Resultados só numa seção posterior.

### Resultados IL (após o pré-registro `f11610d`)

Ambiente local: Python 3.12.3, venv com `requirements-ci.lock` (ruff 0.17.1, coverage 7.16.2, bandit 1.9.4, hypothesis 6.170.5, numpy 2.4.6). O CI usa Python 3.11; esta máquina não o executou. Os números abaixo são desta execução, não um carimbo de CI hospedado.

**Commits.** Pré-registro `f11610d775c50d5ab462287cad5f98e2bbab4b08` (só `CLAIMS.yaml` e esta auditoria). Move-only `33226db5e030beb4881bb31b86b3b0ae1a937f14`. Base `ab3a22bf9697f2a5e45ebbb4d32a423b8150da74`.

**RED**, antes de existir `core/cognitivo/incremental_learner.py`: `PYTHONPATH=. python -m unittest test_incremental_learner_contract.py` → rc 1, 7 testes, 3 `ModuleNotFoundError: No module named 'core.cognitivo.incremental_learner'`. Os quatro testes que importam o monólito passaram: `str` em relevância/importância, plano com reward texto / sem sequência / `goal=None`, item de memória que não é dict, e os contadores de consolidação (só promoção 1/0 e só reforço 0/1).

**GREEN**, o mesmo comando depois do move: 7 testes, rc 0. Nenhum assert foi relaxado.

**Eval** `python tools/cognitivo_eval.py --capability incremental --criteria v3` (v1/v2/v3 não alteram o protocolo; o padrão do argparse continua `v2`): rc 0, `failed: []`. O stdout do monólito ainda imprime o aviso de psutil/FAISS antes do JSON; isso já ocorria ao importar `core.constitutional_brain`.

| id | valor medido | alvo | passou |
|---|---|---|---|
| IL1 | `new=0.9199999999999999`; segunda chamada `promotion=false` e `total_promotions` 1; 0.749/0.4 sem evento; teto 0.99; id `None` vazio; id 0 promove | esses stores e a não-repromoção | sim |
| IL2 | 0.5/0.7 → `0.7799999999999999`; abaixo dos limiares, zero eventos; 0.8/0.8 um único evento de promoção | o boost e o `elif` | sim |
| IL3 | chave `repeat me`; duas gravações quando `retrieve` ecoa o sintético; tag achada → 0; log `PATTERN_DETECTED` | 0.92 e a segunda gravação | sim |
| IL4 | `None`, 0.699 e reward ausente → `None`; reward 0.7 → importância 0.845; `str` → `TypeError`; sem sequência → `AttributeError` | `None`, 0.845 e as exceções | sim |
| IL5 | 3 episódios, 1 promoção, 1 reforço, 1 padrão; flags falsas zeram o padrão e o `DECAY_APPLIED`; `apply_decay` de 1 id devolve 4 e não chama `store` | as contagens e a heurística | sim |
| IL6 | SHA de fonte do módulo = SHA medido no monólito antes do splice = `e09c2d75df964dd52915eaf0c0660dd45a6ac34cbf476804c786423910057542`; SHA comportamental módulo = monólito = `636c95bfaebbd60ca55a5e14f12e9f5ff7b6f7868f5a476b31fa479528db7767`; identidade verdadeira; `clock: wall` | os dois SHA pré-registrados e o mesmo objeto | sim |

**Relógio.** `__init__` continua só com `episodic_memory`, `hierarchical_memory` e `constitutional_log`. Timestamps e `duration` usam `time.time()`. Relógio não foi injetado. O golden de IL6 não grava o número da duração.

**Gaps que o eval mediu e não corrigiu.** A segunda `detect_and_store_patterns` grava de novo quando `retrieve` devolve o sintético `[PATTERN freq=N] {chave}`, porque a busca usa `tag[:30]` = `[PATTERN] {chave}`[:30] e essa substring não está no sintético. `apply_decay` não chama `store`; devolve `max(0, 5-len)` só com log e conjunto não vazio. `bool` segue `int`. Relevância `nan` não promove. Importância `nan` promove com `new_importance` 0.99 neste CPython 3.12.3 (`min(0.99, nan)` fica com o primeiro argumento). Importância `-inf` é gravada como `-inf`. Reward `nan` passa do `< 0.7` e é gravado com texto `reward=nan`. O reason de relevância 0.749 formata `rel=0.75`. Nada disso foi editado no bloco.

**Harness, não o módulo.** O teste dos contadores (1/0 e 0/1) existe porque o cenário misto do IL5 tem uma promoção e um reforço: trocar os dois predicados manteria os dois números em 1 e o SHA. O código de produção não mudou. Coverage 7.16.2, na suíte inteira, reportou `incremental_learner.py` com 123 statements, 32 ramos, 100%.

**Cobertura da suíte**, no commit de move (`NEXUS_INTEGRATED_SUMMARY=/tmp/integrated.json coverage run --branch -m unittest discover -p 'test_*.py'`, depois o demo do núcleo): 583 testes, rc 0, 53.941 s. Demo `python core/constitutional_brain.py` rc 0. O banner impresso por esse demo não é evidência de capacidade. `incremental_learner.py` 100%. `core/cognitivo/*` 1424 statements, 96% (piso 90). `constitutional_brain.py` 68% (piso 62). Fila 95%, VRAM 94%, temporal 93%, vita 76%.

**Mutação.** `python tools/targeted_mutation.py`: `{"declared": 69, "applicable": 69, "killed": 69, "survived": 0, "not_applied": 0, "score_percent": 100.0}`, rc 0. O catálogo do Axis não foi editado. `python tools/cognitivo_mutation.py`: `{"declared": 184, "killed": 184, "survived": 0, "not_applied": 0}`, rc 0. Os 28 mutantes do módulo novo estão nesse total (limiar, teto, id já promovido, padrão, decay heurístico, flags, reexport e o import do monólito). `il_module_pulls_monolith` insere `import core.constitutional_brain`; o AST do eval e o contrato em subprocesso o matam.

**Realidade** (recontagem exigida pelo gate, commitada com o move): núcleo 24886 linhas, 236 classes, 757 funções (antes 25192 / 239 / 764). `incremental_learner.py` 334 linhas, 3 classes, 7 funções. `core/cognitivo` 10 módulos, 3355 linhas. Literais dinâmicos 50. Frases fortes 51. Terminologia científica 233. Prefixo `Real` 63. Testes embutidos 31. `pass` puro 0. Retorno constante 2. `reality_audit` reexecutado depois do commit de move: diff vazio.

**Outros gates locais no commit de move:** `ruff check` nos alvos do workflow, rc 0. `bandit -q -r core vita tools` grava o JSON com rc 1 (há achados LOW; o workflow aceita o relatório não vazio). Contagem 131 LOW, zero MEDIUM/HIGH, nenhum achado em `incremental_learner.py` nem em `tools/cognitivo_eval.py`. `security_triage.py --check` rc 0. `SECURITY_TRIAGE.md` não foi editado. `validar_registro.py --expirados aviso`, rc 0. `bash tools/agente/predeleg.sh HEAD origin/main` no commit de move `33226db` contra `origin/main` `ab3a22b`: `OK: sem conflito`, rc 0.

**Hipóteses.** H1 não se sustenta como falha neste eval: os dois SHA são os que foram congelados no monólito antes do splice. H2, H3, H4 e H5 já tinham sido excluídas pela sonda e o eval repetiu a exclusão. Nenhuma delas vira alegação de aprendizado humano, de consolidação de memória verificada fora deste contrato, nem de prontidão. O nome `RealIncrementalLearner` e a suíte embutida do núcleo continuam sendo demo, não evidência de capacidade.

**Fora do que estes comandos medem:** CI hospedado; Python 3.11 (o 0.99 de `min(0.99, nan)` e o token `-Infinity` foram lidos em 3.12.3); conflito de merge contra as branches dos PRs #38, #39, #40, #41, #42, #44, #46, #48, #49, #50, #60, #77 e #79 (o `predeleg` só compara com `origin/main`); state docs não atualizados, de propósito; ObservationalLearner, ImitationLearner, VicariousLearner, HierarchicalMemory/FAISS, Metacognition, Attention, Working Memory, Lifelong Learning Engine, o atalho `consolidate_learning` do orquestrador e o catálogo `tools/targeted_mutation.py` não foram alterados no comportamento. Não há interface web nesta extração.

PR draft: https://github.com/pedromiorini/Nexus/pull/110

## Episodic Future Thinking — critérios pré-registrados (EF1–EF6)

**Objetivo desta rodada:** extrair, em move-only, o bloco POST31 (`FutureEvent`, `ProspectiveMemory`, `RealEpisodicFutureThinkingSystem`) de `core/constitutional_brain.py` para `core/cognitivo/episodic_future_thinking.py`, com reexport strangler no ponto original e em `core/cognitivo/__init__.py`. Medir EF1–EF6. Se todos passam, documentar e parar. Se algum falha no monólito atual, registrar o gap. Não corrigir comportamento depois de ver o número.
**Por que este bloco, e não outro:** a leitura do monólito em `d0aab24` mostra o bloco separável. Ele não chama classe nenhuma do monólito. Os quatro colaboradores (`memory`, `mental_simulation`, `consciousness`, `emotion`) são só atributos. `memory` e `emotion` entram como valor booleano; nenhum método deles é chamado. `mental_simulation` e `consciousness` são guardados e não são lidos. Os nomes de que o bloco depende (`dataclass`, `List`, `Dict`, `Any`, `uuid`, `time`) já estão no escopo do monólito e são stdlib. Blocos de ferramenta (Tool Use / `ToolDefinition` / permissões) ficam de fora: outro agente trabalha segurança de ferramentas.
**Não escopo:** injetar relógio; passar a consultar memória, emoção, simulação mental ou consciência; trocar substring por palavra inteira; tornar `fulfill_intention` idempotente; rejeitar `bool`/`NaN`/`±inf`/importância fora de [0, 1]; `details or {}`; Mental Simulation (POST30), Analogical Reasoning, Embodied Cognition; o construtor do orquestrador que instancia o sistema (~`enable_episodic_future_thinking`); Axis e `tools/targeted_mutation.py` (mutantes só em `tools/cognitivo_mutation.py`); state docs (`CONTINUATION.md`, `NEXUS_EVIDENCE.md`, `NEXUS_KNOWLEDGE.md`, `SECURITY_TRIAGE.md`); PRs congelados #38–#50, #60, #77, #79 (não rebasear, não editar as branches, não resolver os conflitos deles).
**Eval:** `python tools/cognitivo_eval.py --capability episodic_future`. O id curto é `episodic_future` (não `future` nem `eft`). O protocolo abaixo é único; `--criteria` v1/v2/v3 não o altera. Seeds não se aplicam. `event_id`, `intention_id` e `created_at` vêm de `uuid.uuid4()` e `time.time()` e não entram no golden. O golden guarda `event_id_is_uuid` (string de 36), `created_at_is_float`, `id_is_uuid` e `ids_differ`.
**Base da observação:** `d0aab245f6d0453428d1f530c82ddd23fed18c1b` (merge do #110). Sonda só de leitura, Python 3.12.3, antes deste texto, antes de criar `core/cognitivo/episodic_future_thinking.py` e antes de qualquer harness `episodic_future`. O fingerprint foi calculado duas vezes: executando o texto fatiado do monólito e importando `core.constitutional_brain`. Os dois SHA comportamentais coincidiram. Colaboradores são `object`/`_Boom` (qualquer atributo lido levanta `AssertionError`); nenhum SQLite.

### Varredura §7.2 (2026-10-23, `gh pr list --state open`)

PRs abertos: #38, #39, #40, #41, #42, #43, #44, #45, #46, #47, #48, #49, #50, #60, #77, #79. Os que editam `core/constitutional_brain.py`: #38, #39, #40, #41, #42, #44, #46, #48, #49. São os congelados do pedido; não são a base. A extração precisa do reexport no monólito (§16). Isso tensiona a fila única da §11.2: o conflito com esses PRs fica declarado, sem rebase neles. O mesmo padrão foi o do #108, do #109 e do #110, já mesclados.
Nenhum PR aberto altera `core/cognitivo/episodic_future_thinking.py` (o arquivo ainda não existe), `test_episodic_future_thinking_contract.py`, `tools/cognitivo_eval.py`, `tools/cognitivo_mutation.py`, `core/cognitivo/__init__.py` ou este arquivo. O #79 (draft, não usado como base) altera `docs/agentes/CLAIMS.yaml`. `AUDIT_REALITY_REPORT.md` / `audit_reality.json` aparecem em #38, #39, #40, #41, #42, #44, #46, #48 e #49; se o gate exigir recontagem depois do move, o diff fica só nos números deste módulo. O #46 também altera `docs/agentes/LICOES_APRENDIDAS.md`; esta rodada só escreve lá se um verificador falhar. O claim do IncrementalLearner, já mesclado em `d0aab24` e ainda listado, é substituído pelo claim grok-bot desta tarefa, prazo `2026-10-26T06:00:00Z`.

### Observação (código e sonda, antes do move e antes do eval)

**Fato, lido no bloco ~L5012–L5357:** `__init__(memory=None, mental_simulation=None, consciousness=None, emotion=None)` guarda os quatro por identidade, zera os contadores e as duas listas, e imprime o banner `🧠🔮 POST 31 - RealEpisodicFutureThinkingSystem initialized (Future imagination active)\n`. Não há parâmetro `clock`.

`imagine_future_event`: `details or {}` trata `None`, `0`, `False` e `""` como `{}`. `True` e uma lista são verdadeiros e quebram em `.get` (`AttributeError`). Com `memory` verdadeiro, o passado é o dict fixo `location=familiar_place`, `participants=["friend","colleague"]`, `emotions=["happy","excited"]`, sem chamada de método. Com `memory` falso (`None`, `""`, `0`, `[]`, `False`), o passado é `{}`. Local e participantes usam `details` se a chave existe, inclusive `location=""` e `participants=None` ou uma tupla. `time_point` é guardado como veio: `7` fica `int`.

Emoções: se `emotion` é verdadeiro e o passado tem `emotions`, devolve essa lista e não lê a descrição. Senão, substring em minúsculas: `success` ou `achievement` → `["pride","joy"]`; `challenge` → `["anxiety","excitement"]`; `meeting` → `["curiosity","neutral"]`; caso contrário `["neutral"]`. Descrição `int` sem esse atalho de emoção → `AttributeError`. Descrição `int` com memória e emoção verdadeiras não lê `.lower()` e devolve `["happy","excited"]`.

Vividez: `min(1.0, 0.3 + len(past_elements) * 0.15)`. Passado fixo tem 3 chaves → `0.75`. Passado vazio → `0.3`. O teto `1.0` não é atingido por esse retorno. Plausibilidade: passado não vazio → `0.8`; senão, se a descrição contém `meeting`, `work`, `lunch`, `travel` ou `presentation` → `0.7`; senão `0.5`. `homework`, `network`, `workshop` e `Working late` caem em `0.7` porque contêm `work`. Significância ausente é `0.5`; `0` permanece `0` (`int`); `False` permanece `bool`; `nan`, `inf` e `-1` são guardados.

Cada `imagine` acrescenta o mesmo objeto à lista e soma 1 em `total_events_imagined`. Dois eventos têm ids diferentes. Alterar `participants` de um evento não altera o seguinte. `mental_simulation` e `consciousness` presentes ou `None` não mudam o evento.

`preexperience_event` soma 1 em `total_preexperiences` e devolve `event_id`, `sensory_details` (`visual` = `Imagining {location}`, `emotional` e `social` são as listas do evento), `vividness` e `simulation_success=True`. Não altera o evento. `None` → `AttributeError`.

`get_statistics` com listas vazias: taxa `0.0` e vividez média `0.0` (`max(1, 0)` no denominador da média). Dois eventos de vividez `0.75` e `0.25` dão média `0.5`.

`set_prospective_memory`: `trigger` falso (`""`, `None`, `0`) vira `when`. Importância ausente é `0.5`. `1.5`, `-1`, `True` e `nan` são guardados sem corte. `what=None` é guardado. O objeto devolvido é o que está na lista. `fulfilled` começa `False`.

`check_prospective_triggers`: pula `fulfilled`. Casa por substring, não por igualdade: `when in str(time)`, `where in str(location)`, `trigger in str(event)`, cada um só se o valor do contexto for verdadeiro. `""` está em qualquer string, então `where=""` casa com qualquer local verdadeiro. `"day"` casa com `"today"`. Contexto `{}`, strings vazias ou zeros: lista vazia. Contexto `None`, `str` ou `list` com a lista de intenções vazia não entra no laço e não erro. Com uma intenção em aberto, os três levantam `AttributeError`. Omitir o argumento levanta `TypeError`. A lista devolve os mesmos objetos, na ordem de inserção. No protocolo (Ana `tomorrow`/`office`/`phone`; notes `today`/`""`/`email`; `already` cumprida `tomorrow`/`lab`/`bell`; `day-match` `day`/`nowhere`/`nope`):

| contexto | o que dispara |
|---|---|
| `{}`, strings vazias, zeros | ninguém |
| `time=tomorrow` | call Ana |
| `time=today` | send notes, day-match |
| `location=office` e `location=home office` | call Ana, send notes |
| `event=phone call` | call Ana |
| `event=email` | send notes |
| time+location+event da Ana | call Ana, send notes |

`fulfill_intention` de id ausente devolve `None` e não soma. No id achado, marca `fulfilled` e soma de novo mesmo se já estava cumprida. Uma intenção cumprida duas vezes: `fulfilled_intentions=2`, ativos `0`, taxa `2.0`. Uma de duas cumprida uma vez: ativos `1`, taxa `0.5`. A segunda chamada nessa mesma: contador `2`, taxa `1.0`, ativos ainda `1`.

`project_future_self`: `"year"` na minúscula antes de `"month"`. `1 year`, `YEAR`, `yearly` e `1 year and 1 month` → `increased_experience`, `new_skills`, `evolved_goals`. `next month`, `Month`, `monthly` → `incremental_progress`, `learning`. `tomorrow`, `""`, `week` → `minor_changes`. Crescimento é constante em todos: knowledge `0.8`, skills `0.7`, wisdom `0.6`, continuity `0.7`. Horizonte `int` ou `None` → `AttributeError`.

`compare_future_scenarios` usa `>` estrito; empate fica com o segundo. Produto `vividness * personal_significance`. Eventos construídos com ids fixos: a=`(0.9, 0.4, 0.2)`, b=`(0.3, 0.8, 0.5)` → mais vívido `a`, mais plausível `b`, mais significativo `b`, preferido `a` (`0.18 > 0.15`). Empate de vividez e plausibilidade c=`(0.5, 0.4, 0.4)` contra d=`(0.5, 0.4, 0.5)` → as quatro chaves `d`. Produto empatado c contra e=`(0.4, 0.1, 0.5)` (`0.20 == 0.20`) → preferido `e`, mais significativo `e`, vívido e plausível `c`. Vividez `nan` no primeiro contra b → preferido `b` e mais significativo `n` (`0.9 > 0.5`). `True` contra `False` na significância, vividez igual `0.4`: preferido o `True` (`t`), vívido e plausível o segundo (`u`). Significância `str` → `TypeError`. Objeto sem `personal_significance` → `AttributeError`.

**Gaps medidos, fora de correção nesta rodada.** Memória e emoção não são consultadas; simulação e consciência não são lidas. `work` casa dentro de `network` e `homework`. `where=""` casa com qualquer local. `"day"` casa com `"today"`. `fulfill_intention` não é idempotente e a taxa passa de `1`. `bool` `False` em `details` vira `{}`; `True` quebra. `bool` na significância compara como `int`. `nan` perde a comparação `>`. Descrição que não é string passa se memória e emoção estão ambos verdadeiros. Nada disso será editado no bloco.

### Hipóteses concorrentes (Strong Inference)

| id | hipótese | o que a falsifica | teste barato |
|---|---|---|---|
| H1 | mover o bloco muda o texto ou o fingerprint de comportamento | SHA de fonte ou SHA comportamental ≠ os valores abaixo, ou as três classes reexportadas não são o mesmo objeto | EF6 |
| H2 | `memory` verdadeiro é consultado por método | um objeto que levanta em qualquer atributo ainda produz `familiar_place` / `0.75` / `0.8` | EF1 |
| H3 | `emotion` verdadeiro escolhe a emoção da descrição | com memória e emoção, `"challenge ahead"` continua `["happy","excited"]` | EF1 |
| H4 | o gatilho compara igualdade, não substring | `time=today` inclui `day-match`; `location=office` inclui a intenção com `where=""` | EF3 |
| H5 | cumprir a mesma intenção duas vezes não soma de novo | a intenção sozinha fica com contador `2` e taxa `2.0` | EF4 |
| H6 | `mental_simulation` ou `consciousness` mudam o evento imaginado | o mesmo `imagine("Quiet day")` com os dois objetos e com `None` tem a mesma vista | EF1 |

A sonda já exclui H2, H3, H4, H5 e H6 como descrição do bloco atual. O eval as repete para o move não as reintroduzir. H1 ainda não foi testada: o módulo novo não existe. Hipótese de trabalho, não vinculante para além da tabela: move-only, sem injetar relógio e sem passar a chamar os colaboradores. Os gaps não autorizam editar o bloco.

### Critérios

| id | afirmação | protocolo falsificável | alvo |
|---|---|---|---|
| EF1 | memória verdadeira usa o passado fixo; emoção verdadeira não lê a descrição quando esse passado tem emoções; vividez e plausibilidade seguem as fórmulas lidas | Memória+emoção, `"Quiet day"`: local `familiar_place`, participantes `friend`/`colleague`, emoções `happy`/`excited`, vividez `0.75`, plausibilidade `0.8`, significância `0.5`. Override `location=hall`, `participants=["a"]`, `significance=0.9` mantém as emoções do passado. `"challenge ahead"` com os dois também. Sem memória, `"success today"` → `pride`/`joy`, local `unknown`, vividez `0.3`, plausibilidade `0.5`. `"The Challenge"` → `anxiety`/`excitement` e `0.5`. `"team meeting"` → `curiosity`/`neutral` e `0.7`. `""` → `neutral` e `0.5`. `homework`, `network`, `workshop`, `Working late`, `lunch`, `travel`, `presentation` → `0.7`. Memória falsa (`""`, `0`, `[]`, `False`, `None`) igual a ausente. Memória `1` ou `object()` usa o passado fixo. `details` `None`, `{}`, `0`, `False` e `""` coincidem com o argumento omitido. Significância `0` fica `0`; `False` fica bool. Local `""` e participantes `None` são guardados. `time_point=7` fica `int`. Quatro ids diferem. Acrescentar um participante não altera o evento seguinte. Simulação e consciência não mudam a vista. Descrição `1` com memória e emoção produz o passado fixo | `0.75`, `0.8`, `0.7`, `0.3`, `0.5` e o passado fixo |
| EF2 | pré-vivência devolve o ensaio sensorial e soma o contador; a estatística vazia não divide por zero | Duas pré-vivências do mesmo evento: `simulation_success` verdadeiro, visual `Imagining familiar_place`, contador `2`, eventos imaginados continuam `1`, o evento não muda. Estatística vazia: seis zeros. Dois eventos, o segundo com vividez `0.25` e uma pré-vivência: média `0.5`, pré-vivências `1` | contador `2` e média `0.5` |
| EF3 | gatilho falso copia `when`; a checagem é substring e pula cumpridas; importância não é cortada | Padrão: trigger `tomorrow`, importância `0.5`, `where=""`. `""`, `None` e `0` no trigger caem em `when`. `what=None` é guardado. Importâncias `1.5`, `-1`, `True` e `nan` saem como entraram. A tabela de contextos acima, com a cumprida de fora e os mesmos objetos | a tabela e a importância `1.5` |
| EF4 | cumprir soma outra vez; o horizonte é substring com `year` antes de `month`; o crescimento não depende do horizonte; empate na comparação fica com o segundo | Id ausente: `None` e contador `0`. Uma de duas, uma vez: taxa `0.5`, ativos `1`. A segunda vez: contador `2`, taxa `1.0`, ativos `1`. Intenção sozinha, duas vezes: taxa `2.0`, ativos `0`. Horizontes da observação. Crescimento `0.8`/`0.7`/`0.6` e continuity `0.7` nos dez horizontes. Comparação dos cinco pares com ids `a`–`e`, `n`, `t`, `u` como na observação | taxa `2.0`, continuity `0.7` e o preferido do empate |
| EF5 | tipo errado quebra no uso; contexto vazio de intenções não chega a ler o contexto | Descrição `int` sem emoção, `details` lista, `details` `True`, horizonte `int` e `None`, pré-vivência `None` e comparação sem `personal_significance`: `AttributeError`. Significância `str`: `TypeError`. Contexto `None`/`str`/`list` sem intenções: sem erro. Com uma intenção em aberto: `AttributeError`. Argumento omitido: `TypeError` | essas classes de exceção |
| EF6 | move-only: o texto do bloco e o fingerprint comportamental são os mesmos; reexport é o mesmo objeto | O texto UTF-8 que começa na linha de `═` imediatamente antes de `# MÓDULO 31: EPISODIC FUTURE THINKING SYSTEM (POST 31)` e termina no `}` de `get_statistics` (a chave `"avg_event_vividness"`) tem SHA-256 `a72a34bcc303f98b37cd5e33e030ab69bea8579d4f111c95981b2cf9785cb3e7` no monólito, antes do splice, e o mesmo SHA no módulo novo. As três classes: `core.constitutional_brain` e `core.cognitivo` são o mesmo objeto que `core.cognitivo.episodic_future_thinking`, e `__module__` é `core.cognitivo.episodic_future_thinking`. O módulo importa sem puxar o monólito, numpy ou vita (árvore de `import` / `import from`; menção no docstring não conta). O dict do protocolo, `json.dumps(..., sort_keys=True, ensure_ascii=False)`, tem SHA-256 `062870d6a044174d49150fc6008521105c05557c40c2f46a4e83cd116b761f27`, igual pela classe do monólito e pela do módulo novo. `clock` é `wall` | os dois SHA e a identidade |

**Protocolo do fingerprint comportamental de EF6** (o que entra no dict; sem SQLite):

- `clock`; `init_params`; `banner` exato; nomes dos campos dos dois dataclasses.
- `wired`: as quatro identidades e os contadores zerados.
- `both`: quiet, override, challenge, participantes do evento seguinte, `ids_differ`, `count` 4, o objeto guardado é o devolvido, simulação ignorada.
- `memory_only`, `emotion_only` (doze descrições), `neither`, `falsy_memory`, `truthy_memory`.
- `details`: omitido, `None`, `{}`, `0`, `False`, `""`, significância `0`/`False`/`nan`/`inf`/`-1`, local vazio, participantes `None` e tupla, `time_point` `7`.
- `preexperience` sem o uuid cru; `stats` vazio e o par de vividezes `0.75`/`0.25`.
- `prospective` padrão, trigger falso, importância fora do intervalo, `what=None`.
- `triggers` da tabela; `fulfill` ausente, uma de duas, a segunda vez, e a intenção sozinha duas vezes.
- `project` dos dez horizontes, com crescimento e continuity.
- `compare` dos cinco pares; `gaps` da descrição `int` com memória e emoção; `errors` com o nome da classe.

`bool`/`NaN`/`Infinity`, a substring `work`, o `where` vazio, a taxa `2.0` e o passado fixo fazem parte do golden de identidade. Não são metas de qualidade e não serão corrigidos nesta rodada.

### Regressão exigida

Não há contrato anterior destes três símbolos. O teste embutido 27 do demo continua chamando `imagine_future_event`, `preexperience_event`, `set_prospective_memory`, `project_future_self` e `get_statistics` pelo reexport. O eval dos outros capabilities não é alvo de mudança de critério.

Pré-registro escrito **antes** do move, **antes** do contrato novo de extração e **antes** de rodar `tools/cognitivo_eval.py --capability episodic_future`. O SHA deste commit entra no PR. Resultados só numa seção posterior.
