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
As seções v1, v2 e v3 acima permanecem inalteradas. O M6 é um critério novo, com flag `--criteria m6` no harness.

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
