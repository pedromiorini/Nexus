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
