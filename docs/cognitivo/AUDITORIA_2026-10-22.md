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
