# Benchmark sintético de capacidades observáveis do Nexus

O protocolo original `nexus.capability-benchmark.v1` oferece uma avaliação **reproduzível, local e baseada em tarefas sintéticas** para quatro perguntas delimitadas: comportamento composicional, alegações de intervenção causal, tratamento de incerteza e inferência sob observação parcial. Sua referência histórica de código é o `main` em `5f53a9922b71d6ef3c565a31630f56091a454671` (após a PR #9).

O v1 mede as APIs experimentais existentes; não altera o núcleo de produção, não usa rede ou modelos externos e não converte nomes teóricos em alegações de capacidade. Cada trilha declara o que foi efetivamente medido e o que ainda não é mensurável pela interface. O benchmark POMDP v2 abaixo é um protocolo adicional e independente.

## Executar

Na raiz do repositório, com os requisitos de CI instalados:

```bash
PYTHONPATH=. python -m benchmarks.nexus_capability_benchmark \
  --seed 20261016 \
  --output /tmp/nexus-capability-benchmark.json \
  --dump-cases /tmp/nexus-capability-cases.json
```

O relatório contém versão/schema, seed, versão principal e secundária do Python, commit-base, revisão de CI quando disponível, contagem de casos e SHA-256 do conjunto de tarefas e rótulos. Repetir com a mesma versão do código, Python e seed gera o mesmo taskset e as mesmas métricas. O arquivo de tarefas contém as entradas e rótulos sintéticos para inspeção independente.

Teste do contrato do benchmark:

```bash
PYTHONPATH=. python -m unittest -v test_capability_benchmark.py
```

Use `--require-full-support` para receber código de saída 2 se alguma trilha não possuir uma interface avaliável. O modo padrão registra esse limite e conclui com sucesso, pois identificar capacidade ausente é um resultado válido.

## Trilhas e interpretação

| Trilha | Componente exercitado | Métrica principal | O que não demonstra |
|---|---|---|---|
| Composição | `RealWorldModelSimulator` | acerto exato do estado em sequências de 1–2 ações e em sequências holdout de 3 ações | generalização aprendida: não há treino, aquisição de regra ou dados externos |
| Relações causais | `RealCausalReasoning.extract_causal_relations` | precisão, recall, F1 e falsos positivos em controles associativos/negados | descoberta causal ou validade das relações fora do texto sintético |
| Intervenção | `RealCounterfactualReasoningEngine.model_causal_intervention` com grafo conhecido | precisão/recall/F1 da alegação textual de efeito downstream em mudanças causais, mudanças desconectadas e no-ops | efeito numérico de `do(X=x)`, identificação causal ou previsão contrafactual correta |
| Incerteza | `RealUncertaintyQuantificationEngine` | Brier, ECE, confiabilidade por bins e monotonicidade da dispersão estimada | geração de probabilidades pelo Nexus: o benchmark fornece as probabilidades de entrada |
| Observação parcial | `RealPredictiveCodingSystem.hierarchical_prediction_cycle` | disponibilidade de uma distribuição de crenças e distinção das saídas quando o posterior-alvo muda | acurácia de posterior quando a API não retorna uma distribuição de crença; nesse caso a nota é `null` e o status é `unsupported_interface` |

### Definições operacionais

- **Composição:** a especificação de referência fixa `APPLY_PATCH`, `IGNORE_ALERT` e `WAIT` como primitivas de transição; comprimentos 1–2 são o split visto e comprimento 3 é holdout. O holdout mede reprodução de composições pelo simulador predefinido, não aprendizagem sistemática.
- **Causalidade:** a primeira tarefa distingue relações textuais positivas de associação e negação. A segunda fornece ao engine um pequeno grafo causal conhecido e verifica se a API discrimina variáveis com descendentes, variáveis isoladas e intervenções sem mudança. O teste detecta sobrealegações; não tenta inferir estrutura de dados observacionais.
- **Incerteza:** outcomes Bernoulli são amostrados de probabilidades conhecidas, com seed fixada. Brier avalia previsões probabilísticas; ECE resume a diferença entre confiança e frequência de acerto nos bins definidos. O engine recebe um vetor de probabilidades e retorna somente confiança escalar, então o relatório reconstrói a probabilidade da classe positiva a partir da classe prevista e dessa confiança. A regra estritamente própria de Brier é calculada pelo avaliador, não pelo campo heurístico `calibration_score` do módulo.
- **Observação parcial:** um POMDP sintético binário fornece observações `clear` e `alarm` com posterior exato diferente. A interface atual de codificação preditiva retorna valor/confiança hierárquicos, não uma distribuição de crença sobre estados ocultos. Portanto o benchmark registra a lacuna e não fabrica acurácia de posterior.

## Filtro de crença POMDP — protocolo v2 separado

O protocolo `nexus.pomdp-belief-benchmark.v2` mede uma operação diferente e **não altera o protocolo v1**: uma atualização Bayesiana de um passo, com estados finitos e modelo explicitamente fornecidos. A trilha de observação parcial do v1 continua registrando `unsupported_interface`, pois `hierarchical_prediction_cycle` ainda não retorna uma distribuição de crença.

```bash
PYTHONPATH=. python -m unittest -v \
  test_pomdp_belief_filter_contract.py test_pomdp_benchmark_v2.py
PYTHONPATH=. python -m benchmarks.nexus_pomdp_benchmark_v2 \
  --seed 20261016 \
  --output /tmp/nexus-pomdp-belief-v2.json \
  --dump-cases /tmp/nexus-pomdp-belief-v2-cases.json
```

O taskset determinístico contém 128 modelos sintéticos com 2–8 estados, priors, matrizes de transição por linha e likelihoods variáveis gerados pela seed. O scorer compara cada posterior com uma implementação de referência independente, mede erros absoluto médio/máximo e a taxa de casos dentro da tolerância; quatro controles inválidos verificam rejeição. O filtro aceita no máximo 128 estados, valida dimensões e probabilidades, usa log-sum-exp para manter evidência numericamente estável e rejeita evidência realmente nula.

O gate exige que todos os casos fiquem dentro de `1e-12` e que todos os controles negativos sejam rejeitados; a CLI retorna código 1 se qualquer limiar falhar. O workflow retém o relatório mesmo em falha, se já tiver sido gerado, por 14 dias. O filtro recebe o modelo completo como entrada e não aprende parâmetros, identifica estados, escolhe políticas ou demonstra desempenho em ambiente real. Comparações com o benchmark v1 não são válidas porque os protocolos medem interfaces e comportamentos diferentes.

## Fundamentos teóricos e limites de atribuição

Esses trabalhos motivam as **perguntas experimentais**, mas a execução do benchmark não valida as teorias ou resultados dos autores:

1. Lake, B. M. (2019), [*Compositional generalization through meta sequence-to-sequence learning*](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f4d0e2e7fc057a58f7ca4a391f01940a-Abstract.html), NeurIPS 32. Motivação: separar composição vista e combinação holdout. Nosso protocolo não implementa meta-aprendizagem.
2. Pearl, J. (2010), [*An Introduction to Causal Inference*](https://pmc.ncbi.nlm.nih.gov/articles/PMC2836213/). Motivação: uma intervenção altera a equação/mecanismo interveniente no SCM; mera associação textual ou uma frase genérica de “efeitos downstream” não é estimativa de efeito causal.
3. Gneiting, T. & Raftery, A. E. (2007), [*Strictly Proper Scoring Rules, Prediction, and Estimation*](https://doi.org/10.1198/016214506000001437). Motivação: pontuar previsões probabilísticas com regras próprias, incluindo o escore quadrático/Brier; a calibração é propriedade conjunta das previsões e dos outcomes.
4. Carr, S., Jansen, N. & Topcu, U. (2021), [*Task-Aware Verifiable RNN-Based Policies for Partially Observable Markov Decision Processes*](https://doi.org/10.1613/jair.1.12963), JAIR 72. Motivação: tarefas POMDP têm informação incompleta e exigem uma representação de estado de crença para comparar inferências sobre estados latentes.

## Decisões de reprodutibilidade

- Seed padrão fixa: `20261016`; não há relógio, chamada externa ou aleatoriedade global nos resultados.
- Dataset, split, rótulos, métricas e SHA-256 são gerados no próprio código versionado.
- Nenhum modelo de referência treinado é usado. O baseline uniforme `p=0.5` aparece apenas na trilha de incerteza.
- Resultados são exportados como JSON e retidos pelo workflow CI por 14 dias; não são snapshots manuais nem claims persistentes no repositório.
- O relatório é específico às APIs e tarefas atuais. Alterações de módulo, protocolo ou métrica devem incrementar/revisar o benchmark e seus testes, não comparar números incompatíveis silenciosamente.
