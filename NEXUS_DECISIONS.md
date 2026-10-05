# Nexus — Registro de decisões

## D-001 — Evolução incremental em vez de grande refatoração

- **Decisão:** priorizar patches pequenos e contratos independentes.
- **Contexto:** o núcleo é monolítico e possui alto risco de efeitos laterais.
- **Alternativas:** reescrita ampla ou modularização imediata.
- **Motivo:** mudanças menores são mais reversíveis e auditáveis.
- **Evidência:** suíte de contratos, mutation testing direcionado e CI existente.
- **Reversibilidade:** fácil a moderada.

## D-002 — Fallbacks devem ser explicitamente classificados

- **Decisão:** distinguir backend real, fallback degradado, heurística, simulação, mock e indisponibilidade.
- **Contexto:** dependências opcionais alteram comportamento.
- **Motivo:** impedir que fallback seja apresentado como capacidade completa.
- **Evidência:** contratos multimodais, telemetria GPU e busca SQL.
- **Reversibilidade:** fácil.

## D-003 — Snapshots são transporte, não durabilidade

- **Decisão:** `DeferredTaskQueue` exporta/restaura schema versionado, mas não promete persistência física, locking distribuído ou exactly-once.
- **Motivo:** manter o contrato honesto e a responsabilidade de armazenamento no consumidor.
- **Evidência:** testes de round-trip, validação atômica e documentação do README.
- **Reversibilidade:** moderada.

## D-004 — Auditoria de segurança conservadora

- **Decisão:** preservar findings LOW no relatório e registrar disposições sem suprimi-los automaticamente.
- **Motivo:** triagem contextual não equivale a correção ou clearance.
- **Evidência:** `SECURITY_TRIAGE.md`, Bandit atual e auditor AST B101.
- **Reversibilidade:** fácil.

## D-005 — Métricas observadas substituem constantes de demonstração

- **Decisão:** calcular estatísticas de swarm e memória episódica a partir de decisões e registros SQLite reais.
- **Motivo:** reduzir teatro cognitivo e aumentar observabilidade verificável.
- **Evidência:** `test_swarm_statistics_contract.py` e `test_episodic_memory_statistics_contract.py`.
- **Reversibilidade:** fácil.

## D-006 — Documentação histórica não é fonte de estado atual

- **Decisão:** manter resultados antigos, mas separar estado atual em conhecimento canônico.
- **Motivo:** auditorias históricas registraram 132, 130 e 129 findings; o manifesto CI final anterior registrava 132 LOW; após o hardening do B607, a triagem versionada registra 131 LOW sem `manual_review` residual.
- **Evidência:** comparação entre `NEXUS_AUDIT_FINAL.md`, `SECURITY_TRIAGE.md` e Bandit executado.
- **Reversibilidade:** fácil.

## D-007 — Sinais de atenção opcionais, com fallback explícito

- **Decisão:** `StimulusItem` aceita `contrast` e `motion` opcionais; `BottomUpAttention` usa sinais fornecidos e mantém valores heurísticos somente quando ausentes.
- **Contexto:** contraste e movimento eram constantes sem uma entrada observável.
- **Motivo:** aumentar observabilidade sem quebrar construtores existentes ou inventar um backend perceptual.
- **Evidência:** `test_attention_signal_contract.py` cobre sinais presentes, contexto, limites e fallback.
- **Reversibilidade:** fácil.

## D-008 — Média de sinergia calculada sobre todos os registros

- **Decisão:** `RealIntegrationOrchestrationEngine` calcula `avg_synergy_score` pela média aritmética das integrações registradas.
- **Contexto:** a atualização anterior fazia uma média recursiva sem peso, distorcendo o resultado após mais de uma integração.
- **Motivo:** preservar uma métrica estatística correta sem alterar a API pública.
- **Evidência:** `test_integration_synergy_contract.py` detecta os valores 0.9 e 0.6 e exige média 0.75.
- **Limitação:** os scores individuais continuam heurísticos baseados nos nomes dos módulos; a correção não demonstra sinergia cognitiva.
- **Reversibilidade:** fácil.

## D-009 — Entidades do grafo são identificadas por ID

- **Decisão:** re-adicionar um `entity_id` atualiza a entidade existente, não aumenta `total_entities`, e move o ID entre índices quando o tipo muda.
- **Contexto:** a implementação substituía o índice, mas acumulava a contagem de inserções e deixava o tipo antigo indexado.
- **Motivo:** alinhar estatísticas e índices com a identidade observada no grafo.
- **Evidência:** `test_knowledge_graph_contract.py` cobre duplicata, média de confiança, contagem única e reclassificação.
- **Limitação:** o grafo continua local; isso não define identidade distribuída, consistência concorrente ou conhecimento válido externamente.
- **Reversibilidade:** fácil.

## D-010 — Lista de ações vazia é uma entrada válida no MCTS

- **Decisão:** `available_actions=None` seleciona o espaço de ações padrão; `available_actions=[]` representa deliberadamente um espaço vazio.
- **Contexto:** o planejador usava truthiness e substituía uma lista vazia pelas ações padrão.
- **Motivo:** evitar que uma entrada explícita seja silenciosamente reinterpretada e manter resultados previsíveis para consumidores da API.
- **Evidência:** `test_mcts_action_contract.py` cobre ambos os caminhos e confirma árvore sem filhos para o espaço vazio.
- **Limitação:** isso corrige a semântica de entrada, mas não valida a qualidade dos rollouts ou das recompensas heurísticas do MCTS.
- **Reversibilidade:** fácil.
