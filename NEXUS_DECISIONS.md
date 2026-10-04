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
- **Motivo:** auditorias anteriores registram 132 e 130 findings; a execução atual registra 129 LOW.
- **Evidência:** comparação entre `NEXUS_AUDIT_FINAL.md`, `SECURITY_TRIAGE.md` e Bandit executado.
- **Reversibilidade:** fácil.
