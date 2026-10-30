# Nexus — Auditoria conservadora de realidade

> Este relatório separa o que foi observado no código e nos testes do que é apenas alegação documental. Nomes de módulos, banners e asserts de demonstração não são tratados como prova de AGI, ASI, consciência ou prontidão de produção.

## Evidência estrutural

O núcleo analisado possui **23327 linhas**, **218 classes** e **698 funções**. Foram encontrados **0** métodos compostos apenas por `pass`, **1** funções com retorno constante simples e **31** marcadores de testes embutidos.

| Área | Resultado | Interpretação |
|---|---:|---|
| Testes embutidos marcados | 31 | Evidência de demos internas, não de capacidade geral |
| Funções somente com `pass` | 0 | Lacunas explícitas que exigem revisão |
| Funções com retorno constante (estrito: só literais) | 1 | Possíveis heurísticas, stubs ou simplificações |
| Funções que retornam dict/list/tuple literal com valores dinâmicos | 47 | Em geral getters e estatísticas; não são stubs por si só |
| Padrões proibidos de alegação no código (gate: zero) | 0 | Cobre só as frases listadas em CODE_OVERCLAIMS; não significa ausência de linguagem de capacidade |
| Frases fortes de capacidade no código (inventário; só pode diminuir) | 51 | Ênfase "REAL <Capacidade>", "REAL e FUNCIONAL", "100% funcional"; limite em test_reality_audit.py |
| Terminologia científica no código (informativo) | 233 | Nomes de constructos modelados; não são alegações por si só |
| Classes com prefixo Real (identificadores) | 63 | Nome histórico; renomear exige alias de compatibilidade |
| Módulos extraídos em core/cognitivo (inventário de alegações inclui estes) | 16 (5617 linhas) | Fora das contagens do núcleo; cobertos pelos mesmos padrões de alegação |
| Dependências opcionais sinalizadas | 2 | Fallbacks podem alterar o comportamento |

## Alegações (README e código-fonte)

- **AGI/ASI**: explicitly_disclaimed. A documentação nega explicitamente que o projeto reivindique essa capacidade.
- **consciência/autoconsciência**: explicitly_disclaimed. A documentação nega explicitamente que o projeto reivindique essa capacidade.
- Nenhuma alegação assertiva sem ressalva foi encontrada no README nem no código-fonte.

## Referências quebradas ou verificadas

- `core/constitutional_brain.py`: existe.
- `core/deferred_task_queue.py`: existe.
- `core/vram_defense_guard.py`: existe.
- `vita/nexus_constitutional_bridge_v3.py`: existe.
- `test_recovery_diagnostics_contract.py`: existe.
- `test_deferred_task_queue.py`: existe.
- `tools/reality_audit.py`: existe.
- `test_multimodal_backend_contract.py`: existe.
- `test_swarm_statistics_contract.py`: existe.
- `test_episodic_memory_statistics_contract.py`: existe.
- `test_attention_signal_contract.py`: existe.
- `test_integration_synergy_contract.py`: existe.
- `test_knowledge_graph_contract.py`: existe.
- `test_mcts_action_contract.py`: existe.
- `core/constitutional_brain.py`: existe.
- `tools/reality_audit.py`: existe.
- `test_deferred_task_snapshot.py`: existe.
- `test_memory_sql_contract.py`: existe.
- `test_jailbreak_detector_contract.py`: existe.
- `tools/reality_audit.py`: existe.
- `tools/reality_audit.py`: existe.
- `tools/security_triage.py`: existe.
- `test_reality_audit.py`: existe.

## Riscos prioritários

- Embedded demonstration tests are not independent contract or adversarial tests.
- Constant-return functions and pass-only functions require manual review.
- Optional dependency fallbacks can change behavior and guarantees.

## Próximas ações

- Maintain independent contract tests for critical interfaces.
- Add property-based tests for queue invariants.
- Run Bandit and targeted mutation testing in CI.
- Replace absolute capability language with evidence-qualified documentation.
