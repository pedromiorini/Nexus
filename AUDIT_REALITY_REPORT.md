# Nexus — Auditoria conservadora de realidade

> Este relatório separa o que foi observado no código e nos testes do que é apenas alegação documental. Nomes de módulos, banners e asserts de demonstração não são tratados como prova de AGI, ASI, consciência ou prontidão de produção.

## Evidência estrutural

O núcleo analisado possui **26500 linhas**, **267 classes** e **842 funções**. Foram encontrados **2** métodos compostos apenas por `pass`, **66** funções com retorno constante simples e **30** marcadores de testes embutidos.

| Área | Resultado | Interpretação |
|---|---:|---|
| Testes embutidos marcados | 30 | Evidência de demos internas, não de capacidade geral |
| Funções somente com `pass` | 2 | Lacunas explícitas que exigem revisão |
| Funções com retorno constante | 66 | Possíveis heurísticas, stubs ou simplificações |
| Dependências opcionais sinalizadas | 2 | Fallbacks podem alterar o comportamento |

## Alegações não comprovadas

- **AGI/ASI**: explicitly_disclaimed. A documentação nega explicitamente que o projeto reivindique essa capacidade.
- **consciência/autoconsciência**: unsubstantiated_by_software_tests. A documentação contém a alegação, mas os testes do repositório não constituem evidência suficiente dessa capacidade.

## Referências quebradas ou verificadas

- `core/constitutional_brain.py`: existe.
- `core/deferred_task_queue.py`: existe.
- `core/vram_defense_guard.py`: existe.
- `vita/nexus_constitutional_bridge_v3.py`: existe.
- `test_recovery_diagnostics_contract.py`: existe.
- `test_deferred_task_queue.py`: existe.
- `tools/reality_audit.py`: existe.
- `test_recovery_diagnostics_contract.py`: existe.
- `test_deferred_task_queue.py`: existe.
- `core/constitutional_brain.py`: existe.
- `tools/reality_audit.py`: existe.
- `tools/reality_audit.py`: existe.

## Riscos prioritários

- README contains capability claims stronger than the measured evidence.
- Embedded demonstration tests are not independent contract or adversarial tests.
- Constant-return functions and pass-only functions require manual review.
- Optional dependency fallbacks can change behavior and guarantees.

## Próximas ações

- Maintain independent contract tests for critical interfaces.
- Add property-based tests for queue invariants.
- Run Bandit and targeted mutation testing in CI.
- Replace absolute capability language with evidence-qualified documentation.
