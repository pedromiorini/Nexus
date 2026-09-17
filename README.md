# Nexus Constitutional

[![CI](https://github.com/pedromiorini/Nexus/actions/workflows/nexus-contract-gate.yml/badge.svg)](https://github.com/pedromiorini/Nexus/actions/workflows/nexus-contract-gate.yml)

## Escopo real

O Nexus Constitutional é um **protótipo experimental de arquitetura neuro-simbólica em Python**. O repositório contém componentes de roteamento, governança, memória, filas de tarefas adiadas, telemetria, auditoria persistente e testes de demonstração. Os componentes são úteis para pesquisa de engenharia de software e orquestração cognitiva, mas o projeto **não reivindica AGI, ASI, consciência, autoconsciência ou autonomia geral**.

> Nomes de módulos, banners de execução e testes de demonstração não constituem prova de inteligência geral, consciência ou prontidão para produção.

## Evidências atualmente verificadas

| Área | Evidência | Limite da afirmação |
|---|---|---|
| Núcleo constitucional | A suíte integrada local executa 29 verificações embutidas | São verificações do próprio projeto, não uma avaliação externa de capacidade geral |
| Fila de tarefas adiadas | 14 testes unitários e de integração | Cobre contratos da fila e reprocessamento, não carga distribuída de produção |
| Diagnóstico de recuperação | Exportação versionada `nexus.recovery.diagnostics.v1` e validador formal | O schema garante estrutura, não a correção de decisões cognitivas |
| Auditoria Vita | Eventos, transições e métricas persistidos em SQLite | A análise é local e limitada ao histórico disponível |
| CI | Workflow executa compilação, contrato, regressão, suíte integrada e cobertura | A execução depende do ambiente e das dependências instaladas no runner |
| Cobertura | Relatório de branches é retido como artefato por 14 dias | Cobertura não substitui testes de propriedades, segurança ou mutation testing |

## Componentes principais

`core/constitutional_brain.py` concentra o núcleo legado e o `CentralRouter`. `core/deferred_task_queue.py` implementa prioridade, retry, descarte e callbacks do ciclo de vida. `core/vram_defense_guard.py` fornece decisões de pressão de memória quando os recursos de telemetria estão disponíveis. `vita/nexus_constitutional_bridge_v3.py` registra telemetria, transições, auditoria persistente, análise de recuperação e exportação estruturada.

O arquivo `test_recovery_diagnostics_contract.py` mantém testes independentes do contrato público de observabilidade. O arquivo `test_deferred_task_queue.py` cobre a fila e a integração do reprocessamento. O relatório `AUDIT_REALITY_REPORT.md` é gerado pelo auditor conservador em `tools/reality_audit.py`.

## Como executar

Use Python 3.11 ou versão compatível, defina o caminho de importação e execute:

```bash
export PYTHONPATH=.
python3 -m unittest -v test_recovery_diagnostics_contract.py test_deferred_task_queue.py
python3 core/constitutional_brain.py
python3 tools/reality_audit.py
```

Dependências opcionais podem ativar caminhos adicionais de telemetria ou recuperação. Quando não estão disponíveis, o código pode usar fallbacks; portanto, os resultados devem registrar quais dependências estavam presentes.

## Contrato de diagnóstico

A ponte Vita exporta snapshots com schema versionado:

```python
snapshot = bridge.export_recovery_diagnostics(as_json=True)
validation = bridge.validate_recovery_diagnostics(snapshot)
```

O diagnóstico inclui severidade, eventos analisados, pausas, recuperações, taxa de recuperação, duração de pausas, recorrência crítica e eventos de auditoria. O `CentralRouter` expõe os mesmos contratos por meio de `get_recovery_diagnostics()`, `export_recovery_diagnostics()` e `validate_recovery_diagnostics()`.

## Limitações conhecidas

O núcleo ainda é monolítico e contém componentes experimentais, heurísticas, retornos constantes e caminhos de fallback que exigem revisão individual. A presença de uma classe ou método com nome cognitivo não demonstra a capacidade descrita pelo nome. A suíte integrada é valiosa como teste de não regressão, mas não é uma avaliação independente, adversarial ou científica de inteligência.

As próximas prioridades são elevar a cobertura dos componentes críticos, adicionar testes baseados em propriedades para invariantes da fila e do schema, executar análise estática de segurança e aplicar mutation testing incremental. Só depois de obter evidências específicas deve-se elevar qualquer alegação de capacidade.

## Auditoria de realidade

Execute `python3 tools/reality_audit.py` para gerar `AUDIT_REALITY_REPORT.md` e `audit_reality.json`. O auditor compara alegações documentais, caminhos referenciados, estrutura AST, funções somente com `pass`, retornos constantes, testes embutidos e dependências opcionais. Ele é deliberadamente conservador: sinaliza alegações que precisam de evidência, mas não tenta concluir capacidades cognitivas a partir de nomes ou banners.
