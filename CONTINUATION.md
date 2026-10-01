# Nexus — Continuidade entre agentes

> Este arquivo é o handoff operacional do projeto. Atualize-o no mesmo commit de cada marco relevante para que outro agente Manus possa continuar sem depender do histórico da conversa.

## Estado atual

- **Branch de trabalho:** `main`
- **Remoto:** `https://github.com/pedromiorini/Nexus`
- **Último estado remoto conhecido antes desta rodada:** `7f941db` (`audit: verify capabilities and harden quality gates`)
- **Commit local desta rodada:** o commit mais recente da branch local (`git log -1`) contém `ci: enforce critical coverage and targeted mutation gates`.
- **Fase:** endurecimento de qualidade; cobertura crítica e mutation testing direcionado.
- **Escopo real:** protótipo experimental Python de roteamento, filas, telemetria e contratos de diagnóstico. Não reivindicar AGI, ASI, consciência ou autonomia geral.

## O que já está implementado

- Reprocessamento de tarefas adiadas no `CentralRouter`, com preservação de contexto, callbacks e telemetria.
- Auditoria persistente SQLite e análise longitudinal no bridge Vita.
- Exportação/validação do contrato `nexus.recovery.diagnostics.v1`.
- Auditoria de realidade em `tools/reality_audit.py`.
- Testes de contrato, regressão, Hypothesis, Bandit e cobertura no workflow `.github/workflows/nexus-contract-gate.yml`.
- Testes adicionais da fila e do contrato Vita restaurados/adicionados nesta rodada.

## Trabalho desta rodada

1. Manter este arquivo versionado e atualizá-lo após cada marco.
2. Definir thresholds graduais de cobertura para `core/deferred_task_queue.py` e `vita/nexus_constitutional_bridge_v3.py`.
3. Expandir `tools/targeted_mutation.py` para fila, telemetria Vita e validação de schema.
4. Executar suíte completa, mutation testing, auditoria e validação local do YAML.
5. Commitar e sincronizar a branch `main`; registrar SHA e resultados abaixo.

## Decisões e limites

- Thresholds devem ser mensuráveis pelo `coverage.py`, inicialmente conservadores e explícitos; subir valores somente após medir a suíte no runner.
- Mutation testing será direcionado e sem mutar o núcleo monolítico inteiro.
- O harness deve restaurar arquivos mesmo em falha/interrupção tratável e falhar se houver mutação aplicável sobrevivente ou não aplicada.
- Relatórios gerados (`coverage.xml`, JSON de mutation, SQLite temporário) não devem ser commitados.
- Não expor tokens, credenciais ou conteúdo de `.env` em commits, logs ou handoffs.

## Comandos de validação

```bash
export PYTHONPATH=.
python -m unittest -v test_recovery_diagnostics_contract.py test_deferred_task_queue.py test_reality_audit.py test_deferred_task_properties.py
coverage run --branch -m unittest test_recovery_diagnostics_contract.py test_deferred_task_queue.py test_reality_audit.py test_deferred_task_properties.py
coverage report --include='core/deferred_task_queue.py,vita/nexus_constitutional_bridge_v3.py'
python tools/targeted_mutation.py
python tools/reality_audit.py
python core/constitutional_brain.py
```

## Resultado da última validação

- **33 testes unitários/property:** passaram.
- **Cobertura branch dos módulos críticos:** `core/deferred_task_queue.py` **92,56%** (threshold **90%**); `vita/nexus_constitutional_bridge_v3.py` **70,69%** (threshold **70%**); total crítico **78%**.
- **Mutation testing direcionado:** **10/10 mutações mortas, 0 sobreviventes, 100%**, cobrindo fila, telemetria Vita e validação do schema.
- **Workflow YAML:** parseado com sucesso.
- **Auditoria de realidade:** executada; continua identificando alegações não comprovadas e riscos conhecidos, sem elevar claims.
- **Suíte integrada (`python core/constitutional_brain.py`):** passou, mas seus banners são demos internas e não evidência independente de capacidade cognitiva.
- **Bandit:** relatório gerado; 133 achados existentes/legados permanecem não bloqueantes no workflow (`|| true`) e exigem triagem futura.
- **Ambiente local:** dependências de desenvolvimento instaladas no usuário (`coverage`, `hypothesis`, `bandit`); CI instala as dependências em runner limpo.
- **Publicação parcial:** testes, mutation harness e este handoff foram publicados; a alteração de `.github/workflows/nexus-contract-gate.yml` permanece local por exigir escopo `workflow`.

## Próximo agente

1. Ler este arquivo e `git status --short --branch`.
2. Verificar se há mutações/artefatos temporários pendentes.
3. Reexecutar os comandos de validação acima.
4. Se modificar código, atualizar este arquivo no mesmo commit e registrar o novo SHA.
5. Confirmar `git status` limpo e `git branch -vv` sincronizado antes de encerrar.
