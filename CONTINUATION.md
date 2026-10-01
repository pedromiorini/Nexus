# Nexus — Continuidade entre agentes

> Este arquivo é o handoff operacional do projeto. Atualize-o no mesmo commit de cada marco relevante para que outro agente Manus possa continuar sem depender do histórico da conversa.

## Estado atual

- **Branch de trabalho:** `main`
- **Remoto:** `https://github.com/pedromiorini/Nexus`
- **Último estado remoto conhecido:** `0f4f113` (`docs: record remote CI validation result`)
- **Commit local pendente:** o commit mais recente contém `ci: enforce critical coverage and targeted mutation gates`; ele altera apenas o workflow e exige escopo GitHub `workflow` para publicação.
- **Fase:** endurecimento de qualidade, operacionalização de diagnósticos e triagem de segurança.
- **Escopo real:** protótipo experimental Python de roteamento, filas, telemetria e contratos de diagnóstico. Não reivindicar AGI, ASI, consciência ou autonomia geral.

## O que já está implementado

- Reprocessamento de tarefas adiadas no `CentralRouter`, com preservação de contexto, callbacks e telemetria.
- Auditoria persistente SQLite e análise longitudinal no bridge Vita.
- Exportação/validação do contrato `nexus.recovery.diagnostics.v1`.
- `tools/diagnostics_dashboard.py`: dashboard CLI determinístico que valida o snapshot antes de renderizar severidade, pausas, recuperação e eventos observáveis.
- `tools/security_triage.py`: inventário conservador dos achados Bandit; não suprime nem marca achados como corrigidos automaticamente.
- `SECURITY_TRIAGE.md`: relatório versionado dos achados atuais e suas disposições de revisão.
- Auditoria de realidade em `tools/reality_audit.py`.
- Testes de contrato, regressão, Hypothesis, Bandit e cobertura no workflow `.github/workflows/nexus-contract-gate.yml`.
- Mutation testing direcionado em fila, Vita e schema.

## Trabalho desta rodada

1. Manter este arquivo versionado e atualizá-lo após cada marco.
2. Publicar dashboard, triagem, relatório e testes sem incluir o workflow pendente.
3. Publicar o workflow quando a credencial GitHub tiver escopo `workflow`.
4. Fazer triagem futura dos achados Bandit sem mascarar riscos reais.

## Decisões e limites

- Thresholds críticos: fila ≥90%; bridge Vita ≥70%; medidos com `coverage.py`.
- Mutation testing é direcionado e não muta o núcleo monolítico inteiro.
- O dashboard só exibe métricas observáveis e rejeita snapshots inválidos; não infere capacidade cognitiva.
- A triagem Bandit é classificação operacional, não autorização de supressão; qualquer correção exige revisão e testes.
- Relatórios gerados (`coverage.xml`, JSON de mutation, SQLite temporário) não devem ser commitados; `SECURITY_TRIAGE.md` é uma evidência textual reproduzível.
- Não expor tokens, credenciais ou conteúdo de `.env` em commits, logs ou handoffs.

## Comandos de validação

```bash
export PYTHONPATH=.
python -m unittest -v test_recovery_diagnostics_contract.py test_deferred_task_queue.py test_reality_audit.py test_deferred_task_properties.py test_security_triage.py
coverage run --branch -m unittest test_recovery_diagnostics_contract.py test_deferred_task_queue.py test_reality_audit.py test_deferred_task_properties.py test_security_triage.py
coverage report --include='core/deferred_task_queue.py,vita/nexus_constitutional_bridge_v3.py'
python tools/targeted_mutation.py
python tools/reality_audit.py
python core/constitutional_brain.py
bandit -r core vita tools -f json -o bandit-report.json || true
python tools/security_triage.py bandit-report.json SECURITY_TRIAGE.md
```

Para renderizar um snapshot exportado:

```bash
python tools/diagnostics_dashboard.py snapshot.json
# ou
cat snapshot.json | python tools/diagnostics_dashboard.py -
```

## Resultado da última validação

- **38 testes unitários/property/contrato/triagem:** passaram.
- **Cobertura branch dos módulos críticos:** `core/deferred_task_queue.py` **92,56%** (threshold **90%**); `vita/nexus_constitutional_bridge_v3.py` **70,69%** (threshold **70%**); total crítico **78%**.
- **Mutation testing direcionado:** **10/10 mutações mortas, 0 sobreviventes, 100%**, cobrindo fila, telemetria Vita e validação do schema.
- **Dashboard:** renderização e rejeição de payload inválido cobertas por testes.
- **Triagem Bandit:** **133 achados preservados e classificados**, incluindo o B608 de SQL dinâmico como revisão prioritária; nenhum achado foi suprimido.
- **CI remoto:** workflow `Nexus Contract Gate`, run `36817436102` para `bf87fc7`, terminou em **success** em 27 segundos. Todos os passos passaram.
- **Avisos do CI:** depreciação futura do Node.js 20 nas actions atuais e migração futura de `ubuntu-latest` para Ubuntu 26; não bloquearam o run.
- **Auditoria de realidade:** executada; continua identificando alegações não comprovadas e riscos conhecidos, sem elevar claims.
- **Suíte integrada:** passou, mas seus banners são demos internas e não evidência independente de capacidade cognitiva.
- **Publicação parcial:** testes, mutation harness, dashboard, triagem e handoff serão publicados; alterações de workflow permanecem locais por exigirem escopo `workflow`.

## Próximo agente

1. Ler este arquivo e `git status --short --branch`.
2. Verificar se há mutações/artefatos temporários pendentes.
3. Reexecutar os comandos de validação acima.
4. Se modificar código, atualizar este arquivo no mesmo commit e registrar o novo SHA.
5. Confirmar `git status` limpo e `git branch -vv` sincronizado antes de encerrar.
