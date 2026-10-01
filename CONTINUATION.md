# Nexus — Continuidade entre agentes

> Handoff operacional versionado para continuidade entre agentes Manus.

## Estado atual

- **Branch:** `main`
- **Base remota sincronizada:** `origin/main` em `477a59a` (`ci: integrate continuity quality gates and queue snapshots`)
- **Marco publicado desta rodada:** integração dos artefatos remotos de qualidade, contrato de snapshot da fila e gates de cobertura/mutation.
- **CI remoto:** run `36819282235` terminou com `success`; todos os passos passaram.
- **Escopo real:** protótipo experimental Python de roteamento, filas, telemetria e contratos de diagnóstico. Não reivindicar AGI, ASI, consciência ou autonomia geral.

## O que está implementado

- Reprocessamento de tarefas adiadas no `CentralRouter`, preservando contexto, callbacks e telemetria.
- Auditoria persistente SQLite e análise longitudinal no bridge Vita.
- Exportação/validação do contrato `nexus.recovery.diagnostics.v1`.
- `tools/diagnostics_dashboard.py`: dashboard CLI determinístico que valida snapshots antes de renderizar métricas observáveis.
- `tools/security_triage.py` e `SECURITY_TRIAGE.md`: inventário conservador dos achados Bandit, sem supressão automática.
- `test_memory_sql_contract.py`: teste adversarial para a fronteira SQL do fallback parametrizado da memória.
- `tools/targeted_mutation.py`: mutation testing direcionado para fila, Vita e schema.
- `core/deferred_task_queue.py`: snapshot versionado `nexus.deferred_task_queue.v1`, exportação JSON/objeto, restauração atômica, validação de tipos/capacidade/IDs e opção `replace=True`.
- `test_deferred_task_snapshot.py`: contrato de round-trip, ordem, atomicidade, limites, tipos inválidos e ausência de execução durante restauração.
- Auditoria de realidade em `tools/reality_audit.py`.
- Hardening do fallback SQL em `RealHierarchicalMemory`: consultas estáticas parametrizadas por termo, sem montagem dinâmica de placeholders.

## Gates locais desta rodada

- **51 testes unitários/property/contrato/triagem/SQL:** passaram.
- **Cobertura branch:** `core/deferred_task_queue.py` **98%** (limiar 90%); `vita/nexus_constitutional_bridge_v3.py` **71%** (limiar 70%).
- **Mutation testing:** **10/10 mutações mortas, 0 sobreviventes, 100%**.
- **Bandit:** **132 achados LOW** preservados e classificados; o B608 foi removido após a refatoração, sem suprimir achados.
- **Auditoria de realidade:** executada sem elevar claims cognitivos.
- **Dashboard:** snapshot válido renderizado; payload inválido rejeitado pelos testes.
- **Compilação Python:** passou para módulos, ferramentas e testes alterados.

## Gates do workflow local

O workflow `.github/workflows/nexus-contract-gate.yml` foi ampliado para:

- compilar os novos testes e utilitários;
- executar os contratos SQL, segurança e snapshot;
- exigir fila >=90% e bridge Vita >=70%;
- executar mutation testing direcionado;
- gerar e reter o relatório de triagem Bandit junto ao JSON.
- validar a busca SQL multi-termo e o limite no teste adversarial de memória.

O workflow endurecido está publicado em `477a59a`. O CI registra apenas avisos de migração futura do Node.js 20 nas actions e do rótulo `ubuntu-latest` para Ubuntu 26; nenhum aviso bloqueou a execução.

## Limites e decisões

- Snapshot é contrato de transporte; não é persistência física em disco, locking distribuído ou garantia exactly-once.
- Mutation testing é direcionado e não muta o núcleo monolítico inteiro.
- O dashboard exibe apenas métricas observáveis e não infere capacidade cognitiva.
- O antigo B608 foi eliminado por consultas estáticas parametrizadas por termo; revisão manual de confiança, concorrência e núcleo legado continua aberta.
- Relatórios gerados (`coverage.xml`, JSON de mutation, SQLite temporário e saídas Bandit) não devem ser commitados.
- Não expor tokens, credenciais ou conteúdo de `.env`.

## Próxima ação

Manter a triagem Bandit aberta e executar mutation testing direcionado antes de qualquer alteração no núcleo monolítico. Não tratar os banners da suíte integrada como evidência independente de capacidade cognitiva.
