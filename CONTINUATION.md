# Nexus — Continuidade entre agentes

> Handoff operacional versionado para continuidade entre agentes Manus.

## Estado atual

- **Branch:** `main`
- **Base remota sincronizada:** `origin/main` em `0bfec12` (`test: cover memory SQL fallback injection boundary`)
- **Marco local desta rodada:** integração dos artefatos remotos de qualidade, contrato de snapshot da fila e gates locais de cobertura/mutation.
- **Publicação:** o workflow atualizado ainda depende da tentativa de push com escopo GitHub `workflow`.
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

## Gates locais desta rodada

- **50 testes unitários/property/contrato/triagem/SQL:** passaram.
- **Cobertura branch:** `core/deferred_task_queue.py` **98%** (limiar 90%); `vita/nexus_constitutional_bridge_v3.py` **71%** (limiar 70%).
- **Mutation testing:** **10/10 mutações mortas, 0 sobreviventes, 100%**.
- **Bandit:** **133 achados** preservados e classificados: 132 LOW, 1 MEDIUM; B608 permanece revisão prioritária.
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

O workflow remoto em `0bfec12` ainda não contém essas alterações até que a publicação seja autorizada pelo escopo da credencial.

## Limites e decisões

- Snapshot é contrato de transporte; não é persistência física em disco, locking distribuído ou garantia exactly-once.
- Mutation testing é direcionado e não muta o núcleo monolítico inteiro.
- O dashboard exibe apenas métricas observáveis e não infere capacidade cognitiva.
- A concatenação marcada pelo B608 monta somente placeholders; valores e limites permanecem parametrizados, mas a revisão manual continua aberta.
- Relatórios gerados (`coverage.xml`, JSON de mutation, SQLite temporário e saídas Bandit) não devem ser commitados.
- Não expor tokens, credenciais ou conteúdo de `.env`.

## Próxima ação

Após publicar o commit local, verificar o run do workflow remoto. Se o push for rejeitado por escopo `workflow`, manter o commit local e solicitar uma credencial GitHub com esse escopo antes de publicar o workflow.
