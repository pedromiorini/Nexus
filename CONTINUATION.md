# Nexus — Continuidade entre agentes

> Handoff operacional versionado para continuidade entre agentes Manus.

## Estado atual

- **Branch:** `main`
- **Base remota sincronizada:** `origin/main` em `537cdbe` (`quality: add optional multimodal backend contracts`).
- **Marco desta rodada:** processadores multimodais aceitam backends opcionais para visão e áudio, com fallbacks explícitos e contrato independente; CI remoto `37096821481` terminou com `success`.
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
- Hardening de `_generate_correction`: captura explícita de `ValueError`/`IndexError`, com preservação do conteúdo original em descrições inválidas.
- Revisão dos 16 achados B311: usos classificados como simulação/heurística, sem reutilização autorizada para segredos ou decisões de segurança.
- Revisão dos 111 achados B101 restantes: todos estão no bloco `__main__` demonstrativo e permanecem fora dos contratos independentes de produção.
- `tools/embedded_assert_audit.py`: guardrail AST que impede B101 e qualquer `assert` fora do bloco `__main__` enquanto a migração é incremental.
- `test_central_router_contract.py`: contrato independente para contagem de requisições, cache hit rate e latência média.
- `tools/implementation_gap_audit.py`: inventário AST conservador de funções pass-only e `NotImplementedError`, sem tratar lacunas como implementadas.
- `test_explicit_gap_contract.py`: contratos independentes para propagação de erro e registro de adaptação simulada.
- `test_working_memory_contract.py`: contrato independente para alocação de atenção local e delegação declarada.
- `test_hardware_telemetry_contract.py`: contrato de fallback GPU conservador quando não há backend disponível.
- `test_counterfactual_contract.py`: contratos de world model, causalidade e fallback heurístico.
- `test_multimodal_backend_contract.py`: contratos de backends opcionais de visão/áudio e fallbacks de demonstração.

## Gates locais desta rodada

- **83 testes unitários/property/contrato/triagem/SQL/VRAM/guardrails:** passaram na suíte integral.
- **Cobertura branch:** `core/deferred_task_queue.py` **98%** (limiar 90%); `vita/nexus_constitutional_bridge_v3.py` **71%** (limiar 70%); total dos dois módulos **85%**.
- **Mutation testing:** **10/10 mutações mortas, 0 sobreviventes, 100%**.
- **Bandit:** **129 achados LOW** preservados e classificados; B110 e B608 foram removidos por correções de código, os 16 B311 permanecem sob `simulation_only_random_review` e os 111 B101 restantes sob `embedded_demo_assert_review`.
- **Auditoria de realidade:** executada sem elevar claims cognitivos.
- **Dashboard:** snapshot válido renderizado; payload inválido rejeitado pelos testes.
- **Compilação Python:** passou para módulos, ferramentas e testes alterados.
- **CI alinhado nesta rodada:** `test_vram_defense_guard.py` agora é compilado, incluído na medição de cobertura e executado explicitamente pelo workflow.
- **CI modernizado nesta rodada:** runner fixado em `ubuntu-24.04`; `checkout@v5`, `setup-python@v6` e `upload-artifact@v7` removem a dependência das versões legadas que geravam avisos de Node.js 20.
- **Auditor B101:** 111 achados restantes, `outside_main_block: []`, `assert_lines_outside_main: []`, bloco detectado em `24668–26510`; compilação de módulos, ferramentas e testes passou.
- **Inventário de lacunas:** propagação, adaptação e atualização gerativa registram contratos mínimos observáveis; ferramenta registrada sem executor retorna falha estruturada. O inventário AST agora registra `pass_only_count=0` e `not_implemented_count=0`.
- **Telemetria GPU:** `monitor_hardware` tenta NVML, depois memória reservada CUDA como proxy; sem backend retorna `0.0` sem afirmar disponibilidade.
- **Contrafactuais:** `_predict_outcome` usa `world_model.predict_outcome` somente quando o protocolo existe; `_build_causal_chain` usa `extract_causal_relations` somente quando disponível.
- **Multimodal:** `VisionProcessor` usa `detect_objects`/`analyze_scene` do backend opcional; `AudioProcessor` usa `transcribe_speech`; sem backend, os fallbacks permanecem explicitamente demonstrativos.
- **Auditoria AST:** 0 funções somente com `pass`, 0 `NotImplementedError` explícitos e 63 retornos constantes simples restantes.

## Gates do workflow local

O workflow `.github/workflows/nexus-contract-gate.yml` foi ampliado para:

- compilar os novos testes e utilitários;
- executar os contratos SQL, segurança e snapshot;
- exigir fila >=90% e bridge Vita >=70%;
- executar mutation testing direcionado;
- gerar e reter o relatório de triagem Bandit junto ao JSON.
- validar a busca SQL multi-termo e o limite no teste adversarial de memória.
- validar o fallback original do motor de correção para descrições malformadas.
- preservar os B311 no relatório, com disposição explícita de simulação não criptográfica.
- preservar os B101 no relatório, com disposição explícita de asserts do demo embutido.
- executar o auditor AST e falhar se um B101 ou qualquer `assert` aparecer fora do bloco demonstrativo.
- executar o contrato independente do CentralRouter junto com a suíte expandida.
- executar o contrato multimodal para backends opcionais de visão/áudio junto com a suíte expandida.

O workflow endurecido está publicado em `537cdbe`; as actions estão em versões Node24 e o runner está fixado em `ubuntu-24.04`.

## Limites e decisões

- Snapshot é contrato de transporte; não é persistência física em disco, locking distribuído ou garantia exactly-once.
- Mutation testing é direcionado e não muta o núcleo monolítico inteiro.
- O dashboard exibe apenas métricas observáveis e não infere capacidade cognitiva.
- O antigo B608 foi eliminado por consultas estáticas parametrizadas por termo; revisão manual de confiança, concorrência e núcleo legado continua aberta.
- Os antigos B110 foram eliminados com exceções estreitas; falhas inesperadas continuam exigindo revisão operacional.
- Os B311 não foram suprimidos nem convertidos em aleatoriedade criptográfica; são adequados apenas para os caminhos de simulação revisados.
- Os B101 não foram removidos em massa; a migração para testes isolados deve ser incremental e preservar cobertura comportamental.
- O auditor B101 é um guardrail de localização, não uma liberação de segurança nem substituto dos testes isolados.
- A verificação AST direta reduz dependência do Bandit, mas também não substitui a migração dos asserts para contratos independentes.
- O primeiro assert migrado preserva a falha explícita da demonstração; a garantia comportamental agora vive em `test_central_router_contract.py`.
- Relatórios gerados (`coverage.xml`, JSON de mutation, SQLite temporário e saídas Bandit) não devem ser commitados.
- Não expor tokens, credenciais ou conteúdo de `.env`.

## Próxima ação

Manter a triagem Bandit aberta e executar mutation testing direcionado antes de qualquer alteração no núcleo monolítico. Não tratar os banners da suíte integrada como evidência independente de capacidade cognitiva.
