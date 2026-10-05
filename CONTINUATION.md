# Nexus — Continuidade entre agentes

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 113
bandit_low: 131
-->

> Handoff operacional versionado para continuidade entre agentes Manus.

## Estado atual

- **Branch:** `main`
- **Base remota sincronizada:** `origin/main` contém o gerador de manifesto e a reconciliação documental; use o artefato `nexus-state-manifest` do CI para o SHA exato verificado.
- **Marco desta rodada:** `STATE_MANIFEST.json` passou a ser gerado e retido pelo CI, eliminando a dependência de snapshots manuais para commit, testes, segurança e mutation.
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
- `test_swarm_statistics_contract.py`: contrato de médias observadas do swarm e rejeição de deliberação sem agentes.
- `test_episodic_memory_statistics_contract.py`: contrato de contagem de vínculos, média de importância e fechamento de episódios.
- `test_attention_signal_contract.py`: contrato de sinais opcionais de contraste/movimento e fallbacks heurísticos da atenção.
- `test_integration_synergy_contract.py`: contrato da média aritmética de sinergia sobre todas as integrações registradas.
- `test_knowledge_graph_contract.py`: contrato de identidade única, média de confiança e reclassificação de tipo no grafo.
- `test_sensorimotor_multimodal_contract.py`: contrato de processadores multimodais opcionais e fallback sensorial.
- `test_consensus_application_contract.py`: contratos de parâmetros aceitos, decisões auditadas e rejeição sem mutação.
- `test_shadow_clone_reintegration_contract.py`: contrato de ingestão opcional no bootstrap e fallback honesto.
- `test_reminiscence_bump_contract.py`: contrato de tipicidade e atipicidade do pico etário.
- `NEXUS_CONSTITUTION.md`: limites permanentes e regras epistemológicas do projeto.
- `NEXUS_KNOWLEDGE.md`: fontes de verdade e estado operacional canônico.
- `NEXUS_ARCHITECTURE.md`: mapa de responsabilidades e fronteiras observadas.
- `NEXUS_EVIDENCE.md`: ledger de fatos, evidências e limitações.
- `NEXUS_DECISIONS.md`: decisões arquiteturais com contexto e evidência.
- `NEXUS_ROADMAP.md`: prioridades técnicas orientadas por risco.
- `NEXUS_BOOTSTRAP_AUDIT.md`: auditoria de bootstrap e reconciliação documental.
- `tools/state_manifest.py`: gerador determinístico do manifesto factual por execução.
- `test_state_manifest.py`: contrato do schema básico e da proveniência do manifesto.
- `test_mcts_action_contract.py`: contrato da distinção entre ações padrão (`None`) e espaço de ações vazio (`[]`).
- `tools/state_consistency_audit.py`: compara claims atuais estruturados com o manifesto factual e ignora histórico fora do bloco.
- `test_state_consistency_audit.py`: contratos de sincronização, divergência, ausência de marcador e manifesto inválido.
- `tools/test_summary.py`: executa a suíte final e registra `executed`, `passed`, `failed` e `skipped` para o manifesto.

## Gates locais desta rodada

- **113 testes unitários/property/contrato/triagem/SQL/VRAM/guardrails:** passaram na suíte integral local; o contrato MCTS cobre a distinção entre ações padrão e espaço vazio.
- **Cobertura branch:** `core/deferred_task_queue.py` **98%** (limiar 90%); `vita/nexus_constitutional_bridge_v3.py` **71%** (limiar 70%); total dos dois módulos **85%**.
- **Mutation testing:** **10/10 mutações mortas, 0 sobreviventes, 100%**.
- **Bandit:** **131 achados LOW** após o hardening do B607 em `tools/state_manifest.py`; a triagem classifica 16 B311 como `simulation_only_random_review`, 111 B101 como `embedded_demo_assert_review` e 4 subprocessos como `controlled_subprocess_review`, sem `manual_review` residual. Isso não é clearance de segurança.
- **Auditoria de realidade:** executada sem elevar claims cognitivos.
- **Dashboard:** snapshot válido renderizado; payload inválido rejeitado pelos testes.
- **Compilação Python:** passou para módulos, ferramentas e testes alterados.
- **CI alinhado nesta rodada:** `test_vram_defense_guard.py` agora é compilado, incluído na medição de cobertura e executado explicitamente pelo workflow.
- **CI modernizado nesta rodada:** runner fixado em `ubuntu-24.04`; `checkout@v5`, `setup-python@v6` e `upload-artifact@v7` removem a dependência das versões legadas que geravam avisos de Node.js 20.
- **State consistency gate:** o workflow agora compila e testa o auditor, gera `state-consistency.json` após o manifesto e falha em claims atuais divergentes.
- **Manifesto final:** a suíte integrada e o resumo completo unittest agora terminam antes da geração de `STATE_MANIFEST.json`; `discovered_static` não é apresentado como execução.
- **Segurança:** LOW permanece triado e informativo; qualquer MEDIUM/HIGH no Bandit bloqueia o workflow após a triagem.
- **Auditor B101:** 111 achados restantes, `outside_main_block: []`, `assert_lines_outside_main: []`, bloco detectado em `24668–26510`; compilação de módulos, ferramentas e testes passou.
- **Inventário de lacunas:** propagação, adaptação e atualização gerativa registram contratos mínimos observáveis; ferramenta registrada sem executor retorna falha estruturada. O inventário AST agora registra `pass_only_count=0` e `not_implemented_count=0`.
- **Telemetria GPU:** `monitor_hardware` tenta NVML, depois memória reservada CUDA como proxy; sem backend retorna `0.0` sem afirmar disponibilidade.
- **Contrafactuais:** `_predict_outcome` usa `world_model.predict_outcome` somente quando o protocolo existe; `_build_causal_chain` usa `extract_causal_relations` somente quando disponível.
- **Multimodal:** `VisionProcessor` usa `detect_objects`/`analyze_scene` do backend opcional; `AudioProcessor` usa `transcribe_speech`; sem backend, os fallbacks permanecem explicitamente demonstrativos.
- **Swarm:** `avg_consensus` e `avg_diversity` são médias acumuladas dos resultados de `deliberate`; sem agentes, `deliberate` lança `ValueError` explícito.
- **Memória episódica:** `total_links` e `avg_importance` são consultados das tabelas SQLite; sem episódios, `avg_importance` é `0.0`.
- **Atenção bottom-up:** `StimulusItem` aceita `contrast` e `motion` observados; sem sinais, os fallbacks `0.5` e `0.6` continuam explícitos e testados.
- **Integração/orquestração:** `avg_synergy_score` é a média das integrações registradas; scores individuais continuam heurísticos por nomes de módulos.
- **Grafo de conhecimento:** `entity_id` é único; atualizações não inflacionam a contagem e mudanças de tipo reclassificam o índice.
- **MCTS:** `available_actions=None` usa ações padrão; `available_actions=[]` é respeitado como espaço vazio, sem afirmar qualidade geral de planejamento.
- **Sensorimotor/multimodal:** modalidades `vision`, `audio` e `text` usam processadores compatíveis quando disponíveis; entradas desconhecidas ou ausência de processador seguem fallback de features observadas.
- **Consenso:** alterações aceitas só podem modificar `quorum_size`, `heartbeat_interval` e `election_timeout` dentro de limites; decisões são registradas como eventos locais, sem execução externa.
- **Shadow clones:** conhecimento é enviado ao bootstrap apenas via `ingest_knowledge`; ausência do protocolo gera evento `integrated=false` e não é apresentada como treinamento.
- **Reminiscência:** `is_typical` é `true` somente quando o bucket de maior densidade está entre 10 e 30 anos; picos fora da faixa continuam sendo retornados, mas marcados como atípicos.
- **Correção:** descrições malformadas de mínimo/máximo retornam imediatamente o conteúdo original, preservando o comportamento seguro já testado.
- **Auditoria AST:** 0 funções somente com `pass`, 0 `NotImplementedError` explícitos e 61 retornos constantes simples restantes.
- **Bootstrap audit:** o snapshot de 86 testes e 129 findings LOW é histórico; o estado operacional atual deve ser lido do manifesto do CI associado ao `HEAD`, que registra 113 testes e 131 findings LOW.
- **Manifesto factual:** o CI associado ao `HEAD` gera e retém `nexus-state-manifest`; commits e métricas sem esse artefato são classificados como não verificados.

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
- executar o contrato de estatísticas observadas do swarm junto com a suíte expandida.
- executar o contrato de estatísticas episódicas SQLite junto com a suíte expandida.
- executar o contrato de sinais de atenção junto com a suíte expandida.
- executar o contrato de média de sinergia junto com a suíte expandida.
- executar o contrato de ações do MCTS junto com a suíte expandida.
- executar o contrato de consistência do estado e o gate contra o manifesto da mesma execução.

O workflow endurecido gera o manifesto após Bandit, mutation e auditoria AST; as actions estão em versões Node24 e o runner está fixado em `ubuntu-24.04`.

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

## Histórico de checkpoints

- **2026-10-05 — 105 testes:** run `37258732531`, marco anterior preservado para proveniência.
- **Bootstrap — 86 testes:** run `37097334156`, snapshot histórico; não representa o estado atual.

## Próxima ação

Manter os documentos canônicos sincronizados com o código e executar mutation testing direcionado antes de qualquer alteração no núcleo monolítico. Não tratar os banners da suíte integrada como evidência independente de capacidade cognitiva.
