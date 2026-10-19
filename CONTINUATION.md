# Nexus — Continuidade entre agentes

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 416
bandit_low: 135
-->

## Estado canônico da composição local — 2026-10-18

Candidato isolado em `audit/consolidated-38-45-20261018`, baseado exatamente em
`812ebd02b0c82ecc3169d620e36d25592b1d6727` (`origin/main` local inventariado).
Inclui os heads exatos das PRs #38–#45 por merges locais; não é um merge em
`main`, publicação ou execução de CI remoto. #12–#14 ficam fora: #41 já alinha
os perfis e o lock, e nenhum dos três bumps é dependência funcional obrigatória.
Inventário, contratos, comandos, resultados e plano seguro: [AUDIT_COMPOSITION.md](AUDIT_COMPOSITION.md).

Os registros abaixo de outras branches/rodadas são **históricos**, não resultados
da composição. Os marcadores atuais refletem somente a suíte combinada; `HEAD`
é resolvido pelo manifesto pós-commit, com `ci_run=not_ci`. Não somar contagens
isoladas nem atribuir evidência sintética a capacidade geral.

### Follow-up documental da revisão independente — 2026-10-18

Sobre o candidato congelado `3ddc68d06853ba211bb6c04a02743565c28bd19c`, corrigidas duas questões documentais concretas: a tabela do README agora aponta para a composição local e seus resultados, não para o manifesto histórico `5b8bed8`; a docstring de `RealKnowledgeGraph` preserva os números legados `>93%`, `>95%` em 32k e `<45ms`, mas os classifica explicitamente como não verificados, sem task set, comando, ambiente ou proveniência. Não houve mudança de lógica, assinaturas, testes ou catálogo de mutações. A AST é idêntica à de `3ddc68d` ao excluir somente essa docstring.

Verificação do follow-up: `bash /workspace/nexus-consolidation-evidence/doc-followup/verify.sh`, no mesmo ambiente Python 3.11.16 com lock. Suíte completa **416/416**, sem falhas/skips/XPASS; integração **30 aprovadas + 1 skip FAISS**, sem IDs faltantes/inesperados/conflitantes; Ruff, compileall, realidade regenerada, triagem e `git diff --check` aprovados. Os relatórios de realidade versionados apenas acrescentam as três referências de testes do README. Comandos, logs e resumos separados em `/workspace/nexus-consolidation-evidence/doc-followup/`, sem sobrescrever a evidência anterior.

Cobertura, Bandit, 84 mutantes canônicos e suplemento VRAM de 10 mutantes **não foram medidos novamente** neste follow-up documental; os resultados continuam atribuídos a `3ddc68d` em `AUDIT_COMPOSITION.md`. A revisão independente deve ser reaberta somente sobre este novo diff congelado; seu parecer e o SHA final ficam no diretório externo do follow-up. Permanecem proibidos publicação, alteração de `main`, ações sobre PRs e merge remoto; preservar `/workspace/Nexus` sujo e todas as refs de origem.

> Handoff operacional versionado para continuidade entre agentes Manus.

## Histórico: integração local das duas correções VRAM — 2026-10-18

### Proveniência e contrato

- Base remota confirmada por `git ls-remote --heads origin` e `git fetch origin main`: `812ebd02b0c82ecc3169d620e36d25592b1d6727`. Worktree nova, inicialmente limpa: `/workspace/Nexus-vram-integrated`; branch `fix/vram-integrated-2026-10-18`, derivada diretamente de `origin/main`.
- Origens limpas verificadas antes da composição: `fix/vram-limit-bool-2026-10-18` em `4073eb298cb5fe0b3ca1db58a1b38d91c330afbb`; `fix/vram-snapshot-counters-2026-10-18` em `985917446f25c79d526c9ba2aca7f5d33187dd5d`, após implementação `41bb57143152486588c1b15431b879849ab2ecb0`. Ambas têm merge-base `812ebd02` e os mesmos seis paths alterados. Foram compostos somente seus patches de código/testes; os documentos desta seção foram reconciliados a partir da base, sem transportar os gates isolados como atuais.
- SHA do commit de implementação integrado (código/testes): `9b657011d0b0b0eec61fad53b3d353717c109e4d`. Este SHA identifica a implementação revisada, não o checkpoint documental. O follow-up destes registros é um commit posterior; seu hash é o `HEAD` documental (`git rev-parse HEAD`) e fica identificado separadamente no manifesto de estado pós-commit em `/workspace/nexus-vram-integrated-evidence/STATE_MANIFEST.json`.
- Contrato: limites booleanos geram `TypeError` antes de comparações; defaults, tipos numéricos antes aceitos, `0 < soft < hard <= 1` e ações inclusivas permanecem. Os três counters exigem `type(value) is int` e valor não negativo na construção (`TypeError` para tipo, `ValueError` para negativo), antes de qualquer decisão ou alteração do histórico/telemetria.
- Red/green combinado: com os sete testes novos e sem os patches de produção, 19 testes executados, 145 falhas de subteste e 7 erros; após ambos os patches, 19/19 aprovados. Logs `red-combined.log` e `green-focused.log` no diretório de evidências.

### Resultados medidos no conteúdo combinado

- Suíte completa sob cobertura e resumo final: **310 executados, 310 aprovados, 0 falhas, 0 skips, 0 XPASS**. Integração separada: **30 aprovações, 1 skip FAISS, 0 missing/unexpected/conflicting**.
- Cobertura branch (arredondamento da CLI; piso entre parênteses): fila **95% (90)**, VRAM **94% (90)**, fatos temporais **93% (90)**, Vita **76% (70)**, núcleo **68% (62)**.
- Mutation canônico: **58 declaradas/aplicáveis/mortas**, 0 sobreviventes/não aplicadas. Ensaio suplementar externo `vram_mutation.py`: **10 declaradas/aplicáveis/mortas**, 0 sobreviventes/não aplicadas; sete mutantes de counters e três de limites booleanos, em cópias temporárias. Não foi alterado o catálogo versionado.
- Bandit: **131 LOW, 0 MEDIUM/HIGH**. Triagem/check, auditor B101 (nenhum assert fora do main), gaps (0 pass-only e 0 NotImplementedError), Ruff, compileall, actionlint **1.7.12**, zizmor **1.31.1** offline e dry-run dos requisitos passaram. Bandit gera exit não zero por LOW; o comando aceita relatório não vazio e a política de severidade é conferida separadamente.
- Realidade regenerada sem diff em `AUDIT_REALITY_REPORT.md`/`audit_reality.json`; não há nova alegação cognitiva. Benchmarks CLI herdados executados com seed `20261016`: v1 mantém observação parcial `unsupported_interface`; v2 **128/128** dentro de `1e-12`, erro máximo **2.7755575615628914e-16**, controles inválidos **4/4** rejeitados. Taskset v2 SHA-256 `232371d0e8e948f4b0a004261e3d271b1003aed8d0de8c931d3439f9d370d025`. São checks sintéticos internos, não avaliação independente ou capacidade geral. Os JSONs usam `source_revision=unrecorded-local`; a proveniência real é esta base, o patch congelado e a atestação pós-commit externa, não o `base_commit` histórico do v1.

### Comandos reproduzíveis desta rodada

Diretório: `/workspace/Nexus-vram-integrated`. Script e logs completos: `/workspace/nexus-vram-integrated-evidence/run-gates.sh`. Ambiente reutilizado, sem modificar as origens: `/tmp/nexus-snapshot-venv`, Python **3.12.3**, dependências instaladas previamente com `requirements-ci.lock`, NetworkX **3.6.1** e Ruff **0.16.10**; versões efetivas registradas em `environment.log`.

```bash
export PATH=/tmp/nexus-snapshot-venv/bin:/tmp/nexus-ci-tools:$PATH PYTHONPATH=.
export NEXUS_AUDIT_DB=/workspace/nexus-vram-integrated-evidence/audit.sqlite3
export NEXUS_INTEGRATED_SUMMARY=integrated-summary.json
python -m unittest -v test_vram_defense_guard
bash /workspace/nexus-vram-integrated-evidence/run-gates.sh
# O script executou, com logs separados:
python -m pip freeze
python -m pip install --dry-run --no-deps -r requirements_constitutional.txt
actionlint .github/workflows/nexus-contract-gate.yml
zizmor --offline --no-progress .github/workflows
python -m ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py
python -m compileall -q core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py
python tools/reality_audit.py
git diff --exit-code -- AUDIT_REALITY_REPORT.md audit_reality.json
python -m coverage run --branch -m unittest discover -p 'test_*.py'
python -m coverage run --append --branch core/constitutional_brain.py
for spec in core/deferred_task_queue.py:90 core/vram_defense_guard.py:90 core/temporal_facts.py:90 vita/nexus_constitutional_bridge_v3.py:70 core/constitutional_brain.py:62; do
  python -m coverage report --include="${spec%:*}" --fail-under="${spec##*:}"
done
python -m coverage xml -o /workspace/nexus-vram-integrated-evidence/coverage.xml
python tools/targeted_mutation.py
python /workspace/nexus-vram-integrated-evidence/vram_mutation.py
python -m bandit -r core vita tools -f json -o bandit-report.json || test -s bandit-report.json
python tools/security_triage.py bandit-report.json security-triage-generated.md
python tools/security_triage.py bandit-report.json SECURITY_TRIAGE.md --check
python tools/embedded_assert_audit.py bandit-report.json core/constitutional_brain.py
python tools/implementation_gap_audit.py core/constitutional_brain.py --output implementation-gaps.json
# A política de severidade no script bloqueia qualquer MEDIUM/HIGH.
python -m benchmarks.nexus_capability_benchmark --seed 20261016 --output capability-benchmark.json
python -m benchmarks.nexus_pomdp_benchmark_v2 --seed 20261016 --output /workspace/nexus-vram-integrated-evidence/pomdp-belief-benchmark-v2.json
python tools/test_summary.py --output test-summary.json
python tools/state_manifest.py --output STATE_MANIFEST.json --test-result passed --test-command "python -m unittest discover -p 'test_*.py'" --gates-status success
python tools/state_consistency_audit.py --manifest STATE_MANIFEST.json --output state-consistency.json
```

Os quatro claims canônicos foram sincronizados para **310 testes / 131 LOW**. A consistência é medida antes e depois do commit, com `ci_run=not_ci`; os artefatos pós-commit identificam o SHA final. Os gates adicionais que o workflow repete por módulos já estão incluídos no discover; não foram repetidos todos como invocações independentes. Nenhuma execução de CI remoto foi atribuída a esta branch.

### Diferenças em relação às execuções isoladas e limites

- Limites isolados reportavam 14 testes VRAM e 305 executados/303 aprovados/2 skips NetworkX. Counters isolados reportavam 17 VRAM e 308/308, após corrigir o ambiente. Combinado: **19 VRAM e 310/310**; a ausência dos dois skips vem de NetworkX presente, não de correção adicional.
- Coberturas arredondadas, integração 30+1, mutation canônico 58/58 e Bandit 131 LOW permaneceram iguais aos registros isolados. O ensaio suplementar passa de sete counters para dez mutantes incluindo limites; a comparação é histórica, não nova reexecução das branches de origem.
- Validação somente na construção: não congela snapshots, não valida atribuição posterior, relações físicas entre counters, objetos forjados ou código Python hostil. `available`, timestamp, coleta CUDA e mitigação não mudaram; as conversões CUDA `int(...)` preexistentes permanecem. Sem ensaio GPU real; psutil/FAISS/SentenceTransformers ausentes. Python local 3.12.3 difere do 3.11 de CI. Testes/scan/revisão não provam segurança geral ou prontidão de produção.
- As duas origens e `/workspace/Nexus` sujo devem permanecer idênticos; SQLite, relatórios e logs são externos ou ignorados, nunca parte do commit. Somente revisão final única de Segurança sobre o diff integrado; reabrir apenas por achado concreto. Nenhum push, PR ou merge nesta rodada, nem processamento da autorização privada #38–#44.

## Estado histórico anterior à integração VRAM

- **Branch e base:** `fix/queue-snapshot-hardening-2026-10-17`, baseada em `origin/main` `812ebd02b0c82ecc3169d620e36d25592b1d6727`; alterações locais, sem push, merge ou rebase.
- **D-030:** fronteiras de snapshot endurecidas sem mudar schema `nexus.deferred_task_queue.v1` nem assinaturas públicas; contextos exigem JSON nativo estrito. Extensões string-key em raiz/tarefas são ignoradas sem cópia profunda ou propagação interna; `statistics` exige exatamente quatro chaves. Chaves root/tarefa/estatísticas devem ser `str` exatas antes de qualquer lookup/membership, e JSON textual rejeita duplicatas em todos os níveis. Containers externos `dict`/`list` aceitam subclasses; escalares usam tipos built-in exatos e rejeitam `bool`/subclasses antes de operações ou construção de tarefas. A restauração publica heap, índice e contadores somente após staging completo.
- **Baseline confirmado:** referências aninhadas já eram isoladas; timestamps NaN/±Infinity e bool nos campos inteiros já eram rejeitados. Lacunas reproduzidas: chave extra em estatísticas, tentativa exaurida ainda pendente, hooks de deepcopy/`__eq__`, duplicatas JSON e falha parcial durante staging; revisões independentes encontraram TOCTOU de `task_id` e subclass `int` de `priority` executando `__neg__`, todos agora cobertos e corrigidos.
- **Testes locais:** `unittest` 318 executados, 316 aprovados e 2 skips pelo NetworkX opcional ausente; contratos focados snapshot/fila 51/51 em Python 3.11.16 e 51/51 em Python 3.12.3; cobertura de branch da fila 91%. Suite integrada: 30 aprovados, 1 skip opcional de FAISS, 0 faltantes.
- **Gates:** mutation 80/80 mortos, 0 sobreviventes/não aplicadas; Ruff 0.16.10, actionlint 1.7.12 e zizmor 1.31.1 passaram; Bandit 131 LOW e 0 MEDIUM/HIGH, triagem coerente; compileall passou no Python 3.12.3 e nos três arquivos alterados no Python 3.11.16. Auditoria de realidade regenerada (27.549 linhas, 269 classes, 875 funções), sem diff nos relatórios versionados.
- **Limites:** nenhum teto novo de bytes/profundidade; o parser/encoder JSON padrão pode rejeitar aninhamentos extremos. `enqueue` ainda usa `deepcopy` para contexto e os métodos da fila ainda expõem objetos mutáveis; isso ficou fora do escopo. Não há callsite de produção de `restore_snapshot` no checkout. CI oficial não foi executada porque não houve publicação.
- **Próxima ação:** manter a branch isolada até autorização explícita para publicação/merge; não enviar para `main`. Considerar `enqueue` como uma decisão separada se futuramente for desejada validação JSON estrita também na entrada da fila.

- **Base desta rodada:** `main` está no merge da PR #37, commit `812ebd02b0c82ecc3169d620e36d25592b1d6727`; essa PR incorporou o núcleo do ledger temporal (D-027) e o acúmulo de tipos de relação no grafo (D-028). A branch `feat/temporal-ledger-integration-2026-10-17` parte desse commit.
- **Política de integração:** mudanças permanecem em branches/PRs pequenas e só são mescladas com autorização explícita para a PR específica.
- **D-027/D-028 em `main`:** o `TemporalFactStore` oferece fatos bi-temporais, replay e snapshots encadeados em memória; `RealKnowledgeGraph` preserva tipos de relação múltiplos em uma mesma aresta NetworkX.
- **D-029 nesta branch:** integração opt-in por `RealKnowledgeGraph(..., temporal_store=...)`; o chamador declara explicitamente predicados funcionais e pode fornecer `source` e `valid_from`. Propriedades da relação não são encaminhadas implicitamente como metadata temporal.
- **Validação desta branch:** suíte unittest local 309/309; integração 30 aprovados + 1 skip opcional; contratos focados 11/11; cobertura dos métodos alterados 100% em linhas e branches; mutation testing 62/62 mortos; Ruff passou. Os dois checks oficiais do run `39015023676` passaram no commit de implementação `865063d21de9e986f9a502a790bfac2b68f7f801`; consulte a PR #38 para o estado do HEAD mais recente.
- **Limites atuais:** não há persistência SQLite para o ledger nem ingestão automatizada das alegações de `NEXUS_EVIDENCE.md`; o grafo e o ledger são projeções distintas, sem transação distribuída se uma falha inesperada ocorrer após a escrita temporal. Predicados funcionais continuam configuração explícita do chamador.
- **POMDP herdado:** o filtro v2 usa modelo explícito e não reivindica aprendizagem ou planejamento; o protocolo v1 mantém `unsupported_interface` para observação parcial.
- **Escopo real:** protótipo experimental Python de roteamento, filas, telemetria e contratos de diagnóstico. Não reivindicar AGI, ASI, consciência ou autonomia geral.

## Contratos relevantes na fronteira de cache/ferramentas

- Herdado da PR #27: `test_format_json_contract.py` preserva formatação de JSON finito e exige falha estruturada para NaN/infinitos e dados não serializáveis.
- `test_tool_cache_key_contract.py`: preserva a identidade das chaves pequenas; garante bypass pré-serialização para tamanho, profundidade, nós, itens por contêiner e ciclos; comprova que falhas de cache não bloqueiam o executor, que subclasses de `dict` seguem válidas e que ferramenta não cacheável não forma chave.
- `test_tool_cache_store_contract.py`: injeta `RecursionError` na cópia para cache e exige bypass observável sem descartar o sucesso; a etapa dedicada está no workflow e um mutante protege o caminho.
- `test_tool_parameter_contract.py`: prova execução com subclasse de `list` e cache bypass, aceitação de subclasse de `int`, rejeição de `bool` como `int` e rejeição de classe que apenas imita o nome `list`; dois mutantes protegem o contrato.
- `test_tool_cache_snapshot_contract.py`: verifica identidade única de snapshot nas três fases, isolamento de mutações posteriores/in-place, cópia por valor sem preservar aliases, bypass para valores fora do JSON estrito e compatibilidade de subclasses.

## O que está implementado

- Reprocessamento de tarefas adiadas no `CentralRouter`, preservando contexto, callbacks e telemetria; o sinal interno de reprocessamento não vem mais do contexto do caller e não pode impedir o enqueue normal.
- Auditoria persistente SQLite e análise longitudinal no bridge Vita.
- Exportação/validação do contrato `nexus.recovery.diagnostics.v1`, com tipos/finidade verificados e JSON estrito sem `NaN`/`Infinity`.
- `test_recovery_diagnostics_contract.py`: fuzzing property-based de 60 payloads JSON arbitrários, com rejeição total e estruturada.
- `test_vita_audit_concurrency_contract.py`: contrato de escritas concorrentes e leitura consistente do audit log SQLite do bridge Vita.
- `test_vita_audit_cross_connection_contract.py`: contrato de duas instâncias Vita compartilhando o mesmo arquivo SQLite sem perda de eventos.
- `tools/diagnostics_dashboard.py`: dashboard CLI determinístico que valida snapshots antes de renderizar métricas observáveis.
- `tools/security_triage.py` e `SECURITY_TRIAGE.md`: inventário conservador dos achados Bandit, sem supressão automática.
- `test_memory_sql_contract.py`: protege SQL parametrizado, limite positivo/zero, rejeição de negativos e tipos inválidos e falha antes do backend semântico.
- `tools/targeted_mutation.py`: mutation testing direcionado para fila, snapshots, Vita e schema.
- `core/deferred_task_queue.py`: transições de estado protegidas por `RLock`; conclusão/descarte removem tarefas pendentes do heap, retry é único e limitado, callbacks defeituosos são reportados sem interromper o lote; snapshots usam JSON estrito, estatísticas exatas e restore transacional.
- `RealImmutableLog`: triggers bloqueiam UPDATE/DELETE, o SHA-chain é verificado na inicialização e detecta adulteração/remoção; writes de conexões concorrentes são serializados. Truncamento final ou rehash por quem controla o arquivo ainda exige âncora externa.
- `RealJailbreakDetector`: normalização Unicode, sinais por frases e fronteiras de palavra reduzem falsos positivos e detectam ofuscações testadas em inglês/português; continua heurístico, não é uma fronteira de segurança.
- `test_deferred_task_snapshot.py`: round-trip, ordem, atomicidade nos dois modos, limites, JSON estrito, rejeição de bool/NaN/ciclos/hooks de cópia, estatísticas e mutação concorrente de dicts de entrada; restore não executa tarefas.
- Auditoria de realidade em `tools/reality_audit.py`.
- Hardening do fallback SQL em `RealHierarchicalMemory`: consultas estáticas parametrizadas por termo, sem montagem dinâmica de placeholders.
- Hardening de `_generate_correction`: captura explícita de `ValueError`/`IndexError`, com preservação do conteúdo original em descrições inválidas.
- Revisão dos 16 achados B311: usos classificados como simulação/heurística, sem reutilização autorizada para segredos ou decisões de segurança.
- Revisão dos 111 achados B101 restantes: todos estão no bloco `__main__` demonstrativo e permanecem fora dos contratos independentes de produção.
- `tools/embedded_assert_audit.py`: guardrail AST que impede B101 e qualquer `assert` fora do bloco `__main__` enquanto a migração é incremental.
- `test_central_router_contract.py`: contrato independente para contagem/cache, isolamento de contexto/cache e garantia de que um marcador caller-controlled não suprima o enqueue de emergência.
- `tools/implementation_gap_audit.py`: inventário AST conservador de funções pass-only e `NotImplementedError`, sem tratar lacunas como implementadas.
- `test_explicit_gap_contract.py`: contratos independentes para propagação de erro e registro de adaptação simulada.
- `test_working_memory_contract.py`: contrato independente para alocação de atenção local e delegação declarada.
- `test_hardware_telemetry_contract.py`: contrato de fallback GPU conservador quando não há backend disponível.
- `test_counterfactual_contract.py`: contratos de world model, causalidade e fallback heurístico.
- `test_multimodal_backend_contract.py`: contratos de backends opcionais de visão/áudio e fallbacks de demonstração.
- `test_swarm_statistics_contract.py`: contrato de médias observadas do swarm e rejeição de deliberação sem agentes.
- `test_episodic_memory_statistics_contract.py`: contrato de contagem persistente, rejeição de episódios e memórias inexistentes quando o backend resolve IDs, recuperação com provedor falsey e fechamento de episódios; sem backend, IDs continuam referências opacas não verificadas.
- `test_episodic_memory_concurrency_contract.py`: contrato de ciclo de vida concorrente em uma conexão SQLite compartilhada.
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
- `core/integrated_suite_summary.py`: contabiliza e serializa resultados da suíte integrada sem confundi-los com `unittest`.
- `tools/state_manifest.py`: gerador determinístico do manifesto factual por execução.
- `test_state_manifest.py`: contrato do schema, da proveniência e do resumo integrado do manifesto.
- `test_integrated_suite_summary.py`: contrato de aprovação, skip, ausência, conflito, IDs decimais, rejeição de IDs duplicados e de logs que apenas mencionam marcadores.
- `test_mcts_action_contract.py`: contrato da distinção entre ações padrão (`None`) e espaço de ações vazio (`[]`).
- `tools/state_consistency_audit.py`: compara claims atuais estruturados com o manifesto factual e ignora histórico fora do bloco.
- `test_state_consistency_audit.py`: contratos de sincronização, divergência, ausência, blocos/chaves duplicados e manifesto inválido, incluindo a evidência integrada.
- `tools/test_summary.py`: executa a suíte final e registra `executed`, `passed`, `failed`, `skipped` e `unexpected_successes`; XPASS conta como falha e é coberto por `test_test_summary.py`.

## Validação local pós-PR #21 — 2026-10-16

- **Suíte unitária:** 213 executados, 213 aprovados, 0 falhas, 0 skips e 0 sucessos inesperados.
- **Suíte integrada:** 31 IDs inventariados, 30 aprovados e 1 skip explícito por dependência FAISS opcional ausente.
- **Cobertura:** fila 95%, guard VRAM 94%, bridge Vita 76% e núcleo 67%, todos acima dos pisos do workflow.
- **Mutation testing:** 28/28 mutações aplicáveis mortas, 0 sobreviventes e 0 não aplicadas.
- **Segurança e lint:** 131 achados Bandit LOW, sem MEDIUM/HIGH; Ruff, actionlint, zizmor, auditorias AST e compilação passaram.
- **Auditoria de realidade:** relatórios regenerados; núcleo com 26.946 linhas, 267 classes, 859 funções, 0 `pass`-only e 2 retornos constantes simples.
- **Status:** validação local; o manifesto local identifica `ci_run=not_ci`, sem atribuir estes resultados aos checks oficiais. Os artefatos de realidade versionados correspondem à auditoria local.

## Histórico de gates locais da base `7c8a187`

- **121 testes unitários/property/contrato/triagem/SQL/VRAM/guardrails:** passaram na suíte integral local; os contratos MCTS, memória episódica e Vita cobrem espaço de ações vazio, contagem idempotente de fechamentos, concorrência na mesma conexão e concorrência entre conexões SQLite.
- **Cobertura branch:** `core/deferred_task_queue.py` **98%** (limiar 90%); `vita/nexus_constitutional_bridge_v3.py` **71%** (limiar 70%); total dos dois módulos **85%**.
- **Mutation testing:** **10/10 mutações mortas, 0 sobreviventes, 100%**.
- **Bandit:** **131 achados LOW** após o hardening do B607 em `tools/state_manifest.py`; a triagem classifica 16 B311 como `simulation_only_random_review`, 111 B101 como `embedded_demo_assert_review` e 4 subprocessos como `controlled_subprocess_review`, sem `manual_review` residual. Isso não é clearance de segurança.
- **Auditoria de realidade:** executada sem elevar claims cognitivos.
- **Dashboard:** snapshot válido renderizado; payload inválido rejeitado pelos testes.
- **Compilação Python:** passou para módulos, ferramentas e testes alterados.
- **CI alinhado nesta rodada:** `test_vram_defense_guard.py` agora é compilado, incluído na medição de cobertura e executado explicitamente pelo workflow.
- **CI modernizado nesta rodada:** runner fixado em `ubuntu-24.04`; actions fixadas por SHA de commit (`checkout` v5.1.0, `setup-python` v6.3.0, `upload-artifact` v7.0.2) com `persist-credentials: false`; o Dependabot propõe as atualizações.
- **State consistency gate:** o workflow agora compila e testa o auditor, gera `state-consistency.json` após o manifesto e falha em claims atuais divergentes.
- **Manifesto final:** a suíte integrada e o resumo completo unittest agora terminam antes da geração de `STATE_MANIFEST.json`; `discovered_static` não é apresentado como execução.
- **Segurança:** LOW permanece triado e informativo; qualquer MEDIUM/HIGH no Bandit bloqueia o workflow após a triagem.
- **Auditor B101:** 111 achados restantes, `outside_main_block: []`, `assert_lines_outside_main: []`, bloco detectado em `24668–26510`; compilação de módulos, ferramentas e testes passou.
- **Inventário de lacunas:** propagação, adaptação e atualização gerativa registram contratos mínimos observáveis; ferramenta registrada sem executor retorna falha estruturada. O inventário AST agora registra `pass_only_count=0` e `not_implemented_count=0`.
- **Memória episódica:** a conexão compartilhada usa `RLock` para serializar DDL, escritas, leituras e contadores; `episodes_closed` conta apenas transições `open` → `closed` confirmadas pelo SQLite; o teste concorrente cobre criação, vínculo, fechamento, estatísticas e busca.
- **Bridge Vita:** o audit log compartilhado usa `RLock` para serializar DDL, commits, escritas e leituras; o teste concorrente cobre eventos de política, telemetria e análise de recuperação.
- **Cross-connection SQLite:** cada conexão Vita configura `busy_timeout=5000`; o contrato compartilhado verifica duas instâncias escrevendo no mesmo arquivo sem perda de eventos.
- **Fuzzing de diagnósticos:** 60 exemplos Hypothesis cobrem valores JSON aninhados, tipos escalares, listas e mapas; o validador sempre retorna o contrato booleano/erro sem exceção inesperada.
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
- **Auditoria AST:** 0 funções somente com `pass`, 0 `NotImplementedError` explícitos e 59 retornos constantes simples restantes.
- **Bootstrap audit:** o snapshot de 86 testes e 129 findings LOW é histórico; a branch anterior registrou 159 testes e 131 findings LOW na validação local; o estado oficial de cada `HEAD` deve ser confirmado pelo manifesto do CI associado.
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
- instalar dependências com versão fixa (`requirements-dev.txt`) e falhar em ruff (F, E9, B904, B006, B007 e RUF013), actionlint (binário oficial 1.7.12 verificado por SHA-256 no workflow; o Dependabot não o atualiza) ou zizmor.
- regenerar os relatórios de realidade e falhar se `AUDIT_REALITY_REPORT.md`/`audit_reality.json` versionados divergirem; conferir `SECURITY_TRIAGE.md` com `--check`.
- medir uma única cobertura (testes + suíte integrada) com pisos para fila, guard de VRAM, bridge e núcleo.
- **Compatibilidade Python:** o workflow atual executa somente Python 3.11. `requirements.txt` registra combinações de NumPy testadas em 3.12/3.13, mas o workflow não tem matriz nem job `python-compat`; validar essas versões na CI permanece trabalho futuro.
- falhar se o diretório de trabalho ganhar bancos SQLite ou se arquivos versionados mudarem durante a execução.
- gravar o resumo da suíte integrada (`NEXUS_INTEGRATED_SUMMARY`) e reportá-lo no manifesto em `verification.integrated`, separado de `verification.tests` (que conta só o `unittest`).
- Ao implementar a compatibilidade 3.12/3.13, avaliar execução com `-X dev -W error` e falha em `ResourceWarning`; essa política ainda não está configurada no workflow atual.

O workflow endurecido gera o manifesto após Bandit, mutation e auditoria AST; as actions estão fixadas por SHA e o runner em `ubuntu-24.04`.

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
- O detector de jailbreak é heurístico (primeira camada); `EthicsGuard` e `RealCognitiveBudgetEnforcer` também são baseados em palavras-chave e não foram reescritos nesta rodada.
- O log imutável detecta adulteração e remoção no meio da cadeia; truncar o final exige ancorar o `head` fora do banco.
- Nomes de classe `Real*` e com termos de consciência/emoção foram mantidos (renomear quebraria a API); os textos que afirmavam capacidade foram removidos e o auditor passou a escanear o código.
- `think()` e `_wire_dependencies` seguem com complexidade ciclomática muito alta; o fatiamento por fase é a próxima refatoração estrutural.
- Os trechos (needles) do gate de mutação são texto exato do código: depois de uma refatoração, um mutante `not_applied` significa que o trecho mudou e o mutante deve ser reescrito, não ignorado.
- `tests` no manifesto e nos documentos conta apenas a suíte `unittest`; os checks embutidos da suíte integrada são reportados à parte (no manifesto de referência de `main`: 31 inventariados, 30 aprovados e 1 skip opcional).
- O auditor de realidade separa padrões proibidos (gate zero), frases fortes de capacidade (inventário com limite que só diminui), terminologia científica (informativo) e identificadores `Real*` (neutro); zero padrões proibidos não significa ausência de linguagem de capacidade (restam frases de ênfase como "REAL <Capacidade>").

## Histórico de checkpoints

- **2026-10-08 — 177 testes:** auditoria de código, hardening e revisão independente (D-013 a D-016); o run do CI correspondente é registrado pelo manifesto após o merge.
- **2026-10-05 — 105 testes:** run `37258732531`, marco anterior preservado para proveniência.
- **Bootstrap — 86 testes:** run `37097334156`, snapshot histórico; não representa o estado atual.

## Próxima ação histórica (superada pelo plano da composição)

Entregar a integração VRAM local para aprovação posterior. Não há autorização de push, PR ou merge. A decisão privada sobre PRs #38–#44 não pertence a esta rodada. A instrução histórica sobre PR #36 aberta está superada: #36 e #37 já integram a base `812ebd02`. Não incorporar outras branches nem sobrescrever seus documentos; regenerar evidências se a base mudar. `ToolDefinition.requires_permission`/`permission_level` continuam sem enforcement, fora deste escopo.
