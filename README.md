# Nexus Constitutional

[![CI](https://github.com/pedromiorini/Nexus/actions/workflows/nexus-contract-gate.yml/badge.svg)](https://github.com/pedromiorini/Nexus/actions/workflows/nexus-contract-gate.yml)

## Escopo real

O Nexus Constitutional é um **protótipo experimental de arquitetura neuro-simbólica em Python**. O repositório contém componentes de roteamento, governança, memória, filas de tarefas adiadas, telemetria, auditoria persistente e testes de demonstração. Os componentes são úteis para pesquisa de engenharia de software e orquestração cognitiva, mas o projeto **não reivindica AGI, ASI, consciência, autoconsciência ou autonomia geral**.

> Nomes de módulos, banners de execução e testes de demonstração não constituem prova de inteligência geral, consciência ou prontidão para produção.

## Evidências atualmente verificadas

| Área | Evidência | Limite da afirmação |
|---|---|---|
| Núcleo constitucional | O manifesto do CI de `main` no commit `5b8bed8` registrou 31 verificações integradas: 30 aprovadas e 1 ignorada pela ausência opcional de FAISS; nenhuma falhou | São verificações do próprio projeto, não uma avaliação externa de capacidade geral |
| Fila de tarefas adiadas | 16 testes unitários e de integração | Cobre contratos da fila e reprocessamento, não carga distribuída de produção |
| Diagnóstico de recuperação | Exportação versionada `nexus.recovery.diagnostics.v1` e validador formal | O schema garante estrutura, não a correção de decisões cognitivas |
| Auditoria Vita | Eventos, transições e métricas persistidos em SQLite | A análise é local e limitada ao histórico disponível |
| CI | Workflow executa lint (ruff, actionlint, zizmor), compilação, testes de contrato e regressão, suíte integrada, cobertura com pisos, mutation testing (28 mutantes escolhidos; o score vale só para eles) e auditorias de segurança e de realidade, com ações fixadas por SHA; o gate usa Python 3.11 | A execução depende do ambiente e das dependências instaladas no runner |
| Cobertura | Relatório combinado de linhas e branches (testes + suíte integrada) é retido por 14 dias; pisos de cobertura total por arquivo: fila 90%, guard de VRAM 90%, bridge Vita 70%, núcleo 62% | Cobertura não substitui testes de propriedades, segurança ou mutation testing |

## Componentes principais

`core/constitutional_brain.py` concentra o núcleo legado e o `CentralRouter`. `core/deferred_task_queue.py` implementa prioridade, retry, descarte e callbacks do ciclo de vida. `core/vram_defense_guard.py` fornece decisões de pressão de memória quando os recursos de telemetria estão disponíveis. `vita/nexus_constitutional_bridge_v3.py` registra telemetria, transições, auditoria persistente, análise de recuperação e exportação estruturada.

O arquivo `test_recovery_diagnostics_contract.py` mantém testes independentes do contrato público de observabilidade. O arquivo `test_deferred_task_queue.py` cobre a fila e a integração do reprocessamento. O relatório `AUDIT_REALITY_REPORT.md` é gerado pelo auditor conservador em `tools/reality_audit.py`.

Os processadores multimodais aceitam backends opcionais por protocolo: `VisionProcessor(backend=...)` pode fornecer `detect_objects` e `analyze_scene`, enquanto `AudioProcessor(backend=...)` pode fornecer `transcribe_speech`. Sem esses métodos, o código mantém respostas de demonstração explicitamente limitadas; elas não são inferência visual ou transcrição real. O contrato independente está em `test_multimodal_backend_contract.py`.

As estatísticas de `RealSwarmIntelligence` (`avg_consensus` e `avg_diversity`) são calculadas a partir das decisões produzidas durante as deliberações; não são constantes de demonstração. Uma deliberação sem agentes é rejeitada explicitamente. O contrato está em `test_swarm_statistics_contract.py`.

As estatísticas de `RealEpisodicMemory` (`total_links` e `avg_importance`) são derivadas das tabelas SQLite de episódios e vínculos. No estado vazio, a média retorna `0.0`. `episodes_closed` conta somente transições reais de `open` para `closed`; IDs inexistentes e fechamentos repetidos não incrementam o contador. O contrato independente está em `test_episodic_memory_statistics_contract.py`.

`StimulusItem` aceita sinais opcionais de `contrast` e `motion`. O componente `BottomUpAttention` usa esses valores quando observados, aceita `contrast` no contexto e mantém fallbacks heurísticos explícitos quando os sinais não estão disponíveis. O contrato está em `test_attention_signal_contract.py`.

`RealIntegrationOrchestrationEngine` calcula `avg_synergy_score` como a média das integrações registradas, evitando a média recursiva sem peso anterior. Os scores individuais continuam sendo heurísticas baseadas nos nomes dos módulos; o contrato está em `test_integration_synergy_contract.py`.

`RealKnowledgeGraph` trata `entity_id` como identidade única: re-adicionar uma entidade atualiza o registro, preserva a contagem e remove o ID do índice de tipo anterior quando necessário. `test_knowledge_graph_contract.py` cobre essa invariável e a média de confiança observada.

No `RealMCTSPlanner`, `available_actions=None` usa o espaço padrão, enquanto `available_actions=[]` é respeitado como espaço vazio. Essa distinção evita que uma entrada explícita seja silenciosamente substituída; o contrato está em `test_mcts_action_contract.py`.

## Benchmark científico sintético

`benchmarks/` contém um protocolo local e determinístico para medir composição de ações, extração/alegações causais, saída de incerteza e observação parcial. Ele usa tarefas sintéticas, registra SHA-256 do taskset e é executado no CI; quando uma interface não retorna o tipo de evidência necessário, o resultado é `unsupported_interface`, não uma nota inventada. O protocolo, comandos, métricas, fontes teóricas e limites estão em [`benchmarks/README.md`](benchmarks/README.md). A execução CI é retida como artefato por 14 dias.

## Como executar

Use Python 3.11 (a versão verificada no CI). Python 3.12 e 3.13 podem funcionar, mas não são atualmente verificadas pelo workflow. Instale as dependências, defina o caminho de importação e execute:

```bash
python3 -m pip install -r requirements.txt      # execução (numpy)
python3 -m pip install -r requirements-dev.txt  # testes e CI local (hypothesis, coverage, bandit, ruff...)
export PYTHONPATH=.
python3 -m unittest discover -v -p 'test_*.py'  # todos os testes
python3 core/constitutional_brain.py            # suíte integrada embutida (recusa python -O)
python3 tools/reality_audit.py                  # regenera os relatórios de realidade
```

Os traces de depuração do núcleo ficam silenciosos por padrão; defina `NEXUS_DEBUG=1` para vê-los. `NEXUS_AUDIT_DB` define o caminho do banco de auditoria Vita (padrão: `nexus_audit.sqlite3` no diretório atual).

Dependências opcionais podem ativar caminhos adicionais de telemetria ou recuperação. Quando não estão disponíveis, o código pode usar fallbacks; portanto, os resultados devem registrar quais dependências estavam presentes.

## Contrato de diagnóstico

A ponte Vita exporta snapshots com schema versionado:

```python
snapshot = bridge.export_recovery_diagnostics(as_json=True)
validation = bridge.validate_recovery_diagnostics(snapshot)
```

O diagnóstico inclui severidade, eventos analisados, pausas, recuperações, taxa de recuperação, duração de pausas, recorrência crítica e eventos de auditoria. O `CentralRouter` expõe os mesmos contratos por meio de `get_recovery_diagnostics()`, `export_recovery_diagnostics()` e `validate_recovery_diagnostics()`.

## Snapshot da fila de tarefas adiadas

A `DeferredTaskQueue` expõe `export_snapshot(as_json=True)` e `restore_snapshot(snapshot, replace=False)` para transportar tarefas pendentes entre processos ou pontos de recuperação. O contrato é versionado como `nexus.deferred_task_queue.v1` e preserva prioridade, contexto, tentativas, razão, IDs e ordem de criação. A restauração valida o payload inteiro antes de alterar a fila, rejeita schema incompatível, IDs duplicados, tipos inválidos e excesso de capacidade, e não executa processadores ou callbacks.

Os contextos são copiados nas fronteiras de exportação e restauração para evitar referências mutáveis compartilhadas. Timestamps devem ser finitos; na restauração, contextos precisam ser serializáveis como JSON estrito; limites e contadores devem usar inteiros não negativos (`bool` não é aceito como inteiro do contrato). Snapshots inválidos são rejeitados sem alterar o estado da fila, inclusive com `replace=True`.

Esse recurso **não é persistência física em disco**, não oferece locking distribuído e não garante entrega exatamente uma vez. O consumidor deve armazenar o JSON, controlar concorrência e decidir quando usar `replace=True` para substituir uma fila em memória. Exemplos mínimos:

```python
snapshot = queue.export_snapshot(as_json=True)
recovered = DeferredTaskQueue(max_attempts=3, max_size=256)
recovered.restore_snapshot(snapshot)
```

Verifique o contrato com:

```bash
export PYTHONPATH=.
python3 -m unittest -v test_deferred_task_snapshot.py
```

## Hardening do fallback SQL

A busca por palavras do fallback da memória usa consultas SQL estáticas, uma por termo, com conteúdo e limite sempre parametrizados. Os resultados são unidos por ID e ordenados pelas mesmas métricas de importância e acesso do contrato anterior. Isso elimina a montagem dinâmica de placeholders sinalizada pelo B608 do Bandit.

O teste adversarial `test_memory_sql_contract.py` cobre tokens SQL como dados, busca OR por múltiplas palavras e respeito ao limite. A busca semântica opcional continua sujeita às limitações e aos fallbacks documentados.

## Segurança heurística

O `RealJailbreakDetector` é uma **primeira camada heurística** (EN/PT): normaliza Unicode (NFKC, caracteres de largura zero, homóglifos, acentos, leetspeak), usa frases-alvo com fronteira de palavra e soma pesos de sinais; `sensitivity` (0 a 1) ajusta apenas os sinais fracos. Não é classificador semântico nem substitui controle de privilégio; o corpus de regressão está em `test_jailbreak_detector_contract.py` e contornos novos devem entrar lá. O `RealImmutableLog` é encadeado por hash e *append-only* (triggers SQLite); `verify_chain()` recalcula a cadeia e detecta adulteração, mas truncar o fim do log só é detectável se o `head` retornado for ancorado fora do banco. Relate vulnerabilidades conforme [`SECURITY.md`](SECURITY.md).

## Limitações conhecidas

O núcleo ainda é monolítico e contém componentes experimentais, heurísticas, retornos constantes e caminhos de fallback que exigem revisão individual. A presença de uma classe ou método com nome cognitivo não demonstra a capacidade descrita pelo nome. A suíte integrada é valiosa como teste de não regressão, mas não é uma avaliação independente, adversarial ou científica de inteligência.

As próximas prioridades são elevar a cobertura dos componentes críticos, adicionar testes baseados em propriedades para invariantes da fila e do schema, executar análise estática de segurança e aplicar mutation testing incremental. Só depois de obter evidências específicas deve-se elevar qualquer alegação de capacidade.

## Auditoria de realidade

Execute `python3 tools/reality_audit.py` para gerar `AUDIT_REALITY_REPORT.md` e `audit_reality.json`. O auditor compara alegações documentais, caminhos referenciados, estrutura AST, funções somente com `pass`, retornos constantes, testes embutidos e dependências opcionais. Ele é deliberadamente conservador: sinaliza alegações que precisam de evidência, mas não tenta concluir capacidades cognitivas a partir de nomes ou banners.

O CI regenera esses relatórios e falha se os arquivos versionados divergirem (`git diff`): depois de editar o núcleo, rode `python3 tools/reality_audit.py` e faça commit de `AUDIT_REALITY_REPORT.md` e `audit_reality.json`. A triagem `SECURITY_TRIAGE.md` agrupa os achados por escopo (sem números de linha) e é conferida por `tools/security_triage.py --check`.

O auditor separa padrões proibidos no código (gate: zero), frases fortes de capacidade ("REAL <Capacidade>", "REAL e FUNCIONAL"; inventário com limite que só diminui, em `test_reality_audit.py`), terminologia científica (informativa) e identificadores `Real*` (neutro): zero padrões proibidos não significa ausência de linguagem de capacidade. Com `NEXUS_INTEGRATED_SUMMARY=arquivo.json` a suíte integrada grava um resumo legível por máquina, que o manifesto reporta em `verification.integrated`, separado de `verification.tests` (só `unittest`).

## Manifesto factual e conhecimento

O workflow gera o artefato `nexus-state-manifest` a cada execução. O manifesto contém o commit e branch efetivamente verificados, contagem de testes, resultado, Bandit, mutation testing, métricas AST, dependências opcionais e commits dos documentos. `STATE_MANIFEST.json` é um artefato de execução e não deve ser tratado como snapshot manual versionado; números sem o manifesto correspondente são históricos ou não verificados.

## Conhecimento e continuidade

O estado canônico de engenharia está separado por finalidade: [`NEXUS_CONSTITUTION.md`](NEXUS_CONSTITUTION.md) define limites e regras; [`NEXUS_KNOWLEDGE.md`](NEXUS_KNOWLEDGE.md) registra fontes e estado operacional; [`NEXUS_ARCHITECTURE.md`](NEXUS_ARCHITECTURE.md) descreve responsabilidades e fronteiras; [`NEXUS_EVIDENCE.md`](NEXUS_EVIDENCE.md) mantém a matriz fato/evidência/limitação; [`NEXUS_DECISIONS.md`](NEXUS_DECISIONS.md) registra decisões; [`NEXUS_ROADMAP.md`](NEXUS_ROADMAP.md) prioriza evolução verificável; e [`CONTINUATION.md`](CONTINUATION.md) registra o último checkpoint entre agentes. O diagnóstico que originou essa separação está em [`NEXUS_BOOTSTRAP_AUDIT.md`](NEXUS_BOOTSTRAP_AUDIT.md).
