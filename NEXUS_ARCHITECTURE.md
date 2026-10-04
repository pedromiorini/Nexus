# Nexus — Arquitetura observada

## Visão de camadas

```text
Entradas e contratos
        ↓
CentralRouter / componentes do core
        ↓
Memória, episódios, swarm, multimodalidade e planejamento
        ↓
Vita / SQLite / diagnósticos
        ↓
Ferramentas de auditoria e CI
```

## Responsabilidades

| Área | Local | Responsabilidade observada | Limite |
|---|---|---|---|
| Roteamento | `core/constitutional_brain.py` | registra e direciona requisições | núcleo monolítico e dependente de componentes experimentais |
| Fila | `core/deferred_task_queue.py` | prioridade, retry, capacidade, snapshots | snapshot não é persistência física nem exactly-once |
| Memória | `RealHierarchicalMemory` no core | armazenamento e fallback SQL parametrizado | busca semântica depende de dependências opcionais |
| Episódios | `RealEpisodicMemory` no core | episódios, vínculos e médias SQLite | escopo local, sem garantia distribuída |
| Swarm | `RealSwarmIntelligence` no core | votos e médias observadas | não demonstra inteligência coletiva geral |
| Multimodal | `VisionProcessor`/`AudioProcessor` | protocolos opcionais e fallbacks explícitos | sem backend não há inferência visual ou transcrição real |
| Telemetria | `vita/nexus_constitutional_bridge_v3.py` | eventos, transições e diagnósticos | métricas dependem do histórico disponível |
| Segurança | `tools/security_triage.py` + Bandit | inventário conservador | não é clearance de segurança |
| Realidade | `tools/reality_audit.py` | inventário AST e claims documentais | sinaliza riscos, não prova capacidades |

## Fronteiras de integração

- O `CentralRouter` coordena componentes, mas não transforma seus nomes em capacidades gerais.
- O SQLite fornece persistência local para Vita e episódios; concorrência e durabilidade são responsabilidades do consumidor.
- Backends opcionais são injetados por protocolos pequenos e devem manter estado de disponibilidade explícito.
- O CI compila módulos, executa contratos, cobertura, mutation testing, Bandit e auditorias.

## Risco estrutural principal

`core/constitutional_brain.py` concentra aproximadamente 26,6 mil linhas, 267 classes e 843 funções segundo a auditoria atual. Modularização deve ser incremental, orientada por contratos e precedida por evidência local.
