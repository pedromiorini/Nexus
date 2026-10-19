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

No commit `8c4f6a8057ee4d6a5497fa2df54c26725d9ef66d`, `core/constitutional_brain.py` tinha 27.872 linhas, 269 classes e 887 funções, conforme o manifesto de estado gerado em 2026-10-19. Essas contagens são inventário da superfície, não medida de complexidade ou qualidade. Modularização deve ser incremental, orientada por contratos e precedida por evidência local.


## Contrato operacional do roteador (D-034)

Em cache miss, o preflight de orçamento fecha a rota em negação/ausência/erro. A reserva de uma unidade é atômica no enforcer e ocorre na fronteira da chamada ao reasoner; falha conta como tentativa e cache hit não cobra. Sucesso de orquestração agora exige o produto funcional mínimo por tipo e o Swarm imposto pelo guard ético; falha requerida não é agregada/cacheada como sucesso e permanece sujeita a retry/descarte da fila. Enriquecimento opcional não é requisito de conclusão.

A garantia é local à API do budget e ao CentralRouter, não à rota separada `think()` nem à concorrência geral do processo. CREATIVE e LEARNING preservam apenas seus produtos legados (comparação de futuros e registro de intenção), sem alegação de geração criativa ou aprendizado executado. Mapa, compatibilidade e limites em [D-034](NEXUS_DECISIONS.md#d-034--orçamento-e-conclusão-operacional-do-centralrouter).
