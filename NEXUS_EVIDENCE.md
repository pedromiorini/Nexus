# Nexus — Ledger de evidências

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 222
bandit_low: 131
-->

Snapshot histórico: manifesto `nexus-state-manifest` do commit `5b8bed8125002e25eb066620af67d3a65d44e59f` (CI run `38907987512`), com 200/200 testes `unittest`, 31 verificações integradas inventariadas (30 aprovadas, 1 ignorada por FAISS opcional ausente e 0 falhas) e 131 achados Bandit LOW (0 MEDIUM/HIGH). Esse snapshot e os resultados locais do benchmark v1 não representam o estado atual.
Esta auditoria de segurança parte do `main` pós-PR #22 (`27299281b636deddb5555173a9f191dd01e65013`). Os 222/222 testes e as demais métricas desta branch são locais; ainda não substituem o manifesto de CI associado ao head da revisão.
Os checkpoints de 105 e 86 testes abaixo são históricos de marcos anteriores e não devem ser usados como estado atual.

| Área | Estado | Evidência verificável | Limitação |
|---|---|---|---|
| Fila | verificado no contrato | testes de prioridade, retry, capacidade, snapshot e atomicidade | sem carga distribuída ou exactly-once |
| Log imutável | verificado no contrato | triggers append-only, verificação da cadeia, detecção de adulteração/remoção, reabertura e escritores concorrentes | truncamento final ou rehash por quem controla o arquivo exige âncora externa |
| Detector de jailbreak | testado por regressão EN/PT | corpus benigno/malicioso, NFKC, caracteres invisíveis, homoglifos, leetspeak e entradas longas | heurística, não fronteira de segurança semântica nem substituto de controle de privilégio |
| Diagnósticos | verificado no contrato | schema versionado, validação, dashboard e fuzzing de payloads JSON | estrutura não prova correção cognitiva nem cobre entradas não-JSON arbitrárias |
| Memória SQL | parcial/verificado no fallback | testes adversariais e queries parametrizadas | busca semântica opcional não estava disponível |
| Memória episódica | verificado no contrato | estatísticas do SQLite, ciclo de vida concorrente, validação de `memory_id` quando há backend resolutor e recuperação por provedor falsey | sem backend, o ID é uma referência opaca não validada; sem locking distribuído ou exactly-once |
| CentralRouter | verificado no contrato | contadores/cache e enqueue de emergência mesmo com marcador reservado no contexto do caller | cenários internos; o teste simula a pressão VRAM |
| Swarm | verificado no contrato | consenso/diversidade derivados de deliberações | não demonstra inteligência coletiva geral |
| Multimodalidade | parcial | protocolos de backend e fallbacks testados | sem backend não há percepção/transcrição real |
| Atenção bottom-up | parcial/verificado no contrato | contraste e movimento opcionais, saliência limitada e fallbacks testados | sem sinais observados, usa heurística; não é percepção biológica |
| Integração/orquestração | parcial/verificado no contrato | média de sinergia calculada sobre integrações registradas | score individual ainda é heurístico por nomes; não demonstra inteligência emergente |
| Grafo de conhecimento | parcial/verificado no contrato | entidades únicas por ID, média de confiança e índice de tipo consistente | sem consistência distribuída ou validação externa do conhecimento |
| Planejamento MCTS | parcial/verificado no contrato | `None` usa ações padrão e lista vazia permanece vazia | recompensas e rollouts continuam heurísticos; não é prova de planejamento geral |
| Telemetria | parcial/verificado | Vita, audit log SQLite serializado e guard de VRAM | dependente do ambiente; sem locking distribuído ou exactly-once |
| Segurança | parcial | Bandit 131 LOW, triagem, auditorias AST e testes negativos | não substitui threat model ou revisão manual completa |
| CI | verificado por commit | run associado ao `HEAD` e manifesto retido como artefato quando disponível | qualquer novo commit exige leitura do manifesto correspondente |
| Benchmark sintético | verificado localmente no protocolo v1 | 39 sequências (holdout exato 1,0), extração causal lexical (F1 0,667), alegação de intervenção (F1 0,667), trilha de confiança (Brier 0,1849; ECE 0,01684) e JSON determinístico com SHA-256 | sem treino; causalidade textual/heurística; incerteza recebe probabilidades externas; crença POMDP não suportada; não valida teorias cognitivas |
| Autonomia geral | não demonstrada | nenhuma evidência válida no repositório | não fazer esse claim |
| AGI/ASI/consciência | não demonstradas | explicitamente negadas pela documentação | nomes e banners não contam como evidência |

## Verificação local da branch de correção — 2026-10-16

- Os contratos recém-adicionados reproduziram antes do patch: resposta `deferred` sem ID/enqueue quando o contexto continha `_deferred_reprocess=True`; link a memória inexistente aceito e contado; provedor resolutor falsey tratado como ausente. Após o patch, os contratos focados passaram.
- Suíte `unittest`: 213 executados, 213 aprovados, 0 falhas, 0 skips e 0 sucessos inesperados. Suíte integrada: 31 IDs inventariados, 30 aprovados, 1 skip FAISS opcional, 0 falhas.
- Cobertura: fila 95%, VRAM 94%, Vita 76% e núcleo 67%; mutation testing: 28/28 mortos, 0 sobreviventes/não aplicados; Bandit: 131 LOW e 0 MEDIUM/HIGH.
- Ruff, actionlint, zizmor, compileall, auditorias AST e checks de requirements passaram localmente. `AUDIT_REALITY_REPORT.md` e `audit_reality.json` foram regenerados e refletem 26.946 linhas e 859 funções no núcleo.
- POMDP continua não suportado e sem score no protocolo v1; não foi inferido posterior nem alterado o resultado do benchmark.

## Autoauditoria de recursos do `RealToolUse` — 2026-10-16

- Antes do patch, `execute_tool("math_calculate", {"expression": '"x" * 1000000'})` retornou sucesso com 1.000.000 caracteres e reteve o valor no cache; `2 ** 10000` também retornou sucesso. Depois de marcar uma entrada expirada e fazer outra chamada, a entrada antiga continuou retida.
- O patch rejeita constantes não numéricas, expressões acima de 256 caracteres, AST acima de 64 nós, expoentes acima de 1024 e resultados fora dos limites numéricos; a expressão longa é rejeitada antes de serializar a chave de cache.
- O cache agora expurga TTL vencido, limita 128 entradas, 1 MiB de payload estimado e 64 KiB por resultado; armazena e devolve cópias profundas para impedir que o chamador mutile a entrada após a medição. Resultados grandes de outras ferramentas ainda são retornados ao chamador, mas não armazenados no cache.
- Contratos: 9 novos testes cobrem aritmética comum, string/boolean/NaN, comprimento/complexidade/expoente, rejeição antes da cache key, TTL, alias mutável, limite por resultado e capacidade total. Suíte: 222/222; integração: 30 aprovados/1 skip FAISS; cobertura: fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation: 28/28 mortos, 0 sobreviventes/não aplicadas; Bandit: 131 LOW, nenhum MEDIUM/HIGH.
- Limites remanescentes: `ToolDefinition.timeout_seconds` ainda é declarativo para executores genéricos; o cache limita payload estimado via `sys.getsizeof`, não a memória total do processo; o estado é local e aguarda CI oficial.

## Categorias de resultado

- **Fato:** número ou comportamento reproduzido por comando.
- **Evidência:** contrato ou execução que sustenta apenas o escopo descrito.
- **Inferência:** risco arquitetural derivado do tamanho e acoplamento do núcleo.
- **Hipótese:** benefício futuro de modularização ainda não medido.
- **Intenção:** itens do roadmap, não funcionalidades presentes.
