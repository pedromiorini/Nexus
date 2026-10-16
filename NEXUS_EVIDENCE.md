# Nexus — Ledger de evidências

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 173
bandit_low: 131
-->

Estado oficial de referência: use o artefato `nexus-state-manifest` associado ao `HEAD`. Nesta branch, 173/173 testes passaram localmente; a cobertura da fila foi 96%, a do bridge Vita 74% e a suíte integrada registrou 30 aprovados e 1 ignorado. O benchmark v1 (seed `20261016`) mediu acerto composicional exato 1,0 em 39 casos, F1=0,667 na extração lexical causal, F1=0,667 nas alegações de efeito downstream, Brier=0,1849 e ECE=0,01684 na trilha que recebe probabilidades como entrada; observação parcial foi marcada `unsupported_interface`, sem acurácia inventada. Esses resultados locais não substituem o manifesto de CI do commit.
Os checkpoints de 105 e 86 testes abaixo são históricos de marcos anteriores e não devem ser usados como estado atual.

| Área | Estado | Evidência verificável | Limitação |
|---|---|---|---|
| Fila | verificado no contrato | testes de prioridade, retry, capacidade, snapshot e atomicidade | sem carga distribuída ou exactly-once |
| Log imutável | verificado no contrato | triggers append-only, verificação da cadeia, detecção de adulteração/remoção, reabertura e escritores concorrentes | truncamento final ou rehash por quem controla o arquivo exige âncora externa |
| Detector de jailbreak | testado por regressão EN/PT | corpus benigno/malicioso, NFKC, caracteres invisíveis, homoglifos, leetspeak e entradas longas | heurística, não fronteira de segurança semântica nem substituto de controle de privilégio |
| Diagnósticos | verificado no contrato | schema versionado, validação, dashboard e fuzzing de payloads JSON | estrutura não prova correção cognitiva nem cobre entradas não-JSON arbitrárias |
| Memória SQL | parcial/verificado no fallback | testes adversariais e queries parametrizadas | busca semântica opcional não estava disponível |
| Memória episódica | verificado no contrato | estatísticas derivadas do SQLite, ciclo de vida concorrente serializado, contagem somente em transições `open` → `closed` e testes independentes | escopo local; sem locking distribuído ou exactly-once |
| CentralRouter | verificado no contrato | contadores, cache hit rate e latência | cenários internos |
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

## Categorias de resultado

- **Fato:** número ou comportamento reproduzido por comando.
- **Evidência:** contrato ou execução que sustenta apenas o escopo descrito.
- **Inferência:** risco arquitetural derivado do tamanho e acoplamento do núcleo.
- **Hipótese:** benefício futuro de modularização ainda não medido.
- **Intenção:** itens do roadmap, não funcionalidades presentes.
