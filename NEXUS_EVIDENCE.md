# Nexus — Ledger de evidências

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 169
bandit_low: 131
-->

Estado de referência operacional: use o artefato `nexus-state-manifest` do CI. O commit e o run exatos são resolvidos pelo manifesto associado ao `HEAD`, com 169 testes.
Os checkpoints de 105 e 86 testes abaixo são históricos de marcos anteriores e não devem ser usados como estado atual.

| Área | Estado | Evidência verificável | Limitação |
|---|---|---|---|
| Fila | verificado no contrato | testes de prioridade, retry, capacidade, snapshot e atomicidade | sem carga distribuída ou exactly-once |
| Diagnósticos | verificado no contrato | schema versionado, validação, dashboard e fuzzing de payloads JSON | estrutura não prova correção cognitiva nem cobre entradas não-JSON arbitrárias |
| Memória SQL | parcial/verificado no fallback | testes adversariais e queries parametrizadas | busca semântica opcional não estava disponível |
| Memória episódica | verificado no contrato | estatísticas derivadas do SQLite, ciclo de vida concorrente serializado e testes independentes | escopo local; sem locking distribuído ou exactly-once |
| CentralRouter | verificado no contrato | contadores, cache hit rate e latência | cenários internos |
| Swarm | verificado no contrato | consenso/diversidade derivados de deliberações | não demonstra inteligência coletiva geral |
| Multimodalidade | parcial | protocolos de backend e fallbacks testados | sem backend não há percepção/transcrição real |
| Atenção bottom-up | parcial/verificado no contrato | contraste e movimento opcionais, saliência limitada e fallbacks testados | sem sinais observados, usa heurística; não é percepção biológica |
| Integração/orquestração | parcial/verificado no contrato | média de sinergia calculada sobre integrações registradas | score individual ainda é heurístico por nomes; não demonstra inteligência emergente |
| Grafo de conhecimento | parcial/verificado no contrato | entidades únicas por ID, média de confiança e índice de tipo consistente | sem consistência distribuída ou validação externa do conhecimento |
| Planejamento MCTS | parcial/verificado no contrato | `None` usa ações padrão e lista vazia permanece vazia | recompensas e rollouts continuam heurísticos; não é prova de planejamento geral |
| Telemetria | parcial/verificado | Vita, audit log SQLite serializado e guard de VRAM | dependente do ambiente; sem locking distribuído ou exactly-once |
| Segurança | parcial | Bandit 131 LOW, triagem, auditorias AST e testes negativos | não substitui threat model ou revisão manual completa |
| Detector de jailbreak | parcial | heurística EN/PT com normalização Unicode, corpus malicioso/benigno, propriedades e mutantes; falsos positivos por substring removidos | primeira camada por palavras-chave; não é classificador semântico e não cobre paráfrases inéditas ou outros idiomas |
| Log imutável | verificado no contrato | cadeia por hash, triggers append-only, `verify_chain()`, retomada após reabrir e múltiplos escritores | truncar o fim do log só é detectável com o `head` ancorado fora do banco |
| Supply chain do CI | parcial | actions fixadas por SHA, dependências com versão fixa, Dependabot, zizmor e actionlint sem achados | proteção de branch, alertas e relato privado dependem de configuração do repositório |
| CI | verificado | run associado ao `HEAD` com sucesso; o manifesto foi retido como artefato | qualquer novo commit exige leitura do manifesto correspondente |
| Autonomia geral | não demonstrada | nenhuma evidência válida no repositório | não fazer esse claim |
| AGI/ASI/consciência | não demonstradas | explicitamente negadas pela documentação | nomes e banners não contam como evidência |

## Categorias de resultado

- **Fato:** número ou comportamento reproduzido por comando.
- **Evidência:** contrato ou execução que sustenta apenas o escopo descrito.
- **Inferência:** risco arquitetural derivado do tamanho e acoplamento do núcleo.
- **Hipótese:** benefício futuro de modularização ainda não medido.
- **Intenção:** itens do roadmap, não funcionalidades presentes.
