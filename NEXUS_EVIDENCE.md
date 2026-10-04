# Nexus — Ledger de evidências

Estado de referência: commit `5f261fcaabffe4e265372bfbc1e5b7c7aa303d3f`, suíte local de 86 testes.

| Área | Estado | Evidência verificável | Limitação |
|---|---|---|---|
| Fila | verificado no contrato | testes de prioridade, retry, capacidade, snapshot e atomicidade | sem carga distribuída ou exactly-once |
| Diagnósticos | verificado no contrato | schema versionado, validação e dashboard | estrutura não prova correção cognitiva |
| Memória SQL | parcial/verificado no fallback | testes adversariais e queries parametrizadas | busca semântica opcional não estava disponível |
| Memória episódica | verificado no contrato | estatísticas derivadas do SQLite e teste independente | escopo local |
| CentralRouter | verificado no contrato | contadores, cache hit rate e latência | cenários internos |
| Swarm | verificado no contrato | consenso/diversidade derivados de deliberações | não demonstra inteligência coletiva geral |
| Multimodalidade | parcial | protocolos de backend e fallbacks testados | sem backend não há percepção/transcrição real |
| Atenção bottom-up | parcial/verificado no contrato | contraste e movimento opcionais, saliência limitada e fallbacks testados | sem sinais observados, usa heurística; não é percepção biológica |
| Integração/orquestração | parcial/verificado no contrato | média de sinergia calculada sobre integrações registradas | score individual ainda é heurístico por nomes; não demonstra inteligência emergente |
| Telemetria | parcial/verificado | Vita, SQLite e guard de VRAM | dependente do ambiente e dos recursos disponíveis |
| Segurança | parcial | Bandit 129 LOW, triagem, auditorias AST e testes negativos | não substitui threat model ou revisão manual completa |
| CI | verificado | último run conhecido `37097334156` com sucesso para `e523d9f` | qualquer novo commit exige nova execução |
| Autonomia geral | não demonstrada | nenhuma evidência válida no repositório | não fazer esse claim |
| AGI/ASI/consciência | não demonstradas | explicitamente negadas pela documentação | nomes e banners não contam como evidência |

## Categorias de resultado

- **Fato:** número ou comportamento reproduzido por comando.
- **Evidência:** contrato ou execução que sustenta apenas o escopo descrito.
- **Inferência:** risco arquitetural derivado do tamanho e acoplamento do núcleo.
- **Hipótese:** benefício futuro de modularização ainda não medido.
- **Intenção:** itens do roadmap, não funcionalidades presentes.
