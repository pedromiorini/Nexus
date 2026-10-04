# Auditoria técnica do Nexus

**Data da auditoria:** 17 de setembro de 2026.  
**Escopo:** realidade das capacidades, consistência documental, qualidade dos testes, segurança estática, dependências opcionais e lacunas de engenharia.

## Sumário executivo

O Nexus é, de forma comprovável, um **protótipo experimental de arquitetura neuro-simbólica em Python** com um núcleo monolítico grande, componentes de roteamento, governança, memória, fila de tarefas adiadas, telemetria Vita, auditoria persistente, exportação versionada e mecanismos de proteção associados à pressão de memória. A implementação possui comportamento executável e testes úteis.

A auditoria **não encontrou evidência suficiente** para afirmar AGI, ASI, consciência, autoconsciência, autonomia geral, soberania operacional ou prontidão de produção. Os nomes dos módulos e os banners do executável são mais fortes que a evidência medida. Essa distinção foi incorporada ao README e automatizada por `tools/reality_audit.py`.

A auditoria também identificou uma inconsistência documental concreta: o README antigo referenciava `NEXUS_CORE_v3.94.py`, arquivo inexistente após a consolidação. A documentação foi corrigida para apontar os caminhos atuais e registrar limitações.

## O que foi comprovado

| Capacidade | Evidência executada | Interpretação correta |
|---|---|---|
| Roteamento constitucional | O núcleo integrado executou 29 verificações embutidas com saída de sucesso | Demonstra funcionamento de cenários internos específicos |
| Fila de tarefas adiadas | 14 testes unitários e de integração aprovados | Prioridade, capacidade, retry, descarte, callbacks e preservação de contexto estão cobertos |
| Reprocessamento adaptativo | Testes cobrem estados ativo, degradado e pausado, bloqueio por pressão crítica e métricas | A política local funciona nos cenários simulados; não prova operação distribuída de produção |
| Telemetria Vita | Histórico persistente, análise de recuperação e exportação versionada executados | O diagnóstico é estruturado e auditável dentro do processo |
| Contrato de observabilidade | Schema `nexus.recovery.diagnostics.v1`, validação e round-trip JSON aprovados | A estrutura do payload é verificável; a correção semântica das decisões não é garantida |
| Propriedades da fila | 4 testes Hypothesis aprovados em 40, 30, 25 e 30 exemplos configurados | Invariantes importantes foram exercitadas em múltiplas entradas, sem substituir testes de carga |
| CI | Workflow executa compilação, contrato, auditoria, cobertura, regressão, propriedades, segurança e suíte integrada | Existe um gate automatizado; o scan de segurança permanece não bloqueante enquanto os achados são classificados |

## Resultados quantitativos

A análise AST do núcleo encontrou **26.500 linhas**, **267 classes** e **842 funções**. Foram detectadas **2 funções compostas somente por `pass`**, **66 funções com retorno constante simples** e **30 marcadores de testes embutidos**. Esses números não significam automaticamente defeitos: alguns retornos constantes são heurísticas intencionais, mas todos exigem revisão antes de serem usados como prova de capacidade.

A suíte externa executada nesta auditoria totalizou **29 testes aprovados**: cinco de auditoria de realidade, seis de contrato, quatorze de fila/telemetria e quatro baseados em propriedades. A suíte integrada legada também terminou com **exit code 0** e exibiu `TEST 29 PASSED` para o último cenário. O relatório de cobertura medido nos componentes exercitados registrou aproximadamente **89%** para a fila, **71%** para a ponte Vita, **97%** para propriedades da fila, **94%** para a auditoria e **21%** no conjunto amplo medido, fortemente reduzido pelo núcleo monolítico de aproximadamente dez mil statements instrumentados.

## Verificação de alegações e alucinações

O auditor conservador classifica alegações positivas sem evidência como `unsubstantiated_by_software_tests` e disclaimers explícitos como `explicitly_disclaimed`. Isso é importante porque a documentação histórica misturava descrição de protótipo com linguagem de perfeição, AGI, ASI e consciência.

| Alegação | Estado após a revisão |
|---|---|
| AGI/ASI | Explicitamente negada no README atual; não tratada como capacidade do projeto |
| Consciência/autoconsciência | Ainda aparece em código legado e nomes de demonstração; não há evidência científica ou teste independente que a comprove |
| 100% completo / perfeição absoluta | Removida da descrição factual do README |
| 62/62 módulos funcionais | Não aceita como fato geral; a própria execução imprime `36/55 Modules (65.5%)`, o que contradiz a documentação histórica |
| Zero warnings / zero exceções | Não aceito como garantia; o processo imprime avisos quando dependências opcionais estão ausentes |
| Pronto para produção | Não comprovado; faltam testes adversariais, carga, isolamento, threat model, deployment reproduzível e observabilidade externa |

A contradição mais relevante é entre os banners legados que afirmam “62/62” e a saída efetiva que reporta “36/55 Modules (65.5%)”. Isso não deve ser escondido: foi registrado como lacuna documental e motivou a substituição das alegações absolutas.

## Segurança e qualidade

O Bandit encontrou inicialmente dois usos de MD5 em chaves de cache. Eles foram substituídos por SHA-256 sem alterar o contrato funcional da cache. A análise posterior deixou **131 achados**, todos de severidade baixa ou de confiança média/baixa: 16 usos de pseudoaleatoriedade não criptográfica, dois `try/except/pass`, 112 `assert` em testes embutidos e um B608 de confiança baixa associado à montagem de placeholders SQL. O B608 foi revisado: os valores permanecem parametrizados e apenas a quantidade de placeholders é montada a partir da contagem de palavras. O relatório Bandit continua sendo retido como artefato não bloqueante para permitir revisão incremental.

A auditoria não transforma esse resultado em “zero vulnerabilidades”. O Bandit é uma análise AST de padrões conhecidos; ele não substitui revisão manual, threat modeling, testes de autorização, fuzzing ou análise de dependências. A documentação oficial descreve precisamente esse escopo [3].

## Melhorias implementadas

A revisão aplicou as seguintes mudanças concretas:

1. O README foi reescrito para linguagem baseada em evidências, com matriz de capacidades, limitações e comandos reproduzíveis.
2. `tools/reality_audit.py` foi criado para comparar documentação, caminhos referenciados, estrutura AST, marcadores de testes, funções vazias e retornos constantes.
3. `test_reality_audit.py` adicionou cinco testes independentes para impedir regressões na auditoria documental.
4. `test_deferred_task_properties.py` adicionou quatro propriedades Hypothesis para prioridade, pressão crítica, retry e preservação de contexto.
5. O workflow CI passou a instalar Hypothesis e Bandit, executar a auditoria, reter `AUDIT_REALITY_REPORT.md`, `audit_reality.json` e `bandit-report.json`, além da cobertura existente.
6. Os dois hashes MD5 de cache foram substituídos por SHA-256.
7. O relatório `research_findings.md` documenta as fontes e a justificativa para cobertura de branches, property-based testing, análise AST e mutation testing.

## Lacunas que permanecem

A primeira lacuna é arquitetural: o núcleo tem aproximadamente 26,5 mil linhas e concentra dezenas de componentes, o que aumenta acoplamento, custo de revisão e risco de efeitos laterais. A segunda é epistemológica: os testes legados são demonstrações auto-confirmatórias; eles verificam contadores, objetos e retornos esperados, mas não medem generalização, robustez ou validade externa. A terceira é operacional: dependências opcionais produzem fallback de palavras-chave quando a busca semântica não está disponível, alterando a qualidade do comportamento.

Também faltam testes de carga e concorrência para SQLite e fila, política explícita de retenção e migração do banco de auditoria, threat model para ferramentas e entradas externas, fuzzing do parser e dos contratos, mutation testing direcionado nos módulos críticos e avaliação independente de alegações cognitivas. O scan de segurança ainda reporta asserts embutidos e blocos de exceção silenciosos; eles devem ser tratados gradualmente, sem remover asserts de demonstrações sem preservar a intenção dos testes.

## Soluções pesquisadas e decisão

A documentação oficial do coverage.py define branch coverage como comparação entre transições possíveis e transições observadas, confirmando que a métrica é útil, mas não equivale a correção semântica [1]. A documentação oficial do Hypothesis recomenda declarar propriedades e gerar entradas, incluindo casos de borda que não foram antecipados manualmente [2]. A documentação oficial do Bandit descreve análise AST com plugins para problemas comuns de segurança [3]. O mutmut foi selecionado como candidato para mutation testing incremental, e a literatura recente trata mutation testing como forma de medir se os testes detectam falhas injetadas, além de cobertura [4].

A decisão foi **não adicionar mutation testing amplo nesta rodada**, porque o núcleo é monolítico e o custo/ruído seriam altos. A infraestrutura agora está preparada para aplicar mutation testing primeiro em `deferred_task_queue.py`, na ponte Vita e no validador de schema, onde o contrato é pequeno e a interpretação dos resultados é mais confiável.

## Veredito

O Nexus está **funcional em cenários internos demonstrados**, com melhorias reais de fila, telemetria, auditoria, contratos, testes de propriedades e CI. Ele não está comprovado como inteligência geral, superinteligência, consciência ou sistema autônomo de produção. A documentação atualizada reduz esse risco de alucinação, e o auditor automatizado cria uma barreira contra o retorno de afirmações absolutas.

O próximo marco sugerido anteriormente continua válido: estabelecer um limiar gradual de cobertura apenas para módulos críticos. A ordem recomendada é ponte Vita e validador de schema, depois fila de tarefas, e só então componentes maiores do núcleo. O limiar deve ser acompanhado por mutation testing direcionado, pois cobertura isolada não mede a força dos oráculos.

## Referências

[1]: https://coverage.readthedocs.io/en/latest/branch.html "Coverage.py — Branch coverage measurement"
[2]: https://hypothesis.readthedocs.io/ "Hypothesis — documentação oficial"
[3]: https://bandit.readthedocs.io/en/latest/ "Bandit — documentação oficial"
[4]: https://dl.acm.org/doi/10.1145/3701625.3701659 "Static and Dynamic Comparison of Mutation Testing Tools for Python"
[5]: https://github.com/boxed/mutmut "mutmut — Mutation testing system for Python"

## Atualização posterior — 1º de outubro de 2026

O fallback de busca por palavras em `RealHierarchicalMemory` foi endurecido após revisão do achado Bandit B608. A implementação deixou de montar uma cláusula `WHERE` variável e passou a executar uma consulta estática parametrizada por termo, unindo resultados por ID e preservando a ordenação por importância e acesso. O teste adversarial SQL foi ampliado para cobrir busca OR por múltiplas palavras e limite.

Uma nova execução do Bandit registrou **132 achados LOW e nenhum MEDIUM**; a triagem continua conservadora e não trata isso como prova de ausência de vulnerabilidades. A revisão manual de limites de confiança, concorrência e comportamento do núcleo legado permanece necessária.

## Atualização posterior — correção de exceções silenciosas

O motor `RealErrorDetectionCorrectionEngine` passou a capturar somente `ValueError` e `IndexError` ao interpretar limites numéricos, preservando o conteúdo original para descrições malformadas. O contrato foi coberto por `test_error_correction_contract.py`.

A execução posterior do Bandit registrou **130 achados LOW**, sem B110 e sem B608. Isso reduz padrões conhecidos, mas não substitui revisão manual do núcleo monolítico.

## Atualização posterior — revisão de aleatoriedade não criptográfica

Os 16 achados B311 em `core/constitutional_brain.py` foram revisados como usos de simulação, exploração, ruído heurístico ou escolhas não relacionadas a segredo, autenticação ou decisão de segurança. A triagem agora os classifica como `simulation_only_random_review`, mantendo cada achado aberto e exigindo que futuras mudanças não reutilizem esses caminhos para material secreto ou controles de segurança.

## Atualização posterior — revisão dos asserts B101

Os 112 achados B101 foram localizados exclusivamente no bloco `if __name__ == "__main__"` de demonstração integrada em `core/constitutional_brain.py`, entre as linhas 24843 e 26495. Eles permanecem abertos sob `embedded_demo_assert_review`: não são tratados como contratos independentes de produção e devem ser migrados incrementalmente para testes isolados, sem uma substituição em massa que altere a demonstração sem cobertura equivalente.

## Atualização posterior — guardrail de migração dos asserts

Foi adicionado `tools/embedded_assert_audit.py`, que analisa o AST do arquivo e verifica que cada achado B101 do relatório Bandit permanece dentro do bloco `if __name__ == "__main__"`. A execução atual encontrou **112** B101 e **zero** fora do bloco (`outside_main_block: []`). O auditor não considera os asserts seguros nem os suprime; ele impede a expansão silenciosa do padrão enquanto a migração para testes isolados ocorre incrementalmente.

## Atualização posterior — bloqueio de asserts fora do demo

O `embedded_assert_audit` agora verifica duas fontes: os achados B101 do Bandit e todos os nós `ast.Assert` encontrados diretamente no código. A execução atual confirmou `outside_main_block: []` e `assert_lines_outside_main: []`, impedindo que um assert novo fora do bloco demonstrativo passe despercebido por uma alteração no scanner.

## Atualização posterior — primeiro B101 migrado

O assert do demo que verificava `router_stats['total_requests'] >= 3` foi substituído por uma falha explícita de demonstração (`RuntimeError`) e seu comportamento foi coberto independentemente por `test_central_router_contract.py`, que valida contagem de requisições, cache hit rate e latência média. O inventário B101 caiu de 112 para **111**; o guardrail continua confirmando `outside_main_block: []` e `assert_lines_outside_main: []`.

## Atualização posterior — backends multimodais opcionais

Os métodos `VisionProcessor.detect_objects`, `VisionProcessor.analyze_scene` e `AudioProcessor.transcribe_speech` deixaram de depender exclusivamente de retornos fixos. Eles agora aceitam backends opcionais com protocolos pequenos e preservam fallback de demonstração quando nenhum backend é fornecido. `test_multimodal_backend_contract.py` cobre os dois caminhos. A auditoria AST caiu de 66 para **63 funções com retorno constante simples**, sem alterar a conclusão conservadora sobre capacidades.

## Atualização posterior — métricas observadas do swarm

`RealSwarmIntelligence.get_statistics()` deixou de expor médias fixas (`0.63`) e passou a calcular consenso e diversidade a partir das deliberações realizadas. O contrato `test_swarm_statistics_contract.py` cobre médias acumuladas, estado inicial e falha explícita quando não há agentes.

## Atualização posterior — métricas observadas da memória episódica

`RealEpisodicMemory.get_statistics()` deixou de reportar `total_links: 0` e `avg_importance: 0.7` fixos. Os valores agora são derivados das tabelas SQLite de episódios e vínculos, com `0.0` somente quando não há episódios. O contrato `test_episodic_memory_statistics_contract.py` cobre estado vazio, média persistida, contagem de vínculos e fechamento de episódio. A auditoria AST passou a registrar **61 funções com retorno constante simples**.


## Atualização posterior — sinais observados de atenção

`StimulusItem` passou a aceitar `contrast` e `motion` opcionais. `BottomUpAttention` usa os sinais quando fornecidos, aceita contraste no contexto e preserva fallbacks heurísticos explícitos quando não há observação. `test_attention_signal_contract.py` cobre limites, sinais presentes, contexto e ausência de sinais. A mudança não reivindica percepção real nem altera a classificação conservadora do sistema.
