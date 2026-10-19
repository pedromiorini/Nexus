# Nexus — Continuidade entre agentes

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 477
bandit_low: 135
-->

## Isolamento de head_hash — 2026-10-19

Candidato local `fix/temporal-head-lock-20261019`, worktree `/workspace/Nexus-temporal-head-lock-20261019`, filho direto do SHA exato da #49 `8f8834efba9011cd0d8a4bdfb449d3e862ee6814`. SHA final resolvido por `HEAD` e pelo manifesto externo `/workspace/nexus-temporal-head-lock-20261019-evidence/STATE_MANIFEST.json`. `origin/main` local permanece `812ebd02b0c82ecc3169d620e36d25592b1d6727`. Sem consulta/publicação remota, push, nova PR, alteração de #49/#48/#47, merge, auto-merge ou mudança de main. Checkout sujo `/workspace/Nexus` e demais worktrees preservados; inventários externos de refs/heads e comparação do diff/status protegido registram a conferência. Somente leituras de inspeção foram feitas no checkout protegido.

**Contrato D-027:** `head_hash` usa o mesmo `self._lock` das escritas. Leitor concorrente aguarda commit/rollback e retorna somente o head resultante, nunca um head intermediário ou posteriormente revertido. Mudança de produção restrita ao getter; assinatura, valor normal, ledger, transação e schema permanecem. Testes com Events e sonda do RLock real pausam `_apply` após publicação de `_head`, sem sleeps nem inferência de bloqueio por demora. Cobrem assert inicial, substituição composta (head intermediário distinto do final), corroboração e retração, com sucesso e RuntimeError; verificam snapshot pós-rollback, replay/invariantes e leitura normal/reentrante.

**Red/green:** `bash /workspace/nexus-temporal-head-lock-20261019-evidence/run-red-green.sh` cria archive temporário da base exata, adicionando somente o teste novo: 3 testes, **8 falhas de subteste** na base; candidato **45/45 focados**. `repeated.log`: 100 repetições, 300 testes e 800 cenários concorrentes aprovados. Python 3.12.3: `python3 -m unittest -v test_temporal_head_hash_concurrency_contract test_temporal_facts_contract test_audit_atomic_restore_regression`, **39/39**. Logs externos `red.log`, `red-archive.log`, `green-focused.log`, `green-reproduced.log`, `python-default-focused.log`.

**Gates completos locais:** `bash /workspace/nexus-temporal-head-lock-20261019-evidence/run-gates.sh`, execução `20261019T164801Z-624895`, Python 3.11.16, numpy 2.4.6/NetworkX 3.6.1. Perfil/lock com hashes conferidos em dry-run, sem mudança de dependências. Suíte integral, cobertura e sumário: **477/477**, zero falhas/skips/XPASS; integração separada **30 aprovados + 1 skip FAISS**, sem missing/unexpected/conflicting. Cobertura branch fila/VRAM/temporal/Vita/núcleo **91/94/95/76/70%**, pisos **90/90/90/70/62%**. Mutação canônica inalterada **102/102 mortas**, zero sobreviventes/timeouts/não aplicadas; a remoção do lock é exercitada pelo red/green, não somada ao catálogo. Bandit **135 LOW, 0 MEDIUM/HIGH**. Ruff, compileall, actionlint, zizmor offline, triagem/check, B101, gaps, realidade byte a byte sem diff e diff-check passaram. Benchmarks CLI herdados reexecutados somente como checks sintéticos internos, não nova avaliação de capacidade. As invocações individuais repetidas da CI estão no discover, não todas repetidas separadamente. Logs, comandos exatos, checks e relatórios usam esse prefixo no diretório externo.

**Congelamento/revisão/commit:** `frozen.diff` e seu SHA-256 externo identificam o patch completo. Uma única revisão independente, somente leitura, fica em `independent-review.md`; commit local condicionado à ausência de bloqueadores e à identidade do staged com o congelado. Reabrir somente por achado concreto ou alteração do diff. `bash /workspace/nexus-temporal-head-lock-20261019-evidence/run-final-checks.sh 20261019T164801Z-624895 pre|post` reexecuta focados/sumário/lint/compile/realidade/triagem e, após o commit, confere patch commitado, refs/worktrees, manifesto e consistência. Stage somente dos sete caminhos explícitos; sem artefatos gerados no commit. NEXUS_KNOWLEDGE/SECURITY_TRIAGE mudam apenas o marcador 474→477 exigido pela consistência.

**Limites:** sincronização de uma leitura, não snapshot atômico de várias chamadas nem justiça/latência limitada do lock. Callbacks reentrantes da própria thread escritora, acesso a atributos privados, mutações externas, outras concorrências e atomicidade grafo↔ledger não ganham garantia. Ledger continua em memória, sem autenticação criptográfica ou persistência; não há nova alegação de segurança de produção ou capacidade cognitiva. Gates completos somente no perfil Python 3.11; 3.12 recebeu apenas os 39 testes temporais acima. GPU/backends opcionais não reavaliados; evidência local, não CI remoto.

Os registros seguintes são históricos, não resultados desta branch.

## A07 — Preflight de atributos da projeção NetworkX — 2026-10-19

Branch local nova `fix/kg-projection-preflight-a07-20261019`, worktree `/workspace/Nexus-kg-preflight-a07-20261019`, filha direta do A06 local exato `9fea97cd188b8f7212eb88455747ca8a1b3bb220`. SHA final: `HEAD`, resolvido após o commit pelo manifesto externo `/workspace/nexus-kg-preflight-a07-20261019-evidence/STATE_MANIFEST.json`. `origin/main` local inventariado em `812ebd02b0c82ecc3169d620e36d25592b1d6727`. Não houve consulta/publicação remota, push, PR, merge ou comando no checkout `/workspace/Nexus`; #48 e todas as branches/worktrees preexistentes devem permanecer preservadas. Os inventários externos de refs/worktrees permitem conferir seus heads após o commit.

**Contrato D-029/A07:** `add_relation` prepara atributos e cópia da lista `relations` do NetworkX antes de ledger, criação de entidades, índices ou arestas. Chaves não `str` e colisões com os argumentos `u_of_edge`/`v_of_edge` geram `TypeError` sem mutação de estado. Preserva chaves subclasses de `str`, valores opacos, atributos estruturais prevalecentes, múltiplas relações por par e timestamp capturado no ponto legado. Com store, mantém cópia rasa de propriedades e metadata temporal vazia; sem store, mantém a referência legada de `Relation.properties`. O fallback sem NetworkX conserva propriedades com chaves não string e não recebe validação de kwargs.

**Red/green real:** `bash /workspace/nexus-kg-preflight-a07-20261019-evidence/run-red-green.sh` usa archive da base exata com somente a regressão nova. Quatro testes na base produzem **40 falhas de subteste**; corrigidos, **4/4**. A sonda sem mocks reproduz `properties={7: "bad-key"}`: base cria duas entidades e uma relação indexada, mas nenhuma aresta e `total_relations=0`; com store, também altera o snapshot do ledger. Matriz negativa cobre vazio/populado, endpoints novos/existentes, com/sem store, chaves não string e colisões de argumentos; compara conteúdo e identidade de índices, entidades e relações e verifica zero chamadas a append/criação/aresta. Controles positivos cobrem propriedades reservadas, opacas, subclasses de string, defaults, histórico e fallback. Focados completos: **64/64**.

**Gates medidos:** `bash /workspace/nexus-kg-preflight-a07-20261019-evidence/run-gates.sh`, execução final `20261019T143143Z-609223`, Python 3.11.16, numpy 2.4.6/NetworkX 3.6.1. Lock com hashes conferido em dry-run (13 pacotes, sem alterações). Suíte completa, cobertura e sumário: **474/474**, zero falhas/skips/XPASS. Integração separada: **30 aprovados + 1 skip FAISS**, sem missing/unexpected/conflicting. Coberturas branch fila/VRAM/temporal/Vita/núcleo: **91/94/95/76/70%**, acima dos pisos **90/90/90/70/62%**. Mutação canônica **102/102 mortas**, zero sobreviventes/timeouts/não aplicadas (98 herdadas + 4 novas); a âncora do mutante opt-in temporal foi restringida ao append para não colidir com a preparação nova. Bandit **135 LOW, 0 MEDIUM/HIGH**. Ruff, actionlint, zizmor offline, compileall, requisitos, triagem/check, B101, gaps e diff-check passaram. Benchmarks CLI herdados reexecutados apenas como checks sintéticos internos. Os módulos individuais repetidos pela CI estão no discover, não todos reexecutados separadamente.

A primeira rodada `20261019T142854Z-607463` parou no gate de realidade: a contagem gerada do núcleo passou de 27867 para 27872 linhas. Incluídos os dois relatórios regenerados; a rodada final conferiu reprodução byte a byte desses arquivos. Não houve supressão de finding nem modificação do gate versionado. Em NEXUS_KNOWLEDGE/SECURITY_TRIAGE mudou apenas o marcador 470→474 exigido pela consistência. Logs, comandos e JSONs ficam no diretório externo de evidências.

**Diff congelado e revisão:** o patch completo é congelado em `frozen.diff` com SHA-256 externo antes de uma única revisão independente final, somente leitura; o resultado é registrado em `independent-review.md` no mesmo diretório externo. O commit local fica condicionado à ausência de bloqueadores e à igualdade do patch staged com o congelado. Não reabrir salvo achado concreto ou alteração do diff. `run-final-checks.sh <prefixo-da-execução> pre|post` reexecuta focados/sumário, lint, compile, realidade/triagem e, pós-commit, gera manifesto/consistência, compara o diff commitado com o congelado e confere refs/worktrees. A atestação identifica resultados locais, não CI remoto.

**Limites:** a ordem continua preflight → ledger → grafo; não se promete atomicidade para falhas arbitrárias posteriores, rollback compartilhado ou transação distribuída. Backend, alocação, hooks Python, mutação externa/concorrente e `add_entity` chamado diretamente não ficam protegidos por esta mudança. Nenhuma persistência, sincronização geral ou validação JSON de propriedades foi adicionada. Gates completos somente no perfil Python 3.11; outras versões/GPU não reavaliadas. São contratos de software, não prova de capacidade cognitiva ou segurança de produção.

Os registros seguintes são históricos, não resultados desta branch.

## A06 — Prioridade exata no enqueue — 2026-10-19

Branch local nova `fix/queue-priority-exact-int-20261019`, worktree `/workspace/Nexus-queue-priority-20261019`, criada exatamente de `b2f5bd16cb8b529b8b1df2a61576037837912e39` (head informado da #48; parent `2ba63f3041a7f8e001b69c8363856a5178d8a308`, #47). SHA final: `HEAD`, resolvido pelo manifesto externo pós-commit `/workspace/nexus-queue-priority-20261019-evidence/STATE_MANIFEST.json`. Não confundir com a base nem atribuir CI remoto: estes são resultados locais. `origin/main` local inventariado em `812ebd02b0c82ecc3169d620e36d25592b1d6727`; `main` local já estava em `7c8a187c1465c6db39777f5e9dc16efb112e1121`. Sem consulta/publicação remota, push, PR ou merge; branches preexistentes, #48 e `/workspace/Nexus` sujo preservados.

**Contrato:** `enqueue` exige `type(priority) is int` antes de capacidade, cópia de contexto, UUID, relógio, construtor e heap. `bool`, `float` e subclasses geram `ValueError` sem hooks nem mudanças de heap/índice/IDs em voo/estatísticas, inclusive fila cheia. Prioridades inteiras válidas mantêm ordenação, FIFO por timestamp em empate, round-trip de snapshot, default zero e overflow legado (incrementa `discarded`). Contexto arbitrário permanece aceito em memória quando o dicionário é deep-copyable; não passa a exigir JSON. Decisão limitada registrada no follow-up A06 de D-030.

**Red/green real:** `bash /workspace/nexus-queue-priority-20261019-evidence/run-red-green.sh` extrai archive da base exata, copia apenas a regressão nova e demonstra `True`/`1.5` aceitos pelo enqueue e rejeitados pelo export. Os sete testes novos na base produziram **35 falhas de subteste e 26 erros**; corrigidos, **7/7**. Matriz inclui vazio/pendente/cheio/em voo; verifica identidade das estruturas e tarefas, conteúdo e contadores invariantes. Hooks adversariais e operações de preparação são monitorados; inteiros ±10¹⁰⁰, zero, contexto Python não JSON, ordenação e capacidade válidas são controles positivos. Focados completos: **71/71**.

**Gates medidos:** `bash /workspace/nexus-queue-priority-20261019-evidence/run-gates.sh`, execução `20261019T142022Z-603785`, Python 3.11.16, numpy 2.4.6/NetworkX 3.6.1, lock com hashes conferido em dry-run (13 pacotes; nenhuma mudança). Suíte completa, cobertura e sumário **470/470**, zero falhas/skips/XPASS; integração separada **30 aprovados + 1 skip FAISS**, sem missing/unexpected/conflicting. Coberturas branch fila/VRAM/temporal/Vita/núcleo **91/94/95/76/69%**, acima dos pisos **90/90/90/70/62%**. Mutação canônica **98/98 mortas** (94 herdadas + 4 novas), zero sobreviventes/timeouts/não aplicadas; o mutante preexistente de prioridade do snapshot ganhou âncora específica para não atingir a nova guarda de enqueue. Bandit **135 LOW, 0 MEDIUM/HIGH**; Ruff, actionlint, zizmor offline, compileall, requisitos, triagem/check, B101, gaps, realidade regenerada sem diff e diff-check passaram. CLIs de benchmarks herdados reexecutadas somente como checks sintéticos internos, sem nova avaliação de capacidade. Invocações repetidas por módulo da CI estão no discover, não todas reexecutadas separadamente. Logs, comandos, checks e relatórios estão no diretório externo de evidências; `run-final-checks.sh` registra a conferência final e a atestação pós-commit.

**Documentação e revisão proporcionais:** D-030, ledger e esta continuidade atualizados; em NEXUS_KNOWLEDGE/SECURITY_TRIAGE mudou apenas o marcador de testes 463→470 exigido pelo gate de consistência. Uma conferência paralela somente leitura inventariou gates e atestação, sem revisar código ou executar suítes. Não foi solicitada revisão independente de código: guarda local de tipo exato, sem coerção, sem nova lógica concorrente/dependência e sem alteração do contexto; risco delimitado pelos testes negativos, controles positivos e mutantes. Não atribuir revisão herdada ao novo diff.

**Limites:** não valida mutações externas de objetos `DeferredTask` depois do enqueue nem construção direta da dataclass; não torna contexto arbitrário exportável nem elimina seus hooks de deepcopy. Sem sandbox para código Python hostil, teto de bytes/nós ou garantia de recursos; não altera retry, snapshot, concorrência geral ou exactly-once. Gates completos com lock somente em Python 3.11; outras versões e GPU não reavaliadas. Estado remoto das PRs não consultado; nenhum serviço externo foi alterado. Resultados são contratos de software, não evidência de capacidade cognitiva ou segurança de produção.

Os registros seguintes são históricos, não resultados desta branch.

## A04/A05 — Orçamento e conclusão operacional — 2026-10-19

Branch nova `fix/reasoning-budget-completion-20261019`, worktree `/workspace/Nexus-reasoning-budget-20261019`, derivada exatamente de `2ba63f3041a7f8e001b69c8363856a5178d8a308` (#47). Commit final: `HEAD`, resolvido pelo manifesto externo pós-commit em `/workspace/nexus-reasoning-budget-20261019-evidence/STATE_MANIFEST.json`. #46 em `f025204339b3dc5ae1f345e98be7f6f7fbcb2498`, #47 publicada e `/workspace/Nexus` sujo preservados; nenhuma branch preexistente, `main`, PR ou remoto alterado. Sem push, PR ou merge.

**Contrato D-034:** preflight sem cobrança em cache miss, negando trabalho em ausência/erro/negação ou permissão não booleana; consulta não autoriza a execução posterior. Na fronteira da chamada a `reason`, `try_consume` verifica/debita uma unidade na mesma seção crítica. Falha conta como tentativa, cache hit não cobra; interface/cópia que falha antes do reasoner não cobra. Sucesso exige os módulos funcionais mínimos por tipo e o Swarm imposto pelo guard ético. Falha requerida não agrega/cacheia sucesso nem confirma tarefa adiada: a política existente faz retry e eventual descarte. Retorno ausente, `success` ambíguo/falso, payload com `error` não nulo e resposta textual ausente no reasoner são falhas.

**Red/green factual:** 24 regressões independentes contra archive da base exata: 51 falhas de subteste e 4 erros, sem modificar a base. Corrigido: **52/52 focados** (novo contrato + router + fila). O probe da auditoria, adaptado somente de `first["answer"]` para `first.get("answer")` ao ler a nova falha sem resposta, confirma orçamento negado com zero chamadas, uma unidade usada no sucesso, falha adiada com `completed=0/retried=1/depth=1` e duas tentativas falhas sem cache. Reproduzir: `bash /workspace/nexus-reasoning-budget-20261019-evidence/run-red-green.sh`.

**Gates do conteúdo final:** `bash /workspace/nexus-reasoning-budget-20261019-evidence/run-gates.sh`, execução `20261019T063250Z-587065`, Python 3.11.16, numpy 2.4.6/NetworkX 3.6.1, lock com hashes conferido em dry-run sem mudança de dependências. Suíte completa, cobertura e sumário **463/463**, zero falhas/skips/XPASS; integração separada **30 aprovados + 1 skip FAISS**, sem missing/unexpected/conflicting. Coberturas branch fila/VRAM/temporal/Vita/núcleo **91/94/95/76/70%**, acima dos pisos **90/90/90/70/62%**. Mutação canônica **94/94 mortas**, zero sobreviventes/inconclusivos/não aplicados (84 herdadas + 10 novas). Bandit **135 LOW, 0 MEDIUM/HIGH**; triagem/check, B101, gaps, realidade regenerada, Ruff, compileall, requisitos, actionlint e zizmor aprovados. CLIs de benchmarks reexecutadas somente como checks sintéticos internos, sem alegação nova de capacidade. Logs, comandos, checks, cobertura e JSONs no diretório externo de evidências. Não atribuir CI remoto a estes resultados locais.

**Revisão e correções concretas:** uma revisão independente focada encontrou payload `{"error":"unavailable"}` de módulo obrigatório não-reasoner ainda tratado como sucesso. Regressão adicional reproduziu 15 falhas de subcasos; guarda geral de `error` não nulo corrigiu a fronteira, e confirmação independente limitada ao achado o considerou fechado (24 contratos novos e 2 testes MCTS passaram; mutante específico morto). A primeira rodada de mutação também identificou um teste ausente para recall `None`: 92/93; a regressão de recall já adicionada e a rodada final com o novo mutante de payload de erro resultam em 94/94. Não reabrir áreas não alteradas por inferência destes resultados.

**Limites:** somente CentralRouter e API local de budget; a rota separada `think()` não foi alterada. Lock cobre operações pela API, não mutação externa de `limits`/`usage`, processos distribuídos ou concorrência geral do router/cache/estatísticas. Adaptadores são código confiável, não sandbox. `consume` continua contabilização legada que pode ultrapassar limite; autorização usa `try_consume`. Retries não são exactly-once. O mapa não certifica qualidade/correção do produto: CREATIVE preserva apenas comparação de futuros legada, LEARNING apenas registro da intenção `learning_queued`, sem geração criativa nem treinamento executado. Não reavaliados GPU, controles gerais jailbreak/log ou restante dos achados da auditoria.

Os registros seguintes são históricos, não resultados desta branch.


## Follow-up F-04 sem teto fixo — 2026-10-19

Branch local nova `fix/audit-json-preflight-20261019`, worktree `/workspace/Nexus-audit-json-preflight-20261019`, derivada exatamente de `6d145ccb7a7b2cb50ea0c1bbd1a5b38034d854e1`. O commit final é `HEAD`, resolvido pelo manifesto externo pós-commit em `/workspace/nexus-json-preflight-20261019-evidence/STATE_MANIFEST.json`. O SHA/branch revisados anteriores, todas as refs/PRs e `/workspace/Nexus` sujo devem permanecer preservados; sem push, PR, merge, fechamento ou mudança de `main`.

**Contrato aprovado:** preservar D-030, sem novo teto de profundidade e sem teto 64. Restore serializa apenas os dados capturados/validados no envelope do export antes de construir tarefas ou publicar estado. Não percorre extensões ignoradas. Falha de recursão do encoder gera `ValueError("snapshot excede a capacidade de serialização JSON")`; export testa essa capacidade nos dois modos e usa o mesmo tratamento. Parser textual mantém seu `ValueError` documentado. Schema, formato JSON e contextos exportáveis permanecem compatíveis; contextos Python antes restauráveis, mas não serializáveis, agora falham antes de alterar heap, índice, IDs em voo ou contadores.

**Red/green factual:** o original restaura 5.000 níveis e o export JSON lança `RecursionError` em Python 3.11.16. O encoder 3.12.3 aceita 5.000: o follow-up preserva essa aceitação. Ambos rejeitam o caso de 10.000 níveis controladamente. Seis testes novos cobrem dict/list/misto, replace/merge (incluindo tarefa em voo no merge), tarefa válida antes da inválida, identidade/estado do alvo, erro de export nos dois modos, falha injetada antes do construtor, parser profundo e round-trip a 65/100/300 níveis. Contra a base intocada: 6 testes, 18 falhas de subteste/5 erros em 3.11; 12 falhas/5 erros em 3.12. Corrigido: **77/77** focados nas duas versões. Esses números de profundidade são sondas, não limites do código.

**Gates medidos:** `bash /workspace/nexus-json-preflight-20261019-evidence/run-gates.sh`, execução `20261019T041711Z-550824`, Python 3.11.16, lock com hashes verificado em dry-run (13 pacotes, nenhuma mudança), NetworkX 3.6.1/numpy 2.4.6. Suíte completa, cobertura e sumário: **439/439**, zero falhas/skips/XPASS. Integração separada **30 aprovados + 1 skip FAISS**, nenhum missing/unexpected/conflicting. Coberturas branch fila/VRAM/temporal/Vita/núcleo **91/94/95/76/69%**, acima dos pisos **90/90/90/70/62%**. Mutação canônica **84/84** e suplemento F-04 externo **5/5** mortos, zero sobreviventes/inconclusivos/não aplicados; o suplemento inclui tentativa de introduzir teto 64. Bandit **135 LOW, 0 MEDIUM/HIGH**; triagem/check, B101, gaps, realidade sem diff, Ruff, compileall, requisitos, actionlint 1.7.12, zizmor 1.31.1 e diff-check aprovados. Os módulos repetidos pela CI estão incluídos no discover, não todos reexecutados como comandos individuais. Benchmarks CLI herdados foram reexecutados como checks sintéticos internos, sem nova alegação de capacidade. Logs, comandos, XML de cobertura e JSONs estão no diretório externo; `run-red-green.sh` reproduz a comparação com a base.

**Limites e revisão:** o preflight custa traversal e materialização temporária da string JSON, sem teto de bytes/nós; não é proteção geral contra consumo de recursos, Python hostil ou esgotamento de memória. Capacidade depende de versão/encoder/pilha de chamada; não garante profundidade portátil, parse em outro ambiente, nem export após mutação externa de tarefas. `enqueue` e as garantias temporais anteriores não foram ampliados; nenhum `_json_value` foi alterado. Não reavaliados GPU, transação grafo↔ledger, concorrência geral, PRs ou remotos. A revisão anterior de `6d145cc`, informada como concluída sem bloqueadores, não vale automaticamente para este diff. **Parar com o novo diff/SHA congelados para uma única revisão independente de Segurança**, focada no preflight, atomicidade e compatibilidade; nenhuma revisão independente nova foi realizada nesta etapa. Não reabrir/alterar o diff salvo achado concreto.

As seções seguintes registram checkpoints históricos; decisões pendentes e revisão pendente abaixo descrevem a situação à época, não o follow-up atual.

## Auditoria temporal local — 2026-10-19

Branch nova `fix/audit-atomic-restore-20261019`, worktree `/workspace/Nexus-audit-atomic-20261019`, derivada exatamente de `f025204339b3dc5ae1f345e98be7f6f7fbcb2498`. Implementação/testes: `e8af0ac3da71fa62bebdbcae2f110a7365af2535`; o checkpoint documental posterior é identificado por `HEAD` e pelo manifesto externo pós-commit em `/workspace/nexus-audit-20261019-evidence/STATE_MANIFEST.json`. Base histórica: `812ebd02b0c82ecc3169d620e36d25592b1d6727`. A ref local `main` já apontava para `7c8a187c1465c6db39777f5e9dc16efb112e1121` na abertura; não foi alterada. Sem consulta/publicação remota; preservar candidato #46, refs preexistentes e `/workspace/Nexus` sujo.

**Contrato:** `assert_fact` (incluindo corroboração) e `retract` revertem eventos, head, sequência, watermark, fatos e índice se uma etapa lança exceção, inclusive projeção/cópia de retorno. Rollback usa cópias rasas anteriores, não replay/deepcopy: custa O(fatos + índice) por escrita. Não garante recuperação de esgotamento de memória ou Python hostil. Restore temporal converte `RecursionError` de parse/hash/cópia/replay em `ValueError`; eventos exigem campos exatos por tipo e `kind` string built-in. Exports normais preservam schema, hashes e assinaturas; eventos externos com campos adicionais, antes aceitos, passam a ser rejeitados. Hashes não são autenticação.

**D-030 preservado:** nenhum teto 64, nenhuma alteração da fila ou de `_json_value`. Metadata previamente aceita com 100 contêineres continua em round-trip; limites práticos do interpretador/JSON permanecem. F-04 reproduzido: a fila restaura contexto de 5.000 níveis, mas seu export JSON lança `RecursionError`. Fica aberto por determinação do owner; o patch anexo não foi aplicado integralmente.

**Red/green independente:** candidato Python 3.12.3 + metadata de 600 níveis deixa `close` sem sucessor e o snapshot não restaura (F-01). Python 3.11.16 rejeita esse input antes de mutar, mas falhas injetadas confirmam falta de atomicidade também ali. Os 17 testes finais contra o original: 10 falhas/6 erros em 3.11; 11 falhas/6 erros em 3.12, incluindo subcasos. Com a correção: 17/17 em ambos. Sondas dependentes de 64 ficam somente na evidência externa, fora da suíte aprovada.

**Gates locais:** Python 3.11.16, venv existente sem pip, lock verificado com hashes por `uv pip install --python /workspace/nexus-consolidation-evidence/venv/bin/python --require-hashes -r requirements-ci.lock`; NetworkX 3.6.1/numpy 2.4.6. Roteiro: `bash /workspace/nexus-audit-20261019-evidence/run-gates.sh`, com comandos/logs em `commands.tsv` e `checks.tsv`. Ordem: focados → Ruff/actionlint/zizmor → compileall → suíte integral → cobertura/integração/pisos → mutação → Bandit/triagem/auditorias → realidade → benchmarks/resumo → manifesto/consistência pós-commit. Suíte/resumo **433/433**, zero falhas/skips/XPASS; integração separada **30 aprovados + 1 skip FAISS**, zero IDs faltantes/inesperados/conflitantes. Coberturas fila/VRAM/temporal/Vita/núcleo **91/94/95/76/69%**, pisos **90/90/90/70/62%**. Mutação canônica **84/84**, suplemento temporal externo **12/12** mortos, zero sobreviventes/inconclusivos/não aplicados. Bandit **135 LOW, 0 MEDIUM/HIGH**. Triagem/check, B101, gaps, realidade sem diff, dry-run de requisitos, actionlint 1.7.12, zizmor 1.31.1 e diff-check passam. Repetições por módulo da CI estão no discover, não todas repetidas como invocações independentes. `ci_run=not_ci`; nenhum CI remoto atribuído a esta branch.

**Outros achados/limites:** F-03 já rejeitado pelo candidato (`statistics.depth` extra e JSON profundo → `ValueError`). F-05 não confirma a alegação sobre o perfil: dev inclui `requirements-ci.in`, que já fixa NetworkX; o lock também o inclui. F-06 segue diagnóstico documentado, não cancelamento. Não reavaliados main, PRs individuais, Dependabot, GPU, transação grafo↔ledger, vazamentos gerais ou outras concorrências; psutil/FAISS/SentenceTransformers ausentes. Benchmarks herdados continuam checks sintéticos internos, não validação científica nova. Sem teto de bytes/nós ou garantia de profundidade portátil entre ambientes.

**Decisão pendente do owner:** 64 não decorre dos registros canônicos (D-030 declara sem novo teto). Se um orçamento fixo for autorizado, proposta concreta: no máximo 64 contêineres em um caminho incluindo raiz, 64 aceito/65 rejeitado, com revisão explícita de D-030 e avaliação/migração de snapshots legados. Rejeitaria entradas antes aceitas; o número não foi escolhido por benchmark. Alternativa sem teto fixo: preflight de serialização antes de publicar restore, com falha atômica; também altera a faixa antes aceita. Nenhuma dessas políticas foi implementada.

**Handoff:** congelar diff/SHA após gates para o coordenador obter **uma única revisão independente de Segurança**, focada em rollback/restore. Revisão final ainda pendente; reabrir só por achado concreto ou mudança do diff. Sem push, PR, merge, fechamento de PR ou mudança de main.

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
