# Nexus — Ledger de evidências

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: CURRENT_TESTS
bandit_low: 131
-->

## Auditor de declarações de correção — validação local 2026-10-28

Base `c856e1a899121c1b68be8162f86ec873202816ed`, branch `cursor/correction-claim-audit-0eb5`. O contrato `test_correction_claim_audit_contract.py` falhou por ausência do módulo (1 teste, exit 1) e, com `tools/correction_claim_audit.py`, passou 18 testes. No corpus versionado o auditor devolveu `pass`: 2 declarações não checáveis em `CONTINUATION.md` e 7 lições sem medição de reincidência. Existência de caminho ou símbolo não é correção. Comandos, versões e limites estão na rodada de 2026-10-28 de `CONTINUATION.md`. Isto é evidência de contrato local, não de capacidade cognitiva nem de CI hospedado.

## Duplicatas terminais no resumo integrado — validação local 2026-10-20

Base `d435083bb4c893a58128cee6cfa00f65bba7fe02`, branch `fix/integrated-summary-duplicates-20261020`: regressões reproduziram PASS/PASS e SKIP/SKIP indevidamente aprovados antes do patch. O tally agora expõe `duplicates` e falha com resultados terminais repetidos, preservando stream e categorias existentes. Após a correção: 10 contratos focados aprovados; suíte completa executada duas vezes (com cobertura e `tools/test_summary.py`), 307 executados/305 aprovados/2 skips opcionais/0 falhas; integração real 30 passes/1 skip e nenhuma duplicata. Gates de cobertura, Ruff, compilação, auditorias e 58 mutações preexistentes aprovados; Bandit 131 LOW e nenhum MEDIUM/HIGH. Comandos, limitações e proveniência estão na seção correspondente de `CONTINUATION.md`.

Esta é evidência de contrato de software local, não de capacidade cognitiva nem CI hospedado. O consumidor legado do manifesto permanece fail-closed (`invalid_summary` para falha causada apenas por duplicata), sem contabilização específica nova. Nenhuma sonda sintética, CLI de benchmark ou evolução separada de WorldModel foi executada nesta rodada.

## Igualdade exata na guarda de triagem — 2026-10-20

Na base `d435083bb4c893a58128cee6cfa00f65bba7fe02`, duas regressões novas reproduziram o reconhecimento indevido de comparações não-main. Após cinco condições adicionais no predicado, 12 contratos focados passaram; suíte completa: 305 executados, 303 aprovados, 2 skips opcionais e 0 falhas/XPASS. Bandit mantém 131 LOW, sem MEDIUM/HIGH, e a triagem versionada não mudou de corpo. Comandos, revisão independente e limites estão na [rodada local correspondente de CONTINUATION.md](CONTINUATION.md#rodada-local--igualdade-exata-na-guarda-de-triagem--2026-10-20). Evidência de classificação AST, não de clearance de segurança nem de CI remoto.

## Composição validada localmente sobre a PR #58 — 2026-10-20

Base comum remota `d435083bb4c893a58128cee6cfa00f65bba7fe02`; pai do candidato `fef00882baf19694fe04956d82af096f266ae2d3` (head da #58 OPEN/DRAFT), acrescido somente do diff de overflow `af4f489889fd914e4316bd099a680dad26e47e71` e do contrato da interação. A captura CLI de `RecursionError` e seus seis testes permanecem; os dois loaders agora rejeitam float que decodifica para infinito e constantes não JSON, sem alterar valores finitos nem bridge/schema/writer/profundidade. Worktree `/workspace/Nexus-diagnostics-combined-pr58-20261020`, branch `fix/diagnostics-combined-pr58-local-20261020`.

Evidência nova da árvore combinada: red com 26 testes/32 falhas de subcasos antes dos hooks; green 26/26 em Python 3.11.16 e 3.12.3. Suíte ampla 3.11: **316 executados, 314 aprovados, 2 skips NetworkX, 0 falhas/XPASS**; integração **30 aprovados e 1 skip FAISS**, sem conflitos/ausências/inesperados. Mutação versionada **63/63 mortas** e contraprova externa de recursão **1/1 morta**, em campanhas separadas e sem timeout. Ao remover a captura, reaparecem 2 erros CLI e 6 falhas da interação; baseline e composição intacta passam. Cobertura passou todos os pisos: fila 95%, VRAM 94%, temporal 93%, Vita 76%, núcleo 68%; dashboard 96% informativo.

Compile/Ruff/actionlint/zizmor offline, segurança e auditorias pertinentes passaram; Bandit **131 LOW, nenhum MEDIUM/HIGH/erro de análise**, sem clearance de segurança. Regressão explícita dos auditores/manifesto 38/38; relatórios de realidade sem drift. Comandos exatos, hashes, artefatos e limites estão na seção de composição de `CONTINUATION.md` e em `/workspace/nexus-diagnostics-combined-evidence-20261020`. Resultados abaixo são históricos de árvores isoladas, não resultados combinados.

Não há novo limite de recursos/profundidade nem mudança da precisão/underflow do float. O decoder pode falhar por recursão antes de alcançar qualquer token; callers diretos preservam essa exceção e a CLI a converte em erro operacional. Suíte ampla/mutação somente em 3.11; 3.12 focado. Sem benchmarks novos, CI hospedada, atestação operacional fabricada, push, alteração de PR, merge ou mudança das fontes/main/checkout sujo. Hash do commit e revisão final no handoff externo.

## Histórico da PR #58 isolada — fronteira CLI do dashboard: recursão do decoder — 2026-10-20

Base exata `d435083bb4c893a58128cee6cfa00f65bba7fe02`; candidato local na branch `fix/diagnostics-recursion-error-20261020`. O red registrou dois `RecursionError` escapando de `main` (arquivo e stdin). A captura adicional transforma ambos em exit 2, stdout vazio e stderr iniciado por `dashboard error:`, sem traceback; `load_snapshot` e o parser permanecem inalterados. Os seis novos contratos também preservam propagação direta e snapshots válidos rasos nos dois caminhos.

Evidência local: 19/19 contratos em Python 3.11.16 e 3.12.3; suíte completa em 3.11 com 309 executados, 307 aprovados e 2 skips NetworkX; integração 30 aprovações/1 skip FAISS; mutation existente 58/58 mortas e mutante específico adicional 1/1 morto. Ruff/compile/lint de workflows/auditorias pertinentes passaram, sem drift dos relatórios de realidade; Bandit 131 LOW, nenhum MEDIUM/HIGH. Cobertura acima de todos os pisos existentes. Comandos exatos, hashes funcionais, localização dos logs e limitações estão na seção correspondente de `CONTINUATION.md`; hash do commit final no handoff externo, não autorreferente.

Limite: não há novo teto de profundidade nem garantia de recursos; a fixture profunda foi ajustada para ultrapassar também o limite C do Python 3.12, enquanto o caso exato de 2.000 arrays foi confirmado em subprocessos 3.11. Sem claims de capacidade ou segurança geral, novos benchmarks ou execução de CLIs de benchmark; nenhuma publicação/CI hospedada nesta rodada.

## Histórico da branch-fonte isolada — overflow de float no carregador do dashboard — 2026-10-20

> Evidência histórica exclusiva de `af4f489889fd914e4316bd099a680dad26e47e71`, sobre `d435083bb4c893a58128cee6cfa00f65bba7fe02`; não representa a composição sobre a #58 nem seus resultados atuais. Referências a “esta branch” abaixo descrevem somente a branch-fonte original.

- **Base/isolamento:** `d435083bb4c893a58128cee6cfa00f65bba7fe02`, confirmada por `git ls-remote origin refs/heads/main`; branch independente `fix/diagnostics-float-overflow-20261020`, worktree `/workspace/Nexus-diagnostics-float-overflow-20261020`, criada limpa diretamente nessa base. Nenhum cherry-pick da PR #58 nem alteração das outras worktrees.
- **Correção de contexto confirmada:** `gh pr view <n> --repo pedromiorini/Nexus --json number,baseRefName,headRefName,headRefOid,files` para #52–#58 mostrou que #52–#57 não tocam dashboard/teste de recovery; #58 **toca ambos**, com head `fef00882baf19694fe04956d82af096f266ae2d3`. Essa PR e seus refs/conteúdo permanecem preservados. #57 toca `tools/targeted_mutation.py`: os novos mutantes desta branch também exigirão reconciliação futura nesse arquivo, além dos ledgers compartilhados.
- **Fato/red:** tokens JSON sintaticamente válidos `1e400` e `-1e400`, em `events[].payload` e contêineres alternados dict/list, eram decodificados como infinitos. Antes do patch, `PYTHONPATH=. <PY> -m unittest -v test_recovery_diagnostics_contract` executou 19 testes e registrou 32 falhas de subcasos nos runtimes Python 3.11.16 e 3.12.3: 12 de overflow nos loaders, 8 na CLI e 12 de constantes não padrão. Controles finitos já passavam. A base indicada não configurava `parse_constant` no dashboard; a rejeição prévia do bridge para strings não protegia esse caminho de leitura.
- **Contrato/green:** `parse_float` converte com `float` e rejeita resultado não finito antes de entregar qualquer snapshot; `parse_constant` rejeita `NaN`, `Infinity`, `-Infinity`. Ambos os hooks são configurados em arquivo e stdin. São preservados o tipo/valor dos floats finitos, sinal de zero, subnormal, `1.7976931348623157e308` e seu negativo, valores comuns e inteiro de 1.000 dígitos. O hook independe da profundidade no domínio que o decoder nativo consegue processar; não há travessia pós-decode nem teto novo de profundidade, tamanho, precisão ou inteiros. Schema, bridge, writer e saída para entrada válida não mudam.
- **Regressões:** seis novos testes cobrem overflow positivo/negativo nas profundidades adicionais 0, 1 e 32 em ambos os loaders; subprocessos reais da CLI confirmam exit 2, stdout vazio, stderr com diagnóstico e sem traceback; controles finitos em arquivo/stdin confirmam exit 0 e stderr vazio. O mesmo comando focado passou 19/19 em 3.11 e 3.12 depois da correção.
- **Revisão independente focada:** nenhum bloqueio encontrado na fronteira numérica. A captura de `RecursionError` da #58 é semanticamente ortogonal; uma integração futura deverá preservar essa captura e combinar os testes sobrepostos. Ela não foi copiada para esta branch. Limite: esta revisão não atesta segurança geral nem integração já realizada.
- **Validação ampla, comandos, hashes e limitações:** registrados na rodada correspondente de `CONTINUATION.md`; resultados são locais, não checks hospedados. Os marcadores de contagem foram sincronizados nos quatro documentos canônicos, sem alterar disposições de segurança ou reescrever histórico.

Snapshot histórico: manifesto `nexus-state-manifest` do commit `5b8bed8125002e25eb066620af67d3a65d44e59f` (CI run `38907987512`), com 200/200 testes `unittest`, 31 verificações integradas inventariadas (30 aprovadas, 1 ignorada por FAISS opcional ausente e 0 falhas) e 131 achados Bandit LOW (0 MEDIUM/HIGH). Esse snapshot e os resultados locais do benchmark v1 não representam o estado atual.
A PR #24 partiu do `main` pós-PR #23 (`e3cc760597aa1782aef9879a8f88ee6f48032348`) e foi mesclada em `main` como `dd0a73c6e57dfcccf9b522b3d2dd3a4623cdc732`; os dois checks oficiais passaram. Seus resultados locais de 240/240 testes permanecem históricos; a branch seguinte registra a nova validação em seção própria.
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
| Transferência meta/domínio | verificado no contrato local de 2026-10-21 | `transfer_knowledge` devolve `MetaTransferMapping`; o nome `TransferMapping` permanece a forma do POST 48; bool, NaN, infinito e negativo são rejeitados | `transfer_quality` pode exceder 1; heurística por token, não transferência aprendida; mutação posterior ao init não é revalidada |
| Planejamento MCTS | parcial/verificado no contrato | `None` usa ações padrão e lista vazia permanece vazia | recompensas e rollouts continuam heurísticos; não é prova de planejamento geral |
| Telemetria | parcial/verificado | Vita, audit log SQLite serializado e guard de VRAM | dependente do ambiente; sem locking distribuído ou exactly-once |
| Segurança | parcial | Bandit 131 LOW, triagem, auditorias AST e testes negativos | não substitui threat model ou revisão manual completa |
| CI | verificado por commit | run associado ao `HEAD` e manifesto retido como artefato quando disponível | qualquer novo commit exige leitura do manifesto correspondente |
| Benchmark sintético | verificado localmente nos protocolos v1 e v2 | v1 preserva as métricas composicional/causal/incerteza; v2 mede 128 modelos explícitos, erro absoluto máximo `2,7756e-16`, 128/128 casos dentro de `1e-12` e 4/4 controles inválidos rejeitados | trilha POMDP do v1 continua não suportada; v2 não aprende modelos nem escolhe políticas; não valida teorias cognitivas |
| Autonomia geral | não demonstrada | nenhuma evidência válida no repositório | não fazer esse claim |
| AGI/ASI/consciência | não demonstradas | explicitamente negadas pela documentação | nomes e banners não contam como evidência |

## Resumo de testes inválido não é ausência — validação local, 2026-10-20

Base `d435083bb4c893a58128cee6cfa00f65bba7fe02`, branch `fix/state-manifest-invalid-summary-20261020`. A implementação original classificou JSON malformado, `42`, `null`, booleanos, campos `result` não hashable e leitura negada como `not_generated`; os 14 testes atuais do manifesto contra o arquivo original produziram 11 falhas de subcasos. A correção separa ausência (`FileNotFoundError`) de resumo inválido, valida a raiz objeto e o tipo de `result`, preservando os campos válidos e o schema. Os testes conferem também a propagação integral em `build_manifest()`.

`/tmp/nexus-ci311/bin/python -m unittest -v test_state_manifest.py test_state_consistency_audit.py test_test_summary.py test_integrated_suite_summary.py`: 30 aprovados. Suíte completa: 311 executados, 309 aprovados, 2 skips opcionais NetworkX, 0 falhas/XPASS; integração: 30 aprovados + 1 skip FAISS. Mutação existente 58/58 mortas e seis mutantes adicionais locais do status/guardas/propagação detectados. Bandit: 131 LOW, nenhum MEDIUM/HIGH/erro; auditorias de realidade, asserts e lacunas aprovadas. Comandos, versões, cobertura e limites detalhados em `CONTINUATION.md` na rodada correspondente.

A revisão independente encontrou ainda UTF-8 inválido escapando como exceção e constantes `NaN`/`Infinity` aceitas em extras aninhados. As regressões reproduziram 1 erro e 3 falhas, corrigidos por captura dos erros de decodificação e parser JSON estrito; os 15 métodos finais do manifesto e os gates foram reexecutados.

Esta é evidência de contrato local, não CI hospedado nem garantia de segurança. Não houve reinstalação do lock. O manifesto local permanece `not_ci`, e o auditor operacional o rejeita como esperado somente pela ausência de gates CI concluídos, sem conflitos documentais; seu fail-closed não foi alterado.

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
- Limites remanescentes naquela rodada: `ToolDefinition.timeout_seconds` não interrompia executores; o cache limita payload estimado via `sys.getsizeof`, não a memória total do processo; o estado local aguardava CI oficial. A validação e telemetria advisory de overrun foram implementadas na rodada subsequente.

## Filtro de crença POMDP e benchmark v2 — 2026-10-16

- Reproduções na versão inicial do filtro: 129 estados aceitos, `OverflowError` ao somar probabilidades enormes inválidas e falsa evidência zero com likelihoods ou produtos prior–transição extremamente pequenos.
- Correção: validação de entradas em `[0,1]`, sequências compatíveis, prior normalizado, matriz quadrada estocástica, máximo de 128 estados e normalização posterior em log-space; inputs não são retidos nem mutados.
- Contratos locais: 10 testes do filtro e 8 do benchmark; suíte total 240/240, sem falhas/skips/XPASS; integrada 30 aprovações e 1 skip FAISS opcional. Cobertura: fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation 28/28 mortas, 0 sobreviventes/não aplicadas.
- Benchmark v2 local: 128 casos gerados pela seed `20261016`, 635 probabilidades, MAE `3.001891167837976e-17`, erro máximo `2.7755575615628914e-16`, 128/128 dentro da tolerância e quatro controles negativos rejeitados; taskset SHA-256 `232371d0e8e948f4b0a004261e3d271b1003aed8d0de8c931d3439f9d370d025`.
- A seed fixa do CI (`20261016`) passa explicitamente a precisão e os quatro controles negativos; teste injetando posterior incorreto verifica que a CLI grava o relatório e retorna código 1, cobrindo a falha fechada.
- O benchmark v1 continuou com status `unsupported_interface` para observação parcial e sua implementação/protocolo não foram alterados. As métricas detalhadas acima são locais; posteriormente, a PR #24 foi mesclada após ambos os checks oficiais passarem.

## Contrato de TransferMapping — 2026-10-21

Medição local no commit `4e971f38854dedecac68a55f08125cde01d27992` (Python 3.12.3, branch `cursor/transfer-mapping-contract-5517`). O marcador acima passou a `tests: 320` por causa desta execução. Seções anteriores que citam 303, 275, 254 ou outras contagens descrevem rodadas passadas.

### Fato

- As duas classes `TransferMapping` entram no commit `1f952a9` (2026-08-05). Depois do import, o nome global tem os campos do POST 48. A anotação de retorno de `RealMetaLearningStrategies.transfer_knowledge` conserva a classe do POST 38.
- Comando de reprodução, antes do patch: `PYTHONPATH=. python3` chamando `transfer_knowledge("visual_cortex", "visual_planning")`. Resultado: `TypeError: TransferMapping.__init__() got an unexpected keyword argument 'transfer_strength'`.
- `PYTHONPATH=. python3 -m unittest -v test_transfer_mapping_contract.py` antes da edição do núcleo: `FAILED (failures=17, errors=7)`. No commit citado: `Ran 17 tests in 0.152s` / `OK`.
- `python3 -m ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py` (Ruff 0.16.10): `All checks passed`.
- `PYTHONPATH=. NEXUS_AUDIT_DB=/tmp/nexus-transfer-mapping.sqlite3 python3 tools/test_summary.py --output /tmp/test-summary.json`: executed 320, passed 318, failed 0, skipped 2, unexpected_successes 0, result passed (`Ran 320 tests in 5.177s`).
- `NEXUS_INTEGRATED_SUMMARY=/tmp/integrated-summary.json PYTHONPATH=. python3 core/constitutional_brain.py`: 30 passed, 1 skipped (id `"0"`, log `TEST 0: Keyword Fallback Mode (FAISS not available)`), 0 missing, `result=passed`, exit code 0.
- `python3 tools/reality_audit.py`: núcleo com 27591 linhas, 269 classes e 878 funções. Esses relatórios estão no mesmo commit de código.

### Inferência

- A causa do `TypeError` é a colisão de nome no módulo: a busca global na hora da chamada vê a segunda classe. A anotação, avaliada na definição, ainda aponta para a primeira. Isso sobreviveu à comparação dos `id` das duas classes depois do import completo.

### Hipótese

- **Excluídas nesta rodada:** substituição intencional pelo POST 48 (H1); forma do POST 48 morta (H2); dataclass única com campos opcionais (H4); falha por import parcial (H5). Condições e o que as derrubou estão em `CONTINUATION.md`.
- **Sobrevivente:** as duas formas são pretendidas e o nome duplicado é o defeito (H3). Os testes feitos até aqui não a falsificaram. Isso não é prova de que não exista outro call site externo, fora deste repositório, dependente do nome antigo.

### Limitações

- Bandit, cobertura, mutation testing e CI hospedado não foram reexecutados. `bandit_low: 131` não é medição desta rodada.
- `transfer_quality` acima de 1 continua válido: a heurística de cobertura pode produzi-lo. Não houve normalização silenciosa.
- O ramo integrado que chama `find_transferable_knowledge(..., target_task=question)` não foi alterado. O parâmetro não existe na assinatura. A suíte integrada terminou sem traceback; não se verificou o caso em que já existe domínio fonte registrado.
- A transferência por sobreposição de tokens não é aprendizado entre domínios.

## Categorias de resultado

- **Fato:** número ou comportamento reproduzido por comando.
- **Evidência:** contrato ou execução que sustenta apenas o escopo descrito.
- **Inferência:** risco arquitetural derivado do tamanho e acoplamento do núcleo.
- **Hipótese:** benefício futuro de modularização ainda não medido.
- **Intenção:** itens do roadmap, não funcionalidades presentes.

## Autoauditoria de `ToolDefinition.timeout_seconds` — 2026-10-16

Base local: `main` pós-PR #24, commit `dd0a73c6e57dfcccf9b522b3d2dd3a4623cdc732`; branch isolada `feat/post-pr24-contract-hardening-2026-10-16`.

Antes da correção, o registro aceitava timeout zero, negativo, booleano, float e `NaN`; um valor string podia lançar `TypeError`. A execução síncrona não consultava o orçamento: uma revisão independente reproduziu um executor de cerca de 30 ms com budget de 1 ms retornando sucesso sem sinalizar o excesso.

O novo contrato exige inteiro não booleano em `[1, 60]`, no registro e antes de cada execução, inclusive para detectar mutação da definição depois do registro. O executor é cronometrado com `time.monotonic()`; ao terminar além do budget, por retorno ou exceção, emite `TOOL_EXECUTION_OVERRUN`. Resultados bem-sucedidos são preservados, exceções continuam falhas estruturadas, e o evento informa `execution_completed=true` e `cancelled=false`.

Os seis testes em `test_tool_definition_contract.py` cobrem limites inclusivos, valores inválidos, mutação pós-registro, execução dentro do budget, overrun bem-sucedido e overrun com exceção; cinco mutantes específicos foram adicionados ao mutation gate. Validação local Python 3.12.3: 246/246 testes; suíte integrada 30 aprovações e 1 skip FAISS opcional; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation 33/33 mortas, 0 sobreviventes/não aplicadas; Ruff, actionlint, zizmor, compileall e auditorias AST passaram; Bandit 131 LOW, nenhum MEDIUM/HIGH.

O evento é diagnóstico pós-execução, não deadline nem cancelamento. A documentação oficial de `Future.cancel()` informa que chamada em execução não é cancelada; `Process.terminate()` pode corromper pipes/queues ou tornar locks indisponíveis: https://docs.python.org/3.12/library/concurrent.futures.html e https://docs.python.org/3.12/library/multiprocessing.html. Interrupção segura segue exigindo um contrato cooperativo ou runner isolado.

Os números são locais; a CI do head candidato ainda deve confirmá-los.


## Autoauditoria da fronteira de `RealToolUse.execute_tool` — 2026-10-16

Base local: `main` pós-PR #25, commit `675f5aa95416c3a92998a34dd2f905f529b90980`; branch isolada `feat/post-pr25-hardening-2026-10-16`.

Antes da correção, a chamada `execute_tool("list_operations", None)` reproduziu `TypeError("argument of type 'NoneType' is not iterable")` dentro de `validate_parameters`, antes do `try` do executor; portanto a exceção escapava em vez de virar o `ToolExecutionResult` de falha previsto. A reprodução local também confirmou que um dicionário vazio continua produzindo diagnóstico de parâmetro obrigatório ausente e que a forma válida executa normalmente.

A guarda inicial agora exige `isinstance(params, dict)` e registra `TOOL_PARAM_TYPE_MISMATCH` (`parameter=params`, `expected=dict`, `actual=<tipo>`), fazendo a chamada retornar `success=false`, `error="Parameter validation failed"` e sem invocar o executor. Três novos testes verificam tipos externos inválidos, dicionário incompleto e dicionário válido; um mutante que desativa a guarda é detectado.

Validação local Python 3.12.3: 249/249 testes; integração 30 aprovações/1 skip FAISS; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation gate 34/34 mortas, nenhum sobrevivente/não aplicado; Ruff, actionlint, zizmor, compileall, auditorias AST, capability benchmark e POMDP v2 passaram; Bandit 131 LOW, nenhum MEDIUM/HIGH. Os dois checks oficiais da PR #26 passaram antes do merge em `main` no commit `508020d77d5705f99c76eaca4df05e8e265662f7`.

## JSON estrito e falha da chave de cache — 2026-10-16

Base local: `main` pós-PR #26, commit `508020d77d5705f99c76eaca4df05e8e265662f7`; branch `feat/post-pr26-tool-input-boundary-2026-10-16`.

Estado posterior: a alteração foi publicada como PR #27; os dois checks oficiais passaram e foi mesclada em `main` no commit `c4375d9f42703b11d2632b3a701303c5a69563a4`.

Antes da alteração, `format_json` retornava sucesso com texto contendo `NaN`, `Infinity` e `-Infinity`, pois a serialização Python usava o valor padrão permissivo de `allow_nan`. Também reproduzi `TypeError: Object of type object is not JSON serializable` escapando de `execute_tool` ao serializar a chave do cache antes do `try` do executor. O exemplo usou parâmetros válidos de `list_operations` e uma extensão extra ignorada pelo schema; uma ferramenta marcada como não cacheável também tentava construir uma chave.

O formatador agora define `allow_nan=False`; os valores não finitos se tornam falhas estruturadas pelo tratamento de exceções já existente. A chave é construída somente para ferramentas cacheáveis; `TypeError`, `ValueError`, `OverflowError` ou `RecursionError` no encoder emitem `TOOL_CACHE_KEY_UNAVAILABLE`, ignoram leitura e armazenamento no cache e deixam o executor seguir. Assim, a extensão extra continua ignorada como antes e uma operação válida continua retornando seu resultado sem cache. O caso de dado não serializável enviado ao próprio `format_json` falha de forma estruturada no executor, sem exceção escapar.

Os contratos `test_format_json_contract.py` e `test_tool_cache_key_contract.py` passaram após reproduzirem as falhas antes do patch. O gate direcionado adicionou um mutante para desfazer `allow_nan=False` e dois mutantes para remover o bypass/serialização condicional.

Validação local em Python 3.12: 254/254 testes; suíte integrada 30 aprovados, 1 skip FAISS opcional e 0 falhas; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation testing 37/37 mortas, 0 sobreviventes ou não aplicadas; Bandit 131 LOW, 0 MEDIUM/HIGH; Ruff, actionlint, zizmor, compileall, auditorias AST e CLIs dos benchmarks de capacidade e POMDP v2 passaram. Relatórios de realidade foram regenerados. Os dois checks oficiais da PR #27 passaram antes do merge registrado acima.

Fundamento: a [documentação Python 3.12 de `json`](https://docs.python.org/3.12/library/json.html) descreve o padrão permissivo e o efeito estrito de `allow_nan=False`; a [RFC 8259, seção 6](https://www.rfc-editor.org/rfc/rfc8259) não permite Infinity/NaN na gramática numérica JSON. Não houve normalização silenciosa para strings ou `null`.

Limites: não foi introduzida validação recursiva de schema nem rejeição de chaves desconhecidas. O bypass não limita custo de parâmetros grandes porém serializáveis, e o tamanho estimado do cache não é limite global de memória. O diagnóstico de falha de cache não é falha da ferramenta; exceções reais do executor continuam seguindo seu tratamento original.


## Orçamento de recursos da cache key — 2026-10-16

Base local: `main` pós-PR #27, merge commit `c4375d9f42703b11d2632b3a701303c5a69563a4`; branch `feat/post-pr27-cache-key-resource-bounds-2026-10-16`.
Estado posterior: a branch foi publicada como PR #28, recebeu aprovação dos dois checks oficiais e foi mesclada em `main` no commit `fafcd95e6b40c5d51b34d344f164a9045279d8bb`.

A chave era construída por `json.dumps(params, sort_keys=True)` antes do executor, sem limite de bytes ou profundidade; os budgets existentes só cobriam resultados retidos no cache. A alteração adiciona percurso iterativo prévio com limite conservador de 64 KiB incluindo prefixo, profundidade 64, 4.096 nós/ocorrências e 1.024 itens por contêiner, além de detecção de ciclos. O percurso contabiliza expansão de escapes e tamanho de inteiros sem serializar o objeto completo.

Testes com orçamento excedido (bytes, profundidade, cardinalidade total, itens por contêiner e ciclo) verificam que `json.dumps` não é chamado, que `TOOL_CACHE_KEY_UNAVAILABLE` inclui o motivo e que duas chamadas válidas continuam no executor, sem hit nem armazenamento. O contrato também confirma compatibilidade do hash para entradas pequenas, execução de subclass de `dict`, bypass de valor extra não serializável e ausência de serialização em ferramenta não cacheável. Cinco mutantes direcionados removendo cada guarda/detecção são mortos.

Resultados locais medidos: unittest 257 executados/aprovados, 0 falhas/skips/XPASS; suíte integrada 30 aprovados e 1 skip FAISS opcional; cobertura 95% fila, 94% guard VRAM, 76% Vita, 68% núcleo; mutation 42/42 mortas, 0 sobreviventes/não aplicadas; Bandit 131 LOW, 0 MEDIUM/HIGH. Ruff, actionlint, zizmor, compileall, auditoria de realidade, auditorias AST e CLIs dos benchmarks de capacidade e POMDP v2 na seed `20261016` passaram. A fonte normativa consultada foi https://docs.python.org/3.12/library/json.html.

Limitação: execução fora do cache continua sujeita ao comportamento e custo do executor; a mudança limita somente a pré-verificação/serialização para cache. As métricas acima são locais; os checks oficiais da PR #28 passaram antes do merge.


## Falha de cópia durante armazenamento de cache — 2026-10-16

Base: `main` pós-PR #28, merge commit `fafcd95e6b40c5d51b34d344f164a9045279d8bb`; branch `feat/post-pr28-cache-store-fallback-2026-10-16`.

Reprodução pré-patch: um executor built-in foi substituído por retorno de listas aninhadas com profundidade 600. `_estimate_cache_value_size` mediu 38.445 bytes, abaixo de `MAX_CACHE_ENTRY_BYTES`; a cópia subsequente com `copy.deepcopy` lançou `RecursionError`. Observado: `execute_tool` devolvia `success=False`, erro `maximum recursion depth exceeded` e nenhum payload, embora o executor já tivesse retornado com sucesso.

Após o patch, o mesmo cenário devolve o objeto e sucesso originais, registra `TOOL_CACHE_STORE_UNAVAILABLE` com `reason=RecursionError` e `cache_bypassed=true`, mantém uma contagem de sucesso, não incrementa falhas e deixa o cache vazio. `test_tool_cache_store_contract.py` protege deterministicamente esse contrato por injeção da exceção. Um mutante que restaura a propagação da exceção é morto.

Resultados locais medidos: unittest 258/258; suíte integrada 30 aprovados, 1 skip opcional FAISS; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation 43/43 mortas, 0 sobreviventes e 0 não aplicadas; Bandit 131 LOW, 0 MEDIUM/HIGH; Ruff, actionlint, zizmor, compileall, auditoria de realidade, auditorias AST e benchmarks de capacidade/POMDP v2 na seed `20261016` passaram.

Limite: a captura é específica à cópia profunda antes do armazenamento; não contém exceções arbitrárias do executor, não muda sua saída e não oferece garantia de limite global de memória. Estado posterior: a PR #29 foi mesclada em `main` pelo merge commit `e06511396d407a3440ee1969e0af8d7e055203dc` após os dois checks oficiais passarem.


## Compatibilidade de subclasses em parâmetros tipados — 2026-10-16

Base: `main` pós-PR #29, merge commit `e06511396d407a3440ee1969e0af8d7e055203dc`; branch isolada `fix/tool-validator-container-subclasses-2026-10-16`.

Reprodução pré-patch: `ToolValidator.validate_parameters` comparava o nome da classe observado com o schema, rejeitando uma subclasse `Items(list)` para `list_operations.items`; o executor teria aceitado a lista em `sorted`, mas não era alcançado. O mesmo critério também aceitava indevidamente uma classe não relacionada se seu `__name__` imitasse um nome built-in, como `list`.

Contrato implementado: nomes built-in conhecidos são resolvidos para classes reais e a relação de subclasse é testada pelo tipo concreto. Uma subclasse de `list` executa e retorna `[1, 2]`; como o preflight de cache não serializa containers customizados, a otimização é bypassada e o sucesso do executor permanece. `int` aceita subclasses verdadeiras, mas `bool` continua rejeitado; um tipo falso com nome `list` não satisfaz o schema. `any` e `object` mantêm semântica wildcard; schemas desconhecidos preservam o matching legado por nome.

Resultados locais medidos: unittest 261/261, sem falhas ou skips; suíte integrada 30 aprovações e 1 skip opcional FAISS; cobertura fila 95%, VRAM 94%, Vita 76%, núcleo 68%; mutation gate 45/45 mortas, 0 sobreviventes e 0 não aplicadas; Bandit 131 LOW e 0 MEDIUM/HIGH; Ruff, actionlint, zizmor, compileall, auditorias AST/realidade e benchmarks oficiais passaram. No POMDP v2, 128/128 casos ficaram dentro de `1e-12` (erro máximo `2.7755575615628914e-16`) e 4/4 controles inválidos foram rejeitados.

Base técnica: [built-ins `isinstance`/`issubclass`, Python 3.12](https://docs.python.org/3.12/library/functions.html#isinstance) e [tipo booleano, Python 3.12](https://docs.python.org/3.12/library/stdtypes.html#boolean-type-bool). Limite: o contrato valida apenas o tipo do valor declarado, sem schema recursivo para elementos aninhados; a PR #30 foi mesclada em `main` no commit `e83ce9eb422e4cb9d2fb5ec818cb795351e57a13` após 2/2 checks oficiais aprovados.


## Limite negativo na recuperação SQLite/FAISS — 2026-10-16

Base local: `main` pós-PR #30, merge commit `e83ce9eb422e4cb9d2fb5ec818cb795351e57a13`; branch isolada `fix/memory-retrieve-nonnegative-limit-2026-10-16`.

Reprodução pré-patch com quatro memórias: busca vazia em `limit=0` retornou 0 linhas, `limit=1` retornou 1 e `limit=-1` retornou as 4. A busca por palavra também não rejeitava negativo; no modo semântico, o `k` negativo podia gerar erro e cair nesse mesmo fallback.

A correção valida antes de escolher o backend: `bool` e valores não `int` geram `TypeError`, negativos geram `ValueError` e zero retorna lista vazia sem chamar encoder/FAISS/SQLite. As consultas SQL permanecem parametrizadas. Regressões cobrem busca vazia, keyword, FAISS simulado, tipos inválidos e zero.

Validação local: unittest 265/265; integração 30 aprovados e 1 skip FAISS opcional; cobertura 95% fila, 94% VRAM, 76% Vita, 68% núcleo; mutation gate 47/47 mortos sem sobreviventes/não aplicados; Bandit 131 LOW e 0 MEDIUM/HIGH; Ruff, actionlint, zizmor, compileall, auditorias AST/realidade, CLIs de capacidade e POMDP v2 passaram. A fonte normativa é a [documentação oficial SQLite de SELECT/LIMIT](https://www.sqlite.org/lang_select.html).

Limite: não há teto para valores positivos altos nem garantia sobre memória global. Finding adjacente ainda não implementado: os metadados `requires_permission`/`permission_level` não têm enforcement; todos os built-ins atuais os deixam desativados, e a definição de um modelo de autorização foi adiada por depender de escolha explícita de API.

## Falha de cópia ao recuperar um cache hit — 2026-10-16

Base da atualização: `main` pós-PR #32, merge commit `b07254307a23cdaef68f7c2b0d23010ac0827580`, que já contém a PR #33 (merge commit `62f6af14a2ad1d3659a4a5940e713c588513bf9b`). A PR #32 foi mesclada e D-024 passou a integrar `main`.

Reprodução pré-patch: uma chamada válida a `list_operations` preenchia o cache; uma falha injetada (`RecursionError`) na cópia do hit seguinte escapava de `execute_tool`, pois ocorria fora do `try` que envolve o executor. A segunda chamada não executava a ferramenta; estado observado: `successful_executions=1`, `failed_executions=0`, `cached_executions=0` e a entrada defeituosa permanecia armazenada.

Correção: capturar a exceção apenas ao copiar o resultado do hit, remover a entrada e seu tamanho contabilizado, emitir `TOOL_CACHE_HIT_UNAVAILABLE` com tipo de exceção e `cache_bypassed=true`, e continuar pelo executor. A promoção LRU e os contadores de hit/sucesso só acontecem após uma cópia completa. Se a execução seguinte também não puder ser armazenada, o fallback pré-existente emite `TOOL_CACHE_STORE_UNAVAILABLE` e mantém o sucesso retornado pelo executor.

`test_tool_cache_hit_contract.py` protege a execução real após falha do hit, contadores sem hit espúrio, e evicção quando a cópia da tentativa de re-cache também falha. Dois mutantes direcionados cobrem propagação da exceção e remoção da entrada; o workflow executa este contrato em etapa dedicada.

Validação histórica pré-merge da branch sincronizada: suíte focada de cache 9/9; unittest 271/271 (0 falhas, skips ou XPASS); integração 30 aprovados + 1 skip FAISS opcional; pisos de cobertura 95/94/76/68%; mutation gate 55/55 mortas, sem sobreviventes/não aplicadas; Bandit 131 LOW e nenhum MEDIUM/HIGH, triagem consistente. Ruff, compileall, actionlint, zizmor, auditorias AST/realidade/gaps e CLIs dos benchmarks de capacidade/POMDP v2 passaram. POMDP v2: 128/128 casos dentro de `1e-12` (erro máximo `2.7755575615628914e-16`) e 4/4 controles inválidos rejeitados.

Status de publicação: a [PR #32](https://github.com/pedromiorini/Nexus/pull/32) foi mesclada em `main` no commit `b07254307a23cdaef68f7c2b0d23010ac0827580`; D-024 e D-025 estão juntos na base atual.

Fundamentação: a [documentação oficial `copy` do Python 3.12](https://docs.python.org/3.12/library/copy.html) descreve `deepcopy` como recursiva e observa problemas possíveis com objetos recursivos; a falha específica desta fronteira foi reproduzida por injeção determinística. Limite: esta alteração cobre cópia ao recuperar hit, não falhas do logger. A corrida de mutabilidade de parâmetros foi resolvida pelo snapshot documentado na seção seguinte (PR #33).

## Snapshot JSON limitado para parâmetros de ferramentas — PR #33 mesclada
Base da correção: a PR #33 foi desenvolvida sobre `main` pós-PR #31 e mesclada em `main` no commit `62f6af14a2ad1d3659a4a5940e713c588513bf9b`; a PR #32 foi mesclada depois, no commit `b07254307a23cdaef68f7c2b0d23010ac0827580`. Ambas as decisões estão atualmente em `main`.

A lacuna era temporal: `execute_tool` podia formar uma cache key a partir do valor mutável original e, antes de validar/executar, o chamador alterar esse objeto. A chave então descrevia um valor enquanto o executor observava outro. O contrato `test_tool_cache_snapshot_contract.py` reproduz a divergência e exige identidade do mesmo snapshot nas fases subsequentes; um caso adicional confirma que referências compartilhadas tornam-se cópias independentes por valor.

A correção constrói um snapshot iterativo e limitado para parâmetros cacheáveis estritamente JSON, com os budgets existentes de 64 KiB, profundidade 64, 4.096 ocorrências e 1.024 itens por contêiner. A ordem final é snapshot → validação/schema e guardas semânticas → cache key → lookup → executor. A primeira suíte completa detectou uma regressão de ordenação: expressões de `math_calculate` acima do limite eram serializadas antes da rejeição. A chave foi então movida para depois da validação e da guarda de tamanho, mantendo o mesmo snapshot nas três fases; `test_math_calculate_resource_limits.py` protege esse requisito.

Se o preflight do snapshot falhar (ciclo, tipo não JSON, subclass/container não reconhecido ou budget excedido), a cache é bypassada e a execução segue com o objeto original. Se apenas a geração da chave falhar depois de um snapshot válido, lookup/store são ignorados sem trocar o snapshot usado pela validação e pelo executor.

Uma revisão independente não encontrou bloqueadores, mas identificou uma limitação P2: substituição de valor com o mesmo tamanho enquanto a travessia está em curso pode formar um snapshot misto, pois a cópia não sincroniza o produtor. Depois de construído, o mesmo snapshot é usado por chave, validação e executor; aplicações que exijam um instante atomicamente consistente precisam sincronizar as escritas na origem. Identidade, aliasing e mutações in-place do original não fazem parte da semântica preservada.

Validação local do head original da PR #33 (pré-sincronização da PR #32), Python 3.12: suítes focadas de cache 9/9; suíte completa `unittest` 269/269, sem falhas/skips/XPASS; suíte integrada 30 aprovados e 1 skip FAISS opcional; cobertura fila 95%, VRAM 94%, Vita 76% e núcleo 68%; mutation gate 53/53 mortas, 0 sobreviventes e 0 não aplicadas. Bandit encontrou 131 LOW e nenhum MEDIUM/HIGH; a triagem verificou-se consistente. Ruff, compileall, actionlint, zizmor, auditorias AST/realidade/gaps e CLIs dos benchmarks de capacidade/POMDP v2 na seed `20261016` passaram. O v2 manteve 128/128 casos dentro de `1e-12` (erro máximo `2.7755575615628914e-16`) e rejeitou 4/4 controles inválidos.

## Limite não negativo na busca de memória episódica — 2026-10-16

Base original da alteração: branch `fix/episodic-search-nonnegative-limit-2026-10-16`, criada sobre `main` pós-PR #32 no commit `b07254307a23cdaef68f7c2b0d23010ac0827580` (que contém também a PR #33). A PR #34 foi mesclada em `main` no commit `51373dc77179e4edd094df14290f91aa9c8d8060`.

Reprodução pré-patch: com dois episódios fechados, `search_episodes(status="closed", limit=-1)` retornou os dois; `limit=True` também foi aceito como limite 1. A implementação repassava diretamente o valor para a cláusula SQL `LIMIT`.

Correção: `search_episodes` rejeita bool e tipos não inteiros com `TypeError`, negativos com `ValueError` e devolve `[]` para zero antes de adquirir lock ou executar SQL. Limites positivos e os filtros existentes de query/status permanecem inalterados; nenhum máximo positivo novo foi introduzido.

`test_episodic_memory_search_limits_contract.py` cobre limites negativos nos dois ramos de query e prova ausência de SQL, rejeição de bool/float/string/None, zero sem consulta e preservação de limite positivo. Validação final: 11/11 contratos episódicos focados e unittest 275/275; integração 30 aprovados + 1 skip FAISS opcional; cobertura 95/94/76/68%; mutation 58/58 mortas, sem sobreviventes/não aplicadas; Bandit 131 LOW e nenhum MEDIUM/HIGH. Ruff, compileall, actionlint, zizmor, auditorias AST/realidade/gaps e benchmarks oficiais passaram. O POMDP v2 obteve 128/128 casos dentro de `1e-12` (erro máximo `2.7755575615628914e-16`) e rejeitou 4/4 controles inválidos. O relatório de realidade foi regenerado para refletir as sete linhas acrescentadas ao núcleo.

Fonte normativa: [SQLite Language Reference — SELECT, seção LIMIT](https://www.sqlite.org/lang_select.html), que especifica que um LIMIT negativo não estabelece teto para as linhas retornadas.

## Resumo unittest vazio não aprova o gate — 2026-10-20

**Fato reproduzido:** na base `d435083bb4c893a58128cee6cfa00f65bba7fe02`, uma descoberta sem testes emitia contagens zeradas, `result=passed` e exit 0. Na branch local `fix/test-summary-empty-suite-20261020`, a aprovação do resumo passa a exigir pelo menos um teste executado e sucesso do unittest. Vazio mantém todos os contadores em zero, mas produz `result=failed` e exit 1. Nenhum teste falho artificial é somado, e o resumo da suíte integrada permanece inalterado.

**Evidência:** `$PY -m unittest -v test_test_summary.py` com `PY=/tmp/nexus-ci311/bin/python`: antes da correção, 4 testes com 3 falhas de asserção em 2 casos, demonstrando separadamente resumo vazio, JSON do CLI real e código de saída; depois, 4/4 aprovados. Os controles protegem pass, skip-only, expected-failure-only, XPASS e contagem única de múltiplos subtests falhos. Blobs Git validados: `tools/test_summary.py=0110883147e80e3de56e0b1a387ed87fe78016af`, `test_test_summary.py=207a9aa7e0ec2e13134b823497986e39a92157a2`; hash do commit final no handoff externo. Comandos exatos e ambiente estão na seção correspondente de `CONTINUATION.md`.

**Verificação local:** suíte completa com cobertura e CLI final: 306 executados, 304 aprovados, 2 skips por NetworkX ausente, 0 falhas/XPASS; integrada 30 aprovações e 1 skip FAISS. Cobertura fila/VRAM/temporal/Vita/núcleo: 95/94/93/76/68%, pisos aprovados. Gate de mutação herdado: 58/58 mortos, 0 sobreviventes/não aplicados/timeouts; não inclui mutantes da guarda nova. Ruff, compileall, actionlint, zizmor, auditorias de realidade/asserts/lacunas, Bandit e triagem passaram; 131 LOW e 0 MEDIUM/HIGH/erros de análise não constituem garantia de segurança. Saídas em `/tmp/nexus-empty-suite-20261020`, fora da árvore versionada; os quatro marcadores canônicos foram sincronizados para 306.

**Limites:** ambiente local Python 3.11.16 sem reinstalação do lock; Ruff 0.17.1 em vez do pin 0.16.10, NetworkX opcional não exercitado. Não houve nova sondagem sintética, execução de CLIs de benchmark, publicação ou run remoto. Sem atestação de CI fabricada. O texto padrão do unittest para vazio ainda é `OK`, mas o JSON final é `failed` e a saída do processo é 1. O leitor herdado do manifesto rejeita o vazio falho como `invalid_summary`; continua falhando fechado e não foi modificado. Testes internos e mutação não são evidência de capacidade geral.
## Validação local de LOW no auditor de consistência — 2026-10-20

Base exata `d435083bb4c893a58128cee6cfa00f65bba7fe02`, branch isolada `fix/state-consistency-low-validation-20261020`. O baseline aceitou os quinze valores inválidos das novas regressões quando os documentos repetiam o valor: contagens negativas, booleanos, floats, strings, null, NaN/infinitos e contêineres. Também comparou um LOW inválido com os documentos: 16 falhas de subcasos em 11 métodos antes da correção. O auditor agora exige inteiro built-in não negativo e registra erro antes de qualquer comparação dessa métrica, sem alterar valores válidos.

Evidência local: 25 contratos focados aprovados; suíte completa 306 executados, 304 aprovados e dois skips opcionais NetworkX; integração 30 aprovados e um skip FAISS. Mutação existente 58/58 e cinco mutações locais da nova guarda mortas; zero sobreviventes/não aplicadas. Compilação, Ruff, actionlint, zizmor, triagem Bandit e auditorias AST/realidade aprovados; 131 LOW, zero MEDIUM/HIGH/erros de análise. Isso é evidência de contrato local, não clearance de segurança nem validação de capacidades gerais.

Comandos, blobs Git do código/testes, ambiente, pisos de cobertura e limites estão na rodada correspondente de `CONTINUATION.md`. Não se alterou o gerador do manifesto, o workflow, dependências ou PRs. Sem execução hospedada ou autorização para publicação; o manifesto local conserva `not_ci`, que bloqueia a atestação operacional. O hash do commit final e a revisão independente são entregues no handoff externo.
## Inventário de classe explícita em `raise` — 2026-10-21

Base `d435083bb4c893a58128cee6cfa00f65bba7fe02`; branch local `fix/gap-audit-exception-class-20261021`. O auditor AST retornava zero para `raise NotImplementedError` e uma ocorrência para a chamada equivalente. A [referência Python de raise](https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement) aceita tanto classe quanto instância; a correção reconhece o alvo nome/atributo sem exigir chamada, sem executar fonte ou resolver nomes.

RED na base: cinco métodos novos executados, quatro afetados por oito falhas de assertivas/subcasos, um aprovado. GREEN: 24/24 contratos focados e suíte 308/308 em Python 3.11.16 e 3.12.3; integração 30 aprovadas + 1 skip opcional em ambas. Em 3.11: cobertura do auditor 94%; pisos de fila/VRAM/temporal/Vita/core aprovados (95/94/93/76/68%); mutation existente 58/58 mortas, sem sobreviventes/não aplicadas; Bandit 131 LOW, zero MEDIUM/HIGH/erros. Ruff, compileall, actionlint, zizmor, requirements dry-run, auditorias e benchmarks CLI existentes aprovados. Inventário do núcleo permanece zero pass-only e zero NotImplementedError. Não é evidência de ausência geral de lacunas, segurança completa ou capacidade cognitiva.

Comandos, ambientes, limites e inventário de sobreposição estão em `CONTINUATION.md`; logs e comandos brutos em `/workspace/nexus-gap-audit-class-evidence-20261021/validation.json`, `red.log`, `green311.log`, `green312.log` e `run_gates.py`. Ambientes anteriores do lock reutilizados, sem reinstalação. Nenhum CI hospedado/push/PR/merge foi executado. WIP `window_seconds` não incorporada; `known_at` permanece pendente. Nenhuma decisão anterior foi refutada.
## Guarda demonstrativa direta do módulo — 2026-10-20

Base exata: `d435083bb4c893a58128cee6cfa00f65bba7fe02`; branch local `fix/embedded-assert-module-guard-20261020`. A regressão de função usa exatamente `def helper():\n    if __name__ == '__main__':\n        assert False\n`. Antes do patch, a faixa aninhada era aceita. Depois, nenhuma faixa é criada para essa guarda, a linha 3 fica em `assert_lines_outside_main` e `ok=False`, inclusive sem finding.

Red: os 11 métodos do auditor executaram na implementação original, com 37 falhas de subcasos em três métodos (18 cenários com B101, 18 sem findings e um finding sem assert). Green: 11/11 métodos; contratos focados do auditor, triagem, manifesto e consistência: 32/32. Os nove contextos são função, função assíncrona, classe, `if`, `for`, `while`, `with`, `try` e `else`; cada contexto é testado com/sem uma guarda legítima de módulo adicional. Controles preservam a comparação convencional, aspas simples/duplas e conteúdo aninhado dentro da demonstração legítima; excluem o `else` da guarda e comparações não convencionais.

Resultados locais em Python 3.11.16, sem instalar dependências: suíte completa 310 executados, 308 aprovados, 2 skips por NetworkX opcional ausente, 0 falhas/XPASS; integração 30 aprovações e 1 skip FAISS opcional, sem IDs ausentes, conflitantes ou inesperados. Cobertura combinada: fila 95%, VRAM 94%, fatos temporais 93%, Vita 76%, núcleo 68%; auditor B101 76% (inclui CLI não coberta pela medição unitária; CLI real executada separadamente). Mutation: 59/59 aplicáveis mortos, sem sobreviventes/não aplicados; o novo mutante morreu com saída 1, sem timeout. Ruff, compileall, actionlint, zizmor, auditoria de realidade e auditorias AST passaram. Bandit 131 LOW, 0 MEDIUM/HIGH e nenhum erro de análise; triagem aprovada e 111 B101 preservados na faixa real `25680–27549`, sem findings/asserts fora. Relatórios versionados de realidade inalterados.

Comandos reproduzíveis e limitações operacionais estão na rodada correspondente de `CONTINUATION.md`. Esta é evidência local de um contrato sintático, não CI hospedado nem clearance de segurança. Não houve nova sonda sintética ou execução adicional das CLIs de benchmark; os contratos de benchmark já existentes continuam dentro da suíte completa. Não se fabricou `gates_status=success`: a atestação operacional depende do job hospedado concluído. Quatro marcadores atuais passam de 303 para 310 testes executados; histórico, findings e disposições não são reescritos.
