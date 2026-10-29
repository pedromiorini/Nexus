# Mutation gate do core/cognitivo em paralelo — diagnóstico e pré-registro

Origem: o PR #130 (guarda de margem do timeout) teve o `contract-gate` cancelado por tempo duas vezes no passo `Run core/cognitivo mutation gate`. Pedro decidiu acelerar esse passo em PR separado (ramo a partir de `main` 8f983aa, não do #130), **sem subir o timeout e sem enfraquecer o gate**: mesmo conjunto de mutantes, mesma exigência (todos mortos), resultado equivalente ao serial.

## Propriedade do arquivo (verificado antes de editar)
`tools/cognitivo_mutation.py` **não** é contrato do Axis. O docstring do próprio arquivo o declara "separado de `tools/targeted_mutation.py` (arquivo no escopo do Axis)"; nos docs de agentes as únicas menções ao Axis são `versao_aceita`, `last_seen` e registro. Só `tools/targeted_mutation.py` é Axis e **não é tocado** (nem `run_tests`, `Mutation`, `COPY_IGNORE`, `TIMEOUT_SECONDS`). O PR aberto #116 também edita `tools/cognitivo_mutation.py` (entradas do catálogo); este PR só mexe em `main()`/helpers no fim do arquivo, para limitar o risco de conflito.

## Diagnóstico (evidência real, coletada antes de qualquer mudança)
Fontes: API de jobs do GitHub Actions (`actions/runs/{id}/attempts/{n}/jobs`, duração de cada passo) para as execuções do job `Contract and regression gate` desde 21/10 (lista de execuções do instantâneo `timeout_margin_historico.json` do #130; o resultado por passo está em `docs/agentes/mutacao_cognitivo_passos_hospedados.json`), mais uma execução serial local do código de `main`.

1. **O passo é estritamente serial e cresceu junto com o catálogo.** `main()` copia o repo uma vez e roda um subprocesso `python -m unittest <arquivos>` por mutante, um depois do outro. O catálogo tem hoje **483 mutantes** (34 conjuntos de testes distintos). Mediana hospedada do passo por dia (execuções com sucesso; dados em `docs/agentes/mutacao_cognitivo_passos_hospedados.json`, 207 jobs de 21/10 a 29/10, 108 com o passo concluído): 22/10 **25 s** (máx. 122 s), 23/10 122 s, 24–26/10 ≈ 178 s, 28/10 269 s (máx. 444 s), 29/10 **701 s** (máx. 867 s; últimos 5 sucessos: 867, 779, 862, 518, 829 s, mediana 829 s). Os demais passos pesados ficaram estáveis (mutação direcionada 32–142 s). O job inteiro foi de ~200 s para 1100–1210 s e encosta no `timeout-minutes: 20` (1200 s): é o que cancelou o #130 e a execução de 1210 s. Cresceu mais do que o número de mutantes: os mutantes adicionados depois (simulação mental, memória contextual/emocional) rodam testes de contrato mais pesados (ver item 3).
2. **A premissa "~30 s local" não se confirma.** Executando `python tools/cognitivo_mutation.py` em `main` (8f983aa), Python 3.11, máquina de 8 núcleos: **11 min 56 s** (716 s reais, 19 min 50 s de CPU), 483/483 mortos, soma das durações por mutante = 680 s, média 1,41 s, mediana 1,17 s, máximo 8,39 s, nenhum com código 124. A suíte inteira roda em ~30 s; o gargalo é o **número de mutantes × custo de um subprocesso de testes** (≈ 1,4 s local, ≈ 1,7 s hospedado = 829 s / 483), não um problema específico do runner hospedado.
3. **Onde o tempo vai por mutante.** Custo de interpretador vazio: 0,01 s; import dos testes (`-X importtime`): ≈ 0,2 s; o resto é o próprio teste (muitos testes de contrato abrem subprocessos e importam o pacote). Os 4 maiores conjuntos: `test_mental_simulation_contract.py`+`goal_credit` 54 mutantes, média 4,14 s, 224 s no total; `test_contextual_memory_contract.py` 74 × 2,06 s = 152 s; `test_emotional_memory_weighting_contract.py` 92 × 1,42 s = 131 s; `test_analogical_reasoning_contract.py` 51 × 1,09 s = 56 s. Esses 4 somam ~563 s dos 680 s. Cada mutante **já** roda só os arquivos relevantes (`tests=`), então "rodar só os testes relevantes" **não** é uma alavanca disponível: já é o que acontece. Imports a frio não dominam (0,2 s de ~1,4 s).
4. **Paralelismo: nenhum.** O runner `ubuntu-24.04` de repositório público tem 4 vCPU (a máquina local tem 8); o passo usa 1. Os mutantes são independentes (cada um altera um arquivo numa cópia e restaura), então a limitação é só o laço serial compartilhando uma única cópia.
5. **Limite da evidência:** o relatório `cognitivo_mutation_report.json` não é retido como artefato no CI (só `targeted_mutation_report.json`), então não há duração por mutante hospedada; o custo hospedado por mutante (≈ 1,7 s) é derivado da duração do passo ÷ 483.

**Causa:** laço serial de 483 subprocessos (~1,4–1,7 s cada) numa única cópia do repositório, num runner com 4 vCPU usando 1.

## Correção escolhida
`python tools/cognitivo_mutation.py [--jobs N]`: N trabalhadores (threads que só aguardam subprocessos), **cada um com a sua própria cópia do repositório** (mutantes nunca colidem no mesmo arquivo), fila compartilhada de mutantes, resultados gravados **na ordem do catálogo** independente da ordem de término. Padrão `N = min(os.cpu_count(), 8)` (4 no hospedado; o workflow não muda). `--jobs 1` mantém o caminho serial (uma cópia, laço em ordem). Linha de base (execução de cada conjunto de testes sem mutação) também roda em paralelo e continua abortando tudo se falhar. Não muda: catálogo, `run_tests`, timeout de 180 s por mutante (124 continua "morto"), regra de `needle` com exatamente 1 ocorrência (`not_applied`), formato do relatório e do resumo, saída 0 só se **todos** mortos.

Entrada inválida (`--jobs` ausente/vazio/não inteiro/0/negativo/decimal/zero à esquerda/> 32/duplicado, argumento desconhecido/posicional): **saída 2**, mensagem em stderr, **nada** executado (nem cópia, nem testes, nem relatório). 2 é distinto de 0 (todos mortos) e de 1 (sobrevivente, `not_applied` ou linha de base quebrada).

## Pré-registro dos critérios (commit anterior a qualquer mudança de código)
**Já visto (declarado):** a execução serial local de `main` e as durações dos passos acima. **Não visto:** qualquer execução paralela, qualquer duração hospedada depois da mudança.

- **M1 equivalência (suíte sintética):** para um catálogo pequeno em repositório temporário com mutantes mortos, sobrevivente, `not_applied` (0 e 2 ocorrências) e mutante que dá timeout/saída ≠ 0, `--jobs 1`, `3` e `8` produzem relatório idêntico nos campos `name, target, tests, status, exit_code, matches`, **mesma ordem** e mesmo resumo, e o mesmo código de saída.
- **M2 equivalência (catálogo real):** em Python 3.11, uma execução com `--jobs 4` e uma com `--jobs 8` têm, para os 483 mutantes, exatamente os mesmos `name, target, tests, status, exit_code` e a mesma ordem que o baseline serial salvo de `main` (8f983aa); resumo `483/483/0/0`; **nenhum** `exit_code` 124 (timeout sob carga não pode mascarar sobrevivente).
- **M3 determinismo:** duas execuções `--jobs 4` do catálogo real dão relatório idêntico (mesmos campos de M2, mesma ordem).
- **M4 ganho local:** tempo total de `--jobs 4` ≤ 50 % do serial medido (716 s) = **≤ 358 s**, na mesma máquina.
- **M5 ganho hospedado:** no 1º `contract-gate` concluído do cabeçalho final do PR, passo `Run core/cognitivo mutation gate` **≤ 450 s** (referência serial: 829 s em 29/10 17:40; mediana dos 5 últimos sucessos = 829 s) **e** job inteiro **≤ 900 s** (75 % de 1200 s). Se a execução for cancelada por infraestrutura, uma reexecução é permitida e as duas são relatadas.
- **M6 negativos:** os casos de entrada inválida acima saem 2 sem executar nada; sobrevivente → 1; `not_applied` → 1; linha de base quebrada → aborta antes de gravar relatório; relatório ordenado; trabalhadores com cópias distintas (um mutante nunca vê a mutação de outro).
- **M7 gate não enfraquecido:** diff vazio em `tools/targeted_mutation.py`; nenhum hunk dentro da lista `MUTATIONS` de `tools/cognitivo_mutation.py`; `len(MUTATIONS) == 483`; `timeout-minutes` do workflow continua 20 e o workflow não muda; `TIMEOUT_SECONDS` inalterado.
- **M8 sem regressão:** suíte completa verde em Python 3.11 e 3.12, ruff, compileall, Bandit/`security_triage --check` e `reality_audit` consistentes (`git diff --exit-code`), `claims: []` ao final, CI hospedado verde.

### Disciplina
Se algum critério falhar: parar e relatar opções; **sem errata e sem ajuste** de limiares, definições ou dados depois de ver o resultado. Não toca testes/contrato do Axis, `tools/targeted_mutation.py` nem `versao_aceita`. #79, #115 e #116 ficam intocados.

### Previsões (não vinculantes)
Speedup ≈ 3–3,7× em 4 vCPU (passo ≈ 230–280 s hospedado, ≈ 200–250 s local com 4 trabalhadores). Risco principal: testes que dependem de relógio/CPU sob carga; M2 verifica que nada vira 124.

## Resultados (medidos depois do pré-registro; nenhum critério ou limiar foi alterado)

Implementação em `tools/cognitivo_mutation.py` (`parse_jobs`, `evaluate_mutation`, `run_catalog`, `main(argv)`); testes de contrato em `test_cognitivo_mutation_parallel_contract.py` (16 testes). RED: contra o `tools/cognitivo_mutation.py` de `main`, 38 erros (`parse_jobs`/`run_catalog` inexistentes). GREEN: 16/16, 3 execuções seguidas sem falha (≈ 2,8 s cada). Verificação extra do teste de isolamento: com todos os trabalhadores forçados a compartilhar uma única cópia, ele falha (1 morto, 5 `not_applied`), logo detecta a regressão que protege.

| Critério | Resultado |
|---|---|
| M1 equivalência sintética | PASSA: `--jobs` 1, 3 e 8 dão relatório idêntico (campos, ordem, resumo `6/3/1/2`, saída 1) |
| M2 equivalência real | PASSA: `--jobs 4` (2×) e `--jobs 8` (Python 3.11) idênticos ao baseline serial de `main` nos 483 mutantes (`name, target, tests, status, exit_code`, ordem); resumo 483/483/0/0; nenhum `exit_code` 124 |
| M3 determinismo | PASSA: as duas execuções `--jobs 4` têm relatório idêntico |
| M4 ganho local | PASSA: `--jobs 4` = 186,1 s e 182,2 s (alvo ≤ 358 s; serial 716 s → 26 %) |
| M5 ganho hospedado | PASSA: 1º `contract-gate` do cabeçalho `d693f05`: passo cognitivo **342 s** (alvo ≤ 450 s; antes 829 s) e job **604 s** (alvo ≤ 900 s; antes 1100–1210 s) |
| M6 negativos | PASSA: `--jobs` ausente/vazio/0/negativo/decimal/`01`/>32/duplicado/argumento desconhecido → 2 sem executar nada nem escrever relatório; sobrevivente → 1; `not_applied` (0 e 2 ocorrências) → 1; linha de base quebrada aborta sem relatório; timeout continua "morto" (124); cópias isoladas; concorrência real (barreira de 3) |
| M7 gate não enfraquecido | PASSA: diff vazio em `tools/targeted_mutation.py` e no workflow; o bloco `MUTATIONS` é idêntico byte a byte ao de `main`; `TIMEOUT_SECONDS` 180; `timeout-minutes: 20` |
| M8 sem regressão | PASSA localmente: suíte 816 testes OK em Python 3.11.17 e 3.12.15, ruff, compileall, `reality_audit` sem diff, Bandit 131 LOW (inalterado) com `security_triage --check` ok, `claims: []`; CI hospedado do cabeçalho final: ver o PR |

Observação (não é critério): com `--jobs 8` numa máquina de 8 núcleos o tempo foi **340 s**, pior que `--jobs 4` (186 s), porque os testes de contrato já usam mais de um núcleo por processo (19 min 50 s de CPU em 11 min 56 s no serial). O padrão pré-registrado `min(núcleos, 8)` vale 4 no runner hospedado (onde foi medido), mas numa máquina de 8 núcleos convém passar `--jobs 4`. Opção, não aplicada para não alterar o que foi pré-registrado depois de ver os dados: reduzir `DEFAULT_MAX_JOBS` para 4 em PR seguinte.

Limite honesto: o relatório do cognitivo continua sem retenção como artefato no CI, então ainda não há duração por mutante hospedada; opção futura: incluí-lo no passo "Retain mutation report".
