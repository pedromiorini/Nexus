# Folga do `Contract and regression gate`: mutação cognitiva em jobs paralelos — pré-registro e resultados

Autorização: Pedro, mensagem t162u («Ok», sobre abrir PR para dar folga ao gate). Agent-Id: grok-bot. Base: `main` `fcae00e`.

## Problema (Fato, medido nos jobs hospedados com `gh run view --json jobs`)
O job único `Contract and regression gate` tem `timeout-minutes: 20` (1200 s). Os 8 últimos sucessos de PR com o passo cognitivo (início do passo `Run core/cognitivo mutation gate` hospedado):

| run | head | job (s) | cognitivo (s) | targeted (s) | cobertura+unittest (s) | setup (s) |
|---|---|---|---|---|---|---|
| 41356796818 | ebe947f | 1069 | 666 | 108 | 195 | 8 |
| 41374297755 | 89680a9 | 978 | 595 | 103 | 183 | 13 |
| 41691764779 | 63d3a7e | 768 | 472 | 81 | 129 | 18 |
| 41789420566 | 42a89d8 | 795 | 482 | 89 | 144 | 7 |
| 41795642554 | dc6f070 | 792 | 513 | 78 | 125 | 8 |
| 41817435651 | 0a73d6a | 1102 | 696 | 114 | 193 | 11 |
| 41848524272 | 64eb13a | 666 | 415 | 68 | 114 | 8 |
| 41868010519 | 32b9e04 | 1149 | 709 | 117 | 217 | 11 |

Faixa do job: 666–1149 s (55 %–96 % do limite); o passo cognitivo é 59–62 % do job e varia 1,7× conforme o runner (415–709 s). Os 811 mutantes já rodam em 4 trabalhadores por runner (#131). Setup (checkout, Python, pip) custa 7–18 s: cache de dependências não ajuda. O repositório é público (runners de 4 vCPU); não há proteção de branch nem rulesets (API: 404 e `[]`), logo não há «required check» a renomear.

**Visto antes de escrever isto (declarado):** as durações acima, o workflow, os testes de contrato que o citam (`test_agent_replay_contract.py`, `test_cognitivo_mutation_parallel_contract.py`), `docs/agentes/MUTACAO_COGNITIVO_PARALELA.md`, `docs/agentes/MARGEM_TIMEOUT_CI.md`. **Não visto:** qualquer execução hospedada com jobs separados ou fatias (`--shard` não existe).

## Restrições herdadas (não negociáveis)
1. `test_agent_replay_contract.py` (Axis) exige `state-manifest.needs == {contract-gate, agent-guard-replay}` e a condição `needs.contract-gate.result == 'success' && needs.agent-guard-replay.result == 'success'`. Não será editado. Logo o novo job só pode entrar na atestação **transitivamente**: `agent-guard-replay` passa a ter `needs: [cognitivo-mutation]`. Se qualquer fatia falhar, o replay fica `skipped` e a atestação não roda. Limite honesto: o nome do replay não diz isso; o desenho mais limpo (a atestação listar o novo job) exige mudar o teste do Axis e fica como opção.
2. `test_cognitivo_mutation_parallel_contract.py` exige no workflow as cadeias `timeout-minutes: 20` e `run: python tools/cognitivo_mutation.py`. Mantidas.
3. `tools/targeted_mutation.py`, testes do Axis, #79/#115/#116/#131: intocados. Nenhum teste existente é editado (só arquivos novos). Escopo do registro: `.github/**` e `docs/agentes/**` são do `grok-bot`; `tools/targeted_mutation.py` e `test_agent_registry_contract.py` são do Axis, e o guard do registro (`validar_registro.py`) só valida hash das instruções e claims, não caminhos de workflow. Logo **não** exige aceite do Axis.
4. Regra 6 das instruções (workflow em PR dedicado): este PR altera o workflow **e** adiciona a opção `--shard` em `tools/cognitivo_mutation.py` porque o workflow não funciona sem ela. Desvio declarado; dividir em dois PRs se o Pedro preferir.

## Desenho (decidido antes de qualquer execução fatiada)
- `tools/cognitivo_mutation.py --shard K/N` (1 ≤ K ≤ N ≤ 16): o mutante de índice `i` (0-based, na ordem de `MUTATIONS`) pertence à fatia `(i mod N) + 1`. Sem `--shard` o comportamento é idêntico ao atual. O catálogo não muda.
- Novo job `cognitivo-mutation` com matriz `shard: [1, 2, 3]` (N = 3), `fail-fast: false`, `timeout-minutes: 20`, mesmos passos de setup, `run: python tools/cognitivo_mutation.py --shard ${{ matrix.shard }}/3`, relatório de cada fatia retido como artefato.
- O passo `Run core/cognitivo mutation gate` sai de `contract-gate`; todos os outros passos dele ficam, na mesma ordem, byte a byte. `contract-gate` mantém o `targeted_mutation` (ferramenta do Axis) e todos os artefatos da atestação.
- `agent-guard-replay.needs: [cognitivo-mutation]`. `state-manifest` não muda.
- `margem_main.sh` (proposta de 30/10) **não entra**: depende de `timeout_margin.sh` e acrescenta casos à tabela de replay do Axis; não é pequena nem é necessária para a folga. Fica para PR próprio.

## Critérios (todos precisam passar; medidos depois, sem ajuste)
- **CG1 margem por job:** nas 3 primeiras execuções hospedadas `pull_request` do SHA final do PR (re-execuções do mesmo SHA, as 3 relatadas mesmo se uma for cancelada por infraestrutura), todo job `Contract and regression gate` e toda fatia `cognitivo-mutation` termina `success` com duração ≤ **720 s** (60 % de 1200 s).
- **CG2 margem do conjunto:** nas mesmas 3 execuções, o tempo do início da execução até o fim de `Final state attestation` ≤ **720 s**. Referência antes: o job único chegou a 1149 s.
- **CG3 mesmos mutantes:** (a) teste local: as 3 fatias de `MUTATIONS` são disjuntas e a união é o catálogo inteiro, na ordem; vale também para N = 1…16; (b) hospedado: nas 3 execuções, a soma de `declared` dos relatórios das fatias = `len(MUTATIONS)` do SHA (811 em `main`), `killed` = `declared`, `survived` = `not_applied` = 0, e a união dos nomes = o catálogo (checada por script sobre os artefatos).
- **CG4 nada enfraquecido:** diff vazio em `tools/targeted_mutation.py`; nenhum hunk no bloco `MUTATIONS`; `TIMEOUT_SECONDS` 180; nenhum arquivo `test_*.py` existente alterado; no workflow, cada passo de `contract-gate` de `main` exceto `Run core/cognitivo mutation gate` está presente, na mesma ordem e com o mesmo conteúdo (inclusive todos os `--fail-under`); `contract-gate.timeout-minutes` continua 20; nenhum `continue-on-error` nem `if:` novo nos passos do job novo.
- **CG5 atestação (estrutural):** `state-manifest.needs` e `if` byte a byte como em `main`; `agent-guard-replay.needs` contém `cognitivo-mutation`; as fatias da matriz são exatamente `{1..N}` e o `N` da matriz é o `N` do comando (teste novo que lê o YAML).
- **CG6 atestação (negativa hospedada):** um commit de prova com `--shard 4/3` (entrada inválida, sai 2) faz a fatia falhar, `Agent guard replay` ficar `skipped` e `Final state attestation` ficar `skipped`; o commit é revertido depois e a prova é relatada.
- **CG7 negativos de `--shard`:** `0/3`, `4/3`, `1/0`, `1/17`, `a/b`, `1`, `1/`, `/3`, `1/3/5`, `01/3`, `1/03`, ` 1/3`, `1/3 `, `-1/3`, `1.5/3`, `--shard` sem valor e `--shard` repetido saem **2** sem executar nada nem escrever relatório. `--shard=2/3` e `--shard 2/3` são equivalentes. `1/1` equivale a sem `--shard`.
- **CG8 equivalência sintética:** num catálogo pequeno (mortos, sobrevivente, `not_applied`), a união dos relatórios de 3 fatias (reordenada pelo índice) tem os mesmos `name, target, tests, status, exit_code, matches` que a execução sem `--shard`; o código de saída de cada fatia reflete só as suas falhas.
- **CG9 sem regressão:** suíte completa verde em Python 3.11 e 3.12, ruff, compileall, `reality_audit` sem diff, Bandit/`security_triage --check`, cobertura de `core/cognitivo/*` ≥ 90, `targeted_mutation` 80/80, `cognitivo_mutation` completo local (sem `--shard`) = catálogo/catálogo, `validar_registro` ok, `claims: []`, CI hospedado verde.
- **CG10 (informativo):** relata-se a duração por job nas 3 execuções, o que a guarda `timeout_margin.sh 20` (85 % = 1020 s) diria sobre elas, e o total de minutos de runner antes/depois.

## Previsões (registradas, não vinculantes)
Fatia ≈ 1/3 do passo cognitivo mais o custo de rodar a linha de base das suítes em cada fatia (≈ 51 suítes distintas) → 200–330 s por fatia; `contract-gate` ≈ 440–480 s no runner lento (era 1149 − 709); conjunto ≈ 520–560 s (≈ 45 % do limite). Risco: a linha de base repetida em cada fatia come parte do ganho (medir); minutos de runner sobem ~10–20 % (repositório público: sem custo). Previsão para RED: o teste do YAML e os de `--shard` falham na `main` (opção e job inexistentes); o teste de partição também.

## Disciplina
Se algum critério falhar: parar e relatar opções (uma recomendada); **sem errata e sem ajuste** de limiares (720 s, N = 3), definições ou dados depois de ver o resultado. Não define `versao_aceita` de ninguém.
