# Guarda de artefatos derivados desatualizados — pré-registro e resultados

Origem: proposta de retro `2026-10-29-drift-artefatos` (lição L-010). Pedro aprovou (t137u, item b) trazer `artefatos_drift.sh` e a lição L-010 para o repositório. **Draft: sem merge sem CI verde e um go separado do Pedro.**

## Pré-registro (commit anterior ao contrato e ao script)

**O que foi visto antes de escrever isto (declarado):** a proposta completa (script, diffs de `casos.tsv` e `run_replay.sh`, critérios e resultado dela). Nela, no clone da `main` `23fde14`, com bandit 1.9.4 local: `D-origem` (`afd12a5`) → 7, `D-anterior` (`23fde14`) → 0, `D-incompat` → 3, e o replay completo 20/20. **Não** foi visto nenhum resultado da matriz hermética (AD4–AD6) nem do tipo `drift` do replay em tabela própria.

### Guarda (comportamento exato)
`tools/agente/artefatos_drift.sh <repo_dir>` (var. `PY`, padrão `python3`). Trabalha numa **cópia** do repositório: roda `bandit -r core vita tools`, `tools/security_triage.py <bandit.json> SECURITY_TRIAGE.md --check` e `tools/reality_audit.py`, e compara os relatórios de realidade com o commitado (`git diff`).
- **0** nenhum artefato diverge; **7** algum diverge (`DRIFT: SECURITY_TRIAGE.md` e/ou `DRIFT: relatorios de realidade` em stderr); **3** entrada inválida (diretório ausente, não é repositório git, faltam `tools/security_triage.py` ou `tools/reality_audit.py`, a cópia falha, o bandit não gera relatório, o `reality_audit` falha). 3 é distinto de 0 e de 7.
- Decisão só por código de saída (`set -uo pipefail`), sem pipe no caminho de decisão. Não escreve no repositório original.

### Escopo e decisões (regras escritas antes de ver resultado)
1. O job `Agent guard replay` do CI **não** instala bandit nem dependências do `reality_audit` e tem `timeout-minutes: 5`. Por isso os casos `drift` ficam em **`tools/agente/replay/casos_drift.tsv`**, de uso local (antes do push); `casos.tsv` (a tabela do CI) **não muda** e o workflow **não é editado** (L-004: não misturar workflow com docs). Isto é um desvio declarado do critério 1 da proposta (que punha os casos em `casos.tsv`).
2. O tipo `drift` do replay (`run_replay.sh`) é acrescentado; caminho absoluto roda o guarda nele, SHA abre um worktree no SHA. SHA ausente é entrada inválida (saída 4 do replay, regra existente).
3. `D-origem` usa `afd12a5`, que só existe no histórico do #132. Para reproduzir: `git fetch origin pull/132/head` (ou ter o branch do #132). Fora disso o replay local sai 4 (entrada inválida), não 0.
4. **Fora do escopo declarado** da guarda: o marcador `bandit_low` nos documentos de estado e na fixture de `test_state_stray_marker_contract.py`; os casos de digest/hash de composição (runs `39862499062` e `40707824816`). O #132 mostrou esta lacuna: depois de regenerar a triagem, o gate falhou ainda duas vezes (runs `40999295183` e `41000593789`) por esse marcador.

### Critérios (todos precisam passar; nenhum será ajustado depois de rodar)
- **AD1 origem/anterior/incompatível reais** (`casos_drift.tsv`, local, bandit 1.9.4): `D-origem` → 7, `D-anterior` → 0, `D-incompat` → 3. Se `D-anterior` der 7 por diferença de versão do bandit, o critério falha e é relatado (limite reconhecido da proposta).
- **AD2 sem regressão:** o replay padrão (`casos.tsv`, 17 casos) sai 0 e inalterado; a suíte completa passa em Python 3.11 e 3.12.
- **AD3 modo Git:** `artefatos_drift.sh` é `100755` e roda por execução direta.
- **AD4 matriz hermética** (`test_artefatos_drift_contract.py`; repo falso com `security_triage.py` e `reality_audit.py` de mentira e um `PY` que simula só `-m bandit`): tudo limpo → 0; triagem diverge → 7 com `DRIFT: SECURITY_TRIAGE.md`; relatório de realidade muda → 7 com `DRIFT: relatorios de realidade`; ambos → 7 com as duas mensagens; `reality_audit` falha → 3; bandit sem relatório → 3.
- **AD5 entradas inválidas** (3, nunca 0 nem 7): argumento ausente/vazio; diretório inexistente; arquivo em vez de diretório; diretório que não é repo git; repo sem `tools/security_triage.py`; repo sem `tools/reality_audit.py`.
- **AD6 não escreve no original:** depois de qualquer execução da matriz (incluindo as que divergem), o repo falso fica idêntico (hash de todos os arquivos e `git status` limpo).
- **AD7 tipo `drift` do replay:** tabela temporária com caminho absoluto do repo falso (esperado 7 e 0), `/tmp/…` inexistente (esperado 3) e SHA inexistente (replay sai 4, entrada inválida); tipo desconhecido segue falhando fechado.
- **AD8 lição L-010:** entrada no formato de `LICOES_APRENDIDAS.md` com o bloco de reincidência; `tools/correction_claim_audit.py` e `validar_registro` passam; status `pendente` (ativação é decisão do Pedro).
- **AD9 gates do repositório:** `ruff`, `compileall`, `reality_audit` + `git diff --exit-code`, Bandit/triage/asserts/gaps, `validar_registro`, `cognitivo_mutation` e `targeted_mutation` (não editado) sem mudança de resultado, `claims: []`.

### Previsão do RED
O contrato falha inteiro (o script e o tipo `drift` não existem): AD3–AD7 em erro/falha. AD1 só é medido no GREEN. Qualquer previsão errada será registrada, sem ajuste dos critérios.
