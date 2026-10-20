# Nexus — Continuidade entre agentes

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 316
bandit_low: 131
-->

> Handoff operacional versionado para continuidade entre agentes Manus.

## Composição local sobre a PR #58 — 2026-10-20

- **Proveniência/preflight:** `git ls-remote origin refs/heads/main refs/pull/58/head` confirmou `main=d435083bb4c893a58128cee6cfa00f65bba7fe02` e head #58 `fef00882baf19694fe04956d82af096f266ae2d3`. A API pública `https://api.github.com/repos/pedromiorini/Nexus/pulls/58` confirmou OPEN/DRAFT, base `main` no mesmo SHA e o mesmo head. A tentativa inicial com `gh` foi bloqueada por integração não habilitada; nenhuma autenticação alternativa foi usada. O commit de overflow `af4f489889fd914e4316bd099a680dad26e47e71` também tem pai `d435083…`. A nova branch `fix/diagnostics-combined-pr58-local-20261020` foi criada na worktree `/workspace/Nexus-diagnostics-combined-pr58-20261020` diretamente de `fef00882…`; só o diff de `af4f489^..af4f489` foi composto, com reconciliação dos imports/testes/docs compartilhados e um teste novo da interação. Não houve merge/cherry-pick de outra branch, alteração da PR, fetch, push ou publicação.
- **Contrato combinado:** preservar a captura de `RecursionError` exclusivamente na fronteira CLI, mantendo sua propagação para callers diretos. Adicionar hooks `parse_float`/`parse_constant` nos loaders arquivo/stdin: rejeitar float cujo resultado seja infinito e constantes `NaN`/`Infinity`/`-Infinity`, preservando floats finitos, zero com sinal, subnormal, extremos finitos e inteiros grandes. Erros na CLI retornam 2, stdout vazio e `dashboard error:` em stderr sem traceback. `1e400` é sintaticamente JSON válido; sua rejeição decorre da faixa do consumidor float, não da gramática.
- **Invariantes revisados:** dashboard funcional idêntico ao de `af4f489` mais a captura de recursão de #58; os seis testes de #58 foram preservados textualmente; os seis testes de overflow e cinco mutantes são os da fonte. Um contrato adicional cruza tokens `1.25`, `1e400` e `-1e400` com profundidade além do decoder, ambos os canais e subprocessos reais: a falha de recursão ocorre antes de alcançar o token. Sem alteração de bridge, schema, writer, dependências, política de profundidade, workflow ou núcleo. Os ledgers abaixo conservam resultados isolados como história, não como validação combinada.
- **Red/green combinado:** `PYTHONPATH=. /tmp/nexus-ci311/bin/python -m unittest -v test_recovery_diagnostics_contract` com os 26 testes e parser original de #58: **32 falhas de subcasos**, nenhuma falha nos controles de recursão/finito. Depois de aplicar o patch: **26/26 aprovados** em Python 3.11.16 e 3.12.3 (`/opt/manus/python/bin/python3.12`). O mutante externo que remove só a captura de `RecursionError` da composição produziu **6 falhas nos subcasos do novo teste de interação e 2 erros nos contratos CLI de #58**; baseline aprovado, mutante morto sem timeout e árvore ativa intocada.
- **Comandos reproduzíveis:** `E=/workspace/nexus-diagnostics-combined-evidence-20261020`, `PY=/tmp/nexus-ci311/bin/python`, `PYTHONPATH=.`. `bash $E/run-gates.sh` executa os comandos abaixo na árvore combinada; logs/relatórios/scripts ficam em `$E`, fora do commit. Ambientes existentes reutilizados, sem instalação de dependências.
  - `COVERAGE_FILE=$E/.coverage NEXUS_AUDIT_DB=$E/gates.sqlite3 $PY -m coverage run --branch tools/test_summary.py --output $E/test-summary.json`: **316 executados, 314 aprovados, 2 skips (NetworkX opcional), 0 falhas/XPASS**. Contagem efetivamente medida na composição; não é soma dos resultados isolados.
  - `COVERAGE_FILE=$E/.coverage NEXUS_AUDIT_DB=$E/gates.sqlite3 NEXUS_INTEGRATED_SUMMARY=$E/integrated-summary.json $PY -m coverage run --append --branch core/constitutional_brain.py`: **30 aprovados, 1 skip FAISS opcional, 0 ausências/conflitos/inesperados**. `coverage report --include=<path> --fail-under=<piso>`: fila 95%/90, VRAM 94%/90, temporal 93%/90, Vita 76%/70, núcleo 68%/62; dashboard 96% informativo. Cobertura XML em `$E/coverage.xml`.
  - `$PY $E/run-mutations.py`: harness versionado executado sobre cópia temporária da composição, apenas `REPORT` redirecionado para `$E/targeted_mutation_report.json`: **63/63 aplicáveis mortas, 0 sobreviventes/não aplicadas/timeouts**. Em seguida, mutante adicional de recursão **1/1 morto**, em relatório separado `$E/recursion_mutation_report.json`. Essa contraprova não é contada dentro dos 63 nem altera o harness versionado.
  - `python -m ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py`; `$PY -m compileall -q` sobre os mesmos paths; compile dos três arquivos funcionais em 3.12; `/tmp/nexus-ci-tools/actionlint .github/workflows/*.yml` (1.7.12); `/tmp/nexus-ci-tools/zizmor --offline --no-progress .github/workflows` (1.31.1): **aprovados**.
  - `$PY -m bandit -r core vita tools -f json -o $E/bandit-report.json` seguido de triagem `tools/security_triage.py … SECURITY_TRIAGE.md --check` e check de severidade/erros: **131 LOW, 0 MEDIUM/HIGH, 0 erros de análise**. LOW não representa clearance de segurança. Auditor de asserts: 111 B101 restritos ao demo; auditor de gaps: 0 pass-only/NotImplemented. `tools/reality_audit.py` e `git diff --exit-code -- AUDIT_REALITY_REPORT.md audit_reality.json`: sem drift. Regressão explícita dos auditores/manifesto: **38/38 aprovados**. Comandos completos e saídas em `$E/run-gates.sh` e `$E/gates.log`.
- **SHA-256 da composição funcional:** dashboard `7b0270673261fd8daf7ec1b5f77a142753b5ef42c2156ca577737a7f08c1e610`; recovery contract `bccf6eefd4893575114bb4942518bfa59f3ea948991ff88c6744b7de257137a9`; mutation harness `7d10ffc5e2de75d347b1375079d4bf3749b1b7193fb3a4e28f0d0c7e007d6bd9`. Hash do commit final e revisão independente final no handoff externo, sem autorreferência.
- **Limites combinados:** sem teto novo de tamanho, profundidade, CPU/memória ou inteiros; permanecem limites nativos de recursão/int e arredondamento/underflow float. Não se garante aceitação de JSON arbitrariamente profundo nem isolamento de recursos. Suíte ampla/mutação executadas em 3.11; 3.12 recebeu apenas contratos focados e compile dos arquivos alterados. Sem campanha/CLI de benchmark nova; contratos sintéticos já existentes estão no discover e não provam capacidade cognitiva. Sem CI hospedada ou atestação de `ci.gates_status=success`: testes dos auditores passaram, mas o gate operacional hospedado não foi executado nem fabricado. Marcadores canônicos sincronizados para **316/131**; `HEAD`/`CURRENT_RUN` só são resolvidos por uma execução associada. `/workspace/Nexus` suja, `main` local e ambas as worktrees/branches-fonte devem permanecer exatamente preservadas; nenhuma publicação está autorizada.

## Histórico da branch-fonte isolada — overflow numérico no dashboard — 2026-10-20

> Evidência histórica exclusiva de `af4f489889fd914e4316bd099a680dad26e47e71`, sobre `d435083bb4c893a58128cee6cfa00f65bba7fe02`; não representa a composição sobre a #58 nem seus resultados atuais. Referências a “esta branch” abaixo descrevem somente a branch-fonte original.

- **Base/branch:** `d435083bb4c893a58128cee6cfa00f65bba7fe02`, confirmada no remoto por `git -C /workspace/Nexus ls-remote origin refs/heads/main`. `git worktree add -b fix/diagnostics-float-overflow-20261020 /workspace/Nexus-diagnostics-float-overflow-20261020 d435083bb4c893a58128cee6cfa00f65bba7fe02` criou a worktree limpa diretamente na base; o checkout sujo `/workspace/Nexus` foi preservado. Não houve fetch, push, PR, merge, mudança de main, dependências ou edição em outras worktrees.
- **Paths das PRs confirmados via `gh pr view` (#52–#58):** #52 altera `core/integrated_suite_summary.py`/teste; #53 `tools/test_summary.py`/teste; #54 `tools/security_triage.py`/teste; #55 `tools/state_manifest.py`/teste; #56 `tools/state_consistency_audit.py`/teste; #57 `tools/embedded_assert_audit.py`/teste e `tools/targeted_mutation.py`; #58 **altera dashboard e teste de recovery**, contrariando o contexto inicial, posteriormente corrigido pelo usuário. Todas também tocam documentação. Head #58 `fef00882baf19694fe04956d82af096f266ae2d3`, base `main`; nenhum conteúdo ou ref da #58 foi alterado. A revisão independente restrita aos dois paths não encontrou bloqueios: o tratamento `RecursionError` da #58 é ortogonal e deverá ser preservado numa integração futura autorizada, junto com ambos os conjuntos de testes. Há overlap adicional do harness de mutação com #57 e dos ledgers com todas essas PRs; nenhuma integração foi feita.
- **Contrato:** tokens float que decodificam como infinito são rejeitados nos dois loaders por `parse_float`; constantes não JSON são rejeitadas por `parse_constant`. O erro existente da CLI converte `ValueError` em exit 2, sem stdout/traceback. Finito máximo positivo/negativo, subnormal, zero com sinal, valores comuns e inteiros grandes preservam valor/tipo. Sem limite novo de profundidade, tamanho, precisão ou inteiros; sem mudança de schema, bridge ou writer. Não se alega que `1e400` viola a gramática JSON: o problema é a faixa do tipo float do consumidor.
- **Red/green:** `PYTHONPATH=. <PY> -m unittest -v test_recovery_diagnostics_contract` antes da implementação: 19 testes, 32 falhas de subcasos em cada runtime; depois: 19/19 aprovados com `<PY>=/tmp/nexus-ci311/bin/python` (3.11.16) e `/opt/manus/python/bin/python3.12` (3.12.3). Seis testes novos exercitam arquivo/stdin, overflow positivo/negativo aninhado, constantes, subprocessos CLI e controles finitos, incluindo `1.7976931348623157e308`. Evidência detalhada no ledger.
- **Ambiente dos gates amplos:** `PY=/tmp/nexus-ci311/bin/python`, `PYTHONPATH=.`, `E=/tmp/nexus-float-overflow-evidence`, `NEXUS_AUDIT_DB=$E/audit.sqlite3`, `NEXUS_INTEGRATED_SUMMARY=$E/integrated-summary.json`, `COVERAGE_FILE=$E/.coverage`. Ambiente existente reutilizado, sem instalação/alteração de dependências. Logs e JSONs locais ficaram fora do stage.
  - `python -m ruff check core vita tools test_*.py`, `$PY -m compileall -q core vita tools test_*.py` e `python3.12 -m compileall -q tools/diagnostics_dashboard.py test_recovery_diagnostics_contract.py`: aprovados. O harness adicionado também passou Ruff/compileall.
  - `/tmp/nexus-ci-tools/actionlint .github/workflows/*.yml` e `/tmp/nexus-ci-tools/zizmor --offline --no-progress .github/workflows`: aprovados. O workflow não foi alterado.
  - `$PY -m coverage run --branch -m unittest discover -p 'test_*.py'`; `$PY -m coverage run --append --branch core/constitutional_brain.py`: aprovados. Suíte integrada: 30 aprovações, 1 skip FAISS opcional, 0 ausências/conflitos/inesperados. `coverage report --include=<path> --fail-under=<piso>` passou para fila 95%/90, VRAM 94%/90, fatos temporais 93%/90, Vita 76%/70 e núcleo 68%/62.
  - `$PY tools/test_summary.py --output "$E/test-summary.json"`: **309 executados, 307 aprovados, 2 skips NetworkX opcional, 0 falhas/XPASS**. A suíte completa foi executada em 3.11; 3.12 recebeu os contratos focados, não a suíte ampla.
  - `$PY tools/targeted_mutation.py`: **63/63 aplicáveis mortas**, 0 sobreviventes/não aplicadas; cinco novos mutantes removem a guarda de finidade e os hooks float/constante de cada loader. Todos executados em cópia temporária pelo harness existente, não na árvore ativa. O relatório local `targeted_mutation_report.json` não é versionado.
  - `$PY -m bandit -r core vita tools -f json -o "$E/bandit-report.json"` e `$PY tools/security_triage.py "$E/bandit-report.json" SECURITY_TRIAGE.md --check`: triagem aprovada, **131 LOW, 0 MEDIUM/HIGH, 0 erros de análise**. LOW permanece inventário conservador, não clearance.
  - `$PY tools/embedded_assert_audit.py "$E/bandit-report.json" core/constitutional_brain.py`: 111 B101 no demo, nenhum assert fora do bloco; `$PY tools/implementation_gap_audit.py core/constitutional_brain.py --output "$E/implementation-gaps.json"`: 0 pass-only/NotImplemented; `$PY tools/reality_audit.py` e `git diff --exit-code -- AUDIT_REALITY_REPORT.md audit_reality.json`: aprovados, relatórios versionados inalterados.
- **SHA-256 dos arquivos funcionais verificados:** dashboard `47bd3960441603b472d9cbea8ffc04afa419daf887cbe7b34701bd36496abb69`; contrato recovery `21a4819eb110b138d7b39eb0322dd93633659feeb2058c87132120af3b3ee344`; mutation harness `7d10ffc5e2de75d347b1375079d4bf3749b1b7193fb3a4e28f0d0c7e007d6bd9`. O hash final do commit local fica no handoff externo, evitando autorreferência neste documento.
- **Limites:** não há CI hospedada nem atestação operacional de `ci.gates_status=success`; os contratos dos auditores passaram na suíte. Os quatro marcadores canônicos foram sincronizados para 309 testes/131 LOW sem fabricar run/manifesto. Não foram executadas campanhas/CLIs de benchmark; o discover conserva os testes unitários de contrato já existentes, inclusive os relativos a benchmarks. Limites nativos do decoder/int/recursão e arredondamento/underflow de float permanecem. Esta branch não trata `RecursionError` da CLI; isso pertence à #58 preservada. Nenhuma alegação de segurança global ou capacidade cognitiva decorre destes gates.
- **Conclusão autorizada:** revisar diff/status, stage de paths explícitos e commit **somente local**; aguardar autorização distinta para qualquer publicação ou integração. Não alterar as PRs existentes.

## Histórico da PR #58 isolada — erro de recursão no dashboard — 2026-10-20

- **Base exata:** `d435083bb4c893a58128cee6cfa00f65bba7fe02`, confirmada por `git ls-remote origin refs/heads/main` e `origin/main`. Branch `fix/diagnostics-recursion-error-20261020`, worktree criada limpa em `/workspace/Nexus-diagnostics-recursion-20261020`. `main` local permaneceu em `7c8a187c1465c6db39777f5e9dc16efb112e1121`; não foi atualizado. Checkout sujo `/workspace/Nexus` e outras worktrees preservados. A branch `fix/diagnostics-strict-json-20261020` permanece em `08eabcb403a8de4b011abdc125ea8ae44326bf27`, somente contexto, sem cherry-pick.
- **Contrato/decisão:** adicionar somente `RecursionError` ao tratamento de erro de `main`. Arquivo/stdin que excedam a capacidade do decoder retornam 2, stdout vazio e `dashboard error:` em stderr, sem traceback. `load_snapshot` não muda e continua propagando a exceção para callers diretos. Nenhum limite de profundidade, normalização JSON, dependência ou refatoração foi introduzido.
- **Red/green:** antes do patch, 19 contratos executados, exatamente dois erros por `RecursionError` nos caminhos CLI de arquivo/stdin; controles rasos e callers diretos passaram. Após o patch, 19/19 em Python 3.11.16 e 3.12.3. Fixture final usa `max(10000, sys.getrecursionlimit() + 100)` apenas no teste: neste Python 3.12, o decoder aceita 2.000 arrays, diferentemente do 3.11. O mutante que remove somente a nova captura reproduz os mesmos dois erros com a fixture final. Subprocessos reais em 3.11 com exatamente 2.000 arrays confirmaram os dois caminhos com exit 2/stdout vazio/erro legível/sem traceback.
- **Proveniência funcional (SHA-256):** `tools/diagnostics_dashboard.py`: `2c54cc5878c119860b000aae6a17a476aa0676597b1b2462ac80aaf5ccc01129`; `test_recovery_diagnostics_contract.py`: `f41c499066081e95bc4d3d45257eec5c91e47cba1d149d411324c506fc377ed4`. O hash do commit final fica no handoff externo para evitar autorreferência. Esta seção descreve o candidato local, não um run hospedado.
- **Ambiente/comandos:** `PY=/tmp/nexus-ci311/bin/python`, `PYTHONPATH=.`, `E=/workspace/nexus-diagnostics-recursion-evidence-20261020`, `NEXUS_AUDIT_DB=$E/<gate>.sqlite3`; ambientes preexistentes, sem instalação de dependências. Logs e scripts de reprodução ficam fora do commit em `$E`.
  - `$PY -m unittest -v test_recovery_diagnostics_contract.py` e `python -m unittest -v test_recovery_diagnostics_contract.py`: 19/19 cada; seis novos testes protegem os dois caminhos CLI, propagação direta e snapshots rasos com Unicode, valores finitos, booleanos e null.
  - `COVERAGE_FILE=$E/.coverage $PY -m coverage run --branch tools/test_summary.py --output $E/test-summary.json`: 309 executados, 307 aprovados, 2 skips por NetworkX opcional ausente, 0 falhas/XPASS. A suíte completa inclui testes existentes dos protocolos sintéticos; nenhuma CLI de benchmark foi executada nem benchmark alterado.
  - `COVERAGE_FILE=$E/.coverage NEXUS_INTEGRATED_SUMMARY=$E/integrated-summary.json $PY -m coverage run --append --branch core/constitutional_brain.py`: 30 aprovados, 1 skip FAISS opcional, nenhuma ausência/conflito/ID inesperado. `$PY -m coverage report --include=<arquivo> --fail-under=<piso>` com o mesmo `COVERAGE_FILE`: fila 95% (90), VRAM 94% (90), temporal 93% (90), Vita 76% (70), núcleo 68% (62); dashboard 95% informativo.
  - `$PY tools/targeted_mutation.py`: 58/58 mortas, 0 sobreviventes/não aplicadas. `$PY $E/dashboard_mutation.py`: mutante específico adicional morto, baseline aprovado e cópia temporária isolada; não altera o gate versionado.
  - `bash $E/run-gates.sh`: Ruff, compileall, actionlint 1.7.12, zizmor 1.31.1 offline, auditoria de realidade sem drift dos relatórios, Bandit/triagem, auditorias AST/asserts/gaps e 32 contratos dos auditores/manifesto aprovados. Bandit: 131 LOW, 0 MEDIUM/HIGH, nenhum erro de análise; não é clearance de segurança. Comandos completos estão no script e `gates.log`; nenhum relatório gerado/banco integra o patch.
- **Documentação:** apenas ledger/handoff e os quatro marcadores obrigatórios de contagem atual passam a 309; Constituição/arquitetura/decisões permanecem inalteradas. Marcadores `HEAD`/`CURRENT_RUN` são resolvidos pelo gate associado, não alegam CI remoto já executado. A atestação de job hospedado não foi executada nem fabricada.
- **Limites:** captura não limita CPU/memória nem garante aceitação de JSON arbitrariamente profundo; profundidade suportada continua dependente do decoder/runtime. Suíte completa e mutation em 3.11; somente contratos focados adicionais em 3.12. Sem revisão independente, pois diff funcional é uma única extensão da captura e não revelou risco adicional concreto. Sem push, PR, merge, alteração de main, outras worktrees ou dependências.

## Rodada local — gatilho de PR para qualquer destino — 2026-10-19

- **Base exata:** `812ebd02b0c82ecc3169d620e36d25592b1d6727`, confirmada em `origin/main`, sem fetch nem alteração de `main`. Branch local `fix/ci-pr-target-any-branch-20261019`, worktree `/workspace/Nexus-ci-trigger-any-branch-20261019`, criada limpa. O checkout sujo `/workspace/Nexus` e as demais worktrees não foram alterados.
- **Contrato:** `pull_request` não filtra a branch de destino, permitindo PRs encadeadas; `push` continua restrito a `main`. Foi removida somente a linha `branches: [main]` sob `pull_request`. Não foram alterados tipos de eventos padrão, permissões `contents: read`, concurrency/cancelamento, jobs, steps, pins SHA/checksums ou `persist-credentials: false`; não foi usado `pull_request_target`.
- **Reprodução:** antes da mudança, a leitura YAML encontrou `pull_request.branches == [main]`, violando o contrato. Depois, o teste local abaixo confirmou a remoção exata de uma linha e a preservação byte a byte de todo o restante do workflow:

  ```sh
  python - <<'PY'
  from pathlib import Path
  import subprocess
  p = '.github/workflows/nexus-contract-gate.yml'
  base = subprocess.check_output(['git', 'show', '812ebd02b0c82ecc3169d620e36d25592b1d6727:' + p], text=True)
  assert Path(p).read_text() == base.replace('  pull_request:\n    branches: [main]\n', '  pull_request:\n', 1)
  PY
  ```

- **Comandos e resultados locais:** executados na worktree acima; `PY=/tmp/nexus-ci311/bin/python` (Python 3.11.16), `PYTHONPATH=.`, `NEXUS_AUDIT_DB=/tmp/nexus-ci-trigger-any-branch-20261019.sqlite3` para a suíte e integração.
  - `/tmp/nexus-ci-tools/actionlint .github/workflows/*.yml` (1.7.12) e `/tmp/nexus-ci-tools/zizmor --offline --no-progress .github/workflows` (1.31.1): aprovados, nenhum finding. Leitura com `yaml.BaseLoader` também confirmou apenas `push.branches: [main]` e `pull_request` sem configuração/filtros.
  - `$PY tools/reality_audit.py` e `git diff --exit-code -- AUDIT_REALITY_REPORT.md audit_reality.json`: aprovados, relatórios versionados inalterados.
  - `python -m ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py` (Ruff 0.17.1) e `$PY -m compileall -q core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py`: aprovados.
  - `$PY tools/test_summary.py --output test-summary.json`: 303 executados, 301 aprovados, 2 skips por NetworkX opcional ausente, 0 falhas/XPASS.
  - `NEXUS_INTEGRATED_SUMMARY=integrated-summary.json $PY core/constitutional_brain.py`: 30 verificações aprovadas, 1 skip FAISS opcional, nenhuma ausente/conflitante/inesperada.
  - `$PY -m unittest -v test_reality_audit.py test_state_consistency_audit.py test_state_manifest.py`: 32 contratos aprovados. A atestação operacional via `tools/state_consistency_audit.py --manifest ...` não foi declarada: não há manifesto de um job de gates concluído nesta rodada; não se fabricou `gates_status=success` nem um run hospedado. Os marcadores estruturados atuais permanecem inalterados.
  - `$PY -m bandit -r core vita tools -f json -o bandit-report.json` e `$PY tools/security_triage.py bandit-report.json SECURITY_TRIAGE.md --check`: triagem aprovada, 131 LOW, nenhum MEDIUM/HIGH ou erro de análise. LOW não é clearance de segurança.
  - `git diff --check`: aprovado. Apenas workflow e este handoff fazem parte do patch; relatórios locais ignorados não entram no commit.
- **Limites:** validação local em ambiente já disponível, sem reinstalação do lock; não equivale ao ambiente hospedado. Cobertura, mutation e benchmarks não foram reexecutados para esta exclusão de filtro; código Python e steps não mudaram. O filtro removido permite o disparo, mas não garante execução diante de conflitos, políticas, aprovação de forks ou outros bloqueios do provedor. PRs draft não ganham novos tipos de evento com esta mudança. Não há execução retroativa garantida para PRs abertas, nem alteração das PRs #47–#50. Um run remoto só poderá ocorrer após push/PR autorizado e evento elegível; nenhum push, PR, merge ou auto-merge foi feito.
- **Revisão:** congelar este diff de dois arquivos para uma única revisão independente de CI/security; reabrir somente por achado concreto. Se gates e revisão passarem, produzir somente commit local. O hash final do commit e o resultado da revisão ficam no handoff externo, evitando autorreferência no próprio commit.

## Estado herdado (histórico; não é a validação desta rodada)

- **Base auditada nesta rodada:** `main` estava no merge documental da PR #35, commit `047c4ba6bb4fb22e3dbe3251c8aee9bf986e0583`; a base funcional permanece no merge da PR #34 (`51373dc77179e4edd094df14290f91aa9c8d8060`), que contém as PRs #32, #33 e #34.
- **Política de integração:** mudanças futuras devem permanecer em branches isoladas e ser mescladas somente mediante autorização específica.
- **D-026 implementada:** `RealEpisodicMemory.search_episodes` rejeita `bool`/tipos não inteiros e limites negativos antes de lock/SQL; `limit=0` retorna vazio sem consulta e valores positivos preservam o fluxo anterior, sem novo teto.
- **Hardening herdado:** D-024 (fallback de cache hit) e D-025 (snapshot JSON anti-TOCTOU) estão em `main` após os merges #32 e #33.
- **Validação da PR #34:** unittest 275/275; 11/11 contratos episódicos focados; integração 30 aprovados + 1 skip FAISS opcional; cobertura 95/94/76/68%; mutation 58/58 mortas e 0 não aplicadas/sobreviventes; Bandit 131 LOW, sem MEDIUM/HIGH. Checks oficiais da PR: 2/2 aprovados. POMDP v2: 128/128 dentro de `1e-12`, erro máximo `2.7755575615628914e-16`, 4/4 controles inválidos rejeitados.
- **Limites:** nenhum teto novo para limites positivos altos; o snapshot de parâmetros permanece por valor, sem preservar identidade/aliasing ou sincronizar escrita concorrente. Fallback de cache não limita custo do executor ou memória global; timeout segue diagnóstico, não cancelamento.
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
- `tools/targeted_mutation.py`: mutation testing direcionado para fila, Vita e schema.
- `core/deferred_task_queue.py`: transições de estado protegidas por `RLock`; conclusão/descarte removem tarefas pendentes do heap, retry é único e limitado, callbacks defeituosos são reportados sem interromper o lote; snapshots versionados continuam atômicos.
- `RealImmutableLog`: triggers bloqueiam UPDATE/DELETE, o SHA-chain é verificado na inicialização e detecta adulteração/remoção; writes de conexões concorrentes são serializados. Truncamento final ou rehash por quem controla o arquivo ainda exige âncora externa.
- `RealJailbreakDetector`: normalização Unicode, sinais por frases e fronteiras de palavra reduzem falsos positivos e detectam ofuscações testadas em inglês/português; continua heurístico, não é uma fronteira de segurança.
- `test_deferred_task_snapshot.py`: contrato de round-trip, ordem, atomicidade, limites, tipos inválidos e ausência de execução durante restauração.
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

## Próxima ação

Após revisão e commit local desta correção, aguardar autorização específica antes de qualquer publicação ou integração. Não alterar a cadeia de PRs existente. As referências anteriores à PR #36 são históricas, não uma instrução operacional atual. A auditoria de outras fronteiras fica fora desta rodada; `ToolDefinition.requires_permission`/`permission_level` e validação recursiva dos membros de contêiner não foram examinados aqui.
