# Auditoria — repositório Nexus e materiais de instrução v1/v2 (21/10/2026)

Executor: Grok Bot, clone local `/workspace/nexus/repo`, `main` = `d435083` (merge do PR #51, 19/10/2026 21:52 UTC-3). Nada foi enviado ao remoto. Logs brutos em `/workspace/nexus/v3/logs/`; lista de PRs em `/workspace/nexus/prs_files.json`.

**Limitações desta auditoria (Não verificado):** Python local 3.13.5 (CI usa 3.11); `actionlint`/`zizmor` não rodados; CI hospedado e proteção de branch não consultados; `gh` não autenticado e a API pública bateu no limite (403) após a listagem — arquivos por PR foram obtidos via `refs/pull/*/head` + `git merge-base`, não pela API; estado "draft"/data vem da listagem da API.

## Etapa 1 — Auditoria independente do repositório

### 1.1 Comandos e resultados (Fato, neste commit)
| Comando | Resultado |
|---|---|
| `pip install --require-hashes -r requirements-ci.lock` (venv) | ok; ruff 0.16.10, coverage 7.16.2, hypothesis 6.170.0, bandit 1.9.4, numpy 2.4.6, networkx |
| `ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py` | exit 0 |
| `python -m compileall -q …` | exit 0 |
| `python tools/reality_audit.py` + `git diff --exit-code` relatórios | exit 0, sem diff |
| `coverage run --branch -m unittest discover -p 'test_*.py'` | **Ran 303 tests, OK** (0 skips com o lock; networkx presente) |
| `coverage run --append --branch core/constitutional_brain.py` | exit 0; `EMBEDDED CHECKS: 30 passed, 1 skipped, 0 missing`; banner "36/55 Modules (65.5%)" |
| `coverage report` | `core/constitutional_brain.py` 68% (unittest + demo); total 68% |
| `python tools/targeted_mutation.py` | 58 declarados, 58 mortos, 0 sobreviventes |
| `bandit -r core vita tools` | 131 LOW, 0 MEDIUM/HIGH |
| `security_triage.py … SECURITY_TRIAGE.md --check` | exit 0 |
| `tools/test_summary.py` | 303 executados, 303 passaram |

### 1.2 Estrutura (Fato)
Docs na raiz: 16 `.md` (README, CONTINUATION 224 linhas, NEXUS_EVIDENCE 204, NEXUS_KNOWLEDGE 210, NEXUS_DECISIONS 261, NEXUS_CONSTITUTION, NEXUS_DEVELOPMENT_PROMPT, etc.). Código: `core/constitutional_brain.py` com **27.549 linhas**; módulos extraídos `deferred_task_queue.py`, `temporal_facts.py`, `vram_defense_guard.py`, `integrated_suite_summary.py`; `vita/`, `tools/` (9 scripts de gate), `benchmarks/`. 50 arquivos `test_*.py` na raiz. Um workflow (`nexus-contract-gate.yml`), dois jobs (gate + atestação de estado). Não existem `AGENTS.md`, `CLAUDE.md`, `scripts/`, `LICENSE`. Commits em `main` por autor: pedromiorini 65, Pedro A. M. dos Santos 48, Manus AI 26, Cue Agent 20, Claude 16, Axis 9, dependabot 2.

### 1.3 Achados do AUDIT.md (PR #60) — classificação
| Achado do AUDIT.md | Status | Base |
|---|---|---|
| `TransferMapping` definido 2× (16105, 20287) | **Confirmado** em `main` (corrigido só no PR #61, aberto) | `rg`; `dir()` do módulo mostra um único nome |
| Núcleo só passa o piso de 62% graças à demo; 44% só com unittest | **Confirmado parcialmente**: 68% combinado medido; 44% isolado **não reexecutado** | coverage |
| Docstring "62/62", "Production ready" vs saída "36/55" | **Confirmado** | linhas 8, 132; saída da demo |
| `execute_tool` não aplica `requires_permission`/`permission_level` | **Confirmado** | `rg` no corpo de `execute_tool` (4322+) |
| `CentralRouter.execution_cache` sem limite | **Confirmado** (dict sem teto, linhas 13290/13670); o cache de `RealToolUse` é limitado (`MAX_CACHE_ENTRIES=128`) — AUDIT.md já distinguia | código |
| Dependências divergentes (hypothesis dev 6.168.5 vs lock 6.170.0; networkx só no lock; zizmor 1.30.1 vs 1.31.1) | **Confirmado** | arquivos |
| Dependabot #12–#14 desalinhados do lock | **Confirmado** (#12–#14 tocam só `requirements-dev/.txt/.in`) | diff dos PRs |
| 24 PRs abertos, 21 tocam os 4 docs | **Desatualizado**: hoje **26** abertos, **22** tocam os 4 docs (inclui #60 e #61, deste próprio trabalho) | listagem |
| 9 PRs editam o núcleo | **Desatualizado**: **12** | diff |
| #46 consolida #38–#45, que seguem abertos | **Confirmado** | títulos/bases |
| #45 em conflito com `main` | **Confirmado** (`git merge-tree`) | |
| #47/#48 CONFLICTING | **Não verificado** (estão empilhados sobre branches, não sobre `main`) | |
| 131 LOW / 0 MEDIUM-HIGH | **Confirmado** | bandit |
| Sem LICENSE, `.env` fora do `.gitignore`, `pomdp-belief-benchmark-v2.json` fora do `.gitignore` | **Confirmado** | |
| Sem `busy_timeout` em memória hierárquica/episódica | **Confirmado** (único `busy_timeout` no núcleo na linha 1021) | `rg` |
| LIKE sem escape de `%`/`_` | **Confirmado por leitura** (`LIKE ?` em 230, 2264); comportamento não reexecutado | |
| Timeout não interrompe execução; `_contains_forbidden` por substring; FAISS fallback | **Não verificado** nesta rodada | |
| 157 `print` fora do `__main__` | **Não verificado** exatamente (955 `print(` no arquivo inteiro) | |

Nenhum achado do AUDIT.md foi refutado; dois números estão desatualizados.

### 1.4 Achados novos (desta auditoria)
1. **Causa estrutural dos conflitos (Fato + Inferência, alta relevância).** `tools/state_consistency_audit.py` exige o bloco `NEXUS-CURRENT-STATE` (`tests`, `bandit_low`, …) em `CONTINUATION.md`, `NEXUS_EVIDENCE.md`, `NEXUS_KNOWLEDGE.md`, `SECURITY_TRIAGE.md` (também presente em `NEXUS_DECISIONS.md`). Todo PR que adiciona teste muda `tests: 303→N` nos quatro. Cada PR está limpo contra `main` (16 de 17 testados), mas pares independentes conflitam (`git merge-tree pr/52 pr/53` → conflito em EVIDENCE, KNOWLEDGE, TRIAGE). Inferência: o problema não é disciplina dos agentes, é o desenho do gate; regras de comportamento sozinhas não o resolvem.
2. **A regra central da v2 (§13: "PRs de código não editam os 4 docs") é incompatível com o CI atual** (Inferência derivada do item 1): um PR que adicione teste e não atualize o contador falharia na atestação de estado. Não reexecutei o job hospedado para confirmar (Não verificado).
3. **Versões de ferramenta divergentes nos registros dos agentes:** `CONTINUATION.md` registra Ruff 0.17.1 e Python 3.11.16 em `/tmp/nexus-ci311`; o lock fixa Ruff 0.16.10 (Fato). Evidência de lint registrada não corresponde exatamente ao gate.
4. **`NEXUS_DEVELOPMENT_PROMPT.md`** em `main` é um prompt de tarefa de snapshot da fila já concluída (Fato) — fonte de instrução concorrente e obsoleta.
5. **`NEXUS_CONSTITUTION.md` já existe** e difere da v1 no item 6 (acrescenta "autenticação, autorização") (Fato). v1/v2 não o referenciam.
6. **Gate de mutação próprio já existe** (`tools/targeted_mutation.py`, 58/58) e PRs recentes acrescentam mutantes (#43, #57, #58 tocam o script) (Fato). A recomendação da v2 de `mutmut` duplica capacidade existente.
7. **Os PRs deste próprio esforço (#60, #61) repetem o padrão**: #61 toca os 4 docs + núcleo, que já tem 11 outros PRs abertos (Fato). Ou seja, o protocolo v2 não foi aplicado nem por nós.
8. **Todos os PRs são abertos pela conta `pedromiorini`**; a identidade do agente só aparece nos commits (Fato). "Claim" por `owner` (v2 §11.4) não tem onde ser lido de forma confiável sem convenção no título/corpo.
9. A lista de PRs via API pública sem token esgota em ~60 requisições/h (Fato: 403). Comandos baseados em `gh` não são universalmente executáveis pelos agentes.

### 1.5 Fluxo real que os agentes seguem hoje (Inferência a partir de commits/PRs/CONTINUATION)
1. Criam worktree própria a partir de `origin/main` (CONTINUATION registra base exata e worktree — boa prática já presente).
2. Escolhem uma borda pequena (limite negativo, NaN, recursão JSON…), escrevem contrato `test_*_contract.py`, reproduzem antes, corrigem.
3. Rodam ruff/compileall/suíte/integrada/reality audit/bandit (às vezes com versões fora do lock).
4. Atualizam contador e adicionam rodada longa em `CONTINUATION.md` + linhas em EVIDENCE/KNOWLEDGE/TRIAGE.
5. Abrem PR (frequentemente draft) contra `main` e **param**: não há etapa de integração; ninguém fecha/rebaseia o anterior. Resultado: 26 PRs abertos, último merge em 19/10.

**Onde falha:** (a) etapa 4 garante conflito entre quaisquer dois PRs; (b) não há varredura de PRs abertos antes de escolher tarefa (12 no núcleo); (c) consolidações (#46) e pilhas (#47–#50) multiplicam a fila em vez de reduzi-la; (d) não há dono da integração — o gargalo é merge humano, e a produção de PRs supera a vazão; (e) contagens históricas ficam no topo de documentos lidos como "estado atual".

## Etapa 2 — Auditoria de v1 e v2

Escala: Clareza / Operacionalidade (o agente executa e verifica?) / Aderência ao repo.

### 2.1 v1 (material do Pedro)
| Regra/Seção | Avaliação |
|---|---|
| §1 Identidade | Clara; aderente (README e NEXUS_CONSTITUTION já dizem o mesmo). |
| §2 Constituição 1–8 | Clara; já canônica no repo. Item 7 ("preservar histórico") é, na prática, o gatilho do crescimento dos 4 docs — tensão real com coordenação. |
| Vocabulário | Forte. **A cópia compactada (`INSTRUCOES_AGENTE_NEXUS.md`) perdeu as definições** — só lista os nomes; v3 restaura. |
| §3 Precedência | Clara, mas camadas 5–6 ("Rigor×Exploração", "Rubrica 9/10, SOTA") não têm conteúdo definido — não operacionais. |
| §4 Anti-padrões | Muito boa, verificável por revisão. Falta: demo/docstring como evidência; versões fora do lock. |
| §5 Metodologia | Boa no espírito; custo de contexto moderado; "2–5 hipóteses" é verificável no corpo do PR. (O texto original na conversa foi truncado no trecho "Princípios"; v3 usa a versão compactada.) |
| §6 Meta-Governança | Conceitualmente rica; baixa operacionalidade ("sinais de estagnação" sem métrica). Mantida, compactada. |
| §7 Protocolos | Corretos mas sem comandos; "Ciclo de sessão" manda atualizar CONTINUATION — causa colisão (1.4.1). |
| §9 Tarefa diária | "Se não houver melhoria, só atualize CONTINUATION.md" → geraria PR diário que conflita com os 22 existentes. |
| Lacunas | Nenhum comando, nenhuma regra multiagente, nenhuma regra de segurança operacional (conteúdo externo como dado), nenhum ponto de entrada (`AGENTS.md`). |

**O que a v1 faz melhor:** epistemologia (vocabulário, linguagem não-conclusiva), anti-padrões específicos do Nexus, identidade anti-hype. É o "porquê".

### 2.2 v2 (v1 + §11–18)
| Regra | Avaliação |
|---|---|
| §11.1–2 início de sessão, `scripts/init.sh` | Bom princípio (fonte confere — Anthropic, *Effective harnesses*: git log + progress file + init.sh). **`scripts/init.sh` e `backlog.json` não existem** — regra não executável hoje. |
| §11.3 `gh pr list --json files` | Correto quando `gh` autenticado; falha neste ambiente. v3 dá alternativa via git verificada. |
| §11.4 claim com 72 h | Não verificável (PRs todos sob a mesma conta; sem backlog). Removido para decisão do Pedro. |
| §12 coordenação, fila única do núcleo | **Melhor contribuição da v2**; aderente ao problema real. Mantida e reforçada com números atuais. |
| §13 handoff por fragmentos | Diagnóstico certo, prescrição **incompatível com o gate de estado** (1.4.2). Vira decisão D1 + regra transitória. Fonte "towncrier" é analogia, não evidência sobre agentes (Inferência). |
| §13 backlog JSON | Fonte confere (Anthropic: JSON "less likely to inappropriately change"), mas o arquivo não existe; adiado. |
| §14 ~300 linhas, Strangler Fig | Razoável; DORA trata de lotes pequenos em geral — o número 300 é Hipótese, não da fonte. |
| §15 commit de teste separado | Útil, mas mais rígido que o necessário; v3 exige falha anexada, commits separados recomendados. |
| §15 mutmut | Redundante com `tools/targeted_mutation.py` (1.4.6). Fonte Meta ACH confere (49% dos testes que matam mutantes não aumentam cobertura). |
| §16 segurança | Boa; aderente (`execute_tool`). |
| §17 revisão | Boa. |
| §18 AGENTS.md único | Boa; repo não tem nenhum arquivo de instrução de agente hoje. |
| Fontes | Amostragem: Anthropic *Effective harnesses* ✔ (sustenta init.sh, progress, JSON, uma feature por vez); arXiv 2501.12862 ✔ (sustenta mutação > cobertura). "Claude Academy, Parallel sessions" **não verificada** — não localizei; tratar como não verificada. |
| Custo de contexto | v2 ≈ 13 KB, metade texto da v1 repetido em forma comprimida; seções 11–18 sem comandos verificados. |

**O que a v2 faz melhor:** coordenação multiagente, segurança operacional, checklist de PR, ponto de entrada. É o "como", mas parte escrita sem checar o repo.

### 2.3 Conflitos e redundâncias detectados
- v1 §7 "atualizar CONTINUATION" × v2 §13 "não editar CONTINUATION" × CI exige o contador → resolvido na v3 §11.3.
- v1 §9 PR diário só de CONTINUATION × v2 §12 anti-colisão → v3 §19 (não abrir PR vazio).
- v2 §15 mutmut × gate existente → v3 §8.5.
- Duplicação de Constituição: v1, v2 e `NEXUS_CONSTITUTION.md` → v3 declara a mais restritiva vigente.
