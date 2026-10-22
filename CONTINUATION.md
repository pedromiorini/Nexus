# Nexus — Continuidade entre agentes

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: CURRENT_TESTS
bandit_low: 135
-->

## Revalidação local da composição sobre a3f929f — 2026-10-22

**Identidade:** clone independente `/workspace/Nexus-pr46-revalidation-a3f929f-20261022`,
branch `local/pr46-revalidation-a3f929f-20261022`, HEAD âncora da #46
`f3044badb1f16eb4936dde6db17087f6bc96d049`, main/MERGE_HEAD
`a3f929f046d73461eb1e6b5169dce428c523f6f6`. Merge local pendente,
sem commit. HEAD não identifica os bytes compostos: tree, patches completos contra
ambos os pais, modos e hashes constam em
`/workspace/pr46-revalidation-a3f929f-evidence-20261022/provenance.json` e
`candidate-files.json`. O manifesto local distingue o HEAD âncora da candidate_tree.

**Contrato e resolução:** o merge fresco teve dez conflitos, registrados em
`merge-tree.txt` e `conflicted-paths.txt`. Comparados os nove commits novos de main:
sete paths alterados, nenhum delta novo no núcleo, dependências ou catálogo de
mutação contra `a663a55`. Resoluções anteriores foram reconciliadas com essa
comparação explícita, preservando D-029 temporal e D-034/D-035 da main, união de
mutações/aliases e perfil compartilhado alinhado ao lock. Core, preflight atômico
(incluindo `self`, `u_of_edge`, `v_of_edge`), guardas fail-closed, interseção real
de globs/orçamento e regressões permanecem byte a byte os da tree `697923975efbee368dece179b9139e840919d785`.
A v3.4.1, registro e replay #74 permanecem os da main exata; L-001–L-007 usam o
texto atualizado de main, L-008–L-010 foram preservadas uma única vez como histórico
local, sem promoção. Não houve conflito novo de código exigindo adoção de semântica
sem evidência. O caso `pre deadbeef` continua sendo rejeição exercitada pelo guard;
SHA ausente/falha de worktree em `res`/`merge` é entrada inválida rc 4.

**Evidência nova:** comandos e resultados medidos nesta rodada estão em
`/workspace/pr46-revalidation-a3f929f-evidence-20261022/run-gates.sh`,
`gates.tsv`, arquivos `.command`/`.log`, `STATE_MANIFEST.local.json` e
`state-consistency.json`. As sentinelas são resolvidas desse manifesto local;
`ci_run=not_ci`, nunca CI hospedado. A situação remota e os resultados de CI #74
foram fornecidos pelo owner como contexto; não se atribuem a esta nova tree.
A única revisão independente final é solicitada sobre a tree congelada e seus
dois diffs exatos; seu resultado externo não é antecipado nesta documentação.

**Limites e autorização:** composição/revalidação local somente. Sem commit, push,
PR ou merge remoto; nenhum ref do repositório compartilhado é movido. Os dois
checkouts protegidos, inclusive índice/merge e arquivos não rastreados, são
atestados por inventários completos antes/depois. A main ausente localmente foi
obtida por fetch somente-leitura no clone independente, sem remote persistente.
Gates são contratos de software e sondas sintéticas, não capacidade geral ou
clearance de segurança. O ambiente opcional e os skips são registrados no manifesto;
não há validação GPU/FAISS real, matriz Python ou cancelamento/transação geral.
`preflight_rotina.sh` continua sem examinar untracked; orçamento de globs é limite
operacional aproximado, não garantia formal. O alias histórico do catálogo de
mutação não representa defeito independente adicional.

**Delimitação histórica:** todos os checkpoints e pareceres abaixo, inclusive
548/113 da tree 6979 e BLOCKs anteriores, são preservados como história de suas
próprias árvores. Não validam a composição sobre a3f929f e não se somam às medições
atuais. Não abortar nem concluir automaticamente qualquer merge preservado.


## Correção local da colisão self — 2026-10-21

**Identidade/escopo:** somente `/workspace/Nexus-pr46-compose-main-20261021`,
branch `local/pr46-compose-main-a663a55-20261021`, HEAD âncora
`f3044badb1f16eb4936dde6db17087f6bc96d049`, MERGE_HEAD
`a663a55a560e00ae818e9c8f60aacd4cf9bd93f7`. Partida: tree
`972747bebe1b9a16d79b158ab11caf9365cd618a`, BLOQUEADA pela revisão integral
anterior apesar dos 545 testes/112 mutantes. HEAD não identifica o composto;
árvore final, patches binários contra partida e ambos os pais, modos e hashes
ficam em `/workspace/pr46-composition-evidence-20261021/self-collision-fix/`.
Merge permanece pendente, staged e sem commit. Nenhum push, PR, fetch real,
merge remoto, outra worktree ou `/workspace/Nexus` foi operado nesta rodada.

**Contrato e discriminação:** `properties={'self': 'opaque'}` deve provocar
TypeError de preflight no backend NetworkX antes de qualquer mutação, não o
TypeError tardio de `DiGraph.add_edge`. Hipóteses discriminadas por snapshots:
mera rejeição do backend não preservava o estado; tratar `self` como valor opaco
NetworkX também não era válido. A causa reproduzida foi omitir o receptor bound
da lista de parâmetros reservados. Acrescentado somente `self` à guarda
existente; preservados o snapshot/validação de chaves, demais parâmetros,
valores opacos válidos e contrato legado do fallback. Três métodos novos cobrem
vazio/populado × com/sem store, predicado funcional com endpoints existentes
(inclusive ausência de chamadas a `assert_fact`/`add_edge`) e fallback positivo.
Os arquivos de shell, interseção glob com orçamento fail-closed e seus contratos
permanecem idênticos à partida. O catálogo mantém os 112 nomes anteriores,
ajusta a âncora existente e acrescenta remoção específica da guarda de `self`.

**RED/GREEN e gates:** `red.command`/`red.log`: 7 métodos executados, 5 falhas
de subcasos (quatro combinações de estado/store e um fechamento funcional),
antes da mudança do núcleo. `green.log`: 53/53 contratos do grafo/temporal.
`run-gates.sh`, comandos `*.command`, saídas `*.log` e `gates.tsv`: ambiente
externo Python 3.11.16 com pins do lock, grupo focado **117/117**, suíte sob
cobertura e resumo **548/548**, 0 falhas/skips/XPASS. Integração separada:
**30 aprovações + 1 skip FAISS**, sem IDs ausentes/inesperados/conflitantes/duplicados.
Cobertura CLI: fila 91%, VRAM 94%, temporal 94%, Vita 76%, núcleo 70%, acima
dos pisos 90/90/90/70/62. Mutação: **113 declaradas/aplicadas/mortas**, zero
sobreviventes/timeouts/não aplicadas; aliases históricos não representam defeitos
independentes. Bandit: **135 LOW, 0 MEDIUM/HIGH, 0 erros**, exit bruto 1
separado da política aprovada. Ruff, compileall, actionlint, zizmor offline,
requirements dry-run, realidade, registro, replay 11/11 e controle adulterado,
triagem, B101 e gaps passaram. CLIs sintéticas de benchmarks v1/v2 com seed
20261016 executadas como gates de software; não são validação externa/cognitiva.

**Documentação e revisão:** 517/96 foi identificado explicitamente como
checkpoint histórico em NEXUS_EVIDENCE e NEXUS_KNOWLEDGE, preservando a origem;
545/112 também permanece histórico e bloqueado. Pareceres independentes e
conferência da tree final ficam em `review-integral/`, com o estado exato no
handoff externo. A revisão BLOCK de `972747` não foi convertida em aprovação;
aprovação integral não decorre dos gates nem de parecer limitado a um patch.
`final-checks.sh` sela paths explícitos, confere refs/identidade/correções
preservadas e gera atestação local com `run_id=not_ci`, sem fabricar CI hospedado.

**Limites:** preflight de entradas conhecidas não é transação contra falhas
arbitrárias de backend, objetos customizados ou esgotamento de recursos.
O descompasso writer/reader de suíte vazia permanece fail-closed, sem mudança:
writer emite falha válida de zero testes, leitor a classifica `invalid_summary`;
não há evidência de necessidade de ampliar este patch para isso. Orçamento de
globs permanece aproximado, não prazo real/validação genérica YAML/schema.
Whitespace EOF herdado das instruções main permanece ressalva separada, sem
limpeza lateral ou alteração de seu hash. Sem matriz Python, GPU/FAISS real,
clearance de segurança, CI hospedado ou evidência de capacidade geral.

## Correção local da interseção de claims — 2026-10-21

**Identidade:** candidata exclusiva `/workspace/Nexus-pr46-compose-main-20261021`,
branch `local/pr46-compose-main-a663a55-20261021`, HEAD âncora
`f3044badb1f16eb4936dde6db17087f6bc96d049`, MERGE_HEAD
`a663a55a560e00ae818e9c8f60aacd4cf9bd93f7`. Partida: tree bloqueada
`d60f157162d6d79a9543b73d1047a05b4f44bef2`. HEAD não identifica o composto;
nova tree, patches, hashes, modos, comandos e logs ficam externamente em
`/workspace/pr46-composition-evidence-20261021/glob-overlap-fix/`.
Nenhum commit, push, PR, fetch real, alteração remota ou operação de outra worktree.

**Contrato/correção:** bug herdado de main: `src/a?.py` e `src/?b.py` têm a
testemunha `src/ab.py`, mas o helper, API e CLI aprovavam os claims ativos.
Comparação textual substituída por interseção de autômatos fnmatchcase, com
alfabeto simbólico Unicode; não enumera arquivos existentes. Preserva `/**`
incluindo o pai, expiração e controles disjuntos. As três correções anteriores
e seus contratos permanecem byte a byte idênticos à tree de partida. Claim
temporário Axis encerrado, sem alteração final de CLAIMS.yaml ou das instruções.

**RED/GREEN:** regressões de helper, API, CLI e oracle limitado; 26 métodos finais
da primeira etapa falharam na implementação antiga em cópia temporária
(1.250 falhas de subcasos). A revisão independente da primeira implementação
não encontrou divergência semântica nos casos exercitados, mas apontou risco P2
de custo quadrático sem limite. Acrescentado orçamento de 100.000 unidades por
helper e 1.000.000 compartilhadas por validação, incluindo scanner de classes;
exaustão é erro explícito, nunca ausência de conflito. Cinco métodos de recursos
também têm RED/GREEN. A rodada intermediária interrompida com exit 130, incluindo
mutation baseline RED do scanner durante a inclusão do teste, está preservada
em `pre-scanner-gates/`; não serve de atestação da versão final.

**Evidência atual:** Python 3.11.16 e ambiente externo com pins do lock;
`run-gates.sh`, `*.command`, `*.log` e `gates.tsv`: contratos do registro **31/31**,
grupo focado **114/114**, suíte **545/545**, 0 falhas/skips/XPASS. Integração
separada: **30 aprovados + 1 skip FAISS**, sem IDs ausentes/inesperados/conflitantes.
Cobertura CLI: fila 91%, VRAM 94%, temporal 94%, Vita 76%, núcleo 69%; pisos
90/90/90/70/62 satisfeitos. Mutação: **112 declaradas/aplicadas/mortas**, zero
sobreviventes/timeouts/não aplicadas; preservados os 103 nomes anteriores e seu
alias histórico, com 9 novos (contagem não representa defeitos independentes).
Bandit **135 LOW, 0 MEDIUM/HIGH, 0 erros**, exit bruto 1 separado da política.
Ruff, compileall, actionlint, zizmor offline, requirements dry-run, realidade,
registro, replay 11/11 e controle adulterado, triagem, B101 e gaps passaram.

**Revisão e limites:** parecer adicional em `review-budget/report.md`, somente
sobre os hashes finais de implementação/testes: 31 contratos aprovados, sondagens
quadráticas falharam fechadas, sem bypass relevante encontrado. Não é aprovação
independente da nova tree inteira; o BLOCK de `d60f157` não foi transferido nem
convertido em aprovação. `final-checks.sh` faz atestação/selagem local, sem CI
hospedado. Whitespace EOF herdado das instruções permanece ressalva separada.
Orçamento é custo algorítmico aproximado, não prazo real, limite do parser YAML,
validação genérica do schema ou garantia de recursos/segurança geral. Sem matriz
Python, GPU/FAISS real ou evidência de capacidade. Merge continua pendente e
composto staged, sem autorização para commit/publicação.

## Correção local dos três bloqueios da composição #46 + main — 2026-10-21

**Identidade:** mesma worktree `/workspace/Nexus-pr46-compose-main-20261021` e branch
`local/pr46-compose-main-a663a55-20261021`, HEAD âncora
`f3044badb1f16eb4936dde6db17087f6bc96d049`, MERGE_HEAD
`a663a55a560e00ae818e9c8f60aacd4cf9bd93f7`. Revisão independente anterior BLOCK
sobre tree `5d766cfa057861f4533d4d333c5bd35ffd5f3b1f`: os três achados herdados
foram reproduzidos e corrigidos somente na candidata local, sem abortar o merge,
commit, push, fetch real, alteração remota ou operação de outras worktrees.
HEAD não identifica este composto; tree final, patches, modos e hashes ficam em
`/workspace/pr46-composition-evidence-20261021/fixes-review-block/`.

**Contrato e correção:** `RealKnowledgeGraph.add_relation` faz snapshot e valida
as chaves usadas como kwargs NetworkX antes do ledger/projeção; rejeita chaves
não-string e colisões com `u_of_edge`/`v_of_edge`, mantendo propriedades opacas
válidas e comportamento do fallback. Testa estado vazio e populado, com/sem store,
e preservação do fechamento temporal funcional. `predeleg.sh` distingue sucesso,
conflito (exit 1 com tree e lista demonstrados) e falha operacional: erro/saída
ambígua aborta com 3, conflito de código mantém 2. `preflight_rotina.sh` exige
sucesso de `git status` e aceita apenas `ok|blocked` (default `ok` preservado),
abortando sem fetch nos casos inválidos. Scripts continuam executáveis.

**Fato/evidência local:** RED do grafo: 4 métodos/18 falhas de subcasos;
RED shell: 12 métodos/4 falhas; GREEN focado do grafo com contratos existentes:
26/26, shell 12/12. Novos contratos somam 16 métodos. Revalidação ampla em Python
3.11.16 com o venv externo e pins do lock da composição: **533 executados/aprovados,
0 falhas/skips/XPASS**; integração separada **30 aprovações + 1 skip FAISS**, sem
IDs ausentes/inesperados/conflitantes. Cobertura CLI: fila 91%, VRAM 94%, temporal
94%, Vita 76%, núcleo 70%, acima dos pisos. Mutação: **103 declaradas/aplicadas/mortas,
0 sobreviventes/timeouts/não aplicadas**; 7 mutantes novos, preservando os 96 nomes
anteriores (incluem um par de aliases históricos da mesma alteração semântica).
Bandit **135 LOW, 0 MEDIUM/HIGH, 0 erros**; código bruto 1 mantido separado da
política aprovada. Ruff, compileall, actionlint, zizmor offline, requirements dry-run,
registro, replay 11/11 e rejeição de expectativas adulteradas, B101, gaps e triagem
passaram. Relatórios de realidade regenerados. Comandos e saídas: `run-gates.sh`,
`*.command`, `*.log`, `gates.tsv`; finalização em `final-checks.sh` e respectivos logs.

**Limites e handoff:** a revisão BLOCK anterior permanece histórica, não equivale
a aprovação independente desta nova árvore. Atestação final identifica explicitamente
composição não commitada e gates locais, sem fabricar CI hospedado. Diagnóstico
integral de whitespace mantém a linha vazia EOF das instruções herdadas de main;
nenhuma limpeza lateral das instruções/hashes. Preflight não oferece transação
contra falhas arbitrárias de backend, recursos ou objetos customizados. Sem matriz
Python, GPU/FAISS/SentenceTransformers real, clearance de segurança ou alegação de
capacidade geral. Preservar o merge pendente; não publicar/commitar automaticamente.


## Composição local candidata da #46 com main — 2026-10-21

**Identidade e contrato:** worktree `/workspace/Nexus-pr46-compose-main-20261021`, branch
`local/pr46-compose-main-a663a55-20261021`; HEAD/base da candidata
`f3044badb1f16eb4936dde6db17087f6bc96d049`, incorporando por merge local sem commit
`a663a55a560e00ae818e9c8f60aacd4cf9bd93f7`; merge-base
`d435083bb4c893a58128cee6cfa00f65bba7fe02`. O código testado inclui modificações não
commitadas: HEAD sozinho NÃO identifica a composição. Diff, modos e hashes da árvore
ficam em `/workspace/pr46-composition-evidence-20261021/`. Não há push, alteração de
main ou de PR, rebase das fontes, novo commit ou CI remoto desta composição.

**Resoluções:** decisões preservadas dos dois lados; somente D-029 main do contador
passa a D-034 e D-029 main B101 a D-035, com busca de todas as referências. D-029
temporal da PR e D-030–D-033 permanecem. Perfil CI/dev compartilhado da #46 mantido,
pins alinhados ao lock main: numpy 2.4.6, coverage 7.16.2, hypothesis 6.170.5,
bandit 1.9.4, ruff 0.17.1, networkx 3.6.1. Sem re-resolução do lock: hashes e pins
dos pacotes main preservados; o comentário numpy mantém a proveniência dos dois
arquivos de entrada. Catálogo de mutações é a união de nomes dos dois lados;
`queue_capacity_guard` e `queue_max_attempts_positive` são aliases históricos da
MESMA alteração semântica, com âncora restrita ao construtor para evitar o restore.
Timeout permanece inconclusivo e bloqueia o gate. Os 96 nomes não são 96 defeitos
independentes. Demais mudanças de main auto-mescladas foram testadas em conjunto.

**Evidência real desta tentativa:** ambiente novo externo Python 3.11.16, lock
instalado com hashes; executor `run-gates.sh`, comandos `*.command`, logs `*.log`
e códigos em `gates.tsv` no diretório externo acima. Suíte completa sob cobertura e
resumo final: **517 executados/aprovados, 0 falhas, 0 skips, 0 XPASS**. Integração
separada: **30 aprovações + 1 skip FAISS**, sem IDs ausentes/inesperados/conflitantes.
Cobertura (arredondada CLI): fila 91%, VRAM 94%, temporal 94%, Vita 76%, núcleo 69%,
todos acima dos pisos 90/90/90/70/62. Mutation: **96 declaradas/aplicadas/mortas,
0 sobreviventes, 0 timeouts, 0 não aplicadas**, com o alias semântico já explicado.
Bandit novo: **135 LOW, 0 MEDIUM/HIGH, 0 erros**; exit 1 bruto por LOW registrado,
política de severidade separada aprovada. B101 e gaps aprovados (0 pass-only,
0 NotImplementedError). Ruff, compileall, actionlint 1.7.12, zizmor 1.31.1 offline,
contratos focados, requirements dry-run, registro e replay de 11 casos aprovados;
expectativas adulteradas foram rejeitadas. Relatórios de realidade regenerados
pela árvore composta e estabilidade conferida na finalização.

**Falha observada e correção documental:** a primeira execução completa terminou
com código 1 SOMENTE em `security-check`, porque o documento provisório continha
história fora do formato canônico. O relatório foi regenerado pelo scanner atual;
os snapshots dos dois commits foram preservados abaixo e externamente, sem somar
contagens. O check canônico e os contratos documentais são reexecutados na
finalização; os códigos originais não são apagados. `git diff --check HEAD` também
identifica uma linha vazia no EOF de `docs/agentes/INSTRUCOES_AGENTE_NEXUS_v3.md`,
herdada byte a byte de main. Não alteramos as instruções/hashes do registro por
limpeza alheia à composição; a checagem contra main nesse path é limpa. O gate de
whitespace do primeiro executor inspecionou apenas unstaged: não prova o diff total.

A primeira atestação também rejeitou `SECURITY_TRIAGE.md: bandit_low: 131`:
o gerador canônico preserva o bloco existente, mesmo ao regenerar os findings.
Corrigido somente esse marcador para 135, a medição atual. O manifesto e o log
rejeitados foram preservados como `STATE_MANIFEST.first-rejected.json` e
`state-consistency.first-rejected.json` na evidência externa. Nova finalização,
atestação e congelamento substituem aquela identidade provisória; nenhuma mudança
de código, teste ou resultado do scanner foi necessária.

**Benchmarks e limites:** v1/v2 sintéticos herdados, seed 20261016, sem novo holdout.
v1 mantém `unsupported_interface` para POMDP. v2 128/128 dentro da tolerância
1e-12, erro máximo 2.7755575615628914e-16; controles inválidos rejeitados 4/4,
taskset SHA-256 `232371d0e8e948f4b0a004261e3d271b1003aed8d0de8c931d3439f9d370d025`.
Os JSONs dizem `source_revision=unrecorded-local`; a identidade real é o patch e a
atestação externa, não um benchmark independente nem prova de capacidade geral.
Sem GPU real, FAISS, SentenceTransformers, matriz Python ou auditoria externa.
Checks antigos da PR NÃO validam esta composição. O manifesto nativo usa o HEAD
âncora; o manifesto local anotado e a proveniência externa explicitam árvore suja,
patch e ausência de CI. `clean_required` do gerador não é comprovação de limpeza.
A revisão independente única de QA ainda será solicitada pela coordenação sobre
o diff final congelado; nenhuma revisão independente foi realizada nesta tentativa.



## Registro histórico (não estado atual): revalidação local da #46 após a #51 — 2026-10-19

**Separação de estados:** a base desta rodada é o commit de `main` integrado pela #51,
`d435083bb4c893a58128cee6cfa00f65bba7fe02`, confirmado no objeto local `origin/main`.
O `main` local permanece em `7c8a187c1465c6db39777f5e9dc16efb112e1121` e não foi movido.
O head draft da #46 é `f025204339b3dc5ae1f345e98be7f6f7fbcb2498`, branch
`audit/consolidated-38-45-20261018`, worktree original limpa confirmada; merge-base
`812ebd02b0c82ecc3169d620e36d25592b1d6727`. O candidato vive somente em
`audit/reconcile-pr46-after-ci-20261019`, worktree `/workspace/Nexus-pr46-revalidated-20261019`.
Não representa `main`, atualização da PR ou CI remoto. A árvore original suja
`/workspace/Nexus` e todas as worktrees/refs preexistentes devem permanecer intactas.

**Contrato e resolução:** merge local `--no-commit --no-ff` do head exato da #46 sobre
`d435083`; único conflito em `CONTINUATION.md`, resolvido preservando ambos os
históricos (#51 e composição/VRAM), sem escolher contagens antigas de `main`.
O workflow auto-mesclou: retém `push.branches: [main]`, `pull_request` sem filtro
de destino, ausência de `pull_request_target`, permissões, pins e steps herdados.
Comparado ao head #46, apenas o workflow (remoção do filtro da #51) e este documento
mudam; código, testes, lock e demais arquivos são idênticos. #47–#50 não foram incorporadas.

**Validação atual:** Python 3.11.16 em venv novo externo, lock instalado com
`uv pip install --require-hashes -r requirements-ci.lock`, NetworkX 3.6.1 presente.
Executor e comandos exatos em `/workspace/pr46-validation-20261019/validate.sh`;
logs, comandos por etapa, relatórios e `gates.tsv` no mesmo diretório.
- Contratos focados, `coverage run --branch -m unittest discover -p 'test_*.py'`
  e `python tools/test_summary.py --output test-summary.json`: **416/416**, zero falhas,
  skips e XPASS na suíte completa. Integração separada via
  `NEXUS_INTEGRATED_SUMMARY=integrated-summary.json coverage run --append --branch core/constitutional_brain.py`:
  **30 aprovadas + 1 skip FAISS**, zero IDs faltantes/inesperados/conflitantes.
- Cobertura branch: fila **91%**, VRAM **94%**, temporal **94%**, Vita **76%**, núcleo
  **69%**, acima dos pisos 90/90/90/70/62. `tools/targeted_mutation.py`: **84/84 mortas**;
  suplemento externo VRAM: **10/10 mortas**, zero sobreviventes/timeouts/não aplicadas.
- Ruff 0.16.10, actionlint 1.7.12, Zizmor 1.31.1 offline, compileall, dry-run dos
  requisitos constitucionais, realidade sem diff, triagem canônica, auditor B101 e
  auditor de lacunas aprovados. Bandit 1.9.4: **135 LOW, zero MEDIUM/HIGH e erros**;
  saída 1 esperada por LOW, política de severidade conferida separadamente.
- Benchmarks sintéticos herdados, seed `20261016`: v1 mantém `unsupported_interface`
  na trilha POMDP e limitações dos controles negativos; v2 **128/128** dentro de
  `1e-12`, erro máximo `2.7755575615628914e-16`, controles rejeitados **4/4**,
  taskset SHA-256 `232371d0e8e948f4b0a004261e3d271b1003aed8d0de8c931d3439f9d370d025`.
  Os JSONs iniciais usam `unrecorded-local`; árvore congelada e atestação externa
  fornecem a proveniência desta execução. Não houve novo holdout independente.
- O primeiro comando externo de inventário de versões teve erro de sintaxe de
  f-string; corrigido apenas no executor externo e reexecutado com êxito em
  `environment-retry.log`. Nenhum gate de código falhou ou foi ignorado.

**Checks remotos e limites:** a consulta GitHub desta sessão foi bloqueada por acesso
não habilitado; não há reconfirmação remota nova. Conforme o handoff recebido (não
nova evidência): push de `main` após #51 aprovado; #46 OPEN/draft, CONFLICTING,
checks antigos 2/2 success; #47–#50 OPEN/draft, bases empilhadas e sem checks reportados.
Checks antigos não validam este candidato. A revisão independente única incide
sobre a árvore/diff congelados; parecer, hashes e SHA do commit local ficam fora
da árvore, em `/workspace/pr46-validation-20261019/`, para evitar autorreferência.
O manifesto final será gerado pós-commit, com `ci.run_id=not_ci`; `gates_status=success`
refere-se somente aos gates locais concluídos, não a um job GitHub.
Não há validação de GPU real, FAISS, matriz de outras versões Python ou segurança
externa; métricas sintéticas não demonstram capacidade geral ou prontidão para produção.

**Próximo passo exige autorização:** no máximo commit local desta composição; sem
push, publicação, rebase remoto, atualização de PR, merge remoto ou alteração de `main`.
Antes de qualquer ação futura, mostrar payload, novo head/base, diferenças e checks
atuais ao Nexus e obter autorização individual. A #46 e toda a cadeia permanecem drafts
remotos inalterados; os registros abaixo são históricos, não estado de `main`.


## Histórico da composição local anterior — 2026-10-18

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



## Rodada local — Axis e last_seen canônico — 2026-10-21

- **Autorização e base:** owner aprovou `id: axis`, UID opcional exato `hAVRxhnxpHyGN2av2zCZQn`, correção de `last_seen` no mesmo patch e commit somente local. Worktree reutilizada `/workspace/Nexus-axis-last-seen-20261021`, branch `fix/axis-last-seen-20261021`, base limpa `b2b01f2c6b044f0f7f1b60ec1e9c0d615fa4d296`, também `origin/main` local. Preflight remoto de igualdade foi fornecido pelo owner; nesta execução não houve fetch, consulta GitHub, rebase, alteração de `main` ou refs remotas, push, PR, merge ou auto-merge. PRs #38–#50/#60 e demais worktrees não foram operados.
- **Contrato e diagnóstico:** `last_seen` continua obrigatório e aceita somente string ASCII `YYYY-MM-DD` que represente data civil existente; ausência, null, bool, lista, número, objeto date/datetime YAML, data impossível e formatos alternativos são rejeitados sem coerção. Controles `2026-10-21`, bissexto e limites de ano passam. A hipótese de que o schema já aplicava o formato foi refutada: o validador usava apenas `required`. Parser ISO sozinho é insuficiente, pois aceita compacto e semana ISO. UID é metadado opcional, string não vazia preservada literalmente; não muda identidade dos claims, não impõe unicidade ou autenticação, nem migra o Grok. Não foi adicionada dependência ou validação genérica de todo o schema.
- **Red/green:** baseline rápida (Ruff, compileall, descoberta) 396/396 aprovada. Antes da produção, `$PY -m unittest -v test_agent_registry_contract.py` executou 19 métodos: 32 falhas de asserção e 1 erro (propriedade UID ainda ausente no schema), exit 1; ausência e controles válidos passaram. Após correção: 19/19, exit 0. Matriz real de CLI: 10/10 controles/negativos com exit esperado e arquivos de entrada inalterados. `$PY tools/agente/validar_registro.py --expirados aviso` aprovou dados existentes. Entrada Grok preservada byte a byte e semanticamente, sem UID novo.
- **Reprodução e ambiente:** `E=/workspace/nexus-axis-last-seen-evidence-20261021`; `PY=/workspace/nexus-replay-failclosed-evidence-20261021/venv/bin/python`, Python 3.11.16. Ambiente existente reutilizado em leitura, pins do lock conferidos por `$PY -m pip install --dry-run --require-hashes -r requirements-ci.lock`; sem reinstalação. Ruff 0.17.1, coverage 7.16.2, Bandit 1.9.4, Hypothesis 6.170.5, NetworkX 3.6.1. `$PY $E/run_gates.py` registra comandos completos, códigos e logs por gate; `$PY $E/check_preservation.py` verifica Grok, identidade Axis, estado sujo de `/workspace/Nexus`, refs/worktrees e matriz do CLI (antes do commit).
- **Gates efetivos:** descoberta com cobertura e resumo final 404/404, sem falhas/skips/XPASS. Integração 30 aprovados/1 skip FAISS, sem missing/unexpected/conflicting/duplicates. Pisos aprovados: fila 95%/90, VRAM 94%/90, temporal 93%/90, Vita 76%/70, núcleo 69%/62. Ruff e compileall nos alvos do workflow; actionlint 1.7.12 e zizmor 1.31.1 offline; dry-run de requirements; auditoria de realidade sem drift; registro/claims; triagem e auditores AST aprovados. Replay 11/11; expectativas invertidas retornaram 1 esperado.
- **Mutação e segurança:** 69 declarados/aplicáveis/mortos (64 herdados + 5 específicos), zero sobreviventes/não aplicados/timeouts. O mutante que remove a guarda de tipo de data morre por TypeError em regex, não por asserção; outros quatro novos morrem por asserções. Bandit 131 LOW, zero MEDIUM/HIGH/erros; exit 1 esperado pelos LOW, sem clearance geral. Auditor de asserts: 111 restritos ao demo; gaps pass-only/NotImplemented: zero. Nenhuma mudança de permissões, workflow, dependências, núcleo ou instruções canônicas.
- **Coordenação, revisão e limites:** lease Axis temporário aberto após bootstrap dos testes/registro (não antes do RED), arquivado em `$E/claim-active.yaml` e removido ao concluir; `CLAIMS.yaml` final igual à base vazia. Nenhum claim/registro do Grok alterado. Uma revisão QA independente deve examinar o diff completo congelado; patch, manifesto de hashes/modos e parecer ficam em `$E`, sem alterar o diff após revisão. Hash final/commit são entregues externamente para evitar autorreferência. Manifesto local identifica base com árvore suja e `not_ci`; consistência retorna 1 esperado somente por `ci.gates_status` não ser `success`, sem conflitos documentais. CI hospedado, instalação limpa e CLIs de benchmarks não executados; testes de benchmarks participaram da suíte. Evidência apenas de regressão local, não de capacidade geral nem autenticidade externa do UID. WIP `window_seconds` e demais worktrees preservados.

## Rodada local — restauração do executável de replay — 2026-10-21

- **Escopo e identidade:** continuação local autorizada, sem publicação. Mesma worktree `/workspace/Nexus-replay-failclosed-20261021`, branch `local/replay-failclosed-20261021`, HEAD `ca36166a8bc4f35fc1036867d01e8f6f7054d855`; merge-base com a base histórica `3c3b0388f3eeb97e16330a0cc52c689d2b445d2d` reconfirmada. A ref local `main` já estava em `7c8a187c1465c6db39777f5e9dc16efb112e1121`, distinta daquela base remota histórica; não houve fetch, rebase, mudança de refs nem nova consulta remota nesta retomada.
- **Achado concreto e red/green:** o patch anterior SHA-256 `a175daa4d98704bd0c4a608d18c0517ffffdf62ace488e598b1fd481276df9fd` foi reprovado pela revisão independente por remover o modo executável. Reprodução direta retornou 126/Permission denied; a nova regressão, executada antes da correção, falhou por esse motivo. Restaurado modo de filesystem `755`/Git `100755`, sem alteração dos bytes do runner. `test_documented_runner_is_directly_executable` usa subprocesso direto, sem `bash` ou chmod, com a tabela padrão: 11 casos aprovados. Contrato: preservar a invocação documentada e todos os comportamentos fail-closed/atestação já corrigidos, sem mudar ação, tabela ou dependências.
- **Evidência reproduzível:** `E=/workspace/nexus-replay-execfix-evidence-20261021`, `PY=/workspace/nexus-replay-failclosed-evidence-20261021/venv/bin/python` (Python 3.11.16; ambiente do lock já instalado, sem reinstalação). `$PY $E/run_gates.py`, `$PY $E/run_mutation_probes.py` e `$PY $E/probe_executable_mode.py` registram comandos, saídas e códigos. Logs `direct-before.log`, `regression-red.log`, `regression-green.log` e `mode-after.log` discriminam reprodução, regressão antes e aprovação depois. A sonda de modo remove execução somente de cópia temporária e a nova regressão a detecta por falha de asserção.
- **Resultados atuais:** 14/14 contratos de replay/atestação; conjunto focado 55/55; descoberta com cobertura e resumo final 396/396, zero falhas/skips/XPASS. Integração: 30 aprovadas/1 skip FAISS, sem missing/unexpected/conflicting/duplicates. Replay direto e via `bash`: 11/11, exit 0; expectativas invertidas: exit 1 esperado. Sentinelas `tests: CURRENT_TESTS` nos quatro documentos continuam adequadas; os números 13/54/395 da seção anterior são evidência histórica, superada nesta rodada, não contagem atual.
- **Gates reexecutados:** Ruff e compileall nos alvos do workflow, `bash -n`, dry-run de requirements, actionlint 1.7.12, zizmor 1.31.1 offline, reality audit sem diff, registro e auditorias AST aprovados. Pisos de cobertura aprovados: fila 95%/90, VRAM 94%/90, temporal 93%/90, Vita 76%/70, núcleo 69%/62. Mutação existente 64 declarados/aplicáveis/mortos, zero sobreviventes/não aplicados; seis sondas de guardas/atestação e uma sonda de modo também detectadas. Bandit: 131 LOW, zero MEDIUM/HIGH/erros de análise; exit 1 esperado, triagem aprovada, sem clearance geral.
- **Preservação e limites:** conteúdo do runner e do workflow idêntico ao congelamento anterior; somente modo restaurado, novo teste e acréscimos documentais locais (incluindo L-006). Verificação reproduzível `$PY $E/check_preservation.py` compara hashes anteriores após retirar apenas esses acréscimos e confere WIP original/ref local `main`. Manifesto continua `not_ci`; auditoria local retorna 1 esperado somente por `ci.gates_status` não ser `success`, sem conflitos. Não houve CI remoto, CLI de benchmark, commit, push, atualização da #69, merge ou auto-merge; outros WIPs/worktrees não foram editados.
- **Novo congelamento e revisão:** patch reprovado e seu parecer permanecem intactos no diretório anterior; não representam aprovação do novo diff. Novo patch/manifesto com hashes e modos ficam em `$E`, seguido de uma nova revisão final independente de Segurança sobre esse congelamento, com parecer externo para não alterar o diff revisado. Se a revisão apontar outro achado, reportar sem iniciar novo ciclo de correção.

## Rodada local — replay não-vazio e atestação conjunta — 2026-10-21

- **Autorização e proveniência:** somente patch local, sem publicação. `main` e a base da #69 foram conferidas ao vivo em `3c3b0388f3eeb97e16330a0cc52c689d2b445d2d`; #69 OPEN/draft, MERGEABLE/CLEAN, head `ca36166a8bc4f35fc1036867d01e8f6f7054d855`. Worktree nova `/workspace/Nexus-replay-failclosed-20261021`, branch `local/replay-failclosed-20261021`, criada no head da #69; merge-base igual à `main` conferida. Nenhum commit é necessário: HEAD permanece no candidato; hash do patch congelado e manifesto dos arquivos são entregues no handoff externo, sem autorreferência.
- **Contrato e diagnóstico:** aprovação requer leitura completa de arquivo regular legível, pelo menos um caso reconhecido executado e todos os resultados esperados. Ausência, diretório, permissão negada, vazio, comentários e erro de leitura falham fechados. Tipo desconhecido não pode herdar `rc`; última linha sem newline é executada. Código não-zero esperado da guarda continua aprovado. A atestação final aguarda `contract-gate` e `agent-guard-replay`, exigindo ambos `success`; demais jobs/passos, permissões, pins e gatilhos preservados. Tabela original de 11 casos não alterada.
- **Hipóteses discriminadas:** “vazio já é rejeitado” e “erro de leitura já bloqueia” foram refutadas por subprocessos reais; “atestação inclui replay” foi refutada pelo contrato estrutural do YAML. Antes de qualquer correção, o replay original passou 11/11. Os 13 métodos finais, copiados para worktree descartável do head intacto da #69, produziram 12 falhas de asserção, exit 1 (`red-final.log`); não se atribui falha ao run original não-vazio. Após o patch, os 13 métodos passam; o conjunto focado com registro/manifesto/consistência/resumo passa 54/54.
- **Ambiente e reprodução:** `E=/workspace/nexus-replay-failclosed-evidence-20261021`, `PY=$E/venv/bin/python`, Python 3.11.16. Ambiente novo instalado com `$PY -m pip install --require-hashes -r requirements-ci.lock`; lock deste candidato fixa Ruff 0.17.1, coverage 7.16.2, Bandit 1.9.4, Hypothesis 6.170.5 e NetworkX 3.6.1. `PYTHONPATH=.`, `NEXUS_AUDIT_DB=$E/audit.sqlite3`, `COVERAGE_FILE=$E/.coverage`, `NEXUS_INTEGRATED_SUMMARY=$E/integrated-summary.json`. Ripgrep instalado no ambiente. Scripts reproduzíveis e logs brutos ficam em `$E`, não no patch: `run_baseline.sh`, `run_gates.py`, `run_input_matrix.py`, `run_mutation_probes.py`.
- **Linha de base e suíte:** em worktree descartável da `main` fixada, Ruff/compileall e descoberta passaram 382/382. No candidato corrigido: `$PY -m coverage run --branch -m unittest discover -p 'test_*.py'` e `$PY tools/test_summary.py --output $E/test-summary.json` passaram 395/395, sem skips/falhas/XPASS. `$PY -m coverage run --append --branch core/constitutional_brain.py`: integração 30 aprovados e 1 skip FAISS, sem ausentes/conflitantes/inesperados. Pisos do workflow aprovados: fila 95%/90, VRAM 94%/90, temporal 93%/90, Vita 76%/70, núcleo 69%/62. Isso é evidência de contrato, não de capacidade geral.
- **Replay e negativos:** `bash tools/agente/replay/run_replay.sh`: 11/11, exit 0. A inversão `sed 's/\t0$/\t9/' tools/agente/replay/casos.tsv` seguida do replay executou 11, identificou 5 divergências e retornou 1. Matriz bruta registra stdout/stderr/exit de sucesso, falha esperada, divergência, vazio, comentários, ausente, diretório, permissão negada real, sem newline e tipo desconhecido. Erro após dados parciais é simulado por leitor `cat` controlado no teste; nenhum caso deve ser executado nesse cenário.
- **Gates e segurança estática:** Ruff/compileall nos alvos do workflow, `bash -n`, dry-run de requirements, actionlint 1.7.12, zizmor 1.31.1 offline, reality audit sem diff, registro/claims e auditorias AST aprovados. Bandit: 131 LOW, 0 MEDIUM/HIGH, 0 erros de análise; exit 1 esperado pelos LOW, não clearance. Triagem existente aprovada. Mutação existente: 64/64 aplicáveis/mortos, nenhum sobrevivente/não aplicado; seis mutantes locais adicionais das novas guardas/dependências mortos por falhas de asserção. As sondas externas não ampliam o catálogo versionado de mutação nem autorizam relaxá-lo.
- **Atestação local e limites:** manifesto local preserva `run_id/gates_status=not_ci`; auditor retorna 1 somente por `ci.gates_status` não ser `success`, sem conflitos documentais. Não se fabricou sucesso hospedado. Os contratos do workflow são estruturais; não executam o scheduler GitHub. Tabela/ambiente são confiáveis do repositório: o runner não é sandbox para entradas arbitrárias, e mínimo de um caso não certifica cobertura de todas as guardas. CLI de benchmarks não reexecutada; seus testes existentes participaram da suíte.
- **Preservação e revisão:** PRs abertos #69, #46 e #41 tocam o workflow; esta correção é dependente somente do candidato #69, sem tocar branches alheias. Registro/claim ativo de `grok-bot` lido e preservado; identidade e claim compartilhados não foram alterados para trabalho sem PR. A única revisão independente de Segurança é solicitada sobre o diff congelado completo; seu parecer e hash são entregues externamente para não modificar o diff após a revisão. Sem push, PR, merge, auto-merge ou alteração de `main`; WIPs preexistentes preservados. Nova lição L-005 registrada, sem promoção permanente.

## Rodada local — contador de testes derivado (D1) — 2026-10-21

- **Base:** `d435083` (`main`), branch `fix/handoff-counter-decoupling`, sem push.
- **Mudança:** `tests: CURRENT_TESTS` nos quatro blocos `NEXUS-CURRENT-STATE`; sentinelas restritas ao próprio campo (ver D-034). PRs futuros **não** precisam editar o contador.
- **Comandos:** ruff 0.16.10 (lock) ok; compileall ok; reality audit sem diff; `unittest discover` 309 OK; suíte integrada 30 aprovadas/1 ignorada; mutação 58/58; Bandit 131 LOW; `state_manifest.py` + `state_consistency_audit.py` → `pass`; com literal `tests: 303` injetado → `fail` (conflito detectado).
- **Limitações:** Python 3.13.5 local (CI usa 3.11); CI hospedado, actionlint e zizmor não executados nesta rodada.
## Rodada local — descoberta unittest vazia falha fechada — 2026-10-20

- **Base exata:** `d435083bb4c893a58128cee6cfa00f65bba7fe02`, igual à ref local `origin/main` inspecionada, sem fetch nem alteração de `main`. Branch `fix/test-summary-empty-suite-20261020`, worktree criada limpa em `/workspace/Nexus-test-summary-empty-suite-20261020`. `/workspace/Nexus` permanece intocada; nenhuma edição foi feita nas demais worktrees. Não houve consulta remota, push, PR ou merge. A PR #52 e seu resumo integrado não fazem parte deste patch.
- **Contrato:** `SummaryResult.summary["result"]` exige `testsRun > 0` e `wasSuccessful()`. Descoberta vazia preserva todos os contadores em zero, informa `result=failed` e faz o CLI retornar 1; não inventa um teste falho. Suítes não vazias preservam aprovação, skip, expected failure, XPASS como falha e deduplicação de falhas de subtests.
- **Proveniência do código validado:** blobs Git `tools/test_summary.py=0110883147e80e3de56e0b1a387ed87fe78016af` e `test_test_summary.py=207a9aa7e0ec2e13134b823497986e39a92157a2`. O hash do commit local final e a conclusão da revisão independente são entregues no handoff externo, evitando autorreferência. Os quatro marcadores canônicos de contagem foram sincronizados para 306, sem alterar claims históricos.
- **Ambiente e saídas:** `PY=/tmp/nexus-ci311/bin/python` (Python 3.11.16), `OUT=/tmp/nexus-empty-suite-20261020`, `PYTHONPATH=.`; bancos, JSONs, cobertura e logs de execução ficam fora da árvore versionada. `NEXUS_AUDIT_DB=$OUT/audit.sqlite3`, `COVERAGE_FILE=$OUT/.coverage`, `NEXUS_INTEGRATED_SUMMARY=$OUT/integrated-summary.json` na execução com cobertura.
- **Red/green:** `$PY -m unittest -v test_test_summary.py` antes do patch: 4 testes, 3 falhas de asserção em 2 testes (resumo vazio, resumo do CLI e exit code), saída 1; os controles não vazios passaram. Após a guarda de uma linha: 4/4 aprovados, saída 0. O teste CLI usa subprocesso real com `--pattern 'no_such_tests_*.py'`, saída absoluta em `TemporaryDirectory`, verifica JSON inteiro e exit não-zero separadamente e não faz descoberta recursiva do próprio módulo.
- **Suíte e cobertura:** `$PY -m coverage run --branch -m unittest discover -p 'test_*.py'`; `$PY -m coverage run --append --branch core/constitutional_brain.py`; `$PY -m coverage report`; `$PY tools/test_summary.py --output $OUT/test-summary.json`: 306 executados, 304 aprovados, 2 skips por NetworkX opcional ausente, 0 falhas/XPASS. Integração: 30 aprovadas, 1 skip FAISS, 0 ausentes/conflitantes/inesperadas. Pisos verificados por `$PY -m coverage report --include=ARQUIVO --fail-under=PISO`: fila 95% (90), VRAM 94% (90), temporal 93% (90), Vita 76% (70), núcleo 68% (62). Cobertura total 70%; subprocessos CLI não são instrumentados por esta configuração.
- **Lint/compilação/workflow:** `python -m ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py` (Ruff 0.17.1); `$PY -m compileall -q core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py`; `/tmp/nexus-ci-tools/actionlint .github/workflows/*.yml` (1.7.12); `/tmp/nexus-ci-tools/zizmor --offline --no-progress .github/workflows` (1.31.1): aprovados, sem findings de workflow.
- **Auditorias/segurança:** `$PY tools/reality_audit.py` e `git diff --exit-code -- AUDIT_REALITY_REPORT.md audit_reality.json`: aprovados, relatórios inalterados. `$PY -m bandit -r core vita tools -f json -o $OUT/bandit-report.json` (1.9.4): 131 LOW, 0 MEDIUM/HIGH, 0 erros de análise; código 1 de findings não foi tratado como clearance. `$PY tools/security_triage.py $OUT/bandit-report.json SECURITY_TRIAGE.md --check`, `$PY tools/embedded_assert_audit.py $OUT/bandit-report.json core/constitutional_brain.py` e `$PY tools/implementation_gap_audit.py core/constitutional_brain.py --output $OUT/implementation-gaps.json`: aprovados; 111 asserts demonstrativos, nenhum fora de `__main__`, 0 funções pass-only/NotImplementedError.
- **Mutação:** `NEXUS_AUDIT_DB=$OUT/mutation-audit.sqlite3 $PY -u -c 'from pathlib import Path; from tools import targeted_mutation as m; m.REPORT=Path("/tmp/nexus-empty-suite-20261020/targeted_mutation_report.json"); raise SystemExit(m.main())'`: 58 declarados/aplicáveis/mortos, 0 sobreviventes/não aplicados, código 0, sem mortes por timeout. O harness usa cópia temporária; esses mutantes herdados não cobrem a nova guarda, cuja sensibilidade foi demonstrada pelo red/green.
- **Limites:** validação local, sem reinstalar `requirements-ci.lock`; Ruff local 0.17.1 difere do pin 0.16.10 e NetworkX do lock não está instalado (dois skips, não aprovação desse backend). Nenhuma nova sondagem sintética nem CLI de benchmark foi executada; os testes de benchmark já integrantes da suíte foram preservados. Sem run hospedado/manifesto de job concluído, não se fabricou `gates_status=success` ou atestação operacional. O runner unittest padrão ainda imprime `Ran 0 tests ... OK` para vazio; a decisão contratual é o JSON final `failed` e exit 1, sem alterar `wasSuccessful()`. O leitor herdado do manifesto classifica esse resumo vazio falho como `invalid_summary` (falha fechada); ele não foi alterado, e o CLI já bloqueia o job anterior.
- **Revisão independente:** QA/CI aprovou o diff congelado, sem achados bloqueantes ou defeitos acionáveis; conferiu base/merge-base, guarda, subprocesso real, red/green, temporários, marcadores e interrupção do workflow pelo exit 1. Reexecutou os 4 contratos focados com sucesso, sem alterar arquivos nem repetir gates amplos ou sondagens.
- **Fechamento:** `$PY -m unittest -v test_test_summary.py test_state_manifest.py test_state_consistency_audit.py` aprovado após atualizar os documentos; marcadores comparados às contagens dos JSONs locais, sem fabricar atestação. Stage explícito somente dos seis arquivos deste diff, inspeção do índice e `git diff --cached --check` precedem o commit local; hash e estado final são entregues no handoff externo.
## Rodada local — distinguir resumo inválido de ausente — 2026-10-20

- **Base e isolamento:** `d435083bb4c893a58128cee6cfa00f65bba7fe02`, confirmada por `git rev-parse refs/remotes/origin/main` e `git ls-remote origin refs/heads/main`. Branch inédita `fix/state-manifest-invalid-summary-20261020`, worktree `/workspace/Nexus-state-manifest-invalid-summary-20261020`. Nenhum checkout existente foi editado; `main` e a cadeia de PRs permanecem fora do escopo. O hash do commit local final e a revisão independente são entregues no handoff externo, evitando autorreferência.
- **Contrato:** `_test_summary()` retorna `not_generated` somente para `FileNotFoundError`; erros de leitura/parse, raiz não objeto, campos ausentes, tipos e contagens inconsistentes retornam `invalid_summary`. Objetos válidos preservam campos e resultado com `generated`, inclusive resultado `failed`, zero execuções e campos extras ignorados conforme o contrato anterior. `build_manifest()` mantém o status preciso e o fallback de `result` existente; schema 2 e auditor fail-closed não foram alterados. Não se adotou a proposta de rejeitar suítes vazias da branch separada.
- **Red/green:** os oito novos métodos cobrem ausência, leitura negada, JSON vazio/malformado, oito raízes não objeto, cada campo ausente, tipos/contagens/resultados inválidos e três resumos válidos. Cada fixture verifica diretamente o leitor e o conteúdo completo de `verification.tests`. Antes do patch, 13 métodos produziram 10 falhas de subcasos; após incluir leitura negada, os 14 métodos contra o arquivo original da base em cópia temporária produziram 11 falhas. Com o patch, `$PY -m unittest -v test_state_manifest.py test_state_consistency_audit.py test_test_summary.py test_integrated_suite_summary.py`: 30 testes aprovados, repetidos após gerar os artefatos locais.
- **Correções da revisão independente:** a primeira revisão QA/CI encontrou bytes UTF-8 inválidos escapando como `UnicodeDecodeError` e `NaN`/`Infinity` aceitos em campos extras. Novas regressões produziram 1 erro e 3 falhas em 15 métodos antes da correção. A leitura agora captura `ValueError` (inclui erro de JSON e de decodificação), e `parse_constant` rejeita as três constantes não JSON, inclusive em extras aninhados; strings e demais campos extras JSON válidos continuam ignorados. Todos os gates foram reexecutados após esses ajustes.
- **Ambiente:** `PY=/tmp/nexus-ci311/bin/python` (3.11.16), `PYTHONPATH=.`, bancos fora da worktree por `NEXUS_AUDIT_DB=/tmp/nexus-state-manifest-{suite,mutation}-20261020.sqlite3`. Ambiente preexistente, sem instalar dependências: NumPy 2.4.6, coverage 7.16.2, Hypothesis 6.170.0, Bandit 1.9.4; Ruff 0.17.1 via `python -m ruff`, diferente do pin CI 0.16.10. Não equivale a reinstalação do lock ou execução hospedada.
- **Suíte e cobertura:** `$PY -m coverage run --branch -m unittest discover -p 'test_*.py'` e `$PY tools/test_summary.py --output test-summary.json`: 311 executados, 309 aprovados, 2 skips por NetworkX opcional ausente, 0 falhas/XPASS. `NEXUS_INTEGRATED_SUMMARY=integrated-summary.json $PY -m coverage run --append --branch core/constitutional_brain.py`: 30 checks aprovados, 1 skip FAISS, nenhuma ausência/conflito/ID inesperado. `$PY -m coverage report --include='<arquivo>' --fail-under=<piso>` passou para fila 95%/90, VRAM 94%/90, temporal 93%/90, Vita 76%/70 e núcleo 68%/62; `coverage xml -o coverage.xml` gerado localmente.
- **Gates:** `python -m ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py`; `$PY -m compileall -q` com os mesmos alvos; `/tmp/nexus-ci-tools/actionlint .github/workflows/*.yml` (1.7.12); `/tmp/nexus-ci-tools/zizmor --offline --no-progress .github/workflows` (1.31.1): aprovados. `$PY tools/reality_audit.py` e `git diff --exit-code -- AUDIT_REALITY_REPORT.md audit_reality.json`: relatórios inalterados.
- **Mutação:** `$PY tools/targeted_mutation.py`: 58 declaradas/aplicáveis/mortas, 0 sobreviventes/não aplicadas. Verificação adicional local em cópias temporárias, sem modificar o harness versionado: seis mutantes detectados pelos 15 testes do manifesto (ausência como inválido; parse/leitura como ausente; remoção da guarda de objeto; remoção da guarda de tipo de result; perda do status em `build_manifest`; remoção do parser JSON estrito). Script e logs em `/tmp/nexus-state-manifest-evidence-20261020/`; este teste adicional não amplia o catálogo de mutações do CI.
- **Segurança/auditorias:** `$PY -m bandit -r core vita tools -f json -o bandit-report.json`; `$PY tools/security_triage.py bandit-report.json security-triage-generated.md`; `$PY tools/security_triage.py bandit-report.json SECURITY_TRIAGE.md --check`; `$PY tools/embedded_assert_audit.py bandit-report.json core/constitutional_brain.py`; `$PY tools/implementation_gap_audit.py core/constitutional_brain.py --output implementation-gaps.json`: aprovados, 131 LOW, nenhum MEDIUM/HIGH/erro de análise; asserts fora do demo, pass-only e NotImplemented: zero. LOW não é clearance de segurança.
- **Manifesto e limite de atestação:** `$PY tools/state_manifest.py --test-command "python tools/test_summary.py --output test-summary.json" --output STATE_MANIFEST.json` preservou `generated` e as 311 execuções. `$PY tools/state_consistency_audit.py --manifest STATE_MANIFEST.json --output state-consistency.json` retornou **1 esperado**, sem conflitos documentais e somente `ci.gates_status deve ser 'success'`, pois `run_id` e `gates_status` são `not_ci`. Não se fabricou sucesso/ID de CI para fazer o auditor passar; os caminhos positivos e negativos são cobertos pelos contratos. Os quatro marcadores atuais foram atualizados para 311 sem alterar métricas históricas nem a triagem de findings.
- **Limites finais:** benchmarks de capacidade/POMDP não foram reexecutados (nenhuma mudança nesses caminhos; seus contratos participam da suíte). Artefatos gerados e evidência temporária não entram no commit. Diff/paths devem passar por revisão independente QA/CI antes do commit local; nenhum push, PR, merge, nova dependência ou alteração de workflow foi autorizado.
## Rodada local — validar LOW antes da comparação documental — 2026-10-20

- **Base/isolamento:** `d435083bb4c893a58128cee6cfa00f65bba7fe02`, confirmada em `origin/main` e por `git ls-remote origin refs/heads/main`. Branch `fix/state-consistency-low-validation-20261020`, worktree `/workspace/Nexus-state-consistency-low-20261020`, criada limpa. Worktrees preexistentes preservadas, inclusive `/workspace/Nexus` suja e a referência de manifesto `62afec13b274db044bfdfc21f2e194620560b31a`. PRs #52–#55, `main` e publicação estão fora do escopo. Hash final e revisão independente ficam no handoff externo, evitando autorreferência.
- **Contrato:** `security.bandit.severity.LOW` exige inteiro built-in não negativo. Valor presente inválido retorna `status=fail` e erro legível, sem exceção e sem comparar LOW aos documentos. Os demais campos e documentos continuam auditados; zero/positivos preservam schema, texto e conflitos válidos. Manifestos estruturalmente incompletos e outros campos de segurança mantêm o tratamento herdado.
- **Red/green:** três novos métodos cobrem quinze valores inválidos (negativo, ambos bools, três floats finitos, três strings, null, NaN, ambos infinitos, lista e objeto), documentos concordantes/divergentes e controles `0`, `1`, `131`, `10**30`. `$PY -m unittest -v test_state_consistency_audit.py` contra o baseline: 11 métodos, 16 falhas de subcasos, exit 1. Após o patch, `$PY -m unittest -v test_state_consistency_audit.py test_state_manifest.py test_test_summary.py test_integrated_suite_summary.py`: 25 aprovados. Fixtures são testes de contrato, não sondas ou evidência de capacidade.
- **Ambiente:** `PY=/tmp/nexus-ci311/bin/python` (3.11.16), `PYTHONPATH=.`, NumPy 2.4.6, coverage 7.16.2, Hypothesis 6.170.0, Bandit 1.9.4. Ruff 0.17.1 via `python -m ruff`, diferente do pin CI 0.16.10. Ambiente preexistente, sem instalar dependências/reinstalar lock; não equivale a execução hospedada. Bancos via `NEXUS_AUDIT_DB=/tmp/nexus-state-consistency-low-{suite,mutation,local-mutation}-20261020.sqlite3`, fora da worktree.
- **Suíte/cobertura:** `$PY -m coverage run --branch -m unittest discover -p 'test_*.py'` e `$PY tools/test_summary.py --output test-summary.json`: 306 executados, 304 aprovados, 2 skips NetworkX opcional, zero falhas/XPASS. `NEXUS_INTEGRATED_SUMMARY=integrated-summary.json $PY -m coverage run --append --branch core/constitutional_brain.py`: 30 aprovados, 1 skip FAISS, nenhuma ausência/conflito/ID inesperado. `$PY -m coverage report --include='<arquivo>' --fail-under=<piso>` aprovou fila 95%/90, VRAM 94%/90, temporal 93%/90, Vita 76%/70 e núcleo 69%/62. Auditor alterado: 79% branch combinado, não cobertura completa. `$PY -m coverage xml -o coverage.xml` gerado localmente.
- **Lint/compilação:** `python -m ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py`; `$PY -m compileall -q` com os mesmos alvos; `/tmp/nexus-ci-tools/actionlint .github/workflows/*.yml` (1.7.12); `/tmp/nexus-ci-tools/zizmor --offline --no-progress .github/workflows` (1.31.1): aprovados. `$PY tools/reality_audit.py` e `git diff --exit-code -- AUDIT_REALITY_REPORT.md audit_reality.json`: relatórios inalterados.
- **Mutação:** `$PY tools/targeted_mutation.py`: 58 declaradas/aplicáveis/mortas, zero sobreviventes/não aplicadas. `$PY /tmp/nexus-state-consistency-low-evidence-20261020/mutate_low.py`: cinco adicionais mortas (aceitar bool, aceitar negativo, rejeitar zero, remover validação, comparar inválido). Cópias temporárias, catálogo de CI inalterado; script, relatórios e logs nesse diretório temporário.
- **Segurança/auditorias:** `$PY -m bandit -r core vita tools -f json -o bandit-report.json`; `$PY tools/security_triage.py bandit-report.json security-triage-generated.md`; `$PY tools/security_triage.py bandit-report.json SECURITY_TRIAGE.md --check`; `$PY tools/embedded_assert_audit.py bandit-report.json core/constitutional_brain.py`; `$PY tools/implementation_gap_audit.py core/constitutional_brain.py --output implementation-gaps.json`: aprovados. 131 LOW, zero MEDIUM/HIGH/erros de análise; zero asserts fora do demo, pass-only e NotImplemented. LOW não é clearance de segurança.
- **Proveniência:** blobs Git `9589e95150e85e11b6066d53458e45f1ca002418` (auditor) e `d5037d00d47b586192d9c979446ee7003af56fb8` (testes). Quatro marcadores atuais atualizados para 306; histórico e inventário de findings preservados.
- **Atestação/limites:** `$PY tools/state_manifest.py --test-command "python tools/test_summary.py --output test-summary.json" --output STATE_MANIFEST.json` e `$PY tools/state_consistency_audit.py --manifest STATE_MANIFEST.json --output state-consistency.json`: manifesto local preservou 306 execuções e `not_ci`; auditor retornou exit 1 esperado, sem conflitos e com somente `ci.gates_status deve ser 'success'`. Não se forjou sucesso ou run de CI. Não é atestação hospedada. Benchmarks de capacidade/POMDP não foram reexecutados; seus contratos participam da suíte, seus caminhos não mudaram. Nenhuma dependência, workflow ou PR alterada; nenhum push/merge. Congelar o diff completo para uma revisão independente QA/Segurança; reabrir apenas por achado concreto ou diff alterado. Stage explícito e commit local somente após aprovação.
## Rodada local — duplicatas no resumo integrado — 2026-10-20

- **Base exata:** `d435083bb4c893a58128cee6cfa00f65bba7fe02`; branch `fix/integrated-summary-duplicates-20261020`, worktree exclusiva `/workspace/Nexus-integrated-summary-duplicates-20261020`. O checkout sujo `/workspace/Nexus`, a referência `/workspace/Nexus-audit-json-preflight-20261019` e as demais worktrees foram preservados. Nenhum fetch, push, PR, merge ou alteração de `main`.
- **Contrato:** cada ocorrência terminal reconhecida de PASS/SKIP incrementa a contagem do ID. Mais de uma ocorrência, inclusive PASS/SKIP mistos ou IDs inesperados, aparece uma única vez na lista ordenada `duplicates` e força `result=failed`. `passed`, `skipped`, `missing`, `unexpected`, `conflicting`, regexes, schema 1 e retransmissão original permanecem como antes. Chamadas repetidas a `build_summary` não contam novamente o buffer consumido.
- **Antes/depois:** `$PY -m unittest -v test_integrated_suite_summary.py`: baseline original 6/6; regressões sobre implementação original: 10 testes, 2 falhas por resultado incorretamente `passed` (PASS repetido e SKIP repetido), 5 erros de ausência do novo campo. Após correção: os 10 contratos passam; com `test_state_manifest.py test_test_summary.py`, 18/18. Quatro métodos novos cobrem combinações terminais, IDs inesperados/fracionários, ordenação, chunks, ausência de newline final, idempotência, JSON e stream intacto.
- **Ambiente local:** `PY=/tmp/nexus-ci311/bin/python` (3.11.16), `PYTHONPATH=.`, `NEXUS_AUDIT_DB=/workspace/nexus-summary-duplicates-evidence-20261020/audit.sqlite3`, `COVERAGE_FILE=/workspace/nexus-summary-duplicates-evidence-20261020/.coverage`. Ambiente já disponível, sem reinstalar lock.
- **Gates e comandos reais:** `$PY -m coverage run --branch -m unittest discover -p 'test_*.py'` e `$PY tools/test_summary.py --output /workspace/nexus-summary-duplicates-evidence-20261020/test-summary.json`: ambos executaram 307 testes, 305 aprovados, 2 skips NetworkX opcional ausente, 0 falhas/XPASS. `NEXUS_INTEGRATED_SUMMARY=/workspace/nexus-summary-duplicates-evidence-20261020/integrated-summary.json $PY -m coverage run --append --branch core/constitutional_brain.py`: 31 IDs esperados, 30 passes e 1 skip FAISS opcional; missing/unexpected/conflicting/duplicates vazios, exit 0.
- **Cobertura:** `$PY -m coverage report --include='<arquivo>' --fail-under=<limite>` aprovou fila 95% (>=90), VRAM 94% (>=90), temporal 93% (>=90), Vita 76% (>=70) e núcleo 68% (>=62). Arquivo afetado `core/integrated_suite_summary.py`: 99%; cobertura não é prova de completude semântica.
- **Demais gates:** `python -m ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py` e `$PY -m compileall -q core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py` aprovados; `$PY tools/targeted_mutation.py`: 58/58 mortos, 0 sobreviventes/não aplicados (mutações preexistentes, não específicas do tally). `$PY -m bandit -r core vita tools -f json -o /workspace/nexus-summary-duplicates-evidence-20261020/bandit-report.json`: 131 LOW, 0 MEDIUM/HIGH, 0 erros; triagem `tools/security_triage.py ... SECURITY_TRIAGE.md --check` aprovada, assim como `tools/embedded_assert_audit.py` e `tools/implementation_gap_audit.py`. `$PY tools/reality_audit.py` seguido de `git diff --exit-code -- AUDIT_REALITY_REPORT.md audit_reality.json`: relatórios inalterados. `git diff --check`: aprovado.
- **Documentação:** os quatro marcadores estruturados passam de 303 para 307 testes efetivamente executados. Não foi fabricado manifesto com `gates_status=success` nem run hospedado. Gates de workflow não modificado e instalação de dependências não foram repetidos. Não foram executadas CLIs de benchmarks, sondas sintéticas ou trabalho separado de WorldModel; os contratos preexistentes continuam incluídos na suíte completa.
- **Revisão e limite do consumidor:** revisão independente somente leitura não encontrou bloqueios. `tools/state_manifest.py` ainda classifica um resumo cuja única causa de falha seja duplicidade como `invalid_summary`, em vez de atribuir contagem explícita a duplicatas. Continua fail-closed: o auditor operacional rejeita esse estado e o runner integrado já sai com erro por `result=failed`; o gerador isolado de manifesto não garante exit não zero. Atualizar esse consumidor/mensagem de erro do demo fica fora desta correção mínima. Evidências brutas e script reproduzível: `/workspace/nexus-summary-duplicates-evidence-20261020/`. O hash do commit local final é entregue externamente para evitar autorreferência.
## Rodada local — igualdade exata na guarda de triagem — 2026-10-20

- **Base e isolamento:** `d435083bb4c893a58128cee6cfa00f65bba7fe02`, igual a `origin/main` local; branch nova `fix/security-triage-main-guard-20261020`, worktree `/workspace/Nexus-security-main-guard-20261020`, inicialmente limpa. Sem fetch, push, PR, merge ou alteração de `main`. `/workspace/Nexus` permanece suja e intocada; a worktree de referência `Nexus-test-summary-empty-suite-20261020` permanece limpa em `5c704129e93d4ce73e4144b904c3272a1db582d2`. Nenhuma outra worktree foi usada para escrita.
- **Contrato:** `_is_main_guard` exige `ast.If`, comparação única `Eq`, esquerda `Name('__name__')` e direita `Constant('__main__')`. Não reconhece `!=`, identidade, membership, operadores relacionais, outro valor/nome ou cadeia; a forma convencional com aspas simples/duplas e parênteses permanece reconhecida. São cinco condições adicionais, sem dependência nova nem refatoração.
- **Regressão antes/depois:** `$PY -m unittest -v test_security_triage.py` antes da correção executou 8 testes com 18 falhas de subcasos (exit 1). Depois, `$PY -m unittest -v test_security_triage.py test_embedded_assert_audit.py` aprovou 12/12. As regressões verificam o predicado, guard/corpo no escopo e agrupamento de 2 findings sem perda de contagem. Os arquivos temporários são fixtures de contrato, não um finding observado no source atual; não foi feita sondagem sintética nem executada CLI de benchmark.
- **Ambiente e evidência:** comandos abaixo executados nesta worktree, `PY=/tmp/nexus-ci311/bin/python` (3.11.16), `E=/workspace/nexus-main-guard-evidence-20261020`, `PYTHONPATH=.`. Suíte/integração usam `NEXUS_AUDIT_DB=$E/nexus-audit.sqlite3`, `COVERAGE_FILE=$E/.coverage`, `NEXUS_INTEGRATED_SUMMARY=$E/integrated-summary.json`; mutação usa `$E/mutation-audit.sqlite3`. Logs e relatórios locais ficam em `$E`, fora do stage.
  - `$PY -m coverage run --branch -m unittest discover -p 'test_*.py'` e `$PY tools/test_summary.py --output "$E/test-summary.json"`: ambos aprovados; resumo final 305 executados, 303 aprovados, 2 skips por NetworkX opcional ausente, 0 falhas/XPASS.
  - `$PY -m coverage run --append --branch core/constitutional_brain.py`: 30 verificações aprovadas, 1 skip FAISS opcional, sem ausentes/conflitantes/inesperadas.
  - `$PY -m coverage report --include='<arquivo>' --fail-under=<piso>`: fila 95% (piso 90), VRAM 94% (90), fatos temporais 93% (90), Vita 76% (70), núcleo 68% (62). `$PY -m coverage report --include='tools/security_triage.py'`: 96%; `$PY -m coverage xml -o "$E/coverage.xml"` gerado.
  - `$PY tools/targeted_mutation.py`: 58/58 mutações existentes mortas, 0 sobreviventes/não aplicadas, em cópia temporária isolada. Esse gate não contém mutantes específicos deste predicado; a evidência direta da correção são as regressões antes/depois.
  - `python -m ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py` (Ruff 0.17.1), `$PY -m compileall -q core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py`, `/tmp/nexus-ci-tools/actionlint .github/workflows/*.yml` (1.7.12), `/tmp/nexus-ci-tools/zizmor --offline --no-progress .github/workflows` (1.31.1): aprovados.
  - `$PY tools/reality_audit.py` e `git diff --exit-code -- AUDIT_REALITY_REPORT.md audit_reality.json`: aprovados, relatórios inalterados.
  - `$PY -m bandit -r core vita tools -f json -o "$E/bandit-report.json"`: 131 LOW, 0 MEDIUM/HIGH e nenhum erro de análise (exit de findings tratado e JSON verificado). `$PY tools/security_triage.py "$E/bandit-report.json" SECURITY_TRIAGE.md --check`: aprovado; corpo do relatório permanece igual. `$PY tools/embedded_assert_audit.py "$E/bandit-report.json" core/constitutional_brain.py`: 111 B101, nenhum assert fora do bloco. `$PY tools/implementation_gap_audit.py core/constitutional_brain.py --output "$E/implementation-gaps.json"`: 0 pass-only/NotImplementedError.
- **Documentação mínima:** contagem `tests` dos quatro blocos estruturados passa a 305; histórico e demais claims permanecem. Não há decisão arquitetural nova. O ledger aponta para esta evidência local.
- **Revisão independente:** QA/Segurança aprovou o diff de implementação e testes, sem bloqueadores, e reexecutou os 12 contratos focados com sucesso. A mutação ainda estava pendente durante aquela revisão; seu resultado final acima foi observado posteriormente, sem atribuí-lo ao revisor.
- **Limites:** validação local com ferramentas já disponíveis, sem reinstalar o lock e sem equivalência declarada ao CI hospedado. Sem manifesto/atestação de job remoto ou `gates_status=success` fabricado. Os testes da suíte incluem os contratos de benchmarks existentes, mas suas CLIs não foram executadas. Não muda disposições por ID, tratamento de `else`, escopos/cache ou outras limitações herdadas; triagem continua inventário conservador, não clearance de segurança. Hash do commit local final informado no handoff externo para evitar autorreferência.
## Rodada local — guarda B101 direta do módulo — 2026-10-20

- **Base e isolamento:** `d435083bb4c893a58128cee6cfa00f65bba7fe02`, confirmada por `git ls-remote origin refs/heads/main` e pela ref local `origin/main`. Nova branch `fix/embedded-assert-module-guard-20261020`, worktree `/workspace/Nexus-embedded-assert-module-guard-20261020`, criada limpa nessa base exata. `/workspace/Nexus` permanece suja em `a676ee600e03fd377386cd05adbe20b01908103b`; a referência `/workspace/Nexus-state-consistency-low-20261020` permanece limpa em `a6539319f3d76964be5ed68c05802bf5d8534a37`. Não alterar outras worktrees nem `main`.
- **Contrato e implementação:** somente `ast.If` em `ast.Module.body` com a comparação convencional já reconhecida cria faixa de exceção. Asserts continuam sendo procurados recursivamente; B101 fora da faixa continua sendo reportado mesmo sem `ast.Assert` correspondente. Guardas aninhadas não criam faixas; o corpo positivo de uma guarda legítima preserva sua exceção lexical e o `else` fica fora. Não há mudança em núcleo, workflow, dependências ou triagem de escopos separada.
- **Red/green:** `test_embedded_assert_audit.py` ganhou sete métodos (11 no total). Antes da mudança no auditor, a execução retornou saída 1 e 37 falhas de subcasos em três métodos; após a troca de `ast.walk(tree)` por `tree.body`, 11/11 passaram. Nove contextos aninhados são exercitados com/sem B101 e com/sem guarda de módulo adicional; há controle de B101 sem assert, `else`, comparação não convencional e guarda legítima. Suíte focada ampliada: 32/32.
- **Ambiente:** `PY=/tmp/nexus-ci311/bin/python` (3.11.16), `PYTHONPATH=.`, `NEXUS_AUDIT_DB=/tmp/nexus-embedded-guard-20261020/<gate>.sqlite3`; coverage 7.16.2, Bandit 1.9.4, Hypothesis 6.170.0, NumPy 2.4.6; Ruff 0.17.1 via `python -m ruff`. Ferramentas previamente disponíveis, sem instalação/alteração de dependências. Logs locais em `/tmp/nexus-embedded-guard-20261020/`.
- **Comandos e resultados locais:**
  - `$PY -m unittest -v test_embedded_assert_audit.py test_security_triage.py test_state_manifest.py test_state_consistency_audit.py`: 32 aprovados; `$PY -m unittest -v test_embedded_assert_audit.py`: 11 aprovados no green final.
  - `COVERAGE_FILE=/tmp/nexus-embedded-guard-20261020/.coverage $PY -m coverage run --branch -m unittest discover -p 'test_*.py'`; com o mesmo `COVERAGE_FILE`, `NEXUS_INTEGRATED_SUMMARY=integrated-summary.json $PY -m coverage run --append --branch core/constitutional_brain.py`: suíte completa aprovada; integração 30 aprovados e 1 skip FAISS opcional, 0 ausentes/conflitantes/inesperados. `coverage report --include=<arquivo> --fail-under=<piso>` passou para fila 95%/90, VRAM 94%/90, fatos temporais 93%/90, Vita 76%/70 e núcleo 68%/62. Auditor B101: 76%, sem piso novo; CLI real também executada fora da medição.
  - `$PY tools/test_summary.py --output test-summary.json`: 310 executados, 308 aprovados, 2 skips por NetworkX opcional ausente, 0 falhas/XPASS. Os quatro marcadores atuais de testes foram sincronizados em 310; números históricos permanecem preservados.
  - `$PY tools/targeted_mutation.py`: 59 declarados/aplicáveis/mortos, 0 sobreviventes/não aplicados; cópia temporária isolada. Novo mutante `embedded_assert_only_module_level_guards` restaura a travessia vulnerável e morre com saída 1, não por timeout.
  - `python -m ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py`; `$PY -m compileall -q core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py`; `/tmp/nexus-ci-tools/actionlint .github/workflows/*.yml` (1.7.12); `/tmp/nexus-ci-tools/zizmor --offline --no-progress .github/workflows` (1.31.1): aprovados, nenhum finding dos linters de workflow.
  - `$PY tools/reality_audit.py`; `git diff --exit-code -- AUDIT_REALITY_REPORT.md audit_reality.json`: aprovados e relatórios versionados inalterados. `$PY tools/implementation_gap_audit.py core/constitutional_brain.py --output implementation-gaps.json`: 0 pass-only/NotImplementedError.
  - `$PY -m bandit -r core vita tools -f json -o bandit-report.json` retorna o status informativo de findings LOW; JSON conferido: 131 LOW, 0 MEDIUM/HIGH e 0 erros de análise. `$PY tools/security_triage.py bandit-report.json SECURITY_TRIAGE.md --check`: aprovado. `$PY tools/embedded_assert_audit.py bandit-report.json core/constitutional_brain.py`: `ok=true`, 111 B101, faixa `25680–27549`, nenhum finding/assert fora. LOW não é clearance.
- **Limites:** validação local com ambiente existente, não reinstalação do lock nem CI hospedado. NetworkX e FAISS opcionais indisponíveis; seus caminhos não foram validados aqui. Sem novas sondas sintéticas ou CLIs adicionais de benchmark; os testes de benchmark existentes estão na suíte completa. A atestação operacional de consistência requer `ci.gates_status=success` de um job concluído: não fabricar esse status, run remoto ou manifesto de commit limpo. O auditor é sintático, não uma análise de fluxo/segurança; schema/filename dos findings e a triagem separada não foram endurecidos nesta rodada.
- **Revisão e entrega:** após consolidar estes registros, congelar o diff completo para uma única revisão independente focada de QA/Segurança; reabrir somente por achado concreto ou diff alterado. O resultado da revisão, SHA-256 do diff e hash do commit aprovado ficam no handoff externo, evitando autorreferência ou alteração documental após a revisão. Stage explícito e somente commit local se aprovado; sem push, PR, merge ou mudanças em `main`.
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
## TransferMapping duplicado — 2026-10-21

Validação local no commit de código `4e971f38854dedecac68a55f08125cde01d27992` (branch `cursor/transfer-mapping-contract-5517`, base `main` `d435083`). Python 3.12.3. O commit de documentação posterior não altera o código sob teste. PR #61, sem merge. Nenhum outro PR aberto alterava `TransferMapping`; a PR #60 só adiciona `AUDIT.md`.

### Fato

- `git blame` nas duas classes e `git log -S 'class TransferMapping' -- core/constitutional_brain.py` apontam ambas para `1f952a9` (2026-08-05). Não há commit posterior de substituição.
- Depois do import, `dataclasses.fields(TransferMapping)` é `mapping_id`, `source_domain`, `target_domain`, `transferable_knowledge`, `transfer_quality`. A anotação de retorno de `transfer_knowledge` é outro objeto, com `transfer_strength`, `shared_features` e `adaptation_required`.
- Reprodução, antes do patch:

  ```sh
  PYTHONPATH=. python3 - << 'PY'
  from core.constitutional_brain import RealMetaLearningStrategies, TransferMapping
  import dataclasses
  print([f.name for f in dataclasses.fields(TransferMapping)])
  RealMetaLearningStrategies().transfer_knowledge("visual_cortex", "visual_planning")
  PY
  ```

  Resultado: campos do POST 48 e `TypeError: TransferMapping.__init__() got an unexpected keyword argument 'transfer_strength'`.
- O contrato `test_transfer_mapping_contract.py`, executado antes de editar o núcleo:

  ```sh
  PYTHONPATH=. python3 -m unittest -v test_transfer_mapping_contract.py
  ```

  `FAILED (failures=17, errors=7)`. Os erros de `transfer_knowledge` são o `TypeError` acima. Domínio `bool`/`None`/`int` produz `AttributeError: ... no attribute 'lower'`. Bool, NaN, infinito e negativo são aceitos nos dois construtores.
- O mesmo comando, no commit de código: `Ran 17 tests in 0.152s` / `OK`.
- Ruff 0.16.10:

  ```sh
  python3 -m ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py
  ```

  `All checks passed`.
- Suíte completa:

  ```sh
  PYTHONPATH=. NEXUS_AUDIT_DB=/tmp/nexus-transfer-mapping.sqlite3 python3 tools/test_summary.py --output /tmp/test-summary.json
  ```

  `Ran 320 tests in 5.177s` / `OK (skipped=2)`. Resumo: executed 320, passed 318, failed 0, skipped 2, unexpected_successes 0, result passed.
- Suíte integrada:

  ```sh
  NEXUS_INTEGRATED_SUMMARY=/tmp/integrated-summary.json PYTHONPATH=. python3 core/constitutional_brain.py
  ```

  Saída: `EMBEDDED CHECKS: 30 passed, 1 skipped, 0 missing`. O JSON tem `result=passed`, skip no id `"0"`. O log marca `TEST 0: Keyword Fallback Mode (FAISS not available)`. Exit code 0.
- `python3 tools/reality_audit.py` regenerou os relatórios versionados: 27591 linhas, 269 classes, 878 funções no núcleo. O inventário de frases fortes de capacidade permanece 51.
- `find_transferable_knowledge` no raciocínio integrado (por volta da linha 24738) ainda é chamado com `target_task=`. Esse parâmetro não existe na assinatura. O log integrado não contém traceback.

### Inferência

- O `TypeError` vem da revinculação do nome global: a chamada resolve `TransferMapping` para a classe do POST 48, definida depois. A anotação de retorno, avaliada na definição do método, ainda segura a classe do POST 38. Não é um alias de import parcial: a falha permanece depois do import completo.
- Os 2 skips da suíte unittest coincidem com a ausência de NetworkX (`importlib.util.find_spec("networkx")` é `None`) e com os dois únicos `skipUnless` da descoberta, em `test_knowledge_graph_contract.py`. O resumo de verbosidade 1 não imprimiu os nomes.

### Hipótese

- **H1, excluída:** o POST 48 substituiu de propósito o POST 38. Não há commit de substituição, e o corpo de `transfer_knowledge` ainda passa `transfer_strength`.
- **H2, excluída:** só a forma do POST 38 é pretendida. `find_transferable_knowledge` e `execute_transfer` constroem e leem a forma do POST 48, e essa chamada sucede com domínios válidos.
- **H3, sobrevivente nos testes desta rodada:** as duas formas são pretendidas. O defeito é a colisão de nome. A correção mantém `TransferMapping` como a forma do POST 48 e devolve a forma do POST 38 em `MetaTransferMapping`. Ainda não está falsificada: os dois caminhos válidos passaram e os tipos são distintos.
- **H4, excluída como desenho:** unir os campos com defaults. Os call sites são disjuntos, e defaults aceitariam mapeamento incompleto. O contrato exige tipos distintos e campos obrigatórios.
- **H5, excluída:** falha por ordem de import. A chamada após o import completo ainda levantava `TypeError`.

### O que mudou

- `MetaTransferMapping` guarda a forma do POST 38. `transfer_knowledge` constrói essa classe e rejeita domínio que não seja `str` antes de incrementar contadores.
- `__post_init__` rejeita `bool`, não-reais, NaN, infinitos e negativos, sem converter o valor guardado. `transfer_strength` e `adaptation_required` ficam em `[0, 1]`. `transfer_quality` fica finito e `>= 0`, e pode ser `> 1`.
- O contrato ficou na raiz, não em `tests/`: `unittest discover` não entra em diretório sem `__init__.py`, e o CI usa o glob `test_*.py`.

### Limitações

- Bandit, cobertura e mutation testing não foram reexecutados. `bandit_low: 131` no marcador continua o valor da triagem anterior, não uma medição desta rodada.
- Não há run de CI hospedado para este commit. Os números acima são locais.
- `transfer_quality > 1` continua aceito porque `_calculate_transfer_quality` pode ultrapassar 1 quando a cobertura excede os requisitos. Rejeitar isso quebraria um caminho que hoje sucede. Não houve clamp silencioso.
- Atribuir um campo depois de `__post_init__` não revalida. As dataclasses não foram congeladas.
- A sobreposição de features em `transfer_knowledge` é heurística por token `_`. Não é transferência aprendida.
- O call site `target_task=` do raciocínio integrado não foi corrigido nem exercido com um domínio fonte já registrado. Permanece não verificado se esse ramo levanta `TypeError` quando `find_best_source_domain` devolve um nome.

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

Após revisão e commit local desta correção, aguardar autorização específica antes de qualquer publicação ou integração. Não alterar a cadeia de PRs existente. As referências anteriores à PR #36 são históricas, não uma instrução operacional atual. A auditoria de outras fronteiras fica fora desta rodada; `ToolDefinition.requires_permission`/`permission_level` e validação recursiva dos membros de contêiner não foram examinados aqui.
Entregar a integração VRAM local para aprovação posterior. Não há autorização de push, PR ou merge. A decisão privada sobre PRs #38–#44 não pertence a esta rodada. A instrução histórica sobre PR #36 aberta está superada: #36 e #37 já integram a base `812ebd02`. Não incorporar outras branches nem sobrescrever seus documentos; regenerar evidências se a base mudar. `ToolDefinition.requires_permission`/`permission_level` continuam sem enforcement, fora deste escopo.

## Rodada local — inventário de classe em `raise` — 2026-10-21

- **Base e isolamento:** `d435083bb4c893a58128cee6cfa00f65bba7fe02`, confirmada pela API remota; branch `fix/gap-audit-exception-class-20261021`, worktree nova `/workspace/Nexus-gap-audit-class-20261021`. Inventário de 23 PRs, 48 worktrees principais, 9 checkouts adicionais e 108 refs locais/remotas: sem sobreposição nos dois paths funcionais. Os quatro documentos canônicos já são alterados por 20 PRs, com risco de conflito documental. Preservar a WIP `window_seconds`, o checkout sujo `/workspace/Nexus` e as demais worktrees; `known_at` continua pendente do owner. Nenhum push/PR/merge ou mudança de `main` autorizado nesta rodada.
- **Contrato:** Python aceita classe ou instância no alvo do `raise` ([referência oficial](https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement)). O auditor só reconhecia `ast.Call`: `raise NotImplementedError` gerava zero ocorrências. Duas linhas agora permitem alvo explícito nome/atributo com ou sem chamada, uma vez por `raise`, mantendo schema, linhas, ordem AST, pass-only e retorno CLI. Não executar fonte nem contar causas `from`, strings, handlers, referências ou outras exceções. Sem resolução de aliases/shadowing; atributos permanecem aproximação sintática. A candidata de negação adversativa em `reality_audit` foi adiada por exigir ampliar uma heurística linguística.
- **RED/GREEN:** novo `test_implementation_gap_raise_class_contract.py`, cinco métodos com subcasos de alvos simples/qualificados, chamadas legadas, causas, sync/async, negativos, não execução e CLI. Base intacta: cinco executados, quatro métodos afetados por oito falhas de assertivas/subcasos, um aprovado, saída 1. Candidato: 24/24 contratos do novo teste e dos auditores existentes aprovados em Python 3.11.16 e 3.12.3.
- **Suítes medidas:** 308/308, zero falhas/skips/XPASS em ambas as versões; integração 30 aprovadas + 1 skip FAISS, sem ausentes/conflitantes/inesperadas; Ruff e compileall aprovados em ambas. Os quatro marcadores usam 308 medidos, não os 315 da WIP separada. Em 3.11: cobertura fila 95% (piso 90), VRAM 94% (90), temporal 93% (90), Vita 76% (70), núcleo 68% (62); auditor afetado 94%, sem novo piso. Mutation existente: 58 declaradas/aplicáveis/mortas, zero sobreviventes/não aplicadas; não é campanha nova específica do auditor.
- **Outros gates 3.11:** dry-run de requirements, actionlint, zizmor, realidade com relatórios versionados inalterados, benchmarks CLI existentes (seed 20261016), triagem, embedded-assert e inventário aprovados. Bandit saída 1: 131 LOW, zero MEDIUM/HIGH, zero erros; política de severidade aprovada, sem clearance de segurança. Núcleo permanece 0 pass-only/0 NotImplementedError. Benchmarks são regressões sintéticas, não nova evidência de capacidade. Ambientes do lock anteriores reutilizados, sem reinstalação; NumPy 2.4.6, NetworkX 3.6.1, Hypothesis 6.170.0, Ruff 0.16.10, Coverage 7.16.2 e Bandit 1.9.4. Cobertura/mutation/segurança/benchmarks não foram repetidos em 3.12.
- **Reprodução:** evidência em `/workspace/nexus-gap-audit-class-evidence-20261021/`: contrato prévio, inventário, RED/GREEN, logs, `run_gates.py` e `validation.json` com comandos/saídas/durações. `PY=/workspace/nexus-window-exact-evidence/venv311/bin/python`; `PYTHONPATH=.`; audit DB e cobertura externos. Focado: `$PY -m unittest -v test_implementation_gap_raise_class_contract.py test_implementation_gap_audit.py test_reality_audit.py`; amplo: `$PY /workspace/nexus-gap-audit-class-evidence-20261021/run_gates.py`.
- **Limites e decisão:** o falso negativo foi concretamente demonstrado; nenhuma decisão anterior foi refutada. Sem revisão independente final adicional: correção AST de duas linhas, sem execução do fonte, dependência ou mudança operacional. Hash final, diff congelado e verificação pós-commit ficam no handoff externo para evitar autorreferência. CI hospedado não foi executado; publicação exige autorização do owner para o diff/hash exatos.
- **Atestação local:** `tools/state_manifest.py --output /workspace/nexus-gap-audit-class-evidence-20261021/STATE_MANIFEST-local.json --test-result passed --test-command 'python tools/test_summary.py --output test-summary.json'` retornou 0; `tools/state_consistency_audit.py --manifest .../STATE_MANIFEST-local.json --output .../state-consistency.json` retornou 1, com zero conflitos documentais e exatamente `manifesto: ci.gates_status deve ser 'success'`. O valor real é `not_ci`; não se fabricou run nem status de gates hospedados. Checagem separada dos quatro marcadores confirmou `tests=308` e `bandit_low=131`. O manifesto pré-commit identifica a base, não um commit contendo a WIP.


## Proveniência histórica dos inventários de segurança reconciliados

### Snapshot histórico de segurança — PR #46 `f3044badb1f16eb4936dde6db17087f6bc96d049`

Não é medição da composição, não somar ao inventário atual.

# Bandit Security Triage



> This is a conservative inventory, not a clearance report. Findings remain open until reviewed and fixed or explicitly justified.

- Total findings: **135**
- Severity counts: LOW=135

## Dispositions

| Disposition | Count | Meaning |
|---|---:|---|
| `controlled_subprocess_review` | 6 | Mutation harness subprocess; keep inputs fixed and review execution boundaries. |
| `embedded_demo_assert_review` | 111 | Assertions inside the legacy __main__ demonstration block; keep them out of production contracts and migrate incrementally. |
| `manual_review` | 1 | No project-specific disposition; manual review required. |
| `simulation_only_random_review` | 17 | Randomness reviewed as simulation/heuristic behavior; keep it out of secrets and security decisions. |

## Findings

Agrupados por escopo (sem números de linha, que mudam a cada edição).

| ID | Severity | File | Scope | Count | Disposition |
|---|---|---|---|---:|---|
| `B101` | LOW | `core/constitutional_brain.py` | `<__main__ block>` | 111 | `embedded_demo_assert_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `AudioProcessor.detect_emotion_from_voice` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `CognitiveFlexibility.adapt_to_context` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `ConceptualBlender.generate_multiple_blends` | 2 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `DivergentThinker.generate_alternatives` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `HyperparameterOptimizer.suggest_next_value` | 2 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `ImitationLearner.imitate` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `InhibitoryControl.inhibit_response` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `QuestionGenerator.generate_followup` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `QuestionGenerator.generate_question` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealCreativitySystem.generate_creative_idea` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealMCTSPlanner._expand` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealMCTSPlanner._simulate` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealMentalSimulationSystem.rollout_planning` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealRSI.optimize_step` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `tools/math_verifier_holdout.py` | `freeze` | 1 | `simulation_only_random_review` |
| `B404` | LOW | `tools/math_verifier_holdout.py` | `<module>` | 1 | `controlled_subprocess_review` |
| `B404` | LOW | `tools/state_manifest.py` | `<module>` | 1 | `controlled_subprocess_review` |
| `B404` | LOW | `tools/targeted_mutation.py` | `<module>` | 1 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/math_verifier_holdout.py` | `_git_revision` | 1 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/state_manifest.py` | `_git` | 1 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/targeted_mutation.py` | `run_tests` | 1 | `controlled_subprocess_review` |
| `B607` | LOW | `tools/math_verifier_holdout.py` | `_git_revision` | 1 | `manual_review` |

### Snapshot histórico de segurança — main incorporada `a663a55a560e00ae818e9c8f60aacd4cf9bd93f7`

Não é medição da composição, não somar ao inventário atual.

# Bandit Security Triage



> This is a conservative inventory, not a clearance report. Findings remain open until reviewed and fixed or explicitly justified.

- Total findings: **131**
- Severity counts: LOW=131

## Dispositions

| Disposition | Count | Meaning |
|---|---:|---|
| `controlled_subprocess_review` | 4 | Mutation harness subprocess; keep inputs fixed and review execution boundaries. |
| `embedded_demo_assert_review` | 111 | Assertions inside the legacy __main__ demonstration block; keep them out of production contracts and migrate incrementally. |
| `simulation_only_random_review` | 16 | Randomness reviewed as simulation/heuristic behavior; keep it out of secrets and security decisions. |

## Findings

Agrupados por escopo (sem números de linha, que mudam a cada edição).

| ID | Severity | File | Scope | Count | Disposition |
|---|---|---|---|---:|---|
| `B101` | LOW | `core/constitutional_brain.py` | `<__main__ block>` | 111 | `embedded_demo_assert_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `AudioProcessor.detect_emotion_from_voice` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `CognitiveFlexibility.adapt_to_context` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `ConceptualBlender.generate_multiple_blends` | 2 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `DivergentThinker.generate_alternatives` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `HyperparameterOptimizer.suggest_next_value` | 2 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `ImitationLearner.imitate` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `InhibitoryControl.inhibit_response` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `QuestionGenerator.generate_followup` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `QuestionGenerator.generate_question` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealCreativitySystem.generate_creative_idea` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealMCTSPlanner._expand` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealMCTSPlanner._simulate` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealMentalSimulationSystem.rollout_planning` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealRSI.optimize_step` | 1 | `simulation_only_random_review` |
| `B404` | LOW | `tools/state_manifest.py` | `<module>` | 1 | `controlled_subprocess_review` |
| `B404` | LOW | `tools/targeted_mutation.py` | `<module>` | 1 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/state_manifest.py` | `_git` | 1 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/targeted_mutation.py` | `run_tests` | 1 | `controlled_subprocess_review` |
