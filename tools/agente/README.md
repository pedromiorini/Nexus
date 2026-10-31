# tools/agente — guardas operacionais dos agentes

Scripts shell usados pelos agentes antes de agir no GitHub. Não fazem parte do produto Nexus. O job `Agent guard replay` executa o replay no CI; a atestação final exige sucesso tanto desse job quanto do `contract-gate`. No computador do Grok Bot existe cópia equivalente em `/workspace/nexus/tools_agente/`.

| Script | Uso | Saída |
|---|---|---|
| `predeleg.sh <head> <base>` | Antes de delegar/agir num PR: hashes existem? conflito previsto? | 0 ok (ou conflito só em `.md`); 2 conflito de código → pare; 3 hash inexistente → pare |
| `preflight_rotina.sh` | Preflight da rotina diária (`NUVEM_STATUS=ok\|blocked`) | `MODO=nuvem` 0; `MODO=local` 10; `MODO=abortar` 3 |
| `replay/check_resolution.sh <worktree> [arquivos]` | Após resolver conflito: marcadores, linhas de estado fora do bloco `NEXUS-CURRENT-STATE`, conflito de código | 0 ok; 1 falha; 2 conflito de código |
| `replay/run_replay.sh [casos.tsv]` | Suíte de replay das falhas reais de 21/10/2026 (L-001..L-003) | 0 somente com ≥ 1 caso executado e todos os códigos esperados; 1 para entrada vazia/ilegível ou divergência |
| `artefatos_drift.sh <repo_dir>` | Antes do push: numa cópia do repo, a triagem Bandit (`security_triage.py --check`) e os relatórios de `reality_audit` continuam iguais ao commitado? Requer `bandit`; não cobre o marcador `bandit_low` dos documentos de estado | 0 atuais; 7 algum diverge; 3 entrada inválida |
| `timeout_margin.sh <timeout_min> <duracoes.txt>` | Guarda de margem de timeout do CI: bloqueia se a maior duração (s, uma por linha) passa de 85 % de `timeout_min·60` | 0 dentro da margem; 6 acima de 85 %; 3 entrada inválida |
| `timeout_margin_backtest.py coletar\|avaliar\|atual` | Retroprojeção da guarda sobre o histórico real do `nexus-contract-gate` (`docs/agentes/timeout_margin_historico.json`); `atual` é só relatório | `avaliar`: 0 critérios TM1–TM5 ok; 1 falha; 3 dado inválido. `atual`/`coletar`: 0 (3 se dado inválido) |

`replay/old.sh` é **intencionalmente defeituoso** (reproduz o pipe que engoliu o código de saída — L-003) e só é usado pelos casos do replay; `replay/fakegh` é um `gh` falso que registra ações sem tocar o GitHub. `replay/new.sh` é o fluxo correto.

O runner exige arquivo regular legível e conclui sua leitura antes de executar casos. Arquivo vazio, somente comentários/linhas vazias, ausência, diretório, erro de leitura ou tipo de caso desconhecido falham fechados. A última linha sem newline é executada. Um código não-zero esperado da guarda continua sendo um replay aprovado. Regressões: `python -m unittest -v test_agent_replay_contract.py`.

O runner deve manter modo Git `100755`: `tools/agente/replay/run_replay.sh` funciona diretamente, sem prefixo `bash`. A regressão de execução direta usa a tabela padrão de 17 casos (11 originais + 6 `tmo` da guarda de timeout) e não altera permissões para mascarar falhas.

## Registro e aceite das instruções

`python tools/agente/validar_registro.py --expirados aviso` valida registro e claims sem escrever nos arquivos. No schema 2, aceite anterior só gera aviso se versão, artefato e SHA forem verificáveis; versão desconhecida/futura, arquivo ausente ou SHA divergente continuam falhando. O schema 1 permanece estrito. Contrato, caminhos de migração e limites de confiança em [ACEITE_VERSIONADO.md](../../docs/agentes/ACEITE_VERSIONADO.md). A migração schema 2 foi integrada pela #113. A composição local da #79 seleciona instruções atuais 3.4.2, com Grok em 3.4.2 por aceite explícito verificado e Axis preservado em 3.4.1; somente o aviso de desatualização do Axis é esperado. O validador não consulta claims dos PRs abertos: a união com os claims locais ainda é verificada manualmente (§9C.3).

## Dependências e limitações
- Rodar de dentro de um clone do repositório (o padrão é `git rev-parse --show-toplevel`; pode-se definir `R=`/`REPO=`). Requer `git` ≥ 2.38 (`merge-tree --write-tree`), `rg` e `awk`.
- `replay/casos.tsv` referencia SHAs reais do histórico (`d074c15`, merges do #57/#58/#61). Todos são ancestrais de `main`; basta um clone com histórico completo (não `--depth`). Verificado em clone novo em 21/10/2026: 11/11 casos.
- Não chamam a API do GitHub; o replay usa `fakegh`.
- A tabela e o ambiente de execução são controlados pelo repositório; o runner não é sandbox para tabelas arbitrárias. A contagem não prova cobertura de todas as guardas nem valida segurança geral. Os testes do workflow verificam sua estrutura; não substituem um run hospedado.

A guarda de timeout **não** é passo bloqueante do CI (modo relatório, ver `docs/agentes/MARGEM_TIMEOUT_CI.md`); só os casos `tmo` do replay, que não dependem de durações reais, rodam no CI.

Os casos `drift` do replay (`replay/casos_drift.tsv`) são de uso **local**: o job `Agent guard replay` do CI não instala bandit e continua só com `casos.tsv` (17 casos). `bash tools/agente/replay/run_replay.sh tools/agente/replay/casos_drift.tsv` exige bandit e, para `D-origem`, o SHA `afd12a5` (`git fetch origin pull/132/head`). Ver `docs/agentes/ARTEFATOS_DRIFT.md`.
