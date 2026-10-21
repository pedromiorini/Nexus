# tools/agente — guardas operacionais dos agentes

Scripts shell usados pelos agentes antes de agir no GitHub. Não fazem parte do produto Nexus nem dos gates do CI (o CI roda ruff/compileall/bandit só em Python; estes são `.sh`). No computador do Grok Bot existe cópia equivalente em `/workspace/nexus/tools_agente/`.

| Script | Uso | Saída |
|---|---|---|
| `predeleg.sh <head> <base>` | Antes de delegar/agir num PR: hashes existem? conflito previsto? | 0 ok (ou conflito só em `.md`); 2 conflito de código → pare; 3 hash inexistente → pare |
| `preflight_rotina.sh` | Preflight da rotina diária (`NUVEM_STATUS=ok\|blocked`) | `MODO=nuvem` 0; `MODO=local` 10; `MODO=abortar` 3 |
| `replay/check_resolution.sh <worktree> [arquivos]` | Após resolver conflito: marcadores, linhas de estado fora do bloco `NEXUS-CURRENT-STATE`, conflito de código | 0 ok; 1 falha; 2 conflito de código |
| `replay/run_replay.sh [casos.tsv]` | Suíte de replay das falhas reais de 21/10/2026 (L-001..L-003) | 0 todos os casos com o código esperado |

`replay/old.sh` é **intencionalmente defeituoso** (reproduz o pipe que engoliu o código de saída — L-003) e só é usado pelos casos do replay; `replay/fakegh` é um `gh` falso que registra ações sem tocar o GitHub. `replay/new.sh` é o fluxo correto.

## Dependências e limitações
- Rodar de dentro de um clone do repositório (o padrão é `git rev-parse --show-toplevel`; pode-se definir `R=`/`REPO=`). Requer `git` ≥ 2.38 (`merge-tree --write-tree`), `rg` e `awk`.
- `replay/casos.tsv` referencia SHAs reais do histórico (`d074c15`, merges do #57/#58/#61). Todos são ancestrais de `main`; basta um clone com histórico completo (não `--depth`). Verificado em clone novo em 21/10/2026: 11/11 casos.
- Não chamam a API do GitHub; o replay usa `fakegh`.
