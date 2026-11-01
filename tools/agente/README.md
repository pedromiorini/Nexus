# tools/agente — guardas operacionais dos agentes

Scripts shell usados pelos agentes antes de agir no GitHub. Não fazem parte do produto Nexus. O job `Agent guard replay` executa o replay no CI; a atestação final exige sucesso tanto desse job quanto do `contract-gate`. No computador do Grok Bot existe cópia equivalente em `/workspace/nexus/tools_agente/`.

| Script | Uso | Saída |
|---|---|---|
| `predeleg.sh <head> <base>` | Antes de delegar/agir num PR: hashes existem? conflito previsto? | 0 ok (ou conflito só em `.md`); 2 conflito de código → pare; 3 hash inexistente → pare |
| `preflight_rotina.sh` | Preflight da rotina diária (`NUVEM_STATUS=ok\|blocked`) | `MODO=nuvem` 0; `MODO=local` 10; `MODO=abortar` 3 |
| `replay/check_resolution.sh <worktree> [arquivos]` | Após resolver conflito: marcadores, linhas de estado fora do bloco `NEXUS-CURRENT-STATE`, conflito de código | 0 ok; 1 falha; 2 conflito de código |
| `replay/run_replay.sh [casos.tsv]` | Suíte de replay das falhas reais de 21/10/2026 (L-001..L-003) | 0 somente com ≥ 1 caso executado e todos os códigos esperados; 1 para entrada vazia/ilegível ou divergência |

`replay/old.sh` é **intencionalmente defeituoso** (reproduz o pipe que engoliu o código de saída — L-003) e só é usado pelos casos do replay; `replay/fakegh` é um `gh` falso que registra ações sem tocar o GitHub. `replay/new.sh` é o fluxo correto.

O runner exige arquivo regular legível e conclui sua leitura antes de executar casos. Arquivo vazio, somente comentários/linhas vazias, ausência, diretório, erro de leitura ou tipo de caso desconhecido falham fechados. A última linha sem newline é executada. Um código não-zero esperado da guarda continua sendo um replay aprovado. Regressões: `python -m unittest -v test_agent_replay_contract.py`.

O runner deve manter modo Git `100755`: `tools/agente/replay/run_replay.sh` funciona diretamente, sem prefixo `bash`. A regressão de execução direta usa a tabela padrão de 11 casos e não altera permissões para mascarar falhas.

## Sobreposição de claims

`python tools/agente/validar_registro.py` rejeita dois claims ativos quando seus
globs têm uma string em comum, mesmo que o arquivo ainda não exista. Por exemplo,
`src/a?.py` e `src/?b.py` conflitam por cobrirem `src/ab.py`. Comparar o texto de
um padrão com o outro não basta; o validador percorre o produto dos autômatos dos
dois padrões, com alfabeto simbólico de fronteiras de caracteres Unicode.

A semântica é `fnmatchcase`: sensível a maiúsculas, `*`, `?`, classes, intervalos
e classes negadas; `/` e ponto inicial não recebem tratamento especial, `**`
equivale a `*`. Preserva-se a convenção adicional de `p/**` cobrir também `p`.
Classes inválidas seguem `fnmatch`, inclusive linguagem vazia de `[z-a]`.
Não se trata de `pathlib.glob`, expansão de shell ou inspeção do filesystem.
Por par de padrões, a busca visita no máximo o produto dos números de posições;
as máscaras de caracteres são construídas a partir de fronteiras dos padrões.
O helper dispõe de 100.000 unidades de trabalho; `validate()` compartilha
1.000.000 de unidades entre todos os pares/alternativas de claims ativos.
Cobram-se comprimento de entrada, comprimento × tamanho do alfabeto antes das
máscaras, passos de busca do fechamento de classes e cada retirada da pilha.
Exaustão não significa ausência de
conflito: o helper lança `ValueError`, a API retorna erro explícito e a CLI
falha com código 1, pedindo reduzir/dividir os claims. É um orçamento algorítmico,
não um prazo de tempo real nem limite do parser YAML/validação genérica do schema.
Regressões de helper, API e CLI: `python -m unittest -v test_agent_registry_contract`.

## Dependências e limitações
- Rodar de dentro de um clone do repositório (o padrão é `git rev-parse --show-toplevel`; pode-se definir `R=`/`REPO=`). Requer `git` ≥ 2.38 (`merge-tree --write-tree`), `rg` e `awk`.
- `replay/casos.tsv` referencia SHAs reais do histórico (`d074c15`, merges do #57/#58/#61). Todos são ancestrais de `main`; basta um clone com histórico completo (não `--depth`). Verificado em clone novo em 21/10/2026: 11/11 casos.
- Não chamam a API do GitHub; o replay usa `fakegh`.
- A tabela e o ambiente de execução são controlados pelo repositório; o runner não é sandbox para tabelas arbitrárias. A contagem não prova cobertura de todas as guardas nem valida segurança geral. Os testes do workflow verificam sua estrutura; não substituem um run hospedado.
