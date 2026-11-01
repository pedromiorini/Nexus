---
name: repo-agent-dispatch
description: >-
  Use when launching a cloud coding agent on a repository that has an
  owner-written instruction/constitution file and other agents working in it.
---
# Lançar agente de código com material de instrução
1. Localize o arquivo de instrução vigente do projeto (veja a memória; use a versão mais recente aprovada).
2. Anexe-o via `files` no launch e diga no prompt que ele é a fonte única e que a Constituição prevalece.
3. No prompt: objetivo único e pequeno; fatos já observados (arquivo/linha); exigir checagem de PRs abertos e de arquivos já tocados por eles; teste/contrato falhando antes da correção; atualizar docs de evidência/continuidade; PR sem merge.
4. Peça relatório final em pt-BR separando Fato/Inferência/Hipótese, com comandos, resultados, link do PR e o que segue não verificado.
5. Modelo só se o usuário pediu (resolva com action models).
6. Se o launch falhar por cobrança (uso sob demanda), não repita: avise e ofereça fazer localmente sem PR.
7. Ao concluir: resuma ao usuário com o link do PR e o próximo passo sugerido.
