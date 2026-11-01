---
name: conflict-resolution-handoff-docs
description: >-
  Use ao resolver conflitos de merge em docs de estado/handoff (.md com
  contadores ou sentinelas) para não deixar linhas soltas; pare se houver
  conflito em código.
---
# Passos
1. Liste os conflitos: `git diff --name-only --diff-filter=U`. Algum arquivo que não seja `.md`: PARE e peça aprovação (conflito de código).
2. Use a sentinela (ex.: `CURRENT_TESTS`) só se ela já existir na base; em repo sem sentinela não a introduza.
3. Resolva os docs preservando o bloco de estado da base, uma única vez.
4. Rode `tools/agente/replay/check_resolution.sh <worktree> <arquivos>` e decida pelo CÓDIGO DE SAÍDA, não pela leitura da tela: 2 = conflito de código, pare (tem prioridade); 1 = marcador ou linha de estado fora do bloco, não commite; 0 = siga.
5. Stage explícito (`git add <arquivos>`, nunca `-A`), rode os gates do projeto e leia o diff final antes do push.
# Teste
`tools/agente/replay/run_replay.sh` deve sair 0 (11 casos: #54 sai 1, #57/#61 saem 0, #58 sai 2).
