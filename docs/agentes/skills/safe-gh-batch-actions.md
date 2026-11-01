---
name: safe-gh-batch-actions
description: >-
  Use antes de marcar pronto, mesclar, rebasear ou fechar PRs via gh em
  sequência, para nunca agir antes do CI verde nem engolir códigos de saída.
---
# Passos
1. Script com `set -euo pipefail`; nada de `| grep`/`| tee` decidindo fluxo.
2. Por PR aprovado: `gh pr checks <n> --watch --fail-fast` (saída 0) e `gh pr view <n> --json mergeStateStatus -q .mergeStateStatus` == CLEAN; senão PARE.
3. Só então `gh pr ready` e `gh pr merge`; confirme `state,mergeCommit` e registre o SHA.
4. Falha: pare, desfaça o parcial (`gh pr ready --undo`), reporte; não tente outra via.
5. Token só do ambiente; nunca imprimir.
# Teste (executado 2026-10-21 com gh falso)
Checks falhos: 0 ações; DIRTY: 0 ações; tudo verde e CLEAN: ready + merge.
