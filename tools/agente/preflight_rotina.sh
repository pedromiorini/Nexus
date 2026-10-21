#!/usr/bin/env bash
# Preflight da rotina 03:34. Saida: MODO=nuvem|local|abortar. Entradas: NUVEM_STATUS (ok|blocked, resultado do launch), REPO
set -euo pipefail
REPO=${REPO:-$(git rev-parse --show-toplevel)}
if [[ -n $(git -C "$REPO" status --porcelain --untracked-files=no) ]]; then echo "MODO=abortar motivo=checkout_sujo"; exit 3; fi
git -C "$REPO" fetch -q origin main || { echo "MODO=abortar motivo=fetch"; exit 3; }
if [[ ${NUVEM_STATUS:-ok} == blocked ]]; then
  echo "MODO=local motivo=nuvem_bloqueada acao=worktree_limpo_de_origin/main;sem_push;notificar_pedro"; exit 10; fi
echo "MODO=nuvem"; exit 0
