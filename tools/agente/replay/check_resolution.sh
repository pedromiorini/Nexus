#!/usr/bin/env bash
# NOVO (skill conflict-resolution-handoff-docs, passo 3+6). uso: check_resolution.sh <worktree> [arquivos_em_conflito...]
set -euo pipefail
wt=$1; shift; rc=0; code=0
cd "$wt"
DOCS_RE='\.md$'
for f in "$@"; do
  if ! [[ $f =~ $DOCS_RE ]]; then echo "PARAR: conflito de codigo em $f (pedir aprovacao)"; code=1; fi
done
if rg -n '^(<<<<<<<|=======|>>>>>>>)( |$)' --glob '*.md' --glob '*.py' . >/dev/null 2>&1; then echo "FALHA: marcadores de conflito"; rc=1; fi
for f in $(rg -l 'NEXUS-CURRENT-STATE' --glob '*.md' . 2>/dev/null || true); do
  awk -v F="$f" 'BEGIN{inb=0;bad=0}
   /^<!-- NEXUS-CURRENT-STATE/{inb=1;next}
   inb && /^-->$/{inb=0;next}
   !inb && (/^(commit|ci_run|tests|bandit_low): / || /^-->$/){print "FALHA: linha de estado fora do bloco " F ":" NR ": " $0; bad=1}
   END{exit bad}' "$f" || rc=1
done
[[ $code == 1 ]] && exit 2; exit $rc
