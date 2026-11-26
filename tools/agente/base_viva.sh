#!/usr/bin/env bash
# Guarda de base viva: a branch ainda mescla limpo sobre a main atual?
# uso: base_viva.sh [HEAD] [MAIN]   (padrão: HEAD e origin/main)
# Saídas: 0 mescla limpo; 9 conflito com a main (refazer/rebasear antes de pedir merge); 3 entrada inválida.
set -uo pipefail
H=${1:-HEAD}; M=${2:-origin/main}; R=${R:-.}
for s in "$H" "$M"; do git -C "$R" cat-file -e "$s^{commit}" 2>/dev/null || { echo "base_viva: SHA ausente $s" >&2; exit 3; }; done
out=$(git -C "$R" merge-tree --write-tree --name-only "$M" "$H" 2>&1); rc=$?
if [[ $rc -eq 0 ]]; then echo "base_viva: OK, mescla limpo sobre $M"; exit 0; fi
if [[ $rc -eq 1 ]]; then echo "base_viva: CONFLITO com $M:"; printf '%s\n' "$out" | sed -n '2,/^$/p' | sed '/^$/d' | head -20; exit 9; fi
echo "base_viva: merge-tree falhou (rc=$rc)" >&2; exit 3
