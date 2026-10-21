#!/usr/bin/env bash
# Checagem pre-delegacao (Ajuste C): base/head/hash existem, conflito previsto e classificado
set -euo pipefail
head=$1 base=$2; R=${R:-$(git rev-parse --show-toplevel)}
git -C "$R" cat-file -e "$head^{commit}" && git -C "$R" cat-file -e "$base^{commit}" || { echo "PARAR: hash inexistente"; exit 3; }
out=$(git -C "$R" merge-tree --write-tree --name-only --no-messages "$base" "$head") && { echo "OK: sem conflito"; exit 0; }
files=$(echo "$out" | tail -n +2 | sed '/^$/d')
code=$(echo "$files" | grep -v '\.md$' || true)
if [[ -n $code ]]; then echo "PARAR: conflito de codigo previsto: $code"; exit 2; fi
echo "AVISO: conflito so em docs de handoff: $(echo $files)"; exit 0
