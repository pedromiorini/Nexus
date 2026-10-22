#!/usr/bin/env bash
# Checagem pre-delegacao (Ajuste C): base/head/hash existem, conflito previsto e classificado
set -euo pipefail
head=$1 base=$2; R=${R:-$(git rev-parse --show-toplevel)}
if ! git -C "$R" cat-file -e "$head^{commit}" || ! git -C "$R" cat-file -e "$base^{commit}"; then
  echo "PARAR: hash inexistente"; exit 3
fi
if out=$(git -C "$R" merge-tree --write-tree --name-only --no-messages "$base" "$head"); then
  echo "OK: sem conflito"; exit 0
else
  merge_rc=$?
fi
if [[ $merge_rc -ne 1 ]]; then
  echo "PARAR: falha operacional do merge-tree (exit $merge_rc)"; exit 3
fi
mapfile -t lines <<< "$out"
tree=${lines[0]-}
if [[ ! $tree =~ ^([[:xdigit:]]{40}|[[:xdigit:]]{64})$ || ${#lines[@]} -lt 2 ]]; then
  echo "PARAR: saida ambigua do merge-tree"; exit 3
fi
files=()
for path in "${lines[@]:1}"; do
  if [[ -z $path || $path == /* || $path =~ [[:cntrl:]] || "/$path/" == *"/../"* || "/$path/" == *"/./"* ]]; then
    echo "PARAR: lista de conflitos invalida"; exit 3
  fi
  files+=("$path")
done
if [[ ${#files[@]} -eq 0 ]]; then
  echo "PARAR: lista de conflitos vazia"; exit 3
fi
code=()
for path in "${files[@]}"; do
  [[ $path == *.md ]] || code+=("$path")
done
if [[ ${#code[@]} -gt 0 ]]; then
  printf 'PARAR: conflito de codigo previsto: %s\n' "${code[*]}"; exit 2
fi
printf 'AVISO: conflito so em docs de handoff: %s\n' "${files[*]}"; exit 0
