#!/usr/bin/env bash
# uso: timeout_margin.sh <timeout_min> <duracoes.txt>  -> 0 dentro da margem, 6 acima de 85%, 3 entrada invalida
set -uo pipefail
t=${1:-}; f=${2:-}
[[ $t =~ ^[1-9][0-9]*$ ]] || { echo "entrada invalida: timeout '$t'" >&2; exit 3; }
[[ -n $f && -f $f && -r $f && -s $f ]] || { echo "entrada invalida: arquivo '$f' ausente ou vazio" >&2; exit 3; }
max=0; n=0
while IFS= read -r l || [[ -n $l ]]; do
  [[ -z $l ]] && continue
  [[ $l =~ ^[1-9][0-9]*$ ]] || { echo "entrada invalida: linha '$l'" >&2; exit 3; }
  n=$((n+1)); (( l > max )) && max=$l
done < "$f"
(( n > 0 )) || { echo "entrada invalida: sem duracoes" >&2; exit 3; }
lim=$(( t*60*85/100 ))
if (( max > lim )); then echo "BLOQUEIO: max ${max}s > 85% de ${t}min (${lim}s)"; exit 6; fi
echo "OK: max ${max}s <= ${lim}s"; exit 0
