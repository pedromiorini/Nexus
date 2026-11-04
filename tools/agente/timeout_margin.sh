#!/usr/bin/env bash
# uso: timeout_margin.sh <timeout_min> <duracoes.txt>  -> 0 dentro da margem, 6 acima de 85%, 3 entrada invalida
# inteiros limitados (timeout <= 5 digitos, duracao <= 9 digitos) para a aritmetica do bash nao estourar
# [0-9] segue LC_COLLATE: em en_US.UTF-8 o digito ٣ casa e (( )) falha sem abortar (set sem -e), saindo 0.
set -uo pipefail
t=${1:-}; f=${2:-}
ascii19='123456789'
ascii09='0123456789'
[[ $t =~ ^[$ascii19][$ascii09]{0,4}$ ]] || { echo "entrada invalida: timeout '$t'" >&2; exit 3; }
[[ -n $f && -f $f && -r $f && -s $f ]] || { echo "entrada invalida: arquivo '$f' ausente ou vazio" >&2; exit 3; }
max=0; n=0
while IFS= read -r l || [[ -n $l ]]; do
  [[ -z $l ]] && continue
  [[ $l =~ ^[$ascii19][$ascii09]{0,8}$ ]] || { echo "entrada invalida: linha '$l'" >&2; exit 3; }
  n=$((n+1))
  if ! (( l > max || l <= max )); then
    echo "entrada invalida: linha '$l'" >&2
    exit 3
  fi
  (( l > max )) && max=$l
done < "$f"
(( n > 0 )) || { echo "entrada invalida: sem duracoes" >&2; exit 3; }
lim=$(( t*60*85/100 ))
if (( max > lim )); then echo "BLOQUEIO: max ${max}s > 85% de ${t}min (${lim}s)"; exit 6; fi
echo "OK: max ${max}s <= ${lim}s"; exit 0
