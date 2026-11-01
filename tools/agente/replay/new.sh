set -euo pipefail
for p in $PRS; do
  gh pr checks "$p" --watch --fail-fast >/dev/null
  st=$(gh pr view "$p" --json mergeStateStatus -q .mergeStateStatus)
  [[ $st == CLEAN ]] || { echo "PARAR #$p estado=$st"; exit 2; }
  gh pr ready "$p"; gh pr merge "$p" --merge
done
