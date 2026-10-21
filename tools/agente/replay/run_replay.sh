#!/usr/bin/env bash
# Suite de replay das tarefas do Nexus. uso: run_replay.sh [casos.tsv]
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); R=${R:-$(git rev-parse --show-toplevel)}; T=$(mktemp -d /tmp/replay.XXXX)
casos=${1:-$D/casos.tsv}; fail=0
cleanup(){ for w in $(git -C "$R" worktree list --porcelain | awk -v t="$T" '$1=="worktree" && index($2,t)==1{print $2}'); do git -C "$R" worktree remove --force "$w"; done; rm -rf "$T"; }
trap cleanup EXIT
mkdir -p "$T/bin"; ln -s "$D/fakegh" "$T/bin/gh"
while IFS=$'\t' read -r id tipo args exp; do
  [[ -z $id || $id == \#* ]] && continue; set -- $args
  case $tipo in
    gh)  LOG=$T/$id.log PATH=$T/bin:$PATH CHECKS=$1 STATE=$2 PRS=58 bash "$D/new.sh" >/dev/null 2>&1; rc=$?;;
    res) git -C "$R" worktree add -q --detach "$T/$id" "$1"; "$D/check_resolution.sh" "$T/$id" >/dev/null 2>&1; rc=$?;;
    merge) git -C "$R" worktree add -q --detach "$T/$id" "$1"; (cd "$T/$id"; git -c user.email=r@r -c user.name=r merge --no-commit "$2" >/dev/null 2>&1); U=$(cd "$T/$id"; git diff --name-only --diff-filter=U); "$D/check_resolution.sh" "$T/$id" $U >/dev/null 2>&1; rc=$?;;
    pre) R=$R "$D/predeleg.sh" "$1" "$2" >/dev/null 2>&1; rc=$?;;
  esac
  if [[ $rc == "$exp" ]]; then echo "PASSOU $id rc=$rc"; else echo "FALHOU $id rc=$rc esperado=$exp"; fail=1; fi
done < "$casos"
exit $fail
