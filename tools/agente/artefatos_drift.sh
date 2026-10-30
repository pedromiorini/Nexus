#!/usr/bin/env bash
# uso: artefatos_drift.sh <repo_dir>  -> 0 artefatos atuais; 7 algum divergente; 3 entrada invalida
set -uo pipefail
repo=${1:-}; PY=${PY:-python3}
[[ -n $repo && -d $repo ]] || { echo "entrada invalida: diretorio ausente: $repo" >&2; exit 3; }
git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 || { echo "entrada invalida: nao e repo git: $repo" >&2; exit 3; }
[[ -f $repo/tools/security_triage.py && -f $repo/tools/reality_audit.py ]] || { echo "entrada invalida: faltam tools/security_triage.py ou tools/reality_audit.py" >&2; exit 3; }
W=$(mktemp -d /tmp/drift.XXXX); trap 'rm -rf "$W"' EXIT
cp -a "$repo/." "$W/r" || { echo "entrada invalida: copia falhou" >&2; exit 3; }
cd "$W/r" || exit 3
drift=0
"$PY" -m bandit -q -r core vita tools -f json -o "$W/bandit.json" >/dev/null 2>&1
[[ -s $W/bandit.json ]] || { echo "entrada invalida: bandit nao gerou relatorio" >&2; exit 3; }
if ! "$PY" tools/security_triage.py "$W/bandit.json" SECURITY_TRIAGE.md --check >/dev/null 2>&1; then echo "DRIFT: SECURITY_TRIAGE.md" >&2; drift=1; fi
if ! "$PY" tools/reality_audit.py >/dev/null 2>&1; then echo "entrada invalida: reality_audit falhou" >&2; exit 3; fi
if ! git diff --quiet -- AUDIT_REALITY_REPORT.md audit_reality.json; then echo "DRIFT: relatorios de realidade" >&2; drift=1; fi
[[ $drift -eq 1 ]] && exit 7
exit 0
