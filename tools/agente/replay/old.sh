for p in $PRS; do gh pr checks $p | grep -v pending; gh pr ready $p; gh pr merge $p --merge; done
