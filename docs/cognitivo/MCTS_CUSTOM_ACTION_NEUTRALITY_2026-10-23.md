# MCTS — neutralidade de ações sem modelo

Registro local da correção iniciada em 2026-10-23.

## Base e contrato

Base: `cb74cad7cfcd0ab536178ed984989ef2425c8211` (`origin/main` no preflight). Branch isolada: `fix/mcts-unknown-actions-neutral-20261023`.

`RealMCTSPlanner.plan(..., available_actions=...)` aceita ações fornecidas pelo chamador, inclusive nomes que não existem em `planner_deltas()`. Para uma ação sem modelo, a implementação deve deixar as métricas do estado inalteradas; a ação ainda pode aparecer na sequência porque foi explicitamente permitida pelo chamador. A correção não cria semântica nem capacidade de execução para essas ações.

## Achado e correção

Antes da correção, `plan()` atribuía a cada ação não modelada deltas fixos de `+0.1` para segurança, `+0.1` para estabilidade e `-0.1` para custo, apesar do comentário descrever efeitos neutros. Em uma sonda determinística com objetivo `improve stability`, estado inicial `{security: 0.5, stability: 0.3, cost: 1.0}`, ação única `UNMODELED_ACTION`, `max_depth=1`, `max_iterations=1` e seed 1, a recompensa subia de `0.43` para `0.51` sem evidência de efeito.

A mudança remove os deltas inventados: o mapa de efeitos locais de cada plano começa vazio. Regressões também removem de um teste legado a expectativa de duas repetições da ação customizada, que dependia do bônus artificial; continuam cobertas a lista de ações permitidas e a ausência de vazamento entre planos.

## Verificação local

- RED antes da mudança: `test_unknown_custom_action_has_neutral_effects` falhou; resultado esperado `0.43`, observado `0.51`.
- GREEN focado: `python -B -m unittest -v test_mcts_planner_input_contract.py test_mcts_planner_quality_contract.py` — 18 testes passaram.
- Suíte completa: `python -B -m unittest discover -p 'test_*.py'` — Python 3.11.16: 631 testes, 0 skips; Python 3.12.3: 631 testes, 2 skips. Ambos passaram.
- Avaliação pré-registrada MCTS: M1, M2, M4 e M5 passaram; M3 falhou com regret 0.0 tanto na base intacta quanto no candidato. M3 não foi alterado; não se atribui essa falha à correção. Os tempos de M5 variaram entre execuções e não são tratados como ganho.
- Mutação direcionada que restaura os deltas artificiais: morta pelo novo contrato, 1/1; 0 sobreviventes e 0 não aplicadas.
- Ruff 0.17.1, `compileall` no Python 3.11.16 e `git diff --check` passaram.
- Logs e JSON de execução mantidos fora do repositório em `/workspace/nexus-mcts-unknown-actions-evidence-20261023/`. Não houve CI hospedado, commit nem publicação.

## Limites

O teste demonstra que os deltas artificiais foram removidos para uma ação não modelada e que o conjunto existente de critérios MCTS não regrediu, salvo M3 que já falhava na base. Não avalia qualidade geral do MCTS, validade de ações externas, nem se um executor posterior sabe realizar a ação indicada.
