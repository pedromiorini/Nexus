## Fora de escopo (seções A e B)

- S2a (`expected_reward == 0.0`) e a errata de `counterfactual.py:180`.
- `tools/targeted_mutation.py` e o catálogo dos 80 mutantes direcionados.
- Workflows, timeouts, pisos de cobertura e qualquer arquivo de `.github/`.
- Validadores, schema, aceites e `versao_aceita` do Axis (#79 e `tools/agente/validar_registro.py` além de rodá-lo).
- Arquivos das PRs #79, #115, #116, #148, #166, #167 e #77, no sentido de não editar o trabalho delas nem rebaseá-las.
- Orçamento fail-open e Swarm só em `ETHICAL_DECISION` (deixados em aberto pelo adendo RT).
- Resolver subgoal, preferência `indifferent`, veto por efeito, `compare_futures`.
- Mudar o comportamento de `temporal_reasoning.py` ou fazer o roteador passar a chamar `track_event` / `predict_next_event`.
- Implementar `rejectors(state)` / `required` na seção A.

## Protocolo comum

- Base: `fec42312e63756fa473906fc942b76f61739beb5`.
- Python do CI: 3.11. Ambiente: `PYTHONPATH=.`, `LC_ALL=C`, `NEXUS_AUDIT_DB` fora do repositório.
- O oráculo já congelado do ciclo usa `random.Random(20261401 + k)` para `k = 0..19`. Esta ponte não reabre DC1–DC16 e não cria semente nova para o ciclo.
- Caracterização do roteador, se a seção A for aprovada: uma pergunta fixa `"ping"` e `context={}` (sem aleatório).
- Regra de parada: critério que falhar depois do GREEN, teste já existente que falhar, ou mutante sobrevivente ou `not_applied` ⇒ parar sem push e sem ajustar critério, seed ou limite.

Comandos de gate (seção A e, se um dia for desbloqueada, seção B):

```sh
export PYTHONPATH=. NEXUS_AUDIT_DB=/tmp/nexus_audit.sqlite3 LC_ALL=C
python -m unittest discover -p 'test_*.py'
python tools/cognitivo_mutation.py --jobs 4
python tools/targeted_mutation.py
bandit -r core vita tools -f json -o /tmp/bandit-ponte.json
python tools/security_triage.py /tmp/bandit-ponte.json SECURITY_TRIAGE.md --check
coverage run --branch -m unittest discover -p 'test_*.py'
coverage run --append --branch core/constitutional_brain.py
coverage report --fail-under=0
coverage report --include='core/deferred_task_queue.py' --fail-under=90
coverage report --include='core/vram_defense_guard.py' --fail-under=90
coverage report --include='core/temporal_facts.py' --fail-under=90
coverage report --include='vita/nexus_constitutional_bridge_v3.py' --fail-under=70
coverage report --include='core/constitutional_brain.py' --fail-under=62
coverage report --include='core/cognitivo/*' --fail-under=90
bash tools/agente/replay/run_replay.sh
bash tools/agente/replay/predeleg.sh HEAD origin/main
python tools/agente/validar_registro.py --expirados aviso
```

Bandit: 0 MEDIUM e 0 HIGH. LOW pode continuar (a `main` está em 139). O processo do Bandit sair 1 por causa de LOW não falha o critério se a contagem MEDIUM/HIGH for 0 e `--check` sair 0.
Mutação cognitiva: catálogo não diminui (1133 na base; os 135 `tr_*` e os 81 `dc_*` continuam). Nenhum mutante existente é removido nem enfraquecido. `preflight_catalog` devolve `()`. 100% mortos, 0 sobreviventes, 0 `not_applied`.
Direcionada: 80/80, arquivo intocado.
Replay: `run_replay.sh` sai 0.
Pisos: os `--fail-under` acima saem 0.

---

## Seção A — caracterização (única parte proposta para aprovação)

Arquivo novo, e só ele: `test_router_decision_cycle_bridge_contract.py`. Não edita produção, não edita teste existente, não edita `tools/cognitivo_mutation.py` (não há ramo novo).

Previsão no `fec4231`, antes de escrever o arquivo: os testes abaixo passam na primeira execução, porque descrevem o código atual. Não há RED de comportamento novo. Se algum falhar na primeira execução, a leitura deste rascunho está errada: parar e corrigir o rascunho antes de qualquer código, sem «consertar» o roteador para o teste passar.

- **C1.** `CentralRouter.route("ping", {})` com módulos vazios e o `EthicsGuard` real não chama `run_decision_cycle` nem `select_operator`. O teste substitui as duas funções por um registrador e exige 0 chamadas. O resultado continua um `dict` com `success`.
- **C2.** `CompleteNexusBrain` com os defaults de `enable_temporal_reasoning=True` constrói `temporal_reasoning` e `get_statistics()` devolve as seis chaves já congeladas no contrato TR (`total_events_tracked`, `total_relations_detected`, `total_sequences_built`, `avg_sequence_length`, `active_events`, `temporal_relations`), todas no zero inicial (`0`, `0`, `0`, `0.0`, `0`, `0`). Nenhum evento é rastreado só por construir o cérebro.
- **C3.** O mapa de integração do cérebro continua com a chave `"temporal"` valendo `None` depois da construção. `ModuleType` não ganha membro novo.
- **C4.** DC1–DC16 e TR1–TR6 passam sem edição dos arquivos atuais: `test_decision_cycle_contract.py`, `test_decision_cycle_review_contract.py`, `test_temporal_reasoning_contract.py`, mais `test_central_router_contract.py`, `test_router_fail_closed_contract.py`, `test_router_review_addendum_contract.py` e `test_router_budget_swarm_contract.py`.
- **C5.** Gates do protocolo comum, todos verdes, na mesma árvore. `claims: []` no fim, sem abrir claim (o arquivo está sob a #79).

Comando estreito da seção A, antes da suíte:

```sh
python -m unittest -v test_router_decision_cycle_bridge_contract.py test_central_router_contract.py test_router_fail_closed_contract.py test_router_review_addendum_contract.py test_router_budget_swarm_contract.py test_decision_cycle_contract.py test_decision_cycle_review_contract.py test_temporal_reasoning_contract.py
```
