# Pré-registro — seção B da ponte CentralRouter (B0, só documentação)

Base: `main` `b9de4cfb7e62a83de0b66d4756fd75cf749b396d` (squash da #170).
Aprovação do Pedro (2026-11-10 11:29 America/Sao_Paulo): opções **1a, 2a e 3a** da proposta do Laird.
Este commit não implementa a ponte, não adiciona o arquivo RED (isso é a B1) e não abre claim.
Agent-Id: grok-bot.

Os critérios B1–B5 abaixo estão copiados da proposta. A conferência no fim não os altera. Se um critério falhar no GREEN, a rodada para e reporta opções, sem ajustar critério, semente ou limite.

Sequência da proposta: B0 (este doc) → B1 (arquivo RED, fora daqui) → B2 (GREEN no monólito, bloqueada enquanto #148 e #166 editarem `core/constitutional_brain.py`; AGENTS.md: um PR por vez no monólito) → B3 (hipótese, opção 1c, fora daqui).

## Desenho aprovado

Fontes conferidas em `b9de4cf`. Nenhuma linha citada abaixo divergiu do arquivo.

- **1a.** Em `_route`, a única chamada `analyze_request` (`core/constitutional_brain.py:9837`) passa a ser `_decide_request_type`, antes da chave de cache (`:9757`, dentro de `create_execution_plan`, chamado em `:9843`), do piso do Swarm (`_swarm_mandatory` chamado em `:9891`) e dos estágios (montados em `create_execution_plan`, `:9704` em diante). `analyze_request` (`:9649`) fica intacta. `analyze_request(` aparece duas vezes no arquivo: a definição e essa única chamada.
- **2a.** Um proposer por categoria. As listas atuais viram dado: planning `:9658`, ethical `:9663`, recall `:9668`, analysis `:9673`, creative `:9678`. O proposer padrão é query (`:9682`). `better` é a precedência atual como ordem total: planning, ethical, recall, analysis, creative, query. `allowed` são os 7 valores de `RequestType` (`:9235–9243`): `query`, `planning`, `learning`, `recall`, `ethical`, `analysis`, `creative`. A entrada do ciclo é só o prompt, nunca o `context`. Na base, essa precedência bate com `analyze_request` em 12/12 frases do corpus abaixo; 4 são ambíguas e ficam registradas em `decision.proposed`.
- **3a.** `apply=None` e uma tabela congelada de chave normalizada (`_normalize_action_key`) para `RequestType`. `selected` guarda a primeira grafia; `RequestType(selected)` levantaria `ValueError` quando a grafia não é o valor do enum. `DecisionInputError`: capturar e bloquear com `ROUTER_DECISION_INPUT_INVALID`, `impasse.code = exc.code`, sem o texto da exceção, via `_finish_blocked` (`:9993`) e sem cache. Todo impasse vira `ROUTER_IMPASSE`. `BaseException` propaga. Não voltar para `analyze_request` (seria fail-open).

Trava, como na proposta: `_blocked_response` faz `blocked["module"].value` (`:10107`). A interface precisa aceitar o marcador `str` `"decision_cycle"`, senão `AttributeError` cai no handler de `think()` (`:20960–20979`), que devolve `EXCEPTION_DURING_THINK` com `confidence` 0.5 e não tem chave `success`.

### Gabarito das 12 frases (medido em `b9de4cf`; é o gabarito de B1)

`analyze_request` devolve a primeira categoria que casa. O ciclo com `better` na ordem de 2a devolve a mesma. As quatro ambíguas são as que casam mais de uma lista.

| Prompt | `analyze_request` | Listas que casam |
|---|---|---|
| `should I plan my week` | planning | planning, ethical |
| `What should I do about the pharmacist?` | planning | planning, ethical |
| `pharmacist` | ethical | ethical |
| `copyright policy` | ethical | ethical (`right` dentro de `copyright`) |
| `what did I say about the right choice` | ethical | ethical, recall |
| `design a fair strategy` | planning | planning, ethical, creative |
| `why is the sky blue` | analysis | analysis |
| `remember the last time` | recall | recall |
| `write a poem` | creative | creative |
| `Tell me about item 9` | query | nenhuma |
| `learn this fact` | query | nenhuma |
| `ping` | query | nenhuma |

## Critérios B1–B5 (texto da proposta; RED no main, exceto a guarda B4a)

- **B1** ligação: ciclo chamado 1x por requisição não cacheada; `request_type` bate com gabarito de 12 frases. Patch só enxerga se a chamada passar por `decision_cycle.run_decision_cycle` (from-import no monólito cega B1 e C1 da #170) => errata do C1.
- **B2** impasse bloqueia: `PROPOSER_FAILURE`, `NO_CHANGE`, `CONSTRAINT_FAILURE`, `TIE` => `ROUTER_IMPASSE`; nenhum estágio roda (nem jailbreak); texto da exceção não aparece; 2ª chamada não vem do cache.
- **B3** entrada inválida: `"abc"`, `[1]`, `None` como proposers => `ROUTER_DECISION_INPUT_INVALID`; com tudo falhando, bloqueio do ciclo vem primeiro.
- **B4** sem manipulação: (a) guarda: context com `request_type`/`proposals`/`allowed` não tira prompt ético de ETHICAL; (b) `ethical\u200b`, largura total, `"ethic al"`, `"query;"` => `CONSTRAINT_FAILURE`.
- **B5** observabilidade: `decision.proposed` registra ambiguidade; `impasse.origin = sha256(prompt)[:16]`; `get_statistics()["impasses_by_kind"]` conta por tipo.

Medição anexada à proposta, não reexecutada neste B0 e sem o arquivo RED no repositório: Python 3.13, 8 testes, 24 falhas, 3 erros `KeyError`, sem `ImportError`. A guarda B4a passa na `main` atual porque o `context` já não escolhe o tipo.

## Errata E-C1 (da #170; só este texto; o teste não muda)

`test_router_decision_cycle_bridge_contract.py` substitui `core.cognitivo.decision_cycle.run_decision_cycle` e `select_operator` com `unittest.mock.patch` no atributo do módulo. O patch vê a chamada que consulta esse atributo na hora. Um `from core.cognitivo.decision_cycle import run_decision_cycle` no monólito, resolvido quando o módulo do cérebro é importado, guarda o objeto original. O teste importa `CentralRouter` no topo, antes do `patch`. O nome já importado não passa a apontar para o registrador. Na base `b9de4cf`, a sonda do mesmo desenho mediu o contrário do que o patch parece prometer: o nome importado continua a função real e o acesso pelo módulo vê o substituto. Enquanto o teste não for editado, C1 e B1 ficam cegos a esse from-import. O GREEN da ponte chama o ciclo pelo atributo `decision_cycle.run_decision_cycle`, não por nome importado no topo do monólito.

## Conferência — critérios não alterados

Fatos da proposta que batem com `b9de4cf`:

- #156 está na `main` (`28a7ee84c71947550318530eb67ccb0a51e5b1b5`). O ciclo tem `OPERATOR_NO_CHANGE`, veto aditivo e `allowed`.
- #157 está na `main` (`26e38e538a8e8f3d774922fb15f230caf90d5f0f`).
- `rejectors` não existe no módulo. `required` não é parâmetro de `run_decision_cycle`. `Subgoal` tem só `kind`, `goal` e `candidates` (`decision_cycle.py:98–103`), sem o pedido de origem.
- Orçamento falha fechado. Módulo que não tem sucesso, veredito que não é booleano `True`, ou orçamento que não rodou no fim do plano devolvem `BUDGET_CHECK_FAILED` (`:10072–10084`). A frase «orçamento fail-open em aberto» do doc da seção A está desatualizada e é corrigida num commit seguinte, só nessa linha.
- Swarm já é exigido: `_swarm_mandatory` (`:10021–10035`) e o bloqueio (`:10085–10096`).
- As linhas 9837, 9757, 10107 e 20960–20979 são as citações da proposta.

Acréscimo que a proposta não citou, e que não muda B1–B5: `_finish_blocked` também lê `blocked["module"].value` na linha 10009, ao montar o evento `ROUTER_BLOCKED`, antes de `_blocked_response`. Um marcador `str` `"decision_cycle"` que só `_blocked_response` soubesse ler ainda quebraria nesse log.

Observação que não contradiz 2a: `RequestType.LEARNING` (`"learning"`) entra nos 7 valores de `allowed`, e `analyze_request` não tem lista para ele. `learn this fact` cai em query. Os proposers das listas atuais não emitem `learning`.
