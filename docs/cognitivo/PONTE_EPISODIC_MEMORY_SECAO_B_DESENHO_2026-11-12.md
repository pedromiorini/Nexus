# Desenho — seção B da ligação da memória episódica

Status: desenho local. Não implementa. Não é pré-registro aprovado. Os critérios da seção «Plano contract-first» abaixo **não** estão aprovados.
Base: `main` `e21d25cfc0b2e43b19d82af05cdefc1ddb5bd9be` (pai `8e9feac5447c1c759fe1f0c3abb70e14a5587653`; squash da #178).
Agent-Id: grok-bot.
Caracterização já na main, intocada: `docs/cognitivo/PONTE_EPISODIC_MEMORY_CARACTERIZACAO_2026-11-12.md` e `test_router_episodic_memory_bridge_contract.py` (C1–C15, 26 testes). Este documento não reescreve esses critérios.

A hipótese chamada «B3, opção 1c» na ponte do roteador (`docs/cognitivo/PONTE_CENTRALROUTER_SECAO_B_2026-11-10.md:10`) fica fora daqui. O repositório só a nomeia como hipótese, ao lado do desenho aprovado 1a/2a/3a. Não há, nesse arquivo, o mecanismo da opção 1c. Ela não é desenho desta seção B.

## Fato — o que a main já faz

Conferido em `e21d25c`. De `f2b439c` até este commit, `core/constitutional_brain.py` não está no diff. As linhas citadas na caracterização continuam as mesmas. SHA-256 na main, recalculados nesta sessão:

- doc: `3ee447bfa9ed576df64c56eb4ea8fa15421d06b1f68ab47d6bbba9cb848a5b03`
- teste: `0a48ee9589fd08d7000590c45119f745ebaa761d0776c259ff438a8ccb84991a`

A ligação já existe. Não é o caso da ponte do ciclo, em que a seção A registrava ausência de chamada e a seção B criou a chamada.

- O monólito reexporta as mesmas classes em `core/constitutional_brain.py:2034`. `core/cognitivo/__init__.py` não reexporta `RealEpisodicMemory` (C1).
- Não há parâmetro `enable_episodic`. O cérebro constrói uma `RealEpisodicMemory` e o mapa do roteador aponta o mesmo objeto em `ModuleType.EPISODIC` (`:19587`). A orquestração não tem chave `"episodic"`; `"episodic_future"` é outro módulo (C2).
- `_init_routing_rules` inclui `EPISODIC` só em `LEARNING` (`:9622-9625`) e `MEMORY_RECALL` (`:9627-9629`).
- Os proposers padrão (`_TYPE_ORDER` em `:9770`) não emitem `"learning"`. `"learn this fact"` é `query` e não busca. `"remember the last time"` é `recall` e busca uma vez. `"what did I say about the right choice"` casa recall e ethical, e a precedência escolhe `ethical`, sem busca (C4, C11).
- A única chamada do roteador é `search_episodes(query=prompt, limit=3)`, sem `status` (`:10420-10421`). Não chama `start_episode`, `close_episode`, `link_memory_to_episode` nem `get_episode_memories`.
- O dict público de `route` não tem `episodic`, `episodes` nem `episode_id`. A segunda `route` com `{}` vem do cache e não busca de novo. Contexto não vazio não entra no cache (C5, C6).
- Módulo ausente ou exceção de `search_episodes` degrada (`success` verdadeiro, `degraded` verdadeiro) e não bloqueia. A exceção não entra no cache (C7). Jailbreak ou mapa vazio não alcançam a busca (C3, C8).
- `True`, `False`, `None` e `int` estouram em `:9905`. `bytes` e `list` estouram em `:9370`. Contexto que não é dict estoura em `:9897`. Não há código `ROUTER_*` nesses casos (C9).
- Com memória real, a lista achada não é copiada para o dict. Query vazia com proposer forçado de recall degrada e não cria episódio (C10).
- `think` com roteador abre e fecha episódio (`:20172`, `:20552`). No recall, a busca fica entre os dois. Sem roteador, `learn_from_episode` (`:20448`) chama `get_episode_memories` e não busca (C12, C13).
- `think` copia do roteador uma lista fechada (`:20646-20658`): `answer`, `modules_executed`, `request_type`, `cached`, e, se presentes, `success`, `error`, `reason`, `degraded`, `failed_modules`, `failed_modules_detail`, `confidence`, `swarm_decision`, `mcts_plan`, `swarm_included`, `router_latency_ms`. Uma chave nova no dict de `route` não aparece em `think` por esse caminho.
- `get_conversation_history` e `consolidate_learning` são outro canal: buscam `status="closed"` sem `query` (C14).
- O learner no `route` não é chamado: `:10423-10425` devolve `{"status": "learning_queued"}`.
- `ModuleType` já tem `EPISODIC`. C15 exige `claims: []`. Nesta main o arquivo está assim.

## Fato — o que não está na ligação

- O dict de `route` não publica a lista que `search_episodes` já devolveu.
- Os proposers padrão não selecionam `learning`, embora a regra de roteamento desse tipo inclua `EPISODIC`.
- `auto_log` e `constitutional_log` não são lidos no módulo episódico (caracterização, inventário).
- S2a (`expected_reward == 0.0`) e a errata de `counterfactual.py:180` não fazem parte desta ligação. Fora de escopo. A #77 está aberta e edita o monólito e `core/cognitivo/episodic_memory.py`.

## Inferência

- A busca do `think` em recall ocorre depois de `start_episode` e antes de `close_episode`, e não recebe `status`. O default do método é `"closed"`. A lista devolvida nesse `think` não foi impressa na caracterização. Não se afirma aqui que o episódio corrente ficou de fora; isso continua por medir, se alguém for implementar a opção B2.
- Expor a lista no dict de `route` faz o cache (`:10059`) guardar essa lista junto com o resto. A segunda chamada com `{}` continuaria sem nova busca, como C5 já exige, e devolveria a mesma projeção.
- Qualquer chave `episodes` no sucesso do recall torna falsa a frase de C5 «o dict não tem `episodes`». Implementar isso sem uma errata aprovada de C5 faria o teste já mesclado falhar. Isso não é ajuste silencioso de critério.

## Hipótese

- A opção 1c da ponte do roteador resolveria entradas inválidas do `route` ou mudaria o ponto em que o ciclo é chamado. O texto versionado não descreve o mecanismo. Não entra no desenho recomendado.
- Publicar só `episode_id`, `title`, `trigger_query`, `importance` e `status` basta para um chamador reconhecer o episódio fechado sem congelar relógio. Não foi medido como requisito de um usuário externo.

## Objetivo proposto

Tornar observável, no dict de `route`, a lista que `search_episodes` já devolve quando o módulo episódico termina com sucesso. Não criar proposer de `learning`. Não mudar `start_episode` / `close_episode`. Não fazer a falha episódica bloquear a rota. Não reexportar o módulo no pacote. Não acrescentar membro a `ModuleType`.

Isto é proposta. Não está aprovada.

## Opções

- **B0.** Parar neste documento. Nenhum código. É o que esta tarefa faz.
- **B1 — recomendada, se houver implementação futura.** Depois de aprovação explícita, o `route` que teve `ModuleType.EPISODIC` com `success` verdadeiro passa a incluir a chave `episodes`: lista de dicts, possivelmente vazia, com `episode_id`, `title`, `trigger_query`, `importance` e `status`. Sem `duration`. Sem o objeto `Episode`. A chave fica ausente quando o módulo não rodou ou não teve sucesso. `think` não ganha a chave nesta fatia, porque a cópia em `:20646-20658` não a inclui. C5 recebe errata só na frase da chave `episodes` no sucesso; o resto de C1–C15 permanece.
- **B2.** A busca disparada pelo `think` em recall também enxerga o episódio ainda aberto. Exige medir a lista que a caracterização não imprimiu. Muda a semântica da busca no meio do `think`. Não é a primeira fatia.
- **B3.** Um proposer padrão passa a emitir `learning`. `"learn this fact"` deixaria de ser `query`. Isso altera a ponte do ciclo já mesclada (opções 1a, 2a e 3a), não só a memória episódica. Não recomendada aqui.
- **B4.** Falha de `EPISODIC` passa a bloquear a rota. Contradiz C7 e o RT5 de `test_router_fail_closed_contract.py`. Não recomendada.
- **B5.** Trocar `TypeError` / `AttributeError` / `ValueError` de C9 por um código `ROUTER_*`. É mudança do roteador inteiro. Fica de fora. Não se usa a opção 1c como especificação.

## Decisões que exigem aprovação do Pedro

Nenhuma implementação começa antes destas respostas.

1. B0, B1, B2, B3 ou B4. A recomendação deste texto é B1. Não está aceita.
2. Se B1: o nome da chave é `episodes`, e os cinco campos são os listados acima.
3. Se B1: errata de C5, limitada à presença de `episodes` quando a busca episódica teve sucesso. O arquivo de caracterização só muda nessa frase, num commit separado e visível, depois da aprovação. Até lá, C5 continua como está.
4. Se B1: `think` permanece sem a chave nesta fatia.
5. Qual PR detém a vaga do monólito. Ver conflitos abaixo.
6. B2, B3, B4 e B5 não entram na mesma fatia, salvo aprovação em contrário.

## Arquivos de uma implementação futura e conflitos

B1, se aprovada, tocaria:

- `core/constitutional_brain.py`, na montagem do dict de `route` depois dos estágios (`:10036-10051`), lendo o `ModuleResult` de `EPISODIC`. Não precisa editar `core/cognitivo/episodic_memory.py` nem `core/cognitivo/__init__.py`.
- `test_router_episodic_memory_bridge_contract.py`, só na errata aprovada de C5.
- Um arquivo novo de contrato para os critérios ainda não aprovados da seção seguinte. Não é este documento.

Não misturar na mesma PR: `tools/targeted_mutation.py`, `tools/cognitivo_mutation.py`, workflows, timeouts, pisos de cobertura, `CLAIMS.yaml`, `CONTINUATION.md`, validadores da Axis.

PRs abertas que editam `core/constitutional_brain.py`, na lista desta sessão: #39, #40, #41, #42, #44, #46, #48, #49, #77, #115. A regra do repositório é uma PR por vez nesse arquivo. A #77 também edita `core/cognitivo/episodic_memory.py`, `core/cognitivo/__init__.py`, `test_episodic_like_literal_contract.py`, `test_episodic_memory_extraction_contract.py` e `test_episodic_retrieval_contract.py`. B1 não precisa desses arquivos, mas o monólito em comum já impede o paralelo.

#148 está fechada. #166 está mesclada. Não são a fila atual do monólito. S2a e a errata `:180` continuam fora de escopo.

Nenhuma PR aberta desta lista edita os dois arquivos da #178 nem este desenho.

A branch remota `cursor/episodic-memory-link-criteria-d702` permanece em `81255121dc9b533e003a942683da1bc4bc87684d`. Este desenho não a move e não a apaga.

## Plano contract-first

Proposta. Não aprovada. Não há arquivo de teste novo neste commit.

Ordem, só depois das decisões acima:

1. Errata escrita de C5, no texto, antes de alterar o teste. A frase nova diz: no sucesso da busca episódica a chave `episodes` existe e tem a projeção; nos outros casos de C5 a chave continua ausente. `episode_id` continua ausente no dict de `route`.
2. RED: o teste da errata falha em `e21d25c` porque a chave não existe. A saída dessa falha fica anexada à PR futura. Não se altera o roteador para a falha desaparecer antes do GREEN.
3. GREEN mínimo: só a projeção no dict de `route`. `think`, proposers, `status` da busca, degradação e C9 ficam como C1–C15 os descrevem, fora a errata de C5.
4. Se um critério novo falhar no GREEN, a rodada para e relata. Não se ajusta critério, semente ou limite para ficar verde.

Critérios candidatos, ainda não aprovados:

- **E1.** `route("remember the last time", {})` com um episódio fechado desse título devolve `episodes` igual a uma lista de um dict com os cinco campos, `status` `"closed"`, e o `episode_id` desse episódio. Não há chave `episode_id` no dict da rota. A segunda `route` com `{}` vem do cache, não chama `search_episodes` de novo e repete a mesma lista.
- **E2.** Busca com sucesso e zero episódios fechados: `episodes == []`, `success` verdadeiro, `degraded` falso.
- **E3.** `ping`, `""`, `"learn this fact"`, `"Tell me about item 9"`, `"should I plan my week"` e `"what did I say about the right choice"` continuam sem a chave `episodes`.
- **E4.** Módulo ausente, `search_episodes` que levanta, jailbreak detectado e mapa vazio: sem a chave `episodes`. C7 e C8 permanecem.
- **E5.** `think("remember the last time", use_router=True)` não ganha `episodes`. C12 permanece.
- **E6.** Proposer forçado `("learning",)` em `"ping"`, com episódio fechado de título `"ping"`, também devolve `episodes` com esse episódio. Sem o proposer, `"learn this fact"` continua em E3.

E1 contradiz C5 hoje. Por isso E1 só existe depois da errata aprovada. Até lá, C5 é o contrato vigente.
