# Lições aprendidas com falhas de verificadores

## Propósito
Transformar cada falha apontada por um verificador (CI hospedado, `tools/state_consistency_audit.py`, `tools/targeted_mutation.py`, testes, revisão humana) em conhecimento reutilizável por todos os agentes. A falha de um verificador é tratada como **dado de treino do processo**, não como ruído. Este arquivo não é evidência de capacidade do Nexus; registra só falhas de processo e de engenharia.

Ler este arquivo **antes** de começar qualquer tarefa no repositório.

## Formato de cada entrada
```
### L-NNN — título curto
- Data: AAAA-MM-DD
- Verificador que falhou (ou que deveria ter falhado): ...
- Fato observado: o que foi visto diretamente (comando, saída, commit, PR).
- Causa: Fato quando reproduzida; Inferência quando deduzida.
- Correção: o que foi feito, com PR/commit.
- Regra/checagem derivada: regra para as instruções ou checagem automática proposta.
- Evidência: PR, SHA e/ou ID de execução do CI verificáveis (obrigatório; "evidência: ver texto" só para entradas anteriores à v3.4).
- Derivada de: lições em uso quando esta foi escrita (ex.: L-003) ou "nenhuma".
- Status: pendente | ativa | promovida (onde: instrução §/arquivo, ou teste/gate) | aposentada (data, motivo)
```

## Regra de evidência e ativação (v3.4.1)
- **Evidência bruta obrigatória:** toda lição cita PR, SHA e/ou ID de execução do CI que outro agente consiga conferir. Lição tirada só de resumo, handoff ou memória é **rejeitada** (política local; arXiv 2610.19910, preprint, só por analogia: mede ruído na evidência de evolução, não rejeição de handoff).
- **Status:** `pendente` (nova, não testada) → `ativa` (passou no teste dos três casos) → `promovida` ou `aposentada`. "Promoção sugerida" é nota, não status.
- **Teste dos três casos** (política local, sem verificação automática; não é o replay de guardas): critérios escritos no PR antes de rodar; passa só se o caso de origem fica corrigido, um caso anterior bem-sucedido não regride e um caso semelhante incompatível é rejeitado ou sinalizado.
- **Teto de 15:** contam só `ativa` e as legadas `registrada` (equivalem a `ativa`); `pendente`, `promovida` e `aposentada` não contam. Para ativar além de 15, aposente uma explicitamente (`Status: aposentada`, data e motivo), sem apagar o texto. 15 é teto operacional escolhido para o Nexus, não resultado de paper (arXiv 2608.05810, preprint, só sustenta cautela com acúmulo sem filtro).
- **Registros legados:** L-001..L-007 são anteriores à v3.4: mantêm o status original sem refazer o teste; o campo `Evidência` foi acrescentado só quando ela já constava no texto. Evidência que não é PR/SHA/run verificável externamente fica marcada como legada (L-006).

## Regra de promoção
- Uma lição **repetida 2 vezes** ou de **alto risco** (estado compartilhado do GitHub, segurança, perda de dado, gate enfraquecido) deve ser promovida a regra nas instruções dos agentes **ou** a checagem automática (teste/gate).
- A promoção acontece em **PR próprio**, pequeno, com revisão humana; ao mesclar, o status muda para `promovida` com o link.
- Preferir checagem automática a regra em texto quando ambas forem viáveis (Inferência: regra em texto depende de o agente lê-la; o gate não).
- Não apagar entradas; lições obsoletas recebem uma nota datada.

## Entradas

### L-001 — Contador `tests: N` duplicado nos 4 documentos de handoff
- Data: 2026-10-21
- Verificador: `tools/state_consistency_audit.py` (job "Final state attestation") e `git merge`.
- Fato observado: todo PR que adicionava teste editava `tests: N` em `CONTINUATION.md`, `NEXUS_EVIDENCE.md`, `NEXUS_KNOWLEDGE.md` e `SECURITY_TRIAGE.md`; 23 de 27 PRs abertos tocavam esses arquivos e pares independentes conflitavam (ex.: #52 × #53 com `git merge-tree`).
- Causa (Fato): o gate exigia número literal igual ao manifesto; o número mudava em todo PR.
- Correção: #62 — sentinela `tests: CURRENT_TESTS`, resolvida a partir do manifesto do CI; sentinelas passaram a valer só no próprio campo.
- Regra/checagem derivada: não copiar números gerados pela execução para documentos versionados; usar sentinela resolvida pelo gate. Contrato `test_state_counter_sentinel_contract.py`.
- Evidência: #62; conflito #52 × #53 (`git merge-tree`).
- Status: promovida (gate + contrato, #62).

### L-002 — Linhas soltas no `NEXUS_EVIDENCE.md` após resolver conflito
- Data: 2026-10-21
- Verificador: nenhum falhou — o gate só lê o bloco `NEXUS-CURRENT-STATE` completo; detectado por inspeção (`grep -c '^tests: '` = 2).
- Fato observado: o merge de `main` na branch do #54 (`d074c15`) deixou `tests: CURRENT_TESTS`, `bandit_low: 131` e `-->` fora do bloco; chegou à `main` e foi removido no #63.
- Causa (Fato): o trecho em conflito incluía o fim do bloco de estado + narrativa; o script de resolução concatenou os dois lados (união), duplicando o fim do bloco.
- Correção: #63; script de resolução endurecido (mantém só o fim do bloco vindo de `main`, une a narrativa e recusa qualquer `tests:`/`bandit_low:` fora do bloco).
- Regra/checagem derivada: após resolver conflito em documento com bloco de estado, verificar `grep -c '^tests: ' == 1` e ausência de marcadores `<<<<<<<`/`>>>>>>>` antes do push. Candidata a checagem automática: o gate falhar se `tests:`/`bandit_low:` aparecer fora do bloco (Inferência: teria barrado o #54).
- Evidência: #54 (merge `d074c15`), #63.
- Status: registrada (legada; promoção sugerida: gate, por alto risco).

### L-003 — Pipe engoliu código de saída e marcou o #58 como pronto
- Data: 2026-10-21
- Verificador: `git merge` (conflito real em `tools/targeted_mutation.py`) e o GitHub, que recusou o merge ("Pull Request has merge conflicts").
- Fato observado: num laço de automação, `update.sh N | grep ...` retornou sucesso apesar de o script ter abortado por conflito; o passo seguinte marcou o #58 como pronto e tentou mesclar. Nada foi mesclado; o #58 voltou a rascunho com `gh pr ready --undo`.
- Causa (Fato): sem `pipefail`, o status de um pipe é o do último comando (`grep`), não o do script.
- Correção: comandos que alteram estado do GitHub passaram a rodar com `set -euo pipefail` e sem pipes no caminho de decisão; "pronto" e merge só depois de CI 2/2 verde confirmado no head enviado.
- Regra/checagem derivada: toda automação que faz push, marca pronto, mescla ou fecha PR roda com `set -euo pipefail`; nunca filtrar com `| grep` a saída de um passo cujo código de saída decide o próximo.
- Evidência: #58 (conflito em `tools/targeted_mutation.py`; merge recusado pelo GitHub).
- Status: registrada (legada; promoção sugerida: instruções, por alto risco).

### L-004 — PR do registro misturou workflow com docs (#67)
- Data: 2026-10-21
- Verificador: nenhum; detectado por revisão contra a v3 §11 item 6 ("Workflows, dependências/lock e permissões: PR dedicado, nunca misturado").
- Fato observado: o #67 (mesclado como `3c3b038`) alterou `.github/workflows/nexus-contract-gate.yml` junto com docs, schema, validador e testes.
- Causa (Inferência): o pedido descrevia tudo como um único entregável e a regra não foi checada antes do commit.
- Correção: os PRs seguintes separam workflow de docs (um tema por PR).
- Regra/checagem derivada: antes do commit, se `git diff --cached --name-only` incluir `.github/workflows/` junto com outro caminho, dividir o PR. Candidata a checagem automática no `predeleg.sh`.
- Evidência: #67 (merge `3c3b038`).
- Status: registrada.

### L-005 — Replay sem casos ou sem leitura retornava sucesso
- Data: 2026-10-21
- Verificador que deveria ter falhado: `tools/agente/replay/run_replay.sh`; dependência incompleta do job `Final state attestation` no candidato da #69.
- Fato observado: no head `ca36166a8bc4f35fc1036867d01e8f6f7054d855`, arquivo vazio, comentários, caminho ausente, diretório e arquivo sem permissão de leitura retornaram 0. O replay normal tinha 11 casos e passou; não houve falha observada naquele run. A atestação dependia apenas do `contract-gate`.
- Causa (Fato): `fail=0` permanecia intacto quando nenhuma iteração ocorria ou a leitura falhava; tipo desconhecido podia reutilizar `rc` anterior. `needs`/`if` não incluíam o replay.
- Correção local: snapshot com leitura verificada, mínimo de um caso executado, rejeição de tipo desconhecido e atestação condicionada ao sucesso dos dois jobs; regressões em `test_agent_replay_contract.py`. Base e logs na nova seção correspondente de `CONTINUATION.md`.
- Regra/checagem derivada: aprovação de um replay exige execução não-vazia e leitura concluída; atestação final deve depender de todos os gates que declara cobrir.
- Evidência: head `ca36166a8bc4f35fc1036867d01e8f6f7054d855` (candidato da #69).
- Status: registrada; patch local isolado, sem publicação nem promoção permanente. Revisão de Segurança única solicitada sobre o diff congelado; conclusão entregue no handoff externo.
### L-006 — Teste com bash ocultou perda do modo executável
- Data: 2026-10-21
- Verificador: revisão independente de Segurança do patch local de replay; testes anteriores chamavam `bash` explicitamente.
- Fato observado: o patch `a175daa4d98704bd0c4a608d18c0517ffffdf62ace488e598b1fd481276df9fd` foi reprovado por mudar o modo Git do runner de `100755` para `100644`. Nesta retomada, execução direta retornou 126, Permission denied; a nova regressão falhou antes da correção.
- Causa (Fato): invocar `bash script` não exige o bit executável do script e não exercita a forma direta documentada.
- Correção local: restaurado somente o modo `100755` do runner, sem alterar seu conteúdo; acrescentado `test_documented_runner_is_directly_executable`, sem `bash` nem chmod no teste, sobre os 11 casos padrão.
- Regra/checagem derivada: testar a forma de invocação documentada e incluir modos Git na identidade do diff congelado.
- Evidência: patch `a175daa4d98704bd0c4a608d18c0517ffffdf62ace488e598b1fd481276df9fd` (reprovado). Evidência legada anterior à v3.4: hash de patch local, não commit/PR/run verificável externamente.
- Status: registrada (legada), sem promoção permanente; patch reprovado preservado como histórico, novo diff sujeito à nova revisão final independente de Segurança.

### L-007 — Formato date no schema não era aplicado pelo validador
- Data: 2026-10-21
- Verificador que deveria ter falhado: `tools/agente/validar_registro.py` e contrato do registro na base `b2b01f2c6b044f0f7f1b60ec1e9c0d615fa4d296`.
- Fato observado: o RED de `test_agent_registry_contract.py` aceitou datas impossíveis, formatos alternativos, bool, lista, null e objetos de data carregados do YAML. Ausência já era rejeitada; o controle válido continuou aceito.
- Causa (Fato): o validador consultava os campos `required` do schema, mas não aplicava `type`/`format` de `last_seen`.
- Correção local: checar string, formato ASCII exato e calendário real, sem coerção; regressões e mutantes específicos no mesmo patch autorizado do registro Axis. Logs e comandos na nova rodada de `CONTINUATION.md`.
- Regra/checagem derivada: declaração de formato em schema não substitui teste negativo do caminho efetivamente executado; parser ISO sozinho também aceita formatos compactos/semanais.
- Evidência: base `b2b01f2c6b044f0f7f1b60ec1e9c0d615fa4d296`; correção no PR #70 (head `10a48e81316b3f13c1cff414bbaa6dc91cf791a1`, merge `a663a55a560e00ae818e9c8f60aacd4cf9bd93f7`).
- Status: registrada; sem promoção das instruções, publicação ou alegação de validação genérica de todo o schema.

## Histórico local anterior à recomposição de 22/10/2026

As entradas L-008–L-010 abaixo provêm da tree local `697923975efbee368dece179b9139e840919d785`, anterior à adoção local da v3.4.1. Preservadas sem duplicação e sem promoção; os resultados citados são históricos, não evidência desta nova composição. Não são lições novas ativadas pelo teste dos três casos da v3.4.1.

### L-008 — Composição verde ainda aceitava mutação parcial e falhas operacionais
- Data: 2026-10-21
- Verificador: revisão independente BLOCK da árvore local `5d766cfa057861f4533d4d333c5bd35ffd5f3b1f`; gates anteriores não cobriam os três casos.
- Fato observado: chave não-string em `add_relation` deixava ledger/entidades/índice alterados antes do TypeError NetworkX; merge-tree 128 vazio era aprovado como docs; status Git 128 vazio ou nuvem inesperada era aprovado como modo nuvem.
- Causa (Fato): cópia de dict não valida kwargs; saída vazia era tratada como evidência de ausência de conflito/sujeira sem validar o status do processo; estado da nuvem não tinha domínio fechado.
- Correção local: preflight de kwargs antes das gravações, classificação explícita de exit/saída de merge-tree e validação do status Git/domínio da nuvem. Contratos `test_graph_relation_preflight_contract.py` e `test_agent_guards_contract.py`; 7 mutantes adicionais. RED, GREEN e gates na nova seção de CONTINUATION e no diretório externo `fixes-review-block`.
- Regra/checagem derivada: snapshot de entrada não substitui validação da chamada efetiva; sucesso exige status explícito, não apenas stdout vazio. Testar falha operacional também com saída aparentemente válida.
- Status: registrada; correções apenas na candidata local em merge, sem commit/publicação nem promoção das instruções permanentes.

### L-009 — Comparar textos de globs não detecta sua interseção
- Data: 2026-10-21
- Verificador que deveria ter falhado: `tools/agente/validar_registro.py` e contratos do registro na árvore composta `d60f157162d6d79a9543b73d1047a05b4f44bef2` (bug herdado de main).
- Fato observado: `src/a?.py` e `src/?b.py` cobrem `src/ab.py`, mas helper, `validate()` e CLI aprovavam os dois claims ativos. RED reproduzível guardado em `glob-overlap-fix/red.log` no diretório externo da composição.
- Causa (Fato): `fnmatch` aplicava um padrão ao texto do outro; nenhuma das duas direções decide interseção de linguagens.
- Correção local: produto de autômatos com transições de estrela e alfabeto simbólico Unicode; regressões de duas linguagens, controles disjuntos, API/CLI sem arquivo materializado e expiração, além de nove mutantes específicos. A revisão independente encontrou custo quadrático sem limite no primeiro incremento: acrescentado orçamento compartilhado com exaustão explícita fail-closed e cinco regressões de recursos RED/GREEN, incluindo scanner de classes sem fechamento. Comandos e resultados na nova seção de CONTINUATION.
- Regra/checagem derivada: testar sobreposição de padrões entre si, não somente glob contra caminho literal; não enumerar apenas arquivos já existentes para decidir leases sobre arquivos futuros.
- Status: registrada; somente candidata local, sem commit/publicação nem promoção das instruções. BLOCK da árvore anterior não equivale a aprovação independente da próxima.

### L-010 — Preflight de método bound omitiu o parâmetro self
- Data: 2026-10-21
- Verificador: revisão independente integral BLOCK da tree `972747bebe1b9a16d79b158ab11caf9365cd618a`; 545 testes/112 mutantes anteriores não cobriam a chave `self`.
- Fato observado: novas regressões falharam em quatro combinações vazio/populado × com/sem store e no predicado funcional. A chamada NetworkX rejeitava `self` tarde demais, após mutações de índices/entidades/ledger, inclusive fechamento do fato anterior.
- Causa (Fato): o preflight cobria `u_of_edge`/`v_of_edge`, mas não o parâmetro implícito `self` de `DiGraph.add_edge`.
- Correção local: incluir `self` na guarda anterior a qualquer backend mutante; regressões por snapshot completo, ausência de chamadas mutantes e fallback positivo; mutante específico remove somente essa nova guarda. Evidências em `/workspace/pr46-composition-evidence-20261021/self-collision-fix/` e nova seção de CONTINUATION.
- Regra/checagem derivada: colisões de kwargs de métodos bound incluem o receptor implícito; uma exceção esperada não demonstra atomicidade — comparar todo o estado e, quando pertinente, comprovar que o backend não foi chamado.
- Status: registrada; sem promoção das instruções, commit ou publicação. Revisão BLOCK anterior não aprova a próxima tree; limite de atomicidade restrito a entradas conhecidas, não falhas arbitrárias do backend.
