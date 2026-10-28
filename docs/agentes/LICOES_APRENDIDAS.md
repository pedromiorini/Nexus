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
- Reincidência (eixos separados; um não preenche o outro):
  - Falha-alvo: corrigida | não corrigida | não medido
  - Falhas parecidas: melhoraram | não melhoraram | mistas | não medido
  - Casos novos: melhoraram | pioraram | inalterados | não medido
- Propostas rejeitadas ou que não funcionaram: <texto> | nenhuma registrada
- Status: pendente | ativa | promovida (onde: instrução §/arquivo, ou teste/gate) | aposentada (data, motivo)
```

## Regra de evidência e ativação (v3.4.1)
- **Evidência bruta obrigatória:** toda lição cita PR, SHA e/ou ID de execução do CI que outro agente consiga conferir. Lição tirada só de resumo, handoff ou memória é **rejeitada** (política local; arXiv 2610.19910, preprint, só por analogia: mede ruído na evidência de evolução, não rejeição de handoff).
- **Status:** `pendente` (nova, não testada) → `ativa` (passou no teste dos três casos) → `promovida` ou `aposentada`. "Promoção sugerida" é nota, não status.
- **Teste dos três casos** (política local, sem verificação automática; não é o replay de guardas): critérios escritos no PR antes de rodar; passa só se o caso de origem fica corrigido, um caso anterior bem-sucedido não regride e um caso semelhante incompatível é rejeitado ou sinalizado.
- **Teto de 15:** contam só `ativa` e as legadas `registrada` (equivalem a `ativa`); `pendente`, `promovida` e `aposentada` não contam. Para ativar além de 15, aposente uma explicitamente (`Status: aposentada`, data e motivo), sem apagar o texto. 15 é teto operacional escolhido para o Nexus, não resultado de paper (arXiv 2608.05810, preprint, só sustenta cautela com acúmulo sem filtro).
- **Registros legados:** L-001..L-007 são anteriores à v3.4: mantêm o status original sem refazer o teste; o campo `Evidência` foi acrescentado só quando ela já constava no texto. Evidência que não é PR/SHA/run verificável externamente fica marcada como legada (L-006).

## Reincidência medida à parte

Corrigir a falha-alvo não preenche os outros eixos. Lição nova (`Status` começando por `pendente` ou `ativa`) registra os três, mais as propostas rejeitadas ou que não funcionaram. `não medido` é valor explícito; omitir o bloco nessas lições faz `tools/correction_claim_audit.py` falhar. Lição `registrada`, `promovida` ou `aposentada` sem o bloco permanece `nao_checavel`: o auditor reporta os três eixos como `não medido` e não os infere. L-001..L-007 não foram remedidas.

O replay `tools/agente/replay/run_replay.sh` não executa este auditor. Tipo desconhecido nele falha fechado de propósito, e a tabela cobre guardas de shell (L-001..L-003). O análogo, para lição, é o trio falha-alvo / falhas parecidas / casos novos, no lugar de origem / anterior / incompatível. Motivação externa, não replicada aqui: arXiv 2610.14662 (corrigir a falha-alvo não implicou melhora em casos novos) e arXiv 2610.35950 (declaração de PASS/correção não é observação).

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
- Nota 2026-10-22: o auditor `tools/state_consistency_audit.py` passou a falhar nessas linhas, com o mesmo critério de coluna 0 de `tools/agente/replay/check_resolution.sh`. Contrato `test_state_stray_marker_contract.py`, critérios dos três casos escritos no arquivo antes da correção. Antes: 7 testes, 18 falhas, exit 1; os controles de prosa e dos quatro documentos reais passaram (2/2). Depois: 7/7, exit 0. PR #75 (draft). Status permanece `registrada`: promoção a `promovida` só após merge e revisão humana. O catálogo `tools/targeted_mutation.py` não foi alterado (arquivo reservado pelos PRs #38–#50).

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
