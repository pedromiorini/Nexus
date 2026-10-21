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
- Status: registrada | promovida (onde: instrução §/arquivo, ou teste/gate)
```

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
- Status: promovida (gate + contrato, #62).

### L-002 — Linhas soltas no `NEXUS_EVIDENCE.md` após resolver conflito
- Data: 2026-10-21
- Verificador: nenhum falhou — o gate só lê o bloco `NEXUS-CURRENT-STATE` completo; detectado por inspeção (`grep -c '^tests: '` = 2).
- Fato observado: o merge de `main` na branch do #54 (`d074c15`) deixou `tests: CURRENT_TESTS`, `bandit_low: 131` e `-->` fora do bloco; chegou à `main` e foi removido no #63.
- Causa (Fato): o trecho em conflito incluía o fim do bloco de estado + narrativa; o script de resolução concatenou os dois lados (união), duplicando o fim do bloco.
- Correção: #63; script de resolução endurecido (mantém só o fim do bloco vindo de `main`, une a narrativa e recusa qualquer `tests:`/`bandit_low:` fora do bloco).
- Regra/checagem derivada: após resolver conflito em documento com bloco de estado, verificar `grep -c '^tests: ' == 1` e ausência de marcadores `<<<<<<<`/`>>>>>>>` antes do push. Candidata a checagem automática: o gate falhar se `tests:`/`bandit_low:` aparecer fora do bloco (Inferência: teria barrado o #54).
- Status: registrada (alto risco → candidata a promoção como gate).

### L-003 — Pipe engoliu código de saída e marcou o #58 como pronto
- Data: 2026-10-21
- Verificador: `git merge` (conflito real em `tools/targeted_mutation.py`) e o GitHub, que recusou o merge ("Pull Request has merge conflicts").
- Fato observado: num laço de automação, `update.sh N | grep ...` retornou sucesso apesar de o script ter abortado por conflito; o passo seguinte marcou o #58 como pronto e tentou mesclar. Nada foi mesclado; o #58 voltou a rascunho com `gh pr ready --undo`.
- Causa (Fato): sem `pipefail`, o status de um pipe é o do último comando (`grep`), não o do script.
- Correção: comandos que alteram estado do GitHub passaram a rodar com `set -euo pipefail` e sem pipes no caminho de decisão; "pronto" e merge só depois de CI 2/2 verde confirmado no head enviado.
- Regra/checagem derivada: toda automação que faz push, marca pronto, mescla ou fecha PR roda com `set -euo pipefail`; nunca filtrar com `| grep` a saída de um passo cujo código de saída decide o próximo.
- Status: registrada (alto risco → candidata a promoção nas instruções).

### L-004 — PR do registro misturou workflow com docs (#67)
- Data: 2026-10-21
- Verificador: nenhum; detectado por revisão contra a v3 §11 item 6 ("Workflows, dependências/lock e permissões: PR dedicado, nunca misturado").
- Fato observado: o #67 (mesclado como `3c3b038`) alterou `.github/workflows/nexus-contract-gate.yml` junto com docs, schema, validador e testes.
- Causa (Inferência): o pedido descrevia tudo como um único entregável e a regra não foi checada antes do commit.
- Correção: os PRs seguintes separam workflow de docs (um tema por PR).
- Regra/checagem derivada: antes do commit, se `git diff --cached --name-only` incluir `.github/workflows/` junto com outro caminho, dividir o PR. Candidata a checagem automática no `predeleg.sh`.
- Status: registrada.

### L-005 — Replay sem casos ou sem leitura retornava sucesso
- Data: 2026-10-21
- Verificador que deveria ter falhado: `tools/agente/replay/run_replay.sh`; dependência incompleta do job `Final state attestation` no candidato da #69.
- Fato observado: no head `ca36166a8bc4f35fc1036867d01e8f6f7054d855`, arquivo vazio, comentários, caminho ausente, diretório e arquivo sem permissão de leitura retornaram 0. O replay normal tinha 11 casos e passou; não houve falha observada naquele run. A atestação dependia apenas do `contract-gate`.
- Causa (Fato): `fail=0` permanecia intacto quando nenhuma iteração ocorria ou a leitura falhava; tipo desconhecido podia reutilizar `rc` anterior. `needs`/`if` não incluíam o replay.
- Correção local: snapshot com leitura verificada, mínimo de um caso executado, rejeição de tipo desconhecido e atestação condicionada ao sucesso dos dois jobs; regressões em `test_agent_replay_contract.py`. Base e logs na nova seção correspondente de `CONTINUATION.md`.
- Regra/checagem derivada: aprovação de um replay exige execução não-vazia e leitura concluída; atestação final deve depender de todos os gates que declara cobrir.
- Status: registrada; patch local isolado, sem publicação nem promoção permanente. Revisão de Segurança única solicitada sobre o diff congelado; conclusão entregue no handoff externo.
### L-006 — Teste com bash ocultou perda do modo executável
- Data: 2026-10-21
- Verificador: revisão independente de Segurança do patch local de replay; testes anteriores chamavam `bash` explicitamente.
- Fato observado: o patch `a175daa4d98704bd0c4a608d18c0517ffffdf62ace488e598b1fd481276df9fd` foi reprovado por mudar o modo Git do runner de `100755` para `100644`. Nesta retomada, execução direta retornou 126, Permission denied; a nova regressão falhou antes da correção.
- Causa (Fato): invocar `bash script` não exige o bit executável do script e não exercita a forma direta documentada.
- Correção local: restaurado somente o modo `100755` do runner, sem alterar seu conteúdo; acrescentado `test_documented_runner_is_directly_executable`, sem `bash` nem chmod no teste, sobre os 11 casos padrão.
- Regra/checagem derivada: testar a forma de invocação documentada e incluir modos Git na identidade do diff congelado.
- Status: registrada, sem promoção permanente; patch reprovado preservado como histórico, novo diff sujeito à nova revisão final independente de Segurança.
