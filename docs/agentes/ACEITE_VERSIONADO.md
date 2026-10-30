# Aceite gradual com artefatos verificáveis

Uma mudança nas instruções atuais não representa aceite dos agentes. O registro fixa **qual versão, quais bytes e qual arquivo** cada agente declarou aceitar. Versão anterior verificável gera aviso; falta de evidência continua bloqueando o gate.

O schema 2 foi integrado pela #113 na main `cb74cad7cfcd0ab536178ed984989ef2425c8211`. Esta composição posterior da #79 é somente local, sem commit ou publicação: seleciona a versão atual 3.4.2 e preserva Axis em 3.4.1 e registra somente o aceite próprio do Grok em 3.4.2, após verificação de sua declaração explícita. A PR #79 publicada permanece separada e intacta; promover o texto não é aceite individual nem autorização para publicar.

## Contrato do schema 2

`instrucoes_atuais` mantém `arquivo` (ponto de entrada legível, ainda mutável) e `versao`. `instrucoes_versoes` é um catálogo de versões publicadas, cada uma com `arquivo` e `sha256` fixados. Cada agente mantém `instrucoes: {arquivo, sha256, versao_aceita}`; seu caminho deve ser o do artefato exato no catálogo, não o ponto de entrada mutável.

O caminho obrigatório do artefato é `docs/agentes/versoes/<versao>/<sha256>.md`. O SHA é calculado dos bytes, sem normalização de encoding ou newline. Alterar o conteúdo sob a mesma identidade falha. Não se sobrescreve um artefato; novos bytes exigem novo artefato e nova versão. O catálogo e os registros existentes devem ser preservados nas revisões futuras.

As versões do schema 2 são strings numéricas canônicas `MAJOR.MINOR.PATCH`, sem zeros iniciais, prefixos, espaços, pre-release ou build metadata. A comparação é numérica por componentes: `3.4.10` sucede `3.4.2`. Não há dependência SemVer adicional.

| Situação | Resultado |
|---|---|
| Aceite da versão atual, arquivo e SHA exatos | OK, sem aviso de versão |
| Aceite anterior presente no catálogo, arquivo e SHA exatos | OK, com aviso de desatualização |
| Versão anterior desconhecida, versão futura (mesmo catalogada), versão malformada | Erro |
| SHA do agente diferente do artefato aceito | Erro, não só aviso |
| Artefato ausente, diretório, link simbólico no arquivo ou em seus pais, conteúdo adulterado | Erro |
| Catálogo ausente/inválido ou versão atual ausente nele | Erro, sem fallback para arquivo atual |
| Ponto de entrada atual ausente ou com SHA diferente do artefato atual | Erro, mesmo se todos os agentes ainda aceitarem versão anterior |

Todos os artefatos catalogados são verificados, inclusive versões sem agentes. Catálogo não deve conter versões futuras. O ponto de entrada atual também deve ser relativo e canônico, sem `..`, caminho absoluto ou links simbólicos. Os artefatos não podem usar caminhos alternativos ao formato acima. Avisos de versão não suprimem outros erros; o CLI retorna 1 se existir qualquer erro.

`--expirados aviso` só muda o tratamento de claims expirados. Claims sobrepostos/indeclarados, UID, data e avisos de autores mantêm seu contrato. O validador é somente leitura: não altera aceites, hashes, claims ou arquivos.

## Compatibilidade

O schema 1 continua aceito com a regra anterior: todo agente precisa apontar para o arquivo atual e ter sua versão/hash. Não se relaxa o SHA nesse modo, pois ele não tem artefato histórico independente. Um registro legado válido continua válido; aceite anterior continua erro. `schema_version` deve ser inteiro exato 1 ou 2: booleanos, floats, strings e versões desconhecidas são rejeitados.

A mudança para schema 2 é explícita. Não há detecção implícita nem fallback quando o catálogo estiver incompleto. Consumidores antigos do validador não entendem os ponteiros versionados: atualizar o validador/schema antes ou junto da migração do registro. O JSON Schema descreve a estrutura; a validação semântica efetivamente executada está em `tools/agente/validar_registro.py`. Isso não é aplicação genérica de todo o JSON Schema a todos os campos legados.

## Histórico: migração incluída na #113, 3.4.1 sem novo aceite

1. Copiar byte a byte o ponto de entrada da main `d0aab245f6d0453428d1f530c82ddd23fed18c1b` para o artefato `docs/agentes/versoes/3.4.1/6778552adef5433c85bf9209b494456f4f497ffdf6f8e190b19eab45d54c20a6.md`.
2. Adicionar ao catálogo somente a versão `3.4.1` com esse caminho/SHA e selecionar schema 2.
3. Mudar somente os caminhos de `instrucoes.arquivo` dos dois agentes para esse artefato. `versao_aceita: 3.4.1`, hashes, identidade, UID, datas e demais metadados permanecem iguais. Isso realoca bytes já aceitos; não é novo aceite nem nova declaração do Grok.
4. Manter o ponto de entrada atual intacto e a versão atual em `3.4.1`. Não editar o texto canônico apenas para documentar a migração: isso mudaria justamente os bytes já aceitos.

## Composição local da #79: instruções atuais 3.4.2, aceites individuais distintos

A origem dos bytes é `docs/agentes/INSTRUCOES_AGENTE_NEXUS_v3.md` no head `d1a4e6d2e5c0d8ba3254b44f73360e7592b7e832` da #79. O ponto de entrada e `docs/agentes/versoes/3.4.2/8f03762d2a2d128eea955597f1b55210e387c08385bdc5544514ea2f39701b5d.md` contêm exatamente esses bytes. A entrada 3.4.2 é acrescentada ao catálogo; o artefato/catalogação 3.4.1 e o registro inteiro do Axis são preservados. Somente o trio arquivo/SHA/versão aceita do Grok passa ao artefato 3.4.2, conforme seu aceite explícito.

O §9C.3 introduz claims no branch do PR, removidos no último commit antes do merge. A leitura de claims nos PRs abertos e a checagem de sobreposição entre PRs continuam manuais; o validador lê apenas o arquivo local. O claim preexistente do Grok (Tool safety A/B, expira em 2026-10-26T06:00:00Z) é preservado integralmente; somente o cabeçalho recebe a política nova. Sua remoção depende de decisão própria, não deste transplante documental.

**Compatibilidade textual:** o §9C.4 da versão arquivada ainda diz que mudança na v3 exige atualizar hashes no mesmo PR. Essa redação é legada do caminho mutável/schema 1; não se edita o artefato exato para corrigir isso. Sob schema 2, a promoção exige snapshot + catálogo + ponto de entrada coerentes, nunca bump automático de `versao_aceita` ou SHA de agentes. Aplicar esta interpretação operacional não constitui aceite 3.4.2 por Axis ou Grok. Uma correção do texto exigirá nova versão/artefato e aceites separados.

O Grok declarou aceite do SHA exato em [#78, comentário 6367039207](https://github.com/pedromiorini/Nexus/issues/78#issuecomment-6367039207). O SHA literal informado é `8f03762d2a2d128eea955597f1b55210e387c08385bdc5544514ea2f39701b5d`, confirmado nos bytes e no caminho do artefato candidato. Após autorização específica do owner para refletir esse aceite somente no candidato local, o registro do Grok foi atualizado; isto não autoriza publicação. A declaração é assinada textualmente `de: grok-bot` sob a conta GitHub `pedromiorini`; o validador não autentica identidade de agentes por conta própria. Axis permanece em 3.4.1 sem aceite 3.4.2.

## Como promover uma versão depois

Com autorização específica do owner, criar novo artefato de uma versão realmente publicada e acrescentá-lo ao catálogo; atualizar o ponto de entrada e `instrucoes_atuais` na mesma mudança. Não editar nem remover os artefatos antigos. Manter cada agente no aceite anterior até sua declaração explícita, com caminho/SHA correspondentes. O aviso permite essa transição sem falsificar consentimento.

Quando um agente aceitar expressamente a nova versão, mudar apenas seu trio versão/caminho/SHA. Não inferir aceite de leitura do arquivo, CI verde, título de PR ou aprovação do owner para implementar o gate. Em particular, o diff da #79 atribuindo 3.4.2 ao Axis **não é evidência de aceite**. Uma integração futura deve reconciliar essa PR separadamente, preservando o aceite 3.4.1 do Axis enquanto ele não declarar outro.

Se faltar um artefato histórico, recuperá-lo de um commit confiável e conferir o SHA já declarado. Nunca reconstruir bytes por aproximação nem substituir o hash do agente para passar. Alternativamente, manter o schema 1 e exigir todos os aceites explícitos antes de promover instruções; não há modo permissivo sem evidência.

## Reversão e limites

Antes de publicar qualquer versão nova, esta migração pode ser revertida inteira, restaurando schema 1 e os caminhos originais sem mudar os aceites. Depois de haver aceites mistos, schema 1 volta a bloquear versões anteriores: não fazer downgrade automático nem atualizar todos os registros para contornar isso.

A identidade por conteúdo impede sobrescrever bytes mantendo o mesmo caminho/SHA, mas **não autentica o autor nem torna o Git fisicamente imutável**. Um editor autorizado ainda pode substituir simultaneamente catálogo, versão e aceites. Revisão do owner, proteção do histórico e evidência externa de consentimento continuam necessárias; o validador não consulta GitHub, assinaturas ou histórico Git. A vinculação histórica versão→SHA é política append-only de revisão, não uma raiz de confiança externa implementada aqui. Links são rejeitados no snapshot observado; não há proteção contra alteração concorrente do filesystem durante a leitura. Catálogos são dados confiáveis do repositório, sem teto novo de tamanho/recursos.

Os casos da classe `InstructionVersionsContract` usam dados sintéticos; `test_agent_pr79_composition_contract.py` verifica os bytes reais desta composição local. Nenhum dos dois publica uma versão ou autentica consentimento. A evidência é de contrato de software local, não de capacidade geral ou segurança completa.

## Verificação

```sh
python -m unittest -v test_agent_instruction_versions_contract test_agent_registry_contract test_agent_pr79_composition_contract test_grok_individual_acceptance_contract
python tools/agente/validar_registro.py --expirados aviso
python tools/targeted_mutation.py
```

O teste de origem exige aceite antigo verificável após mudança do arquivo atual (aviso); o controle exige aceite vigente exato (OK sem aviso); semelhantes incompatíveis exigem erro (SHA errado, versão desconhecida/futura, artefato ausente). Os testes adicionais cobrem catálogo, ordenação numérica, tipos, links, CLI, ausência de escrita e compatibilidade legada. Comandos/resultados da rodada ficam na nova seção de `CONTINUATION.md`; logs brutos e revisão independente são entregues separadamente.
