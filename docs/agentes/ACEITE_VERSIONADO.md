# Aceite gradual com artefatos verificáveis

Uma mudança nas instruções atuais não representa aceite dos agentes. O registro fixa **qual versão, quais bytes e qual arquivo** cada agente declarou aceitar. Versão anterior verificável gera aviso; falta de evidência continua bloqueando o gate.

O schema 2 foi integrado pela #113 na main `cb74cad7cfcd0ab536178ed984989ef2425c8211`. Esta composição seleciona a versão atual 3.4.2 e registra o aceite explícito do Grok e a declaração direta do owner para Axis, ambos para o artefato exato abaixo. Os dois registros Cursor permanecem em 3.4.1. Este registro não atesta o head, checks ou mergeabilidade remotos da #79; confirme esses estados no GitHub. Publicação e integração são ações distintas que exigem autorização específica.

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

## Estado da proposta da #79 antes do aceite direto do Axis

A origem dos bytes é `docs/agentes/INSTRUCOES_AGENTE_NEXUS_v3.md` no head `d1a4e6d2e5c0d8ba3254b44f73360e7592b7e832` da #79. O ponto de entrada e `docs/agentes/versoes/3.4.2/8f03762d2a2d128eea955597f1b55210e387c08385bdc5544514ea2f39701b5d.md` contêm exatamente esses bytes. A entrada 3.4.2 é acrescentada ao catálogo; o artefato/catalogação 3.4.1 é preservado. No head remoto `e4ae8d9`, Axis ainda estava em 3.4.1 e somente o trio arquivo/SHA/versão aceita do Grok fora atualizado. Esse é o estado histórico da proposta, não o estado desta candidata.

O §9C.3 introduz claims no branch do PR, removidos no último commit antes do merge. A leitura dos claims das PRs abertas e a checagem de sobreposição continuam manuais; o validador lê apenas o arquivo local. Na revalidação desta candidata, `CLAIMS.yaml` da base `1795ee0`, da #79 (`e4ae8d9`) e da #197 (`f2d9dca`) estavam vazios; a #198 já estava mesclada, e a candidata mantém `claims: []`. O claim do Grok mencionado em registros anteriores não está ativo nesses estados atuais.

**Compatibilidade textual:** o §9C.4 do snapshot ainda diz que mudança na v3 exige atualizar hashes no mesmo PR. Essa redação é legada do caminho mutável/schema 1; não se edita o artefato exato para corrigi-la. Sob schema 2, a promoção exige snapshot, catálogo e ponto de entrada coerentes, nunca bump automático de `versao_aceita` ou SHA. A interpretação operacional, isoladamente, não constitui aceite; o aceite de Axis desta candidata decorre somente da declaração direta abaixo. Uma correção do texto exigiria nova versão/artefato e aceites correspondentes.

O Grok declarou aceite do SHA exato em [#78, comentário 6367039207](https://github.com/pedromiorini/Nexus/issues/78#issuecomment-6367039207). O SHA literal é `8f03762d2a2d128eea955597f1b55210e387c08385bdc5544514ea2f39701b5d`, confirmado nos bytes e no caminho do artefato. Com autorização específica do owner para refletir esse aceite só no candidato local, o registro do Grok foi atualizado; isso não autoriza publicação. A declaração é assinada textualmente `de: grok-bot` sob a conta GitHub `pedromiorini`; o validador não autentica identidades. A permanência do Axis em 3.4.1 no head remoto e na proposta anterior refletia a ausência de evidência naquele momento; esse estado foi superado pela declaração abaixo.

## Aceite direto do owner para Axis — 2026-11-18

O owner declarou diretamente: **“Aceito a v3.4.2 para as instruções do Axis”** e autorizou registrar esse aceite para o artefato exato `docs/agentes/versoes/3.4.2/8f03762d2a2d128eea955597f1b55210e387c08385bdc5544514ea2f39701b5d.md`, SHA-256 `8f03762d2a2d128eea955597f1b55210e387c08385bdc5544514ea2f39701b5d`. Esta é a fonte de consentimento registrada para Axis; não se atribui ao agente uma declaração independente. No registro, só o trio de instruções de Axis muda dentro da entrada desse agente; identidade, UID, competências, skills, escopo, limites e demais metadados permanecem iguais, assim como o artefato histórico 3.4.1. Nesta composição, Grok e Axis ficam em 3.4.2; os dois agentes Cursor continuam em 3.4.1 e são os únicos avisos esperados. A autorização para registrar o aceite não incluía publicação; o owner autorizou separadamente o commit e a atualização da branch da #79, mantendo-a draft e sem merge.

## Como promover uma versão depois

Com autorização específica do owner, criar novo artefato de uma versão realmente publicada e acrescentá-lo ao catálogo; atualizar o ponto de entrada e `instrucoes_atuais` na mesma mudança. Não editar nem remover os artefatos antigos. Manter cada agente no aceite anterior até existir evidência explícita para aquele agente. O aviso permite essa transição sem falsificar consentimento.

Quando um agente ou o owner declarar expressamente aceite para um agente específico, mudar somente o trio versão/caminho/SHA correspondente e registrar a fonte. Não inferir aceite de leitura do arquivo, CI verde, título de PR ou aprovação genérica para implementar o gate. O diff da #79 que atribuiu 3.4.2 ao Axis não era evidência; a declaração direta acima é. Uma futura integração deverá partir desta candidata ou reconciliar sua evidência sem restaurar o estado antigo da base da #79.

Se faltar um artefato histórico, recuperá-lo de um commit confiável e conferir o SHA já declarado. Nunca reconstruir bytes por aproximação nem substituir o hash do agente para passar. Alternativamente, manter o schema 1 e exigir todos os aceites explícitos antes de promover instruções; não há modo permissivo sem evidência.

## Reversão e limites

Antes de publicar qualquer versão nova, esta migração pode ser revertida inteira, restaurando schema 1 e os caminhos originais sem mudar os aceites. Depois de haver aceites mistos, schema 1 volta a bloquear versões anteriores: não fazer downgrade automático nem atualizar todos os registros para contornar isso.

A identidade por conteúdo impede sobrescrever bytes mantendo o mesmo caminho/SHA, mas **não autentica o autor nem torna o Git fisicamente imutável**. Um editor autorizado ainda pode substituir simultaneamente catálogo, versão e aceites. Revisão do owner, proteção do histórico e evidência externa de consentimento continuam necessárias; o validador não consulta GitHub, assinaturas ou histórico Git. A vinculação histórica versão→SHA é política append-only de revisão, não uma raiz de confiança externa implementada aqui. Links são rejeitados no snapshot observado; não há proteção contra alteração concorrente do filesystem durante a leitura. Catálogos são dados confiáveis do repositório, sem teto novo de tamanho/recursos.

Os casos da classe `InstructionVersionsContract` usam dados sintéticos; `test_agent_pr79_composition_contract.py` verifica os bytes reais desta composição local. Nenhum dos dois publica uma versão ou autentica consentimento. A evidência é de contrato de software local, não de capacidade geral ou segurança completa.

## Verificação

```sh
python -m unittest -v test_agent_instruction_versions_contract test_agent_registry_contract test_agent_pr79_composition_contract test_grok_individual_acceptance_contract
python tools/agente/validar_registro.py --expirados aviso
bash tools/agente/replay/run_replay.sh
```

O teste de origem exige aceite antigo verificável após mudança do arquivo atual (aviso); o controle exige aceite vigente exato (OK sem aviso); semelhantes incompatíveis exigem erro (SHA errado, versão desconhecida/futura, artefato ausente). Os testes adicionais cobrem catálogo, ordenação numérica, tipos, links, CLI, ausência de escrita e compatibilidade legada. Como esta candidata altera registro, documentação e contratos, sem tocar módulos de produção, a mutação direcionada e os gates de segurança/realidade não se aplicam; resultados e limites dos gates compatíveis ficam na nova seção de `CONTINUATION.md`.
