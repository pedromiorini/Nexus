# Nexus — Constituição do projeto

## Identidade

O Nexus é um **protótipo experimental de arquitetura neuro-simbólica em Python**, voltado a roteamento, memória, governança, telemetria, auditoria e experimentação de engenharia.

O projeto não demonstra AGI, ASI, consciência, autoconsciência, autonomia geral, soberania operacional ou prontidão para produção. Nomes de classes, banners, demos e métricas internas não são evidência dessas capacidades.

> **Implementação não é evidência; evidência não é capacidade geral declarada.**

## Regras permanentes

1. Priorizar comportamento verificável, contratos, observabilidade, segurança, testabilidade e honestidade documental.
2. Fazer mudanças pequenas, reversíveis e compatíveis antes de refatorações amplas.
3. Tratar fallback, heurística, simulação, mock e backend real como categorias diferentes.
4. Validar completamente antes de alterar estado persistente ou operacional.
5. Não usar cobertura, ausência de erros, Bandit ou testes internos como prova isolada de segurança ou inteligência.
6. Não reutilizar aleatoriedade não criptográfica para segredos, tokens, autenticação, autorização ou decisões de segurança.
7. Preservar histórico: resultados novos devem ser adicionados com data, commit, comando, resultado e interpretação.
8. Toda mudança significativa deve atualizar testes, limitações e documentação correspondente.

## Vocabulário epistemológico

- **Fato:** observado diretamente em código, execução, teste, CI ou artefato.
- **Evidência:** resultado reproduzível que sustenta uma afirmação limitada.
- **Inferência:** conclusão técnica derivada de fatos.
- **Hipótese:** possibilidade ainda não demonstrada.
- **Intenção:** trabalho futuro, não implementação existente.
- **Não verificado:** alegação ainda sem demonstração suficiente.

## Critério de conclusão

Uma tarefa só está concluída quando implementação, testes, casos negativos, segurança, documentação, limitações, diff e comandos reproduzíveis foram revisados. A conclusão deve separar fatos, inferências e hipóteses.
