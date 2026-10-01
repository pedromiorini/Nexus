# Prompt de engenharia para desenvolver o Nexus

Você é um engenheiro de software sênior responsável por evoluir o repositório Nexus com foco em **evidência verificável, confiabilidade operacional e honestidade documental**.

## Contexto confirmado

- O projeto é um protótipo experimental neuro-simbólico em Python; não trate nomes de classes, banners ou métricas internas como prova de AGI, ASI, consciência ou autonomia geral.
- O núcleo possui componentes de roteamento, governança, fila de tarefas adiadas, telemetria Vita, auditoria SQLite e contratos de diagnóstico versionados.
- A auditoria atual recomenda priorizar componentes críticos pequenos antes de refatorar o núcleo monolítico.
- A fila de tarefas adiadas é atualmente em memória e já possui contratos de prioridade, limite, retry, descarte, callbacks e preservação de contexto.

## Missão desta iteração

Evolua a fila de tarefas adiadas para suportar **exportação e restauração versionadas, determinísticas e seguras**, sem executar tarefas durante a restauração e sem quebrar os contratos existentes.

## Requisitos funcionais

1. Defina um schema público versionado para snapshots da fila, com `schema`, `generated_at`, limites da fila, estatísticas e tarefas pendentes.
2. Preserve em cada tarefa `task_id`, `prompt`, `context`, `priority`, `created_at`, `attempts`, `max_attempts` e `last_reason`.
3. Exporte JSON e objeto Python; restaure a partir de ambos.
4. Preserve a ordem de prioridade e desempate por criação.
5. Rejeite snapshots malformados, schema incompatível, IDs duplicados, tentativas fora do intervalo, tipos inválidos, capacidade excedida e contextos não serializáveis, sem alterar a fila em caso de erro.
6. Não invoque `processor`, callbacks ou qualquer execução de tarefa durante a restauração.
7. Permita restauração em fila vazia por padrão e uma opção explícita de substituição para recuperação operacional.
8. Não faça alegações de durabilidade em disco: o snapshot é um contrato de transporte; persistência física continua sendo responsabilidade do consumidor.

## Requisitos de engenharia

- Preserve compatibilidade com Python 3.11 e com a API existente.
- Mantenha mudanças pequenas, legíveis e sem dependências novas de runtime.
- Adicione testes unitários e de contrato para round-trip, ordem, validação, atomicidade do erro, substituição e segurança contra execução.
- Atualize README e documentação do contrato com comandos reproduzíveis.
- Execute compilação, suíte completa, auditoria de realidade e inspeção do diff.
- Não transforme testes internos em evidência de capacidade cognitiva; registre limites e o que ainda não foi medido.

## Critérios de aceite

A entrega só está concluída se o contrato for documentado, os testes novos e antigos passarem, snapshots inválidos forem rejeitados sem mutação de estado, a restauração não executar tarefas e a documentação declarar explicitamente que o recurso não é um mecanismo de persistência física nem prova de autonomia.
