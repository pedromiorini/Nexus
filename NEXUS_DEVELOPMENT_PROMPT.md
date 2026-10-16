# Prompt de engenharia — hardening do snapshot da fila Nexus

Você é um engenheiro de software sênior responsável por evoluir o repositório Nexus com foco em comportamento reproduzível, confiabilidade operacional e documentação honesta.

## Contexto confirmado

- Nexus é um protótipo experimental de arquitetura neuro-simbólica em Python; não trate nomes, banners, métricas ou testes internos como prova de AGI, consciência ou autonomia geral.
- A fila em `core/deferred_task_queue.py` já implementa o contrato `nexus.deferred_task_queue.v1`, exportação/restauração em objeto e JSON, restauração validada, limite de capacidade e opção `replace=True`.
- A missão histórica de implementar snapshots já está cumprida. Não duplique a funcionalidade nem altere o schema/API sem necessidade.
- O contrato e os testes atuais passam, mas dois defeitos foram reproduzidos: mutar o contexto aninhado do snapshot exportado também muta a tarefa da fila; e o validador aceita timestamp `NaN` e contador `True`.

## Missão desta iteração

Endureça as fronteiras de exportação/restauração do snapshot para que os dados sejam isolados e os valores numéricos sejam válidos, mantendo compatibilidade com Python 3.11 e com o schema/API existentes.

## Requisitos

1. Exportar e restaurar contextos sem compartilhar referências mutáveis com a fila nem com o objeto de snapshot recebido/devolvido.
2. Rejeitar timestamps `generated_at`/`created_at` não finitos (`NaN`, `Infinity`, `-Infinity`) e contextos que não possam ser codificados como JSON estrito.
3. Validar limites e contadores estatísticos como inteiros reais não negativos — `bool` não é inteiro válido para o contrato — e validar `max_attempts` das tarefas com a mesma regra estrita.
4. Validar o snapshot inteiro e preparar os objetos restaurados antes de alterar a fila; falhas também devem preservar integralmente estado, fila e estatísticas quando `replace=True`.
5. Preservar ordenação, tentativas, IDs, limites, estatísticas, callbacks e comportamento de execução já existentes. Restauração não executa tarefas.
6. Não adicionar dependências de runtime nem alegar persistência física. O snapshot continua sendo um contrato de transporte.
7. Adicionar testes de regressão para isolamento de referências, números inválidos, validação estrita de inteiros e atomicidade.
8. Atualizar a documentação pública do contrato e registrar comandos/resultados reproduzíveis.

## Execução e critérios de aceite

- Reproduza os casos antes do patch e confirme que os novos testes falham sem a correção.
- Execute testes de snapshot e regressão da fila, suíte completa, compilação Python, auditoria de realidade e inspeção do diff; execute mutation/security gates quando as ferramentas/dependências necessárias estiverem disponíveis.
- Não versionar relatórios temporários, bancos de teste nem artefatos de execução.
- Concluir apenas se testes novos e antigos passarem, snapshots inválidos não alterarem a fila, referências não vazarem entre fronteiras e a documentação distinguir transporte de persistência física.
