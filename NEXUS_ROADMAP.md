# Nexus — Roadmap técnico orientado por evidência

## Prioridade 0 — manter a fonte de verdade operacional

- Reexecutar testes, auditoria de realidade e Bandit em cada marco relevante.
- Atualizar `CONTINUATION.md` com commit, comando, resultado e limitações.
- Não usar números históricos como números atuais.

## Prioridade 1 — contratos e segurança

- Migrar incrementalmente asserts do bloco demonstrativo para contratos independentes.
- Manter auditoria AST impedindo novos asserts fora do bloco permitido.
- Fazer revisão manual de entradas externas, subprocessos e fronteiras SQLite.

## Prioridade 2 — reduzir risco do núcleo monolítico

- Escolher um componente pequeno por vez.
- Extrair somente após contrato público e teste de regressão.
- Medir efeitos laterais antes e depois; não iniciar uma reescrita ampla.

## Prioridade 3 — observabilidade

- Derivar novas métricas de eventos ou registros reais, nunca de constantes arbitrárias.
- Registrar disponibilidade e qualidade de backends opcionais.
- Separar métrica operacional de alegação cognitiva.

## Prioridade 4 — dependências opcionais

- Documentar comportamento com e sem `psutil` e FAISS/SentenceTransformers.
- Testar fallback e backend real por contratos separados quando houver backend disponível.

## Fora do roadmap verificável

AGI, ASI, consciência, autonomia geral e alegações de inteligência geral não são objetivos demonstrados por este repositório e não devem ser usados como critérios de conclusão.
