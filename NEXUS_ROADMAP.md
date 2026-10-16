# Nexus — Roadmap técnico orientado por evidência

## Prioridade 0 — manter a fonte de verdade operacional

- Reexecutar testes, auditoria de realidade e Bandit em cada marco relevante.
- Atualizar `CONTINUATION.md` com commit, comando, resultado e limitações.
- Não usar números históricos como números atuais.

## Prioridade 1 — contratos e segurança

- Migrar incrementalmente asserts do bloco demonstrativo para contratos independentes.
- Manter auditoria AST impedindo novos asserts fora do bloco permitido.
- Fazer revisão manual de entradas externas, subprocessos e fronteiras SQLite.
- Revisar budgets de ferramentas: `math_calculate` agora limita tamanho/complexidade e resultados numéricos; cache expurga TTL, limita entradas/payload e isola resultados mutáveis. `ToolDefinition.timeout_seconds` ainda não interrompe executores genéricos e segue como lacuna arquitetural explícita.

## Prioridade 2 — reduzir risco do núcleo monolítico

- Escolher um componente pequeno por vez.
- Extrair somente após contrato público e teste de regressão.
- Medir efeitos laterais antes e depois; não iniciar uma reescrita ampla.

## Prioridade 3 — observabilidade

- Derivar novas métricas de eventos ou registros reais, nunca de constantes arbitrárias.
- Registrar disponibilidade e qualidade de backends opcionais.
- Separar métrica operacional de alegação cognitiva.
- Manter `benchmarks/` versionado, determinístico e com outputs retidos no CI; comparar números somente entre versões compatíveis do protocolo.
- Definir um contrato explícito de estado de crença antes de pontuar inferência POMDP no `RealPredictiveCodingSystem`:
  - atualização `b_{t+1}(s') ∝ O(o|s',a) Σ_s T(s'|s,a)b_t(s)`, seguida de normalização explícita;
  - espaço de estados finito, prior normalizado, transições por linha estocásticas e likelihoods em `[0,1]` fornecidos como entrada;
  - rejeitar dimensões incompatíveis, valores negativos/não finitos e evidência de observação zero; não fabricar posterior uniforme;
  - usar protocolo v2 para qualquer nova pontuação e não alegar que o filtro aprende o modelo ou resolve uma política. Referência: [Ross et al. (2011), JMLR](https://www.jmlr.org/papers/volume12/ross11a/ross11a.pdf).

## Prioridade 4 — dependências opcionais

- Documentar comportamento com e sem `psutil` e FAISS/SentenceTransformers.
- Testar fallback e backend real por contratos separados quando houver backend disponível.

## Fora do roadmap verificável

AGI, ASI, consciência, autonomia geral e alegações de inteligência geral não são objetivos demonstrados por este repositório e não devem ser usados como critérios de conclusão.
