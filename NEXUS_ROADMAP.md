# Nexus — Roadmap técnico orientado por evidência

## Prioridade 0 — manter a fonte de verdade operacional

- Reexecutar testes, auditoria de realidade e Bandit em cada marco relevante.
- Atualizar `CONTINUATION.md` com commit, comando, resultado e limitações.
- Não usar números históricos como números atuais.

## Prioridade 1 — contratos e segurança

- Migrar incrementalmente asserts do bloco demonstrativo para contratos independentes.
- Manter auditoria AST impedindo novos asserts fora do bloco permitido.
- Fazer revisão manual de entradas externas, subprocessos e fronteiras SQLite.
- `RealHierarchicalMemory.retrieve` rejeita `bool`, tipos não inteiros e valores negativos antes de FAISS/SQLite; `limit=0` devolve vazio. Ainda não há teto para limites positivos altos; só adicionar um com contrato público explícito.
- `ToolDefinition.requires_permission` e `permission_level` são metadados, sem enforcement no caminho de execução; os built-ins atuais mantêm `requires_permission=False`. Antes de registrar ferramenta protegida, definir principal/contexto de autorização, fonte das concessões, semântica de níveis e falha fechada.
- `RealToolUse.execute_tool` exige um `dict` na fronteira externa; `format_json` usa serialização estrita (`allow_nan=False`) e converte `NaN`/`±Infinity` em falha estruturada. A geração da cache key é precedida por traversal iterativo com budgets de 64 KiB, profundidade 64, 4.096 ocorrências e 1.024 itens por contêiner, além de detecção de ciclos; bypass gera `TOOL_CACHE_KEY_UNAVAILABLE` e mantém o executor. Chaves elegíveis preservam encoder/hash legados; ferramentas não cacheáveis não serializam parâmetros.
- Se `copy.deepcopy` falhar ao preparar resultado para cache, a falha é registrada como `TOOL_CACHE_STORE_UNAVAILABLE` e somente o armazenamento é ignorado; o resultado bem-sucedido do executor é preservado. Isso não limita o custo do executor nem a memória global.
- `ToolValidator` resolve schemas built-in conhecidos para classes reais e aceita subclasses verdadeiras; `bool` continua inválido para `int`. `any`/`object` permanecem curingas, schemas desconhecidos mantêm matching legado por nome e a validação não é recursiva. O preflight do cache é independente e pode fazer bypass de subclasses sem bloquear o executor.
- Revisar budgets de ferramentas: `math_calculate` limita tamanho/complexidade e resultados numéricos; cache expurga TTL, limita entradas/payload e isola resultados mutáveis. `ToolDefinition.timeout_seconds` rejeita valores que não sejam inteiros em `[1, 60]` e registra `TOOL_EXECUTION_OVERRUN` quando o executor síncrono termina acima do budget, por retorno ou exceção. O diagnóstico não interrompe o trabalho; cancelamento genérico segue como lacuna arquitetural explícita.

## Prioridade 2 — reduzir risco do núcleo monolítico

- Escolher um componente pequeno por vez.
- Extrair somente após contrato público e teste de regressão.
- Medir efeitos laterais antes e depois; não iniciar uma reescrita ampla.

## Prioridade 3 — observabilidade

- Derivar novas métricas de eventos ou registros reais, nunca de constantes arbitrárias.
- Registrar disponibilidade e qualidade de backends opcionais.
- Separar métrica operacional de alegação cognitiva.
- Manter `benchmarks/` versionado, determinístico e com outputs retidos no CI; comparar números somente entre versões compatíveis do protocolo.
- **POMDP/belief state:** o filtro puro de um passo e o benchmark separado `nexus.pomdp-belief-benchmark.v2` estão implementados. O contrato aplica `b_{t+1}(s') ∝ O(o|s',a) Σ_s T(s'|s,a)b_t(s)`, valida prior/transições/likelihoods, limita o espaço a 128 estados, usa log-sum-exp e rejeita evidência nula. O scorer/CLI falha fechado se algum caso exceder a tolerância ou algum controle inválido for aceito; a seed do workflow é coberta diretamente por teste.
- A trilha de observação parcial do protocolo v1 permanece `unsupported_interface`; o v2 mede somente um modelo explicitamente fornecido e não reivindica aprendizagem, descoberta de estados ou planejamento de política. Referência: [Ross et al. (2011), JMLR](https://www.jmlr.org/papers/volume12/ross11a/ross11a.pdf).

## Prioridade 4 — dependências opcionais

- Documentar comportamento com e sem `psutil` e FAISS/SentenceTransformers.
- Testar fallback e backend real por contratos separados quando houver backend disponível.

## Fora do roadmap verificável

AGI, ASI, consciência, autonomia geral e alegações de inteligência geral não são objetivos demonstrados por este repositório e não devem ser usados como critérios de conclusão.
