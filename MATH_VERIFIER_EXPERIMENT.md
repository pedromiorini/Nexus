# Verificador de soma: contrato offline, não adaptação em produção

## Decisão

`math_calculate` já é uma ferramenta executável em `RealToolUse.execute_tool`.
Ela usa AST restrita e aritmética inteira de Python. Porém, o ciclo
`RealMetacognitionSystem.execute_with_monitoring` não seleciona nem altera o
executor dessa ferramenta. Instanciar ambos os componentes não cria essa ligação.
Não há alternativas de execução atribuíveis às estratégias existentes que
permitam medir uma melhoria causada pela adaptação. Creditar `analytical` por
uma soma correta seria confundir acerto da ferramenta com eficácia da estratégia.
Por isso, **nenhum callsite de produção foi adicionado e nenhuma adaptação foi
ativada**. O novo módulo e o harness são apenas contrato e experimento offline.
A API existente recebe `evaluator=`, não `result_evaluator=`; este último é o
nome local do callback explicitamente fornecido pelo harness.

## O que pode ser verificado

O domínio é somente `a + b`, com um espaço de cada lado do sinal, dois inteiros
ASCII não negativos canônicos de até 12 dígitos. Fora disso a fábrica devolve
`None`: não há fallback para autoavaliação, prosa ou heurística.

O oráculo usa `decimal.Context.add` com precisão isolada de 13 dígitos e traps
para operações inexatas/arredondadas. Não usa o parser AST, a implementação do
executor, resultados históricos, a saída do modelo ou seu sucesso autodeclarado.
A comparação aceita somente resultado inteiro exato; booleanos e floats não
recebem crédito. Falha de execução recebe zero. Wrapper/tipo/ferramenta inválidos
falham de forma explícita.

A independência é de implementação/algoritmo (decimal versus AST e inteiros de
Python), não de máquina ou cadeia de confiança: ambos rodam no mesmo processo
Python. O callback não é sandbox nem prova criptográfica de correção. O wrapper
operacional não traz request ID; o pareamento é responsabilidade do harness,
que precisa conferir IDs, expressão, cardinalidade e integridade antes do score.
Um chamador arbitrário pode forjar arquivos e hashes: os seals são controles
contra alteração acidental e vazamento de processo, não assinatura de um terceiro.

## Protocolo congelado antes dos resultados finais

1. Seleção do domínio, testes de desenvolvimento e revisão acontecem antes do
   holdout. Não há tuning de tolerância, estratégia ou threshold.
2. `freeze` cria um diretório novo com 128 pares únicos aleatórios, operandos
   de 10 a 12 dígitos, seed de entropia nova e manifesto com hashes das entradas,
   fontes operacionais e lock de dependências, hash do bundle de fontes, versão do
   harness/oráculo e versão exata do Python. Não importa nem invoca o callback,
   não calcula gabaritos e não usa nem abre splits oficiais.
3. `predict` valida os hashes e executa somente a rota real
   `RealToolUse.execute_tool("math_calculate", ...)`. Congela as previsões e seu
   seal **antes** de importar ou criar qualquer callback/oráculo para o holdout.
   `predict` público não aceita executor injetável e usa o executor real; somente
   um helper privado é usado pelos testes sintéticos.
4. `score` verifica o pareamento e o seal, abre os gabaritos pela primeira vez
   e faz replay offline no monitor com callback explícito. O process type
   `tool_integer_addition` não é aplicável a `analytical`; deve haver zero
   atualizações de estratégia. Pareamento exige cardinalidade idêntica, IDs únicos
   e completos e expressão exata associada a cada ID. As previsões finais não
   recebem feedback nem são repetidas para tuning.
5. Resultado ruim não autoriza editar fontes e reaproveitar os mesmos casos.
   Mudanças de implementação exigem outro protocolo e holdout ainda não aberto.
   Reproduzir o score dos arquivos arquivados é auditoria, não uma nova medição
   cega. Depois de `score`, este conjunto está exposto e nunca mais é intocado.

Semente, manifesto, hashes de fontes/versões, entradas, previsões e relatório
devem ser arquivados juntos. O commit de fontes é fixado antes do experimento; o
relatório pode ser adicionado em um commit posterior sem mudar as fontes congeladas.
O experimento mede acerto pareado da ferramenta e encadeamento do monitor,
**não** ganho de aprendizagem, comparação baseline/adaptado, robustez geral ou
capacidade de resolver problemas descritos em linguagem natural.

## Fundamento das escolhas

A [documentação de `decimal`](https://docs.python.org/3/library/decimal.html)
explica aritmética decimal exata, contexto de precisão e traps para inexatidão.
A [orientação do scikit-learn sobre vazamento](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
explica por que dados de teste não devem orientar escolhas de modelo ou tuning.
Essas referências fundamentam o contrato e a separação de fases; não são evidência
de que Nexus melhorou ou generalizou.

## Resultado final

Pendente: esta seção deve ser preenchida apenas depois de congelar as fontes e
as previsões e executar o score uma vez, sem alterações motivadas pelos labels.

## Parecer pré-holdout

**Apto para um ensaio controlado, não para alegações de melhoria ou adaptação.**
Os bloqueios centrais de vazamento estão cobertos pelo protocolo: `freeze` e
`predict` não chamam o oráculo; o score só importa/cria callbacks após validar o
seal; fonte, versão e ambiente são vinculados; pareamento é um-para-um; e o replay
usa evaluator explícito com verificação de procedência e zero crédito estratégico.
Ainda não há holdout final nem resultado. Hashes locais/seals não são assinatura
contra adulteração deliberada por quem controla os arquivos.

Antes de um holdout final: (1) revisar e fixar um commit limpo com harness,
oráculo e lock; (2) confirmar que a execução do ensaio usará apenas o CLI
`predict` (injeção do executor somente nos testes); (3) executar a suíte sintética
no mesmo Python/ambiente travado e arquivar o log; (4) congelar uma única vez,
selar previsões antes do score e arquivar todos os artefatos; (5) depois do score,
tratar os casos como expostos e abrir outro holdout independente para qualquer
mudança motivada pelos resultados. Não usar nem modificar labels finais/oficiais.
