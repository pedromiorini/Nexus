---
name: repo-evidence-audit
description: >-
  Use when asked to audit a code repository end to end with reproducible,
  falsifiable evidence instead of opinions.
---
# Auditoria de repositório baseada em evidência
1. Clone/leia: estrutura, README, docs de continuidade/evidência, CI, dependências, git log, PRs abertos (arquivos tocados por cada um).
2. Reproduza o ambiente do CI (mesma versão de linguagem e lock) e rode lint, testes, suíte integrada, scanners de segurança e de dependências; registre comando + saída resumida + commit.
3. Se houver auditoria anterior, classifique cada achado como confirmado/refutado/não verificado.
4. Categorias: build, qualidade, segurança (permissões aplicadas no código, segredos, entrada externa), dependências, testes (cobertura ≠ correção; considere mutation testing), performance, DevOps/docs, coordenação multiagente (PRs duplicados/conflitantes).
5. Severidade Crítico/Alto/Médio/Baixo com arquivo:linha; plano priorizado com quick wins.
6. Linguagem não conclusiva: Fato, Evidência, Inferência, Hipótese, Não verificado. Nunca use números históricos como atuais.
7. Não altere código de produção; entregável é relatório (PR só com o relatório, se permitido).
