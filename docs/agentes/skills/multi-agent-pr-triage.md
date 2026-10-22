---
name: multi-agent-pr-triage
description: >-
  Use when a repository has many open PRs from several AI agents and you need to
  find duplicates, conflicts, superseded PRs and what to review or close first.
---
# Triagem de PRs multiagente
1. Liste PRs abertos com autor/agente, branch, data, status de CI, mergeabilidade e arquivos tocados.
2. Agrupe por arquivos sobrepostos; marque duplicados, PRs substituídos por um de consolidação e conflitos antigos (>48 h).
3. Para cada PR: escopo único? testes que falhavam antes? enfraqueceu teste? mexe em CI/deps/permissões (exige revisão humana)? números de evidência reexecutados no commit?
4. Proponha ordem de merge (menor risco e sem sobreposição primeiro) e quais fechar/rebasear, com motivo.
5. Não faça merge, close ou comentário sem confirmação explícita do usuário; entregue a tabela e a recomendação.
