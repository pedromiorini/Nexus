---
name: contract-first-bugfix
description: >-
  Use when fixing a bug in fragile or monolithic code where the fix must be
  proven by a failing contract test first (strong inference).
---
# Correção com contrato primeiro
1. Descreva o fenômeno só com observações e um comando de reprodução.
2. Formule 2–5 hipóteses concorrentes, cada uma com condição de falsificação e custo; use git log/blame e call sites para excluir.
3. Escreva um contrato independente (`test_*_contract.py`) com casos negativos e tipos inválidos (bool como número, NaN, inf, negativos), falha fechada; property-based quando couber. Mostre-o falhando em commit separado.
4. Aplique a menor correção compatível e reversível; sem mudanças de passagem.
5. Rode lint, suíte completa e integrada; opcionalmente mutation testing só no módulo alvo.
6. Registre data, commit, comandos, resultados, hipóteses excluídas/sobreviventes e limitações nos docs de evidência/continuidade.
7. PR sem merge, listando o que segue não verificado.
