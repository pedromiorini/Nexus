# Política de segurança

O Nexus é um **protótipo de pesquisa** e laboratório público. Não o use em produção nem em
contextos críticos para segurança. As proteções do código (detector de jailbreak, guardas
éticos, log encadeado) são heurísticas: não oferecem garantias e não substituem controle de
acesso, isolamento e revisão humana.

## Versões suportadas

Somente a branch `main`.

## Como reportar uma vulnerabilidade

Use o **relato privado de vulnerabilidades do GitHub**: aba *Security* → *Report a
vulnerability*. Não abra issues públicas com detalhes exploráveis. Se a opção não aparecer,
o mantenedor precisa habilitá-la em *Settings → Code security → Private vulnerability
reporting*. As respostas são feitas em regime de melhor esforço.

## Escopo útil

- Execução de código, injeção de SQL ou escrita fora dos caminhos esperados.
- Adulteração não detectada do log encadeado (`RealImmutableLog.verify_chain`).
- Segredos ou dados pessoais expostos no repositório.
- Contornos do detector de jailbreak são bem-vindos como **testes de regressão**
  (corpus em `test_jailbreak_detector_contract.py`), não como vulnerabilidades: a detecção
  por palavras-chave é, por desenho, uma primeira camada.
