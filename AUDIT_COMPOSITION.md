# Nexus — Candidato local consolidado para auditoria

A composição usa a base exata `812ebd02b0c82ecc3169d620e36d25592b1d6727`, o `origin/main` local confirmado no início. A branch `audit/consolidated-38-45-20261018` vive exclusivamente em `/workspace/Nexus-consolidated-20261018`. Não houve consulta remota para atualizar esse inventário, push, PR, fechamento, rebase de origens nem alteração de `main`.

## O que foi incluído

Todos os heads abaixo são ancestrais do candidato por merges locais `--no-ff`, preservando as branches e os commits originais. A ordem local foi #44, #43, #38, #39, #40, #41, #42, #45; não se reutilizou a composição descartável `775a0cd` nem seu JSON POMDP não rastreado.

| PR | Branch de origem | Head exato |
|---|---|---|
| #38 | `feat/temporal-ledger-integration-2026-10-17` | `3315c2509478984e9ba2f42684053e0ae594261d` |
| #39 | `fix/knowledge-graph-fallback-hops-2026-10-17` | `7690b67d7aa48d69697b94388bd8f1717bc68031` |
| #40 | `fix/knowledge-graph-missing-nodes-2026-10-17` | `b3497f21f475582fdb2a704d3435ef2ee2acc646` |
| #41 | `chore/dependency-profiles-2026-10-17` | `f56c39c0894f03a8503ad6a79f1758c959212539` |
| #42 | `fix/metacognition-score-provenance-2026-10-17` | `504b0c1756e8447afdd8406caff4109134b51d18` |
| #43 | `fix/queue-snapshot-hardening-2026-10-17` | `1619b48c6822f2bdb10fae241c8ca03df2117751` |
| #44 | `fix/reconcile-exclusive-audit-2026-10-17` | `0643509639a6f7459cee945e6286840bf5ac12c5` |
| #45 | `fix/vram-integrated-2026-10-18` | `1bf03d7f43db317349dff6fefc1add3a014e52cc` |

O head #45 inclui a implementação `9b657011d0b0b0eec61fad53b3d353717c109e4d` e seu follow-up documental. Não é substituído pelo SHA de implementação isolado.

## Dependências mantidas separadas

| PR e head exato excluído | Decisão baseada nos manifests |
|---|---|
| #12 `4103c9e4bdd5dd5e90efe490c031f9bf6114d936` | Ruff 0.17.0 muda input/dev, mas não o lock 0.16.10. Não é requisito funcional. Atualizar input e lock juntos em proposta separada. |
| #13 `4853df0056504477b3fa6cf757e0d8f8ec4ee0e2` | Hypothesis 6.169.1 no dev é anterior ao 6.170.0 já fixado no lock e alinhado por #41. Não rebaixar o perfil para preservar uma proposta obsoleta. |
| #14 `6ddd840e265a30487a108359450551aa80fbc5ad` | Parte de `5b8bed8125002e25eb066620af67d3a65d44e59f`; eleva runtime para NumPy >=2.5.4 e dev para 2.5.4 sem atualizar o lock 2.4.6. O runtime >=1.26,<3 já permite 2.5.4; elevar o mínimo exige justificativa de compatibilidade e validação própria. |

O perfil #41 fica integral: runtime `numpy>=1.26,<3`; CI/dev NumPy 2.4.6, Hypothesis 6.170.0, Ruff 0.16.10, NetworkX 3.6.1, Coverage 7.16.2, Bandit 1.9.4; Zizmor 1.31.1. A instalação de teste usa o lock com hashes. Não se alegam vulnerabilidades ou compatibilidade de versões não testadas.

## Reconciliação mínima

- **#38/#41:** conservar o parâmetro keyword-only `temporal_store` e corrigir apenas a docstring que dizia que NetworkX pertencia à biblioteca padrão. Nenhuma nova dependência de runtime.
- **#39/#40:** o primeiro ensaio conjunto encontrou quatro falhas de subteste porque #39 ainda exigia `NodeNotFound` ou um caminho fantasma para um ID inexistente. #40 define explicitamente resultado vazio em ambos os backends. O teste herdado foi alinhado a esse contrato e reforçado para verificar contadores e ausência de criação de entidades; os casos de distância real, ciclos e limite de saltos continuam intactos. Não se alterou produção para satisfazer expectativa obsoleta.
- **#43/#44:** conservar staging atômico/JSON estrito, redação dos detalhes de erro e classificação de timeout como inconclusivo/bloqueante. Catálogos de mutação de snapshot e temporal foram acumulados, não substituídos.
- **Documentos:** preservar registros históricos e distinguir o estado combinado no início dos quatro documentos compartilhados. D-029 temporal e D-030 snapshot continuam; os IDs colidentes de metacognição tornam-se D-031/D-032. D-033 registra esta composição. Relatórios de realidade são regenerados, não resolvidos escolhendo contagens de uma PR.
- **Regressões adicionais:** `test_consolidated_pr_contract.py` cobre store reidratado + preflight do grafo em ambos os backends, consultas ausentes/fora do orçamento sem escrita temporal e retry + snapshot + restore sem persistir detalhes privados ou compartilhar contexto exportado.

## Evidência combinada

Os resultados finais são registrados abaixo após a execução completa. Logs, ambiente isolado, relatórios, manifesto pós-commit e revisão congelada ficam fora da árvore versionada em `/workspace/nexus-consolidation-evidence/`. O manifesto é gerado no commit final; `ci_run=not_ci` distingue esta execução dos checks remotos históricos.

- **Python 3.11.16 / lock instalado com hashes:** suíte final 416 executados, 416 aprovados, 0 falhas/skips/XPASS. O primeiro coverage run tinha 413 testes antes da adição das três regressões específicas; a rodada final de cobertura inclui as 416.
- **Integração:** 31 IDs inventariados, 30 aprovados e 1 skip explícito por FAISS opcional ausente; nenhum ID faltante/inesperado.
- **Cobertura branch:** fila 91%, VRAM 94%, temporal 94%, Vita 76%, núcleo 69%; pisos 90/90/90/70/62 mantidos.
- **Mutação canônica:** 84 declaradas/aplicadas/mortas; 0 sobreviventes, timeouts e não aplicadas. **Suplemento VRAM:** 10/10 mortas, igualmente sem inconclusivas.
- **Segurança:** Bandit 135 LOW, 0 MEDIUM/HIGH; a saída 1 do scanner reflete achados LOW, não falha da política de severidade. Triagem regenerada em formato canônico e conferida, B101 e auditoria de lacunas aprovados; nenhuma supressão nova. A conferência inicial da triagem detectou o parágrafo manual inserido acima da área gerada; ele foi retirado da triagem (permanece nos outros documentos), sem modificar o gerador ou ignorar o gate.
- **Demais gates:** Ruff 0.16.10, actionlint 1.7.12, Zizmor 1.31.1, compileall, requisitos, realidade e consistência documental aprovados. A realidade é regenerável sem diff após o commit.
- **Benchmarks sintéticos herdados:** POMDP v1 continua `unsupported_interface`; v2 128/128 dentro de `1e-12`, erro absoluto máximo `2.7755575615628914e-16`, controles inválidos 4/4 rejeitados. O v1 também mantém controles negativos com falsos positivos: isso não é resolvido por esta composição nem valida raciocínio causal geral.
- **Proveniência final:** o hash do candidato é o `repository.commit` do `STATE_MANIFEST.json` externo; a revisão independente acontece apenas após o congelamento do diff. Nenhuma CI remota foi executada.

### Reprodução

No Python 3.11, crie um ambiente novo e instale `requirements-ci.lock` com `--require-hashes`. Use `PYTHONPATH=.` e `NEXUS_AUDIT_DB` fora da worktree. O executor externo `validate.sh` registra comandos, códigos de saída e logs individuais. As etapas principais são:

```bash
python -m unittest -v test_consolidated_pr_contract test_knowledge_graph_hops_contract
coverage run --branch -m unittest discover -p 'test_*.py'
NEXUS_INTEGRATED_SUMMARY=integrated-summary.json coverage run --append --branch core/constitutional_brain.py
coverage report --include=core/deferred_task_queue.py --fail-under=90
coverage report --include=core/vram_defense_guard.py --fail-under=90
coverage report --include=core/temporal_facts.py --fail-under=90
coverage report --include=vita/nexus_constitutional_bridge_v3.py --fail-under=70
coverage report --include=core/constitutional_brain.py --fail-under=62
python tools/targeted_mutation.py
ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py
actionlint .github/workflows/*.yml
zizmor --offline --no-progress .github/workflows
python -m compileall -q core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py
python tools/reality_audit.py
bandit -r core vita tools -f json -o bandit-report.json
python tools/security_triage.py bandit-report.json SECURITY_TRIAGE.md --check
python tools/embedded_assert_audit.py bandit-report.json core/constitutional_brain.py
python tools/implementation_gap_audit.py core/constitutional_brain.py --output implementation-gaps.json
python tools/test_summary.py --output test-summary.json
python tools/state_manifest.py --output STATE_MANIFEST.json --test-result passed --test-command "python tools/test_summary.py --output test-summary.json" --gates-status success
python tools/state_consistency_audit.py --manifest STATE_MANIFEST.json --output state-consistency.json
```

Os dois benchmarks são executados com seed `20261016`; os JSON registram taskset hash e, na reexecução pós-commit, `CI_COMMIT_SHA` identifica o candidato sem alegar execução remota. O suplemento VRAM externo declara dez mutantes e usa cópias temporárias, sem alterar o catálogo canônico.

## Limites residuais

O ledger continua em memória, sem SQLite; grafo e ledger são projeções distintas, sem transação compartilhada após um append bem-sucedido. Predicados funcionais permanecem explícitos. Restore não cria persistência física nem exactly-once; a origem precisa sincronizar mutações para obter visão coerente. `enqueue` continua usando `deepcopy` e objetos de tarefa continuam mutáveis. VRAM é validado na construção, sem GPU real, sem coerência física dos contadores ou imutabilidade posterior. Os callbacks avaliadores são código confiável em processo, não sandbox. Os artefatos de holdout herdados de #42 são históricos, não um novo holdout independente desta composição.

A validação local em Python 3.11 não substitui CI remoto, uma matriz 3.12/3.13 nem auditoria externa de segurança. Testes sintéticos, cobertura e Bandit não provam capacidade geral ou prontidão para produção.

## Integração/publicação segura — somente após nova autorização

1. Registrar o hash final e a revisão focada, sem modificar o candidato auditado. Confirmar se `origin/main` e os oito heads ainda coincidem com o inventário; qualquer avanço requer nova composição e gates.
2. Publicar somente a branch candidata, nunca atualizar `main` diretamente. Submeter a composição para revisão e executar os dois checks oficiais sobre a árvore combinada. Não depender dos checks individuais antigos.
3. Integrar a composição aprovada com ancestralidade preservada, após revisar os conflitos explícitos acima e a evidência exata do commit. Não mesclar novamente as oito PRs de forma independente sobre uma árvore diferente sem revalidar a composição.
4. Somente depois da integração confirmada, decidir administrativamente o destino das PRs originais. #12/#14 precisam propostas coerentes com novo lock; #13 está funcionalmente superada por #41. Nenhuma dessas ações foi executada agora.

`/workspace/Nexus` sujo, todas as refs preexistentes e a composição histórica permanecem preservados. Os únicos commits novos pertencem à branch candidata nesta nova worktree.
