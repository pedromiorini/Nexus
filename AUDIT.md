# Auditoria técnica do Nexus

**Data:** 21 de outubro de 2026  
**Commit auditado:** `d435083bb4c893a58128cee6cfa00f65bba7fe02` (`main`)  
**Escopo:** leitura estática, execução local e revisão de branches/PRs. Nenhum código de produção foi alterado.  
**Ambiente desta rodada:** Linux, Python 3.12.3, `numpy` 2.4.6 e o perfil `requirements-dev.txt`. O CI oficial usa Python 3.11 e `requirements-ci.lock`.

Este relatório é um snapshot novo. Ele não substitui o histórico em `NEXUS_AUDIT_FINAL.md` (setembro/outubro de 2026), `AUDIT_REALITY_REPORT.md` nem `SECURITY_TRIAGE.md`. Onde os números divergem, vale o que foi reexecutado aqui ou o gate de `main` citado abaixo.

## Resumo executivo

O Nexus é um protótipo experimental, em Python, de arquitetura neuro-simbólica: roteamento, memória, fila de tarefas adiadas, telemetria Vita, contratos de diagnóstico e ferramentas de auditoria. O README atual descreve esse limite com honestidade e o CI de `main` está verde. A implementação útil está concentrada em poucos módulos pequenos e bem testados; o restante vive num núcleo de 27.549 linhas.

A auditoria não encontrou segredos commitados, CVE conhecido nas dependências diretas do projeto, injeção de SQL clássica nem serviço de rede exposto. Encontrou um defeito funcional reproduzido (`TransferMapping` sombreado), um cache de roteador sem teto, metadados de permissão que não são aplicados, perfis de dependência divergentes e uma fila de 24 pull requests abertos que editam os mesmos documentos canônicos.

**Nota geral de saúde: 6,5 / 10.**

A nota sobe pelo gate de contrato (lint, hashes, Bandit sem achados MEDIUM/HIGH, mutation testing, cobertura com piso, actions pinadas por SHA) e pela documentação operacional que já separa alegação de evidência. A nota desce pelo monólito, pela cobertura de contrato do núcleo (44% sem a suíte embutida), pelo bug de sombreamento sem teste, pelo drift entre `requirements-dev.txt` e o lock, e pelo risco de integração entre agentes. Não é um sistema pronto para produção; também não é um repositório abandonado.

## 1. Visão geral

### Propósito

Biblioteca e laboratório de pesquisa para orquestração cognitiva local: `CentralRouter`, memória hierárquica/episódica, grafo, ferramentas com validação, fila adiada, guarda de VRAM, log encadeado, detector heurístico de jailbreak e ponte Vita com diagnósticos versionados. O próprio README nega AGI, ASI, consciência e prontidão de produção. O docstring de `core/constitutional_brain.py` ainda afirma o contrário (seção 3).

### Stack

| Camada | Escolha observada |
|---|---|
| Linguagem | Python. CI em 3.11 (`nexus-contract-gate.yml`, linha 31). Esta máquina tem 3.12.3. |
| Runtime direto | `numpy` (`requirements.txt`, `requirements_constitutional.txt`) |
| Persistência | SQLite na biblioteca padrão (`:memory:` ou arquivo) |
| Opcionais | `psutil`, `faiss` / `sentence-transformers`, `networkx` (só no lock de CI) |
| Dev/CI | `coverage` 7.16.2, `hypothesis` (pin diverge; ver seção 5), `bandit` 1.9.4, `ruff` 0.16.10, `zizmor` |
| Empacotamento | Não é pacote instalável. `pyproject.toml` só configura Ruff e coverage. `core/__init__.py` e `vita/__init__.py` estão vazios. |

Não há `Dockerfile`, `docker-compose`, `LICENSE`, `CODEOWNERS`, pre-commit nem type checker (`mypy`/`pyright`).

### Arquitetura e pastas

```text
Entradas (unittest, suíte embutida, CLIs de benchmark)
        ↓
core/constitutional_brain.py     núcleo monolítico + CentralRouter
core/deferred_task_queue.py      fila em memória, snapshot versionado
core/vram_defense_guard.py       pressão de VRAM (torch opcional)
core/temporal_facts.py           ledger bi-temporal em memória
core/integrated_suite_summary.py contagem da suíte embutida
        ↓
vita/nexus_constitutional_bridge_v3.py   auditoria SQLite e diagnóstico
        ↓
tools/                               reality, Bandit triage, mutation, manifesto
benchmarks/                          protocolos sintéticos determinísticos
test_*.py                            50 arquivos na raiz, não em tests/
.github/workflows/nexus-contract-gate.yml
```

Contagem AST desta rodada: 70 módulos Python, 36.182 linhas no total, 27.549 no núcleo, 269 classes e 875 funções no núcleo (o relatório de realidade versionado concorda: 27.549 / 269 / 875). Há 63 classes com prefixo `Real`.

### Pontos de entrada

| Comando | Papel |
|---|---|
| `python3 -m unittest discover -v -p 'test_*.py'` | 303 testes de contrato |
| `python3 core/constitutional_brain.py` | suíte integrada embutida; recusa `python -O` (`constitutional_brain.py:25681`) |
| `python3 tools/reality_audit.py` | regenera `AUDIT_REALITY_REPORT.md` e `audit_reality.json` |
| `python3 tools/test_summary.py` | resumo da suíte, usado pelo CI |
| `python -m benchmarks.nexus_capability_benchmark` e `nexus_pomdp_benchmark_v2` | benchmarks do CI |

Variáveis: `PYTHONPATH=.`, `NEXUS_DEBUG=1` (traces), `NEXUS_AUDIT_DB` (SQLite da Vita; padrão `nexus_audit.sqlite3` no diretório atual), `NEXUS_INTEGRATED_SUMMARY` (JSON da suíte embutida).

### Como rodar

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
export PYTHONPATH=.
export NEXUS_AUDIT_DB=/tmp/nexus_audit.sqlite3
python3 -m unittest discover -v -p 'test_*.py'
python3 core/constitutional_brain.py
```

`networkx` não entra nesse perfil. Dois testes de grafo ficam pulados. O CI instala `requirements-ci.lock` com hashes, inclusive `networkx==3.6.1` e `hypothesis==6.170.0`.

## 2. Saúde do build

Não existe alvo de “build” de pacote. A verificação equivalente é instalação, compilação, lint e testes.

| Verificação | Resultado nesta rodada |
|---|---|
| `pip install -r requirements-dev.txt` | Sucesso. `numpy` 2.4.6, `coverage` 7.16.2, `hypothesis` 6.168.5, `bandit` 1.9.4, `ruff` 0.16.10, `zizmor` 1.30.1. |
| `pip install --dry-run --require-hashes -r requirements-ci.lock` | Resolver aceito no Python 3.12. Instalaria ainda `hypothesis==6.170.0`, `networkx==3.6.1` e `PyYAML==6.0.3`. |
| `pip install --dry-run --no-deps -r requirements_constitutional.txt` | `numpy` já satisfeito. |
| `ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py` | **Passou.** “All checks passed!” O conjunto do Ruff é estreito (seção 3). |
| `python3 -m compileall -q` nos mesmos alvos do CI | **Passou.** |
| `unittest discover -p 'test_*.py'` | **Passou:** 303 testes, 12,35 s, `OK (skipped=2)`. Os skips são `test_knowledge_graph_contract.py:38` e `:49` (`networkx` ausente). |
| Suíte integrada `core/constitutional_brain.py` | **Passou** (exit 0). Resumo em `NEXUS_INTEGRATED_SUMMARY`: 30 aprovados, 1 skip (FAISS opcional, checagem `"0"`), 0 ausentes. A saída ainda imprime `36/55 Modules (65.5%)` (`constitutional_brain.py:27547`). |
| Cobertura de branches, unittest apenas | Total **49%**. Núcleo **44%**, abaixo do piso de 62% do CI. Fila 95%, VRAM 94%, ledger temporal 93%, Vita 76%. |
| Cobertura após acrescentar a suíte integrada | Núcleo **69%**. Os pisos do workflow (fila 90, VRAM 90, temporal 90, Vita 70, núcleo 62) são atingidos **só com a suíte embutida**. |
| `bandit -r core vita tools` | Exit code 1 do Bandit (há achados), mas **0 MEDIUM/HIGH**. 131 LOW, confiança HIGH. A política do CI (`nexus-contract-gate.yml:224`) passaria. |
| `zizmor --offline` 1.30.1 | **Passou.** “No findings to report.” O workflow piná o binário 1.31.1, não este pacote pip. |
| `actionlint` | **Não executado.** O binário pinado não está neste ambiente. O run de `main` `39350179279` (20/10/2026, `d435083`) concluiu com sucesso e inclui esse passo. |
| Mutation testing (`tools/targeted_mutation.py`) | **Não reexecutado** (58 mutantes, cada um com subprocesso de testes). Cobertura desse arquivo na suíte unittest: **0%**. O mesmo run verde de `main` inclui o gate. |
| `pip-audit -r requirements-dev.txt` e `-r requirements-ci.lock` (`--disable-pip --no-deps`) | **Nenhuma vulnerabilidade conhecida** nas dependências do projeto. |

`pip-audit --local` lista CVE em pacotes do sistema operacional desta VM (`pip` 24.0, `urllib3` 2.0.7, `jinja2` 3.1.2, `setuptools` 68.1.2, `PyJWT`, `oauthlib`, `wheel`). Eles não estão em `requirements*.txt` nem no lock. Não são achados do repositório.

## 3. Qualidade de código

### Achados

| Sev. | Achado | Onde |
|---|---|---|
| **Alto** | Duas dataclasses homônimas `TransferMapping` com campos incompatíveis. A segunda substitui a primeira no escopo do módulo. `RealMetaLearningStrategies.transfer_knowledge` constrói a classe com `transfer_strength`, `shared_features` e `adaptation_required`. Em runtime o construtor vigente exige `mapping_id`, `transferable_knowledge` e `transfer_quality`. Reproduzido: `TypeError: TransferMapping.__init__() got an unexpected keyword argument 'transfer_strength'`. Nenhum teste chama esse método. | `core/constitutional_brain.py:16105`, chamada em `:16334`, redefinição em `:20287` |
| **Alto** | O piso de cobertura do núcleo (62%) só fecha porque o CI mede também a demonstração embutida. Os contratos unittest deixam o arquivo em 44% (5.645 statements não executados de 11.011). `CompleteNexusBrain.think` tem 807 linhas e complexidade ciclomática aproximada de 110; `_wire_dependencies` tem complexidade aproximada de 136. | `core/constitutional_brain.py:24578` (`think`), `:24291` (`_wire_dependencies`), `:25680` (`__main__`); pisos em `.github/workflows/nexus-contract-gate.yml:99` |
| **Médio** | Docstring do módulo afirma “62/62 (100%)”, “ABSOLUTE PERFECTION”, “29/29 + 0 WARNINGS” e “Production ready”. A execução desta rodada imprime “36/55 Modules (65.5%)” e pula o teste 0 por FAISS ausente. O README já corrige a narrativa; o arquivo que os agentes editam não. | `core/constitutional_brain.py:8`, `:9`, `:132`, `:27547` |
| **Médio** | Ruff seleciona só `F`, `E9`, `B904`, `B006`, `B007` e `RUF013`. Estilo, imports, complexidade e padrões `E`/`W`/`I`/`UP` não são gate. Não há mypy. Das 1.350 funções do repositório, 768 (56,9%) anotam retorno; 103 têm parâmetros sem anotação completa. O núcleo usa `Any` com frequência (81 ocorrências de `: Any`). | `pyproject.toml:7-11`; anotações em `core/constitutional_brain.py` (ex.: `:147`) |
| **Médio** | 26 `except Exception` no código de produção, nenhum `except` nu e nenhum `except: pass`. Vários são intencionais (cache, fila, rollback do log). O fallback de FAISS engole qualquer erro e segue para palavra-chave, mudando a semântica da busca. | `core/constitutional_brain.py:405`, `:259`, `:1090`, `:13602`, `:13667`; `core/deferred_task_queue.py:60`, `:123`, `:138`; `core/vram_defense_guard.py:11`, `:130` |
| **Baixo** | 157 chamadas `print` fora do bloco `__main__`. Construtores anunciam “POST N initialized” e poluem stdout de quem importa a biblioteca (visto na suíte: dezenas de linhas por teste). | ex. `core/constitutional_brain.py:263`, `:13062`, `:13320` |
| **Baixo** | `tools/targeted_mutation.py` não é exercitado pelos unittest (cobertura 0% nesta medição). `tools/diagnostics_dashboard.py` está em 58%. Há PRs abertos (#58, #52–#57, #59) em cima desses ferramentas; não é lacuna ignorada, é trabalho paralelo ainda não integrado. | `tools/targeted_mutation.py`, `tools/diagnostics_dashboard.py` |

Não há métodos duplicados dentro da mesma classe. A única classe de módulo definida duas vezes é `TransferMapping`. Funções só com `pass`: 0, segundo `AUDIT_REALITY_REPORT.md`. Funções com retorno constante estrito: 2. Frases fortes de capacidade ainda no código: 51 (inventário do mesmo relatório, limite que só pode cair).

Consistência de estilo: os módulos novos (`deferred_task_queue.py`, `temporal_facts.py`, `vram_defense_guard.py`, `vita/`) usam `from __future__ import annotations`, tipos e erros explícitos. O núcleo mistura isso com banners, `print` e `Dict`/`List` da `typing`. A diferença é de geração, não de um guia único aplicado ao arquivo grande.

## 4. Segurança

Não há listener HTTP, Flask, FastAPI nem `eval`/`exec`/`pickle`/`shell=True` no código do produto. A busca SQL observada usa parâmetros (`?`). `pip-audit` não achou CVE nas listas do projeto. Busca por chaves, tokens e blocos `BEGIN PRIVATE KEY` no conteúdo versionado não retornou segredo. `.gitignore` cobre `.env` indiretamente só se alguém nomear `*.log`/`*.sqlite3`; um `.env` solto **não** está ignorado (não existe `.env` commitado).

| Sev. | Achado | Onde |
|---|---|---|
| **Médio** | `ToolDefinition.requires_permission` e `permission_level` (`read`/`write`/`admin`) não são lidos em `execute_tool`. Os quatro built-ins nascem com `requires_permission=False`. O campo descreve uma fronteira que o caminho de execução não aplica. O roadmap já registra isso; continua verdadeiro em `main`. | `core/constitutional_brain.py:3542`, `:3774`, `:4322` |
| **Médio** | `timeout_seconds` é validado em `[1, 60]`, mas o estouro só gera o evento `TOOL_EXECUTION_OVERRUN` com `cancelled: False`. O executor síncrono não é interrompido. `math_calculate` tem teto de AST; `string_transform` e `list_operations` não limitam o tamanho da entrada. | `core/constitutional_brain.py:3649`, `:4466` |
| **Médio** | `ToolValidator._contains_forbidden` documenta fronteira de palavra e implementa `palavra in texto`. A substring `"system"` dispara em descrições legítimas; ofuscações com espaço (`"ev al"`) passam. A execução real não depende dessa lista: o mapa de executores é fixo (`:4438`). A lista cria confiança maior do que a garantia. | `core/constitutional_brain.py:3614`, `:3632`, `:4438` |
| **Médio** | Metacaracteres `LIKE` (`%`, `_`) da consulta entram crus no padrão. Não é injeção de SQL (o valor é parâmetro), mas uma consulta `"%"` casa com todo o conteúdo. O mesmo padrão está na busca episódica. Limites positivos altos continuam sem teto (decisão já documentada em `NEXUS_ROADMAP.md`). | `core/constitutional_brain.py:428`, `:2266` |
| **Médio** | `sqlite3.connect(..., check_same_thread=False)` na memória hierárquica e na episódica sem `PRAGMA busy_timeout`. O log imutável e a ponte Vita definem `busy_timeout = 5000`. Outra conexão no mesmo arquivo pode falhar imediatamente com “database is locked”. O lock em processo cobre o uso num único objeto. | `core/constitutional_brain.py:237`, `:2129`; contraste `:1021` e `vita/nexus_constitutional_bridge_v3.py:30` |
| **Baixo** | Bandit: 111× B101 (`assert` no `__main__`), 16× B311 (`random` em simulação), 2× B404 e 2× B603 (subprocesso com argv fixo em `tools/state_manifest.py:29` e `tools/targeted_mutation.py:274`). Triagem versionada em `SECURITY_TRIAGE.md` confere com essa contagem. | `SECURITY_TRIAGE.md`; `core/constitutional_brain.py` a partir de `:25680` |
| **Baixo** | `RealJailbreakDetector` é heurística EN/PT (NFKC, homóglifos, leetspeak). O código e o `SECURITY.md` dizem que não substitui controle de privilégio. `EthicsGuard` bloqueia por palavras (`harm`, `ignore previous`, etc.). Tratá-los como autorização seria incorreto. | `core/constitutional_brain.py:1254`, `:13057` |
| **Baixo** | `RealImmutableLog.verify_chain` detecta adulteração no meio da cadeia. Truncar o fim ou reescrever a cadeia inteira só é detectável se o `head` for ancorado fora do arquivo. Já está documentado. | `core/constitutional_brain.py:1008`, `:1096` |
| **Baixo** | `NEXUS_AUDIT_DB` aceita um caminho arbitrário e a ponte cria/escreve o SQLite ali. É configuração local, sem sanitização de path. Aceitável enquanto for biblioteca de um único usuário. | `vita/nexus_constitutional_bridge_v3.py:25` |

Não há achado **Crítico**: não há RCE exposto, segredo no git, nem CVE nas dependências pinadas do repositório.

## 5. Dependências

| Pacote | `requirements-dev.txt` | `requirements-ci.lock` (o que o CI instala) | Último no PyPI nesta data | PR Dependabot aberto |
|---|---|---|---|---|
| numpy | 2.4.6 (e `requirements.txt` permite `>=1.26,<3`) | 2.4.6 | 2.5.4 | #14 → 2.5.4, só em `requirements.txt` / `requirements-dev.txt` |
| hypothesis | **6.168.5** | **6.170.0** | 6.170.5 | #13 → **6.169.1**, atrás do lock |
| ruff | 0.16.10 | 0.16.10 | 0.17.1 | #12 → 0.17.0, não recompila o lock |
| coverage | 7.16.2 | 7.16.2 | 7.16.2 | — |
| bandit | 1.9.4 | 1.9.4 | 1.9.4 | — |
| networkx | ausente | 3.6.1 | 3.7 | — |
| zizmor (pip) | 1.30.1 | não está no lock; o workflow baixa o binário **1.31.1** | 1.31.1 | — |

| Sev. | Achado | Onde |
|---|---|---|
| **Médio** | Três fontes de verdade. O comentário de `requirements-dev.txt:1` diz que aquelas pins são as do CI. O workflow instala o lock (`nexus-contract-gate.yml:34`). `hypothesis` já divergiu em duas versões menores. O Dependabot olha os arquivos soltos e abriu um bump que **reduziria** o pin de dev em relação ao que o CI já resolve, sem tocar o lock. | `requirements-dev.txt:6`, `requirements-ci.in:6`, `requirements-ci.lock:130`, PRs #12, #13, #14 |
| **Médio** | `networkx` é dependência direta do caminho de grafo (`constitutional_brain.py:556`) e do CI, mas não do perfil documentado no README. Resultado local: 2 skips. Quem reproduz “como executar” não reproduz o CI. | `requirements-ci.in:9`, `test_knowledge_graph_contract.py:38` |
| **Baixo** | `requirements.txt` sem pin exato deixa `pip install -r requirements.txt` flutuar dentro de `numpy>=1.26,<3`. O perfil dev em seguida força 2.4.6. Instalação só de runtime não é reproduzível. | `requirements.txt:3` |
| **Baixo** | Licenças observadas, todas permissivas ou de arquivo: numpy BSD-3/MIT/Zlib, coverage Apache-2.0, hypothesis **MPL-2.0**, bandit Apache-2.0, ruff MIT, zizmor MIT, PyYAML MIT, rich MIT. Não há `LICENSE` do próprio Nexus, então o projeto não declara com qual licença o código pode ser reutilizado. MPL-2.0 do Hypothesis não contamina o código do Nexus (é ferramenta de teste). | metadados dos pacotes instalados; ausência de `LICENSE` na raiz |

Não há conflito de resolver que impeça a instalação. Não há dependência direta morta óbvia: `numpy` é importado; as ferramentas de dev são usadas pelo CI. `zizmor` no pip e o binário do workflow são o mesmo produto em versões diferentes, não dois produtos.

## 6. Testes

50 arquivos `test_*.py` na raiz, 69 classes, **303 métodos de teste**. Três arquivos usam Hypothesis: `test_deferred_task_properties.py`, `test_recovery_diagnostics_contract.py`, `test_jailbreak_detector_contract.py`.

Cobertura de branches medida aqui (unittest + suíte integrada):

| Arquivo | Cobertura | Piso do CI | Leitura |
|---|---:|---:|---|
| `core/deferred_task_queue.py` | 95% | 90 | Contrato forte |
| `core/vram_defense_guard.py` | 94% | 90 | Contrato forte; ramo `torch` real não roda sem GPU |
| `core/temporal_facts.py` | 93% | 90 | Contrato forte |
| `vita/nexus_constitutional_bridge_v3.py` | 76% | 70 | Acima do piso, ainda com 60 statements sem execução |
| `core/constitutional_brain.py` | 69% com a demo; **44% só com unittest** | 62 | A demo segura o piso |
| `tools/security_triage.py` | 96% | — | |
| `tools/diagnostics_dashboard.py` | 58% | — | PR #58 em aberto |
| `tools/targeted_mutation.py` | 0% na suíte | — | Só o job de CI o executa |
| Total `core`+`vita`+`tools` | 49% unittest; ~70% nos arquivos críticos após a demo | — | |

| Sev. | Lacuna | Onde |
|---|---|---|
| **Alto** | Nenhuma cobertura de `RealMetaLearningStrategies.transfer_knowledge`. O sombreamento de `TransferMapping` passa invisível. | `test_*.py` (ausência); chamada em `constitutional_brain.py:16313` |
| **Alto** | A maior parte das ~60 classes `Real*` do núcleo não tem contrato independente. A suíte embutida (`:25680`–fim) verifica contadores e objetos da demonstração. 111 `assert` ainda estão nesse bloco (B101). | `core/constitutional_brain.py:25680`; `SECURITY_TRIAGE.md` |
| **Médio** | Sem teste de crescimento do `execution_cache` do `CentralRouter`, de estouro de memória da fila de histórico do VRAM, nem de metacaractere `LIKE`. | `constitutional_brain.py:13290`, `vram_defense_guard.py:84` |
| **Médio** | Caminho semântico FAISS/SentenceTransformers e `psutil` não são exercitados aqui (imports opcionais ausentes). O skip é explícito e honesto; a qualidade da busca cai para palavra-chave sem um teste de paridade. | `constitutional_brain.py:196`, `:173` |
| **Baixo** | Testes na raiz, sem `tests/` e sem marcadores pytest. Funciona com unittest; dificulta seleção e paralelismo. | `test_*.py` |

O que está bem coberto, e não deve ser refeito às cegas: fila e snapshot, limites de `math_calculate`, cache de ferramentas, log imutável, jailbreak, diagnóstico Vita, manifesto de estado. Vários PRs #52–#59 acrescentam contratos exatamente nessas bordas.

## 7. Performance

Não houve benchmark de carga. Os gargalos abaixo são estruturais, visíveis no código.

| Sev. | Achado | Onde |
|---|---|---|
| **Médio** | `CentralRouter.execution_cache` não tem tamanho máximo. A TTL (300 s) só é consultada quando a **mesma** chave volta a ser pedida. Chaves únicas ficam no dicionário até o processo morrer. Cada inserção faz `copy.deepcopy` do resultado. | `core/constitutional_brain.py:13290`, `:13595`, `:13670` |
| **Médio** | `VramDefenseGuard.history`, `mitigation_history` e `audit_log` só crescem. `evaluate()` é chamado no caminho de roteamento. | `core/vram_defense_guard.py:59`, `:84`, `:133` |
| **Médio** | Fallback de memória executa uma query `LIKE '%palavra%'` por token, sem índice útil para prefixo curinga, e atualiza `access_count` em seguida. Consultas longas multiplicam round-trips. | `core/constitutional_brain.py:427` |
| **Baixo** | `TemporalFactStore` documenta reconstrução `known_at` em O(eventos). Adequado ao protótipo; não a um log grande. | `core/temporal_facts.py:13` |
| **Baixo** | O workflow roda `unittest discover` e depois repete dezenas de arquivos em passos separados, mais a suíte integrada, mais até 58 mutantes com cópia do repositório (`TIMEOUT_SECONDS = 180` por mutante) dentro de um job com `timeout-minutes: 12`. O risco é timeout ou cancelamento, não lentidão do produto. | `.github/workflows/nexus-contract-gate.yml:19`, `:94`; `tools/targeted_mutation.py:48` |
| **Baixo** | `think` e a fiação de módulos constroem dezenas de objetos e imprimem banners a cada processo. A suíte integrada desta rodada terminou em menos de 2 s, então não é um problema de CPU hoje. É custo de manutenção. | `constitutional_brain.py:24578` |

O cache de `RealToolUse` está limitado (`MAX_CACHE_ENTRIES = 128`, `MAX_CACHE_BYTES`, TTL com poda). Esse caminho recebeu hardening recente e não é o problema. O cache do roteador não herdou o mesmo teto.

## 8. DevOps e documentação

O que está sólido:

- Um workflow, `permissions: contents: read`, `persist-credentials: false`, actions pinadas por SHA, `actionlint` e `zizmor` com checksum, instalação com `--require-hashes`, artefatos retidos 14 dias, concorrência com cancelamento em PR.
- Dependabot semanal para pip e GitHub Actions (`.github/dependabot.yml`).
- `.gitignore` cobre bytecode, venv, caches, `*.sqlite3`, `*.db` e a maior parte dos JSON gerados pelo CI.
- README, `SECURITY.md`, constituição, decisões, evidências, roadmap e `CONTINUATION.md` existem e estão alinhados entre si melhor do que o docstring do núcleo.
- CI de `main` no commit auditado: [run 39350179279](https://github.com/pedromiorini/Nexus/actions/runs/39350179279), sucesso em 20/10/2026.

| Sev. | Achado | Onde |
|---|---|---|
| **Médio** | Sem `LICENSE`. Reuso e contribuição ficam sem termos. | raiz do repositório |
| **Baixo** | `pomdp-belief-benchmark-v2.json` é gerado pelo CI e apagado no fim do job, mas não está no `.gitignore` (o JSON de capability está). Uma execução local pode commitá-lo por acidente. | `.gitignore:27`; workflow `:190` e `:269` |
| **Baixo** | Sem Docker, sem matriz Python (só 3.11), sem ambiente de dev declarativo além dos `requirements*`. Esta auditoria passou em 3.12.3; isso não substitui o runner. | `.github/workflows/nexus-contract-gate.yml:31` |
| **Baixo** | Proteção de branch não pôde ser lida (`gh api` retornou 403 para o token desta sessão). Não dá para afirmar se `main` exige review ou status check. | API de branch protection |
| **Baixo** | `.env` não está no `.gitignore`. Não há `.env` versionado. | `.gitignore` |
| **Baixo** | Documentos canônicos (`CONTINUATION.md`, `NEXUS_EVIDENCE.md`, `NEXUS_KNOWLEDGE.md`, `SECURITY_TRIAGE.md`) são reescritos por quase todo PR. Isso é processo, detalhado na seção 9. | esses quatro arquivos |

## 9. Colaboração multiagente

Autores no histórico (contagem de commits, não de PRs): Pedro Alexandre Miorini dos Santos (várias identidades de e-mail), Manus AI, Cue Agent, Claude, Axis, dependabot. Os PRs abertos aparecem sob a conta `pedromiorini`; os commits dentro deles carregam os agentes. O último merge em `main` é o #51 (Axis, CI para qualquer branch de destino), em 20/10/2026.

**24 pull requests abertos. 24 branches remotas ainda não contidas em `origin/main`.**

| Sev. | Achado | Evidência |
|---|---|---|
| **Alto** | Quase toda contribuição reescreve os mesmos quatro documentos de handoff. 21 dos 24 PRs abertos tocam `CONTINUATION.md`, `NEXUS_EVIDENCE.md`, `NEXUS_KNOWLEDGE.md` e `SECURITY_TRIAGE.md`. GitHub marca cada um como MERGEABLE contra `main` **hoje**. O segundo merge encontra conflito ou reverte o handoff do primeiro. Os drafts #52–#59 (20–21/10) estão todos com CI verde e são o caso mais claro: correções pequenas e independentes no código, colisão garantida na documentação. | PRs #38–#59 |
| **Alto** | Trabalho duplicado e pilha quebrada em volta do núcleo. Nove PRs editam `core/constitutional_brain.py` (#38, #39, #40, #41, #42, #44, #46, #48, #49). A PR **#46** (draft, CI verde, base `main`) se apresenta como consolidação das PRs **#38–#45**, mas #38, #39, #40, #41, #42, #43, #44 e #45 continuam abertas. #45 está **CONFLICTING** com `main`. A pilha seguinte está partida: #47 (base `audit/consolidated-38-45-20261018`) e #48 (base `fix/audit-json-preflight-20261019`) estão **CONFLICTING**. #49 e #50 estão MERGEABLE nas bases intermediárias e **sem checks** (não houve run; o disparo para qualquer base só entrou em `main` no #51, sem reexecução retroativa). | PRs #38–#50 |
| **Médio** | Dependabot (#12 ruff, #13 hypothesis, #14 numpy) está verde e MERGEABLE desde 16/10, e desencontrado do lock (seção 5). Integrar #13 como está não atualiza o CI e ainda aponta para uma versão de Hypothesis anterior à do lock. | PRs #12–#14 |
| **Baixo** | Vários agentes (Cue, Axis, Manus, Claude) entregam no mesmo estilo de “um contrato + atualização dos quatro documentos”. A qualidade individual dos patches recentes é alta (CI verde, escopo curto). O defeito é a ausência de fila única: ninguém fecha o PR anterior antes de abrir o próximo sobre o mesmo handoff. | histórico e PRs #52–#59 |

Leitura prática: `main` em si está consistente (o gate passou). O risco está na integração futura, não num conflito já commitado em `main`.

## Plano de ação

Ordem sugerida: o que remove defeito ou risco de merge com diff pequeno, antes de qualquer extração do núcleo. Esforço é tamanho do patch e da coordenação, não calendário.

### Quick wins

1. **Corrigir o sombreamento de `TransferMapping` e travar com teste.** Renomear uma das duas dataclasses (a de `RealMetaLearningStrategies` ou a de `RealTransferLearningEngine`) e adicionar um teste que chame `transfer_knowledge`. Esforço: **pequeno** (um arquivo, um teste). Severidade que isso fecha: Alto. Não duplica PR aberta; nenhuma das 24 toca esse símbolo.
2. **Serializar a fila de PRs antes de novos patches no núcleo.** Escolher um único veículo para #38–#45 (a consolidação #46 **ou** os PRs individuais, não os dois). Rebasear ou fechar #47 e #48, que já estão CONFLICTING. Para #52–#59, integrar um por vez e rebases dos demais só no handoff, ou extrair o trecho de “estado atual” para um artefato gerado. Esforço: **médio** (coordenação; o código de cada PR é pequeno). Não editar `constitutional_brain.py` em paralelo a essa decisão.
3. **Alinhar os três perfis de dependência.** Recompilar `requirements-ci.lock` com `uv pip compile`, igualar `hypothesis`/`ruff`/`numpy` em `requirements-dev.txt`, incluir `networkx` no perfil local **ou** documentar o skip como esperado no README, e fechar os PRs Dependabot #12–#14 se o lock resultante for mais novo do que eles. Esforço: **pequeno**. Não mergear #13 sozinho.
4. **Teto no cache do `CentralRouter` e nos históricos do `VramDefenseGuard`.** Reutilizar a ideia já usada em `RealToolUse` (máximo de entradas e poda por TTL). Teste que enfileira chaves distintas e exige limite. Esforço: **pequeno**.
5. **Escapar `%` e `_` nos `LIKE` de memória e episódio.** Teste com consulta `"%"` que não pode devolver o corpus inteiro por metacaractere. Esforço: **pequeno**.
6. **Higiene de repositório.** Acrescentar `LICENSE` (a licença que o mantenedor escolher), ignorar `pomdp-belief-benchmark-v2.json` e `.env`. Esforço: **pequeno**.

### Em seguida

7. **Permissão de ferramenta: aplicar ou apagar.** Ou `execute_tool` falha fechado quando `requires_permission` é verdadeiro e não há concessão, ou os campos saem da API até existir um principal. Hoje eles documentam uma garantia falsa. Esforço: **médio** (contrato público, já citado no roadmap).
8. **Substituir o docstring-banner (linhas 1–132) por um texto no tom do README** e fazer o reality audit falhar se “Production ready” ou “62/62 (100%)” voltar ao módulo. Esforço: **pequeno** no diff, **médio** na coordenação, porque vários PRs abertos editam o mesmo arquivo e vão reintroduzir o banner no rebase.
9. **`PRAGMA busy_timeout` nas conexões de memória hierárquica e episódica**, no mesmo padrão do log e da Vita. Esforço: **pequeno**.
10. **Alargar o Ruff em passos** (`E4`, `I`, `UP` só em `core/deferred_task_queue.py`, `temporal_facts.py`, `vita/` e `tools/`, não no monólito inteiro). Esforço: **médio**.
11. **Continuar a migração dos 111 asserts do `__main__`**, um contrato por vez, como já fazem #52–#59. Esforço: **grande** no acumulado, **pequeno** por passo. Não fazer a troca em massa.

### Fora desta leva

12. **Extrair módulos do `constitutional_brain.py`.** O roadmap já pede um componente por vez, com contrato antes da mudança. Esforço: **grande**. Fazer isso com 9 PRs abertos sobre o mesmo arquivo aumenta o conflito sem reduzir o risco.
13. **mypy no núcleo.** Só vale depois de um recorte com anotações reais. Esforço: **grande**.
14. **Docker / matriz 3.11–3.13.** Útil quando alguém precisar reproduzir o runner. Esforço: **médio**. O ganho imediato é menor do que os itens 1–6.

## Limitações desta auditoria

- Python 3.12.3, não o 3.11 do CI. Compilação, Ruff, 303 testes e a suíte integrada passaram nessa versão; isso não prova o runner.
- `actionlint` e os 58 mutantes não foram reexecutados. O run verde de `main` em `d435083` é a evidência disponível para esses dois passos.
- `zizmor` local é 1.30.1; o CI usa o binário 1.31.1.
- Não houve leitura linha a linha das 27.549 linhas. A varredura foi AST, busca dirigida (SQL, subprocesso, segredos, `except`, caches), execução e amostragem dos caminhos de ferramenta, memória, log e roteador.
- `pip-audit` do ambiente local mistura pacotes do sistema. O resultado que vale para o repositório é o das listas `requirements-dev.txt` e `requirements-ci.lock`.
- A API de branch protection retornou 403. Não há conclusão sobre reviews obrigatórios.
- Outros agentes continuam abrindo PRs. A lista da seção 9 é a de 21/10/2026 por volta de 06:40 UTC.

## Apêndice — comandos desta rodada

```bash
python3 -m pip install --user -r requirements-dev.txt
export PYTHONPATH=.
export NEXUS_AUDIT_DB=/tmp/nexus-audit.sqlite3
ruff check core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py
python3 -m compileall -q core vita tools test_*.py benchmarks/nexus_pomdp_benchmark_v2.py
coverage run --branch -m unittest discover -p 'test_*.py'
# 303 testes, OK (skipped=2), núcleo em 44%
NEXUS_INTEGRATED_SUMMARY=/tmp/integrated-summary.json \
  coverage run --append --branch core/constitutional_brain.py
# exit 0; 30 passed, 1 skipped; núcleo em 69%
bandit -r core vita tools -f json -o /tmp/bandit-report.json
# 131 LOW, 0 MEDIUM/HIGH
zizmor --offline --no-progress .github/workflows
pip-audit -r requirements-dev.txt --disable-pip --no-deps
pip-audit -r requirements-ci.lock --disable-pip --no-deps --require-hashes
# No known vulnerabilities found
```

Reprodução do defeito de `TransferMapping`:

```bash
PYTHONPATH=. python3 -c '
import core.constitutional_brain as b
print(list(b.TransferMapping.__dataclass_fields__))
b.RealMetaLearningStrategies().transfer_knowledge("math", "physics")
'
# TypeError: unexpected keyword argument 'transfer_strength'
```
