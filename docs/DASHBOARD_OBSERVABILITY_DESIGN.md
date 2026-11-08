# Especificação de observabilidade do Nexus

**Status:** proposta local de produto/arquitetura. O adendo do catálogo descreve uma extensão local estática do dashboard, sem nova instrumentação runtime. “Planejado” neste documento significa proposta ainda sujeita a decisão; não é compromisso de roadmap.

**Base factual:** branch `feat/first-steps-recovery-dashboard-20261107`, `HEAD`/base `632ce8456ff76810654aa19b950ed1960e9754d0`. A árvore já continha alterações locais do MVP HTML em `tools/diagnostics_dashboard.py`, `test_recovery_diagnostics_contract.py`, `docs/GETTING_STARTED.md` e `docs/recovery-diagnostics-html-dashboard.md`; a extensão atual é aditiva no dashboard/testes e nesta especificação, preservando os guias de onboarding/uso.

## Ideia refinada

O dashboard deve ser um **painel de observabilidade do software**, não uma representação da “mente” de Nexus. Uma tela inicial pode combinar: (1) indicadores operacionais do snapshot disponível; (2) um catálogo das funções cognitivas-operacionais, com evidência de código/testes e visibilidade de runtime claramente separadas; e (3) um rastro temporal de eventos estruturados e seguros, quando essa instrumentação existir.

Em vez de uma pizza “pronto versus falta”, prefira uma matriz ou tabela de registros candidatos. Gráficos de cobertura funcional ficam bloqueados até aprovação dos critérios exclusivos, da atribuição primária e da unidade de contagem. “Existe código”, “há teste de contrato”, “houve evento observado” e “a função funciona bem em tarefas externas” são afirmações diferentes; não devem virar uma nota única de prontidão.

## Fatos observados no checkout

A Constituição define o Nexus como protótipo experimental e rejeita inferências de AGI, consciência, autonomia geral ou prontidão de produção a partir de nomes, métricas internas e testes. A árvore atual também tem módulos cognitivo-operacionais em `core/cognitivo/`, além do núcleo ainda concentrado em `core/constitutional_brain.py`. Os contratos abaixo evidenciam comportamentos delimitados, não cognição humana ou desempenho geral.

| Área candidata | Fontes e contratos presentes no checkout | O que a evidência local permite dizer | Visibilidade no snapshot atual |
|---|---|---|---|
| Roteamento e fila de tarefas | `core/constitutional_brain.py`, `core/deferred_task_queue.py`; `test_central_router_contract.py`, `test_deferred_task_queue.py`, `test_deferred_task_snapshot.py` | Há contratos para encaminhamento, contexto, prioridade, retry, capacidade e snapshots da fila. Snapshot da fila não equivale a entrega distribuída ou exactly-once. | Não instrumentado neste snapshot. Eventos Vita de reprocessamento não atestam execução de rotas/fila. |
| Decisão, efeitos e permissões de ferramentas | `core/cognitivo/decision_cycle.py`, `core/cognitivo/action_effects.py`, `core/tool_evidence_gate.py`; `test_decision_cycle_contract.py`, `test_action_effects_contract.py`, `test_tool_permission_failclosed_contract.py`, `test_evidence_omission_contract.py` | Contratos testam seleção/vetos, efeitos declarados, validação e bloqueio de executor sem concessão. Não medem qualidade de decisões no mundo. | Não instrumentado no snapshot de recuperação. |
| Atenção e workspace | `core/cognitivo/attention.py`, `core/constitutional_brain.py`; `test_attention_contract.py`, `test_attention_signal_contract.py`, `test_global_workspace_contract.py` | Existem sinais opcionais de contraste/movimento, fallback heurístico e contratos de competição/broadcast. Isso não é percepção biológica nem prova de integração completa em todas as rotas. | Não instrumentado no snapshot de recuperação. |
| Memória de trabalho | `core/cognitivo/working_memory.py`; `test_working_memory_contract.py` | Há contratos delimitados para capacidade, alocação de atenção, ensaio e itens ativos. | Não instrumentado no snapshot de recuperação. |
| Memória episódica e contextual | `core/cognitivo/episodic_memory.py`, `core/cognitivo/contextual_memory.py`; `test_episodic_retrieval_contract.py`, `test_episodic_memory_statistics_contract.py`, `test_episodic_memory_concurrency_contract.py`, `test_contextual_memory_contract.py` | Há contratos de armazenamento, estatísticas, recuperação e concorrência locais. Resultados dependem do backend/configuração e dos dados disponíveis. | Não instrumentado no snapshot de recuperação. |
| Conhecimento semântico e fatos temporais | `core/cognitivo/semantic_networks.py`, `core/temporal_facts.py`, grafo no núcleo; `test_knowledge_graph_contract.py`, `test_temporal_facts_contract.py`, `test_temporal_graph_integration_contract.py`, `test_semantic_networks_contract.py` | Há contratos de grafo e de fatos com histórico/intervalos; teste local não valida a verdade externa do conhecimento nem consistência distribuída. | Não instrumentado no snapshot de recuperação. |
| Causalidade, contrafactuais e simulação | `core/cognitivo/causal_world_model.py`, `core/cognitivo/scm_counterfactual.py`, `core/cognitivo/counterfactual.py`, `core/cognitivo/mental_simulation.py`; `test_causal_world_model_extraction_contract.py`, `test_scm_counterfactual_contract.py`, `test_counterfactual_contract.py`, `test_mental_simulation_contract.py` | Há algoritmos e contratos locais para intervenções, comparação de cenários e simulação. Não equivalem a modelos causais validados no mundo. | Não instrumentado no snapshot de recuperação. |
| Planejamento MCTS | `core/cognitivo/mcts_planner.py`, `core/cognitivo/action_effects.py`; `test_mcts_planner_quality_contract.py`, `test_mcts_action_contract.py`, `test_mcts_world_effects_contract.py` | Há contratos de entradas, horizonte e efeitos/recompensas modelados; rollouts e recompensas continuam heurísticos, sem evidência de planejamento geral. | Não instrumentado no snapshot de recuperação. |
| Predição e aprendizagem incremental | `core/cognitivo/predictive_coding.py`, `core/cognitivo/incremental_learner.py`; `test_predictive_learning_contract.py`, `test_predictive_coding_extraction_contract.py`, `test_incremental_learner_contract.py` | Há contratos sobre estados internos e atualizações definidas; isso não demonstra aprendizagem robusta em dados externos ou generalização. | Não instrumentado no snapshot de recuperação. |
| Analogia, projeção de futuro e ponderação de memória | `core/cognitivo/analogical_reasoning.py`, `core/cognitivo/episodic_future_thinking.py`, `core/cognitivo/emotional_memory_weighting.py`; `test_analogical_reasoning_contract.py`, `test_episodic_future_thinking_contract.py`, `test_emotional_memory_weighting_contract.py` | Existem componentes e testes de contrato específicos; nomes de módulos não demonstram experiência subjetiva, emoção humana ou previsão confiável. | Não instrumentado no snapshot de recuperação. |
| Backends multimodais | `core/constitutional_brain.py`; `test_multimodal_backend_contract.py`, `test_sensorimotor_multimodal_contract.py` | Os testes distinguem backend disponível de fallback explícito. Sem backend disponível, não se deve apresentar percepção/transcrição como realizada. | Não instrumentado no snapshot de recuperação. |
| Resiliência operacional | `core/vram_defense_guard.py`, `core/deferred_task_queue.py`; `test_vram_defense_guard.py`, `test_deferred_task_queue.py`, `test_hardware_telemetry_contract.py` | Há contratos para pressão de recursos, mitigação, retry e telemetria delimitada. Simulações e disponibilidade de hardware devem ser rotuladas. | Não instrumentado neste snapshot. Registros Vita se vinculam ao diagnóstico de recuperação; não atestam defesa VRAM nem resiliência geral. |
| Diagnóstico de recuperação | `vita/nexus_constitutional_bridge_v3.py`, `tools/diagnostics_dashboard.py`; `test_recovery_diagnostics_contract.py` | A ponte exporta e valida `nexus.recovery.diagnostics.v1`; o teste cobre schema, métricas, CLI e HTML local. | Parcial somente quando há `policy_transition` ou `telemetry`; sem observação quando ausentes. Snapshot operacional limitado, não fluxo cognitivo ao vivo. |

A tabela é um inventário candidato de áreas sobrepostas, não uma lista exaustiva de métodos, componentes ativos numa execução, nem placar de inteligência. A existência do arquivo de teste indica que há um contrato escrito; não afirma, por si só, que a suíte foi executada nesta tarefa ou que todo caminho de produção foi testado.

### O que os números atuais realmente contam

`NexusConstitutionalBridge.export_recovery_diagnostics()` exporta schema, horário de geração, fonte, diagnósticos e até os últimos `limit` eventos de auditoria (padrão 128). Os eventos têm `event_type`, `timestamp` e `payload`. A análise usa os eventos recentes e, se solicitada, aplica uma janela temporal; o array `events` exportado não recebe esse mesmo filtro de janela.

- `events_analyzed` é a quantidade de registros considerados após o limite e eventual filtro de janela — não é o total histórico do Nexus.
- `severity_counts` conta uma classe por evento analisado; eventos sem regra específica entram como informativos. `severity` resume essa distribuição.
- `pauses` conta transições para `paused` a partir de outro estado; `recoveries` conta transições de `paused` para `active`.
- `recovery_rate` é implementado como `recoveries / pauses`, ou `0.0` quando não há pausa. É uma razão, não necessariamente uma porcentagem ou probabilidade: pode superar 1 em sequências incompletas, e o zero também representa denominador ausente.
- `avg_pause_seconds` e `total_pause_seconds` consideram somente pares de pausa/retomada encontrados no recorte analisado; um par cortado pelo limite ou pela janela pode faltar. `open_pause` é a pausa ainda aberta dentro do recorte, não necessariamente o estado de todo o sistema.
- `critical_events` agrega pausas e certos alertas de telemetria de reprocessamento. Não é uma medida de risco cognitivo geral.
- O HTML atual mostra contagens/indicadores e metadados, e deliberadamente não renderiza `events[].payload`. O CLI textual continua padrão; HTML é opt-in com `--format html`.

Esses valores são utilizáveis em cartões e gráficos do snapshot real, desde que a tela mostre janela/limite, origem e horário de geração, denominações e o rótulo “snapshot”. Um ponto com valor zero nunca deve significar automaticamente “a função está inativa” ou “nada aconteceu”. Tendências de longo prazo e atualização ao vivo não podem ser derivadas de um único snapshot limitado.

## Modelo proposto de evidência e maturidade

Cada cartão do catálogo teria duas classificações independentes para evitar misturar implementação com atividade observada:

1. **Evidência de implementação:** `implementado/testado` quando o código está presente e ao menos um teste de contrato exercita a afirmação delimitada; `parcial` quando só alguns caminhos, fallbacks ou condições são cobertos, ou a função é explicitamente heurística/condicional; `planejado` quando há apenas proposta de comportamento futuro. Um teste aprovado não demonstra eficácia fora do contrato.
2. **Observabilidade de runtime:** `instrumentado` quando há evento/métrica versionado, teste do produtor e denominadores verificáveis; `parcial` quando só um caminho ou subconjunto de campos aparece; `não instrumentado` quando não há sinal runtime confiável. Esse último rótulo descreve ausência de telemetria, não ausência da função.

No catálogo v1, o denominador do resumo seria **N = número de IDs de área explicitamente cadastrados**, uma linha por ID, sem contar classes, métodos, arquivos, testes ou eventos. Todas as linhas entram exatamente uma vez em cada eixo; a soma dos estados de implementação é N, e a soma dos estados de observabilidade também é N. A versão do catálogo precisa ser mostrada. Se o escopo mudar, cria-se uma versão nova; não se altera retroativamente o denominador. O inventário acima contém 13 cartões candidatos, mas só passa a ser o denominador oficial quando o catálogo for aprovado e versionado.

O painel deve exibir números absolutos e denominador (por exemplo, “3/13 áreas com algum sinal de runtime”), com a lista das áreas atrás do resumo. Não agregar resultados de testes numa porcentagem de “cognição pronta”. Contagens de testes só são comparáveis com a mesma seleção, comando, revisão e execução; skips e falhas precisam aparecer. “Planejado” neste documento é sugestão, não cronograma nem autorização para instrumentar.

## Gráficos: snapshot atual versus instrumentação nova

| Visualização | Pode usar dados reais atuais? | Contrato e limites |
|---|---|---|
| Cartões de eventos analisados, severidade, pausas, recuperações, duração média e estado observado | Sim, do snapshot v1 | Mostrar janela/limite, horário e denominador; manter razão de recuperação como numerador/denominador, inclusive `0/0` ou “sem pausas”. |
| Barras por severidade e barras de pausas/recuperações | Sim | São contagens do recorte. Não converter a razão em pizza/progresso de “pronto”. |
| Linha do tempo curta de eventos Vita | Sim, de forma restrita no HTML estático local | O rastro usa apenas `policy_transition` e `telemetry`, mostra rótulos traduzidos e timestamp Unix da ocorrência, preserva a ordem de registro e omite payloads/campos arbitrários e tipos não permitidos. É limitado aos eventos do recorte exportado; não é feed ao vivo. |
| Mapa de funções por evidência de código/testes | Sim, como catálogo estático revisado para uma revisão do repositório | Não representa atividade em execução. Referenciar paths/testes e fixar a revisão; não inferir “ativo” de classe existente. |
| Gráfico de atenção, memória, predição, planejamento ou uso multimodal em runtime | Não | Requer produtores e contratos de telemetria específicos para cada função. Não usar números de teste como substituto de uso em produção. |
| Evolução histórica, latência por etapa e estado ao vivo | Não | Requer coleta com timestamps/ordenação, retenção, definição de unidade, denominadores, origem e mecanismo de atualização; snapshot v1 não é esse fluxo. |

## Substituto seguro para “pensamento interno”

A área visual deve se chamar **Rastro de eventos observáveis** ou **Registro de decisões e transições**. Ela mostra fatos verificáveis de execução: etapa/componente, evento permitido, resultado delimitado, horário, regra/limiar aplicado e referência opaca para a evidência que sustenta a decisão. Um exemplo futuro seria “ação bloqueada · permissão ausente · política `tool_permission_required` · evidência `ref:…`”, desde que esses campos sejam efetivamente emitidos e testados. Isso informa o motivo operacional sem simular uma voz interna.

A especificação proposta para novos eventos inclui: schema e versão; identificador de execução opaco; sequência monotônica por execução; timestamp; componente e tipo de evento de allowlist; estado/resultado enumerado; `reason_code` estruturado; valor e denominador apenas quando definidos; referência opaca/hash de evidência; versão do produtor. A interface deve mostrar somente campos explicitamente permitidos. Os valores textuais livres, `reason` atual, `policy` e demais `payloads` não são uma fonte segura para apresentação: a ponte persiste dicionários arbitrários no audit log, e truncar texto não é sanitização.

Nunca exibir prompts, cadeia de pensamento, segredos, credenciais, conteúdo bruto de ferramenta/memória, dados pessoais ou payloads arbitrários. Não inventar justificativas quando o produtor não emitiu um código verificável; mostrar “motivo estruturado indisponível”. Validar enumerações e referências antes de renderizar, omitir campos desconhecidos e falhar fechado. Agregação por padrão, limite de retenção, minimização e controle de acesso precisam ser definidos antes de qualquer feed persistente ou ao vivo.

## Analogia temporal adequada

Use uma **linha do tempo de caixa-preta operacional / diário de bordo do software**, não um EEG. Ela representa eventos que componentes registraram, com pontos, transições e intervalos sem sinal; não mede ondas cerebrais, atividade contínua, consciência ou experiência subjetiva. No snapshot atual, a linha deve dizer “últimos eventos incluídos no recorte”, não “agora”. Um feed futuro só pode dizer “ao vivo” quando tiver mecanismo de transporte e indicador de frescor, lacunas de sequência e desconexão; silêncio de telemetria não pode ser desenhado como inatividade.

## Fases pequenas e critérios de aceitação

### Fase 1 — Contrato do catálogo estático

**Entrega proposta:** IDs estáveis para as áreas, revisão de paths/testes, as duas classificações independentes e denominador versionado. Sem alterar produtores runtime.

**Aceitação:** cada ID tem fonte, teste de contrato e limite descrito; `N` é reproduzível e todas as linhas têm exatamente um estado por eixo; referências ausentes/obsoletas são marcadas, não preenchidas por inferência; a interface distingue revisão do catálogo de estado runtime.

### Fase 2 — Contrato de evento seguro para um fluxo Vita

**Entrega proposta:** acrescentar uma representação exportável segura para transições de política/telemetria sem mudar o consumidor v1. Manter payload bruto fora da nova projeção visual. Instrumentar primeiro um fluxo estreito, com enums e códigos de motivo definidos.

**Aceitação:** testes provam sequência/tempo e contadores com denominadores definidos; eventos desconhecidos ou tipos de campo inválidos são omitidos/rejeitados sem exposição; inserir sentinelas de segredo/payload no registro bruto não as faz aparecer no JSON/HTML seguro; testes cobrem fluxo sem eventos, janela vazia, recorte incompleto e timestamp inválido; o contrato v1 e a saída textual default permanecem compatíveis.

### Fase 3 — Rastro e gráfico do snapshot

**Entrega proposta:** acrescentar uma visualização acessível do rastro Vita seguro e o painel de evidência do catálogo. Continuar gerando HTML estático de um snapshot, sem assumir rede, servidor ou JavaScript.

**Aceitação:** cada contagem é reconciliável ao recorte que a gerou; a tela mostra período, origem e denominadores; nenhum payload bruto, pensamento interno fabricado ou estado de outras áreas é renderizado; valores ausentes, zero observações e `0/0` têm rótulos distintos; leitura por teclado e versão móvel preservadas.

**Incremento local nesta branch:** o HTML inclui um rastro estático derivado do snapshot já validado. A allowlist contém somente `policy_transition` e `telemetry`; o rastro mostra seus rótulos em português e o timestamp Unix da ocorrência, em ordem de registro. Exibe até 20 eventos permitidos mais recentes do recorte, preservando a ordem entre eles, e informa quantos registros foram omitidos por tipo não permitido ou pelo limite. Snapshot sem eventos e snapshot contendo apenas tipos não permitidos têm estados textuais diferentes. Payload, nomes de tipos não permitidos e campos adicionais não são exibidos. O schema v1, o produtor, o CLI textual padrão e suas saídas permanecem inalterados; não há JavaScript, dependência ou feed ao vivo. A revisão visual manual de teclado/dispositivos móveis não faz parte desta verificação local.

### Fase 4 — Instrumentar áreas adicionais, uma por vez

**Entrega proposta:** escolher cada função por necessidade concreta e criar um evento mínimo com produtor, contrato e limites de privacidade antes de desenhar seu gráfico. Não instrumentar todo o monólito de uma vez.

**Aceitação por função:** evento só é emitido quando o caminho correspondente realmente executa; testes positivos, negativos e de fallback confirmam campos/denominadores; execução sem backend ou sem eventos permanece explicitamente “sem observação”; a página não transforma score heurístico em eficácia; impacto de armazenamento/custo e retenção é medido antes de ampliar coleta.

### Fase 5 — Atualização contínua, somente se aprovada

**Entrega proposta:** definir transporte, buffering, sequência, reconexão, retenção, privacidade e indicador de frescor; então considerar atualização periódica. Não pressupor framework novo.

**Aceitação:** a interface evidencia atraso, desconexão e lacunas; não chama snapshot antigo de “agora”; falha ou indisponibilidade do feed não altera a saída textual nem invalida diagnósticos estáticos; testes cobrem reordenação, duplicatas, perda e retomada; controles de acesso e retenção são definidos antes da operação.

## Limites e decisão em aberto

A decisão de produto para etapas além deste incremento continua em aberto. Esta branch local acrescenta somente o catálogo estático e um rastro HTML limitado ao que o snapshot v1 já contém; não cria evento seguro novo nem instrumentação. Qualquer expansão para telemetria cognitiva, armazenamento adicional ou atualização ao vivo exige contrato e autorização próprios.

Nada nesta especificação declara o Nexus, seus módulos cognitivos, seus testes ou seu fluxo de recuperação completos, eficazes em geral, conscientes, em tempo real ou prontos para produção.


## Contrato local do catálogo e dos rótulos do dashboard

A extensão local do dashboard usa o catálogo estático e versionado `nexus.function-evidence.v1`, com IDs estáveis e rótulos em português. As 13 áreas deste inventário continuam candidatas; a versão do catálogo não aprova o inventário como exaustivo nem como medida de maturidade. O denominador exibido é `N = quantidade de IDs cadastrados` (não total de testes, métodos, eventos ou funções cognitivas), e os resultados são contagens `n/N`, sem porcentagem de inteligência, consciência ou prontidão.

Esse denominador é estritamente administrativo, de registros no catálogo candidato. As áreas se sobrepõem por comportamento e incluem funções fora de `core/cognitivo/`; as 13 linhas não constituem uma partição única defensável do universo funcional. Critérios exclusivos, atribuição primária e unidade de contagem funcional ainda não foram aprovados. Até isso ocorrer, não derivar gráfico de pizza, percentual ou cobertura funcional de `n/13`, nem converter contagens de módulos/testes em prontidão. A ressalva de revisão independente é estática e não constitui resultado de testes.

No `HEAD` exato `632ce8456ff76810654aa19b950ed1960e9754d0`, `git ls-tree` identifica 17 arquivos `.py` em `core/cognitivo/`, excluindo `__init__.py`; todos os 17 paths estão referenciados no catálogo. **17 módulos** e **13 áreas candidatas** são unidades distintas. A referência local atrasada `origin/main` não foi usada para essa contagem.

O primeiro eixo apresenta evidência de implementação/contrato como referências existentes a código e testes. Uma referência não afirma que o teste foi executado nesta revisão, que todos os caminhos estão integrados, nem que há eficácia geral. O segundo eixo é calculado por snapshot: só usa `event_type` dos registros validados. `Parcial` exige ao menos um evento allowlisted relevante no snapshot; área com evento esperado mas nenhum registro no recorte diz `Sem observação neste snapshot (0 eventos)`; área sem sinal correspondente diz `Não instrumentado`. A ausência de telemetria não significa que a função não exista. Nenhum `payload` de evento é lido para essa classificação ou renderizado.

O mapa local explícito `event_type → área` é conservador:

| `event_type` presente | ID correspondente | O que evidencia neste recorte |
|---|---|---|
| `policy_transition` | `recovery-diagnostics` | Fluxo operacional de política/recuperação |
| `telemetry` | `recovery-diagnostics` | Registro operacional de reprocessamento Vita |

As outras 12 áreas permanecem `Não instrumentado` por esse snapshot. Em particular, esses eventos não atestam execução de roteamento/fila, defesa VRAM ou função cognitiva genérica; um registro não torna todos os cartões `Parcial`. Tipos alheios e alegações contidas no payload não alteram a classificação.

Os estados `0`, `Desconhecido`, `Sem denominador` e `Não instrumentado` são distintos. Contagem zero permanece `0`; campo ausente/indeterminado é `Desconhecido`; razões de recuperação com zero pausas são `Sem denominador (0/0)`, e não `0.000`; estado sem telemetria compatível é `Não instrumentado`. Quando houver pausas, a razão é acompanhada por numerador/denominador. Rótulos textuais e tabelas HTML semânticas tornam estados e contagens compreensíveis sem depender de cor.

### Rastro HTML estático de eventos (incremento local)

O rastro é uma lista ordenada, gerada somente a partir de `events[]` depois da validação do snapshot v1. Ele não classifica eventos com base no payload. A allowlist de apresentação é fechada: `policy_transition` aparece como “Transição de política” e `telemetry` como “Telemetria de reprocessamento”; esses tipos continuam associados somente a `recovery-diagnostics`. Nenhum campo além do tipo permitido e do timestamp da ocorrência é usado para compor cada linha. O timestamp é apresentado como valor Unix em segundos, sem conversão para fuso horário.

Quando há mais de 20 eventos permitidos, o rastro exibe os 20 permitidos mais recentes dentro do array recebido, preservando a ordem de registro original, e informa quantos anteriores foram omitidos pelo limite. Tipos não permitidos nunca têm seus nomes ou conteúdo mostrados; sua quantidade é informada separadamente. Um array vazio mostra “Nenhum evento incluído neste snapshot”; um array não vazio sem tipos permitidos mostra “Nenhum evento permitido neste snapshot” e a contagem omitida. Nenhum desses estados indica atividade atual ou ausência de atividade no sistema. A lista diz respeito apenas ao recorte presente no snapshot, não a um “agora” nem a um feed ao vivo.

Por compatibilidade, a saída textual padrão foi mantida literalmente: o renderer legado ainda usa zero para contadores opcionais ausentes. A distinção visual de campos ausentes e `0/0` deste incremento vale para o HTML; o comportamento textual legado é uma limitação conhecida, não evidência de zero observações.

O snapshot continua identificado por origem e horário de geração, sem ser apresentado como atividade atual. Os gráficos de severidade mostram as três classes canônicas do v1 e agregam chaves adicionais aceitas pelo validador em uma linha visível `Não reconhecida`; essa linha participa da escala e os nomes extras não são renderizados. Assim, nenhum valor oculto redimensiona as barras visíveis, e o schema `nexus.recovery.diagnostics.v1` não muda.

Como referência de projeto, a página OpenTelemetry de Events declara status **Development**; não é uma norma estável ou obrigatória do Nexus [1]. Suas recomendações úteis incluem nome estável e documentado, sem valores dinâmicos no nome, timestamp da ocorrência e atributos estruturados para valores variáveis; operações com duração são representadas por spans [1] [2]. Para instrumentação futura, o Nexus deve primeiro definir um esquema local pequeno e versionado e só então decidir eventual compatibilidade OTel. Esta extensão apenas consome o snapshot v1 existente, sem adicionar OpenTelemetry nem outra dependência. A orientação USWDS favorece visualizações simples, contexto em linguagem clara e uma tabela acessível equivalente [3].

Referências consultadas:

[1]: https://opentelemetry.io/docs/specs/semconv/general/events/ "OpenTelemetry — Semantic conventions for events"
[2]: https://opentelemetry.io/docs/specs/semconv/general/trace/ "OpenTelemetry — Trace semantic conventions"
[3]: https://designsystem.digital.gov/components/data-visualizations/ "U.S. Web Design System — Data visualizations"
