"""Renderizador determinístico para snapshots de recuperação do Nexus.

O dashboard é deliberadamente operacional: apresenta métricas observáveis do
contrato versionado e não infere capacidades cognitivas a partir delas.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vita.nexus_constitutional_bridge_v3 import NexusConstitutionalBridge


@dataclass(frozen=True)
class FunctionEvidenceEntry:
    """Entrada candidata do catálogo; referências não afirmam execução de testes."""

    id: str
    label: str
    code_refs: tuple[str, ...]
    test_refs: tuple[str, ...]
    implementation_limit: str


FUNCTION_EVIDENCE_CATALOG_VERSION = "nexus.function-evidence.v1"
FUNCTION_EVIDENCE_CATALOG = (
    FunctionEvidenceEntry(
        "routing-and-deferred-tasks", "Roteamento e fila de tarefas",
        ("core/constitutional_brain.py", "core/deferred_task_queue.py"),
        ("test_central_router_contract.py", "test_deferred_task_queue.py", "test_deferred_task_snapshot.py"),
        "Snapshot de fila não comprova entrega distribuída ou exactly-once.",
    ),
    FunctionEvidenceEntry(
        "tool-decisions-and-permissions", "Decisão e permissões de ferramentas",
        ("core/cognitivo/decision_cycle.py", "core/cognitivo/action_effects.py", "core/tool_evidence_gate.py"),
        ("test_decision_cycle_contract.py", "test_action_effects_contract.py", "test_tool_permission_failclosed_contract.py", "test_evidence_omission_contract.py"),
        "Contratos cobrem seleção e bloqueios delimitados; não medem qualidade no mundo.",
    ),
    FunctionEvidenceEntry(
        "attention-and-workspace", "Atenção e workspace",
        ("core/cognitivo/attention.py", "core/constitutional_brain.py"),
        ("test_attention_contract.py", "test_attention_signal_contract.py", "test_global_workspace_contract.py"),
        "Sinais opcionais de contraste/movimento mantêm fallback heurístico.",
    ),
    FunctionEvidenceEntry(
        "working-memory", "Memória de trabalho",
        ("core/cognitivo/working_memory.py",),
        ("test_working_memory_contract.py",),
        "Contratos delimitam capacidade e alocação; não demonstram memória humana.",
    ),
    FunctionEvidenceEntry(
        "episodic-and-contextual-memory", "Memória episódica e contextual",
        ("core/cognitivo/episodic_memory.py", "core/cognitivo/contextual_memory.py"),
        ("test_episodic_retrieval_contract.py", "test_episodic_memory_statistics_contract.py", "test_episodic_memory_concurrency_contract.py", "test_contextual_memory_contract.py"),
        "Resultados dependem do backend e dos dados disponíveis.",
    ),
    FunctionEvidenceEntry(
        "semantic-knowledge-and-temporal-facts", "Conhecimento semântico e fatos temporais",
        ("core/cognitivo/semantic_networks.py", "core/temporal_facts.py"),
        ("test_knowledge_graph_contract.py", "test_temporal_facts_contract.py", "test_temporal_graph_integration_contract.py", "test_semantic_networks_contract.py"),
        "Contratos não validam verdade externa nem consistência distribuída.",
    ),
    FunctionEvidenceEntry(
        "causal-modeling-and-simulation", "Causalidade, contrafactuais e simulação",
        ("core/cognitivo/causal_world_model.py", "core/cognitivo/scm_counterfactual.py", "core/cognitivo/counterfactual.py", "core/cognitivo/mental_simulation.py"),
        ("test_causal_world_model_extraction_contract.py", "test_scm_counterfactual_contract.py", "test_counterfactual_contract.py", "test_mental_simulation_contract.py"),
        "Algoritmos e contratos locais não equivalem a modelos causais validados no mundo.",
    ),
    FunctionEvidenceEntry(
        "mcts-planning", "Planejamento MCTS",
        ("core/cognitivo/mcts_planner.py", "core/cognitivo/action_effects.py"),
        ("test_mcts_planner_quality_contract.py", "test_mcts_action_contract.py", "test_mcts_world_effects_contract.py"),
        "Rollouts e recompensas heurísticos não provam planejamento geral.",
    ),
    FunctionEvidenceEntry(
        "prediction-and-incremental-learning", "Predição e aprendizagem incremental",
        ("core/cognitivo/predictive_coding.py", "core/cognitivo/incremental_learner.py"),
        ("test_predictive_learning_contract.py", "test_predictive_coding_extraction_contract.py", "test_incremental_learner_contract.py"),
        "Atualizações internas não demonstram aprendizagem robusta nem generalização.",
    ),
    FunctionEvidenceEntry(
        "analogy-and-future-thinking", "Analogia e projeção de futuro",
        ("core/cognitivo/analogical_reasoning.py", "core/cognitivo/episodic_future_thinking.py", "core/cognitivo/emotional_memory_weighting.py"),
        ("test_analogical_reasoning_contract.py", "test_episodic_future_thinking_contract.py", "test_emotional_memory_weighting_contract.py"),
        "Módulos e contratos não demonstram previsão confiável ou experiência subjetiva.",
    ),
    FunctionEvidenceEntry(
        "multimodal-backends", "Backends multimodais",
        ("core/constitutional_brain.py",),
        ("test_multimodal_backend_contract.py", "test_sensorimotor_multimodal_contract.py"),
        "Backends opcionais e fallbacks permanecem categorias distintas.",
    ),
    FunctionEvidenceEntry(
        "operational-resilience", "Resiliência operacional",
        ("core/vram_defense_guard.py", "core/deferred_task_queue.py"),
        ("test_vram_defense_guard.py", "test_deferred_task_queue.py", "test_hardware_telemetry_contract.py"),
        "Simulações e telemetria parcial não equivalem a resiliência geral.",
    ),
    FunctionEvidenceEntry(
        "recovery-diagnostics", "Diagnóstico de recuperação",
        ("vita/nexus_constitutional_bridge_v3.py", "tools/diagnostics_dashboard.py"),
        ("test_recovery_diagnostics_contract.py",),
        "O snapshot é limitado e não representa um feed ao vivo nem eficácia geral.",
    ),
)

# Eventos Vita não atestam execução de roteamento, defesa VRAM ou cognição.
RUNTIME_EVENT_AREA_MAP = {
    "policy_transition": ("recovery-diagnostics",),
    "telemetry": ("recovery-diagnostics",),
}

EVENT_TRACE_LABELS = {
    "policy_transition": "Transição de política",
    "telemetry": "Telemetria de reprocessamento",
}
MAX_VISIBLE_TRACE_EVENTS = 20


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("JSON float must decode to a finite value")
    return value


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant: {token}")


def load_snapshot(source: str) -> Any:
    if source == "-":
        return json.load(sys.stdin, parse_float=_parse_finite_float, parse_constant=_reject_json_constant)
    return json.loads(
        Path(source).read_text(encoding="utf-8"),
        parse_float=_parse_finite_float, parse_constant=_reject_json_constant,
    )


def render_dashboard(snapshot: Any) -> str:
    bridge = NexusConstitutionalBridge(":memory:")
    validation = bridge.validate_recovery_diagnostics(snapshot)
    if not validation["valid"]:
        raise ValueError("invalid recovery diagnostics: " + ", ".join(validation["errors"]))
    diagnostics = snapshot["diagnostics"]
    severity_counts = diagnostics.get("severity_counts", {})
    lines = [
        "NEXUS RECOVERY DASHBOARD",
        f"schema: {snapshot['schema']}",
        f"generated_at: {snapshot['generated_at']:.3f}",
        f"severity: {diagnostics['severity']}",
        f"events_analyzed: {diagnostics['events_analyzed']}",
        f"latest_state: {diagnostics.get('latest_state', 'unknown')}",
        f"pauses: {diagnostics.get('pauses', 0)}",
        f"recoveries: {diagnostics.get('recoveries', 0)}",
        f"recovery_rate: {diagnostics.get('recovery_rate', 0.0):.3f}",
        f"open_pause: {diagnostics.get('open_pause', False)}",
        f"avg_pause_seconds: {diagnostics.get('avg_pause_seconds', 0.0):.3f}",
        f"critical_events: {diagnostics.get('critical_events', 0)}",
        f"severity_counts: info={severity_counts.get('info', 0)} warning={severity_counts.get('warning', 0)} critical={severity_counts.get('critical', 0)}",
        f"audit_events: {len(snapshot['events'])}",
    ]
    return "\n".join(lines)


def _chart_row(label: str, count: int | None, maximum: int) -> str:
    width = (count * 100 // maximum) if count is not None and maximum else 0
    count_label = "Desconhecido" if count is None else str(count)
    return (
        "<tr>"
        f'<th scope="row">{html.escape(label, quote=True)}</th>'
        '<td><span class="bar-track" aria-hidden="true">'
        f'<span class="bar-fill" style="width: {width}%"></span></span></td>'
        f'<td class="count">{html.escape(count_label, quote=True)}</td>'
        "</tr>"
    )


def _recovery_ratio_label(diagnostics: dict[str, Any]) -> str:
    if "pauses" not in diagnostics or "recoveries" not in diagnostics:
        return "Desconhecido"
    pauses = diagnostics["pauses"]
    recoveries = diagnostics["recoveries"]
    if pauses == 0:
        return f"Sem denominador ({recoveries} recuperações / 0 pausas)"
    return f"{recoveries}/{pauses} (recuperações/pausas)"


def _render_ref_list(paths: tuple[str, ...]) -> str:
    if not paths:
        return "<span>Desconhecido</span>"
    items = "".join(f"<li><code>{html.escape(path, quote=True)}</code></li>" for path in paths)
    return f'<ul class="refs">{items}</ul>'


def _catalog_fraction(count: int, denominator: int) -> str:
    if denominator == 0:
        return "Sem denominador (0 IDs)"
    return f"{count}/{denominator}"


def _runtime_visibility(entry: FunctionEvidenceEntry, area_event_counts: dict[str, int]) -> tuple[str, str, str]:
    if not any(entry.id in area_ids for area_ids in RUNTIME_EVENT_AREA_MAP.values()):
        return (
            "not_instrumented", "Não instrumentado",
            "O snapshot v1 não oferece um sinal de runtime correspondente a esta área.",
        )
    observed = area_event_counts.get(entry.id, 0)
    if observed == 0:
        return (
            "no_observation", "Sem observação neste snapshot (0 eventos)",
            "Há tipos de evento relacionados, mas nenhum aparece neste recorte.",
        )
    return (
        "partial", "Parcial",
        f"{observed} registro(s) operacional(is) de política/recovery/reprocessamento; não atestam execução cognitiva.",
    )


def _render_function_evidence_section(snapshot: dict[str, Any]) -> str:
    area_event_counts = {
        area_id: 0
        for area_ids in RUNTIME_EVENT_AREA_MAP.values()
        for area_id in area_ids
    }
    for event in snapshot["events"]:
        event_type = event.get("event_type")
        for area_id in RUNTIME_EVENT_AREA_MAP.get(event_type, ()):
            area_event_counts[area_id] += 1

    denominator = len(FUNCTION_EVIDENCE_CATALOG)
    with_references = sum(bool(entry.code_refs and entry.test_refs) for entry in FUNCTION_EVIDENCE_CATALOG)
    runtime_states = {"partial": 0, "no_observation": 0, "not_instrumented": 0}
    rows = []
    for entry in FUNCTION_EVIDENCE_CATALOG:
        state, runtime_label, runtime_note = _runtime_visibility(entry, area_event_counts)
        runtime_states[state] += 1
        rows.append(
            "<tr>"
            f'<th scope="row" id="catalog-{html.escape(entry.id, quote=True)}">'
            f'<code>{html.escape(entry.id, quote=True)}</code><br>{html.escape(entry.label, quote=True)}</th>'
            f"<td>{_render_ref_list(entry.code_refs)}</td>"
            f"<td>{_render_ref_list(entry.test_refs)}</td>"
            '<td><strong>Código e contrato referenciados</strong><br>'
            f'{html.escape(entry.implementation_limit, quote=True)}</td>'
            f'<td><strong>{html.escape(runtime_label, quote=True)}</strong><br>'
            f'{html.escape(runtime_note, quote=True)}</td>'
            "</tr>"
        )

    implementation_summary = (
        f"{_catalog_fraction(with_references, denominator)} IDs têm referências a código e teste de contrato. "
        "Referenciar um teste não afirma que ele foi executado nesta revisão."
    )
    runtime_summary = (
        f"Parcial: {_catalog_fraction(runtime_states['partial'], denominator)}; "
        f"sem observação: {_catalog_fraction(runtime_states['no_observation'], denominator)}; "
        f"não instrumentado: {_catalog_fraction(runtime_states['not_instrumented'], denominator)}."
    )
    return f"""<section id="functions-and-evidence" aria-labelledby="functions-and-evidence-title">
      <div class="section-heading">
        <p class="eyebrow">Catálogo candidato · {html.escape(FUNCTION_EVIDENCE_CATALOG_VERSION, quote=True)}</p>
        <h2 id="functions-and-evidence-title">Funções e evidências</h2>
        <p class="intro">Inventário estático versionado, ainda não aprovado como lista exaustiva. Denominador N = {denominator} IDs cadastrados; não é porcentagem de inteligência, consciência ou prontidão.</p>
        <p class="intro">As áreas se sobrepõem: n/N conta somente registros candidatos e não mede cobertura de funções. Critérios exclusivos, atribuição primária e unidade de contagem funcional ainda dependem de aprovação.</p>
      </div>
      <div class="catalog-summary card">
        <h3>Evidência de implementação/contrato</h3>
        <p>{html.escape(implementation_summary, quote=True)}</p>
        <h3>Visibilidade de runtime</h3>
        <p>{html.escape(runtime_summary, quote=True)}</p>
        <p>O mapa <code>event_type → área</code> atribui <code>policy_transition</code> e <code>telemetry</code> somente ao diagnóstico operacional de recuperação. Não demonstra execução de funções cognitivas, roteamento ou defesa VRAM. Zero eventos não é tratado como área não instrumentada; payloads não são lidos nem exibidos.</p>
      </div>
      <div class="table-scroll" role="region" tabindex="0" aria-label="Tabela de funções e evidências">
        <table class="catalog-table">
          <caption>Um registro por ID estável. Caminhos são referências a código e contratos, não resultado de execução de testes nesta revisão.</caption>
          <thead><tr><th scope="col">ID e área</th><th scope="col">Código</th><th scope="col">Testes/contratos</th><th scope="col">Evidência de implementação/contrato</th><th scope="col">Visibilidade de runtime</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>"""


def _unknown_events_omission_label(count: int) -> str:
    if count == 1:
        return "1 registro com tipo não permitido foi omitido"
    return f"{count} registros com tipo não permitido foram omitidos"


def _render_event_trace_section(snapshot: dict[str, Any]) -> str:
    """Renderiza somente eventos allowlisted do snapshot, sem ler payloads."""
    events = snapshot["events"]
    permitted_events = [
        event for event in events if event["event_type"] in EVENT_TRACE_LABELS
    ]
    unknown_count = len(events) - len(permitted_events)
    visible_events = permitted_events[-MAX_VISIBLE_TRACE_EVENTS:]
    truncated_count = len(permitted_events) - len(visible_events)

    if not events:
        status = "Nenhum evento incluído neste snapshot."
    elif not permitted_events:
        status = "Nenhum evento permitido neste snapshot"
        status += f"; {_unknown_events_omission_label(unknown_count)}."
    else:
        status = (
            f"Exibindo {len(visible_events)} de {len(permitted_events)} "
            "eventos permitidos neste recorte."
        )
        if unknown_count:
            status += f" {_unknown_events_omission_label(unknown_count)}."

    if truncated_count:
        if truncated_count == 1:
            status += (
                f" 1 evento permitido anterior foi omitido pelo limite de "
                f"{MAX_VISIBLE_TRACE_EVENTS} linhas visíveis."
            )
        else:
            status += (
                f" {truncated_count} eventos permitidos anteriores foram omitidos "
                f"pelo limite de {MAX_VISIBLE_TRACE_EVENTS} linhas visíveis."
            )

    rows = "".join(
        "<li>"
        f'<span class="event-timestamp">Timestamp da ocorrência (Unix): '
        f'{html.escape(str(event["timestamp"]), quote=True)} s</span> '
        f'<span class="event-label">{html.escape(EVENT_TRACE_LABELS[event["event_type"]], quote=True)}</span>'
        "</li>"
        for event in visible_events
    )
    event_list = f'<ol class="event-trace">{rows}</ol>' if rows else ""
    return f"""<section id="event-trace" aria-labelledby="event-trace-title">
      <div class="section-heading">
        <p class="eyebrow">Recorte do snapshot</p>
        <h2 id="event-trace-title">Rastro de eventos observáveis</h2>
        <p class="intro">Eventos permitidos incluídos neste recorte, em ordem de registro. O timestamp é o da ocorrência, em Unix (segundos). Não é um feed ao vivo.</p>
      </div>
      <p class="trace-status">{html.escape(status, quote=True)}</p>
      {event_list}
    </section>"""


def render_html_dashboard(snapshot: Any) -> str:
    """Renderiza um HTML estático e autocontido para um snapshot validado."""
    bridge = NexusConstitutionalBridge(":memory:")
    validation = bridge.validate_recovery_diagnostics(snapshot)
    if not validation["valid"]:
        raise ValueError("invalid recovery diagnostics: " + ", ".join(validation["errors"]))

    diagnostics = snapshot["diagnostics"]
    severity_counts = diagnostics["severity_counts"]
    severity_labels = {"info": "Informativa", "warning": "Alerta", "critical": "Crítica"}
    extra_severity_count = sum(
        count for name, count in severity_counts.items() if name not in severity_labels
    )
    severity_values = [severity_counts[name] for name in severity_labels]
    severity_values.append(extra_severity_count)
    severity_maximum = max(severity_values)
    severity_rows = "\n".join(
        _chart_row(severity_labels[name], severity_counts[name], severity_maximum)
        for name in ("info", "warning", "critical")
    )
    severity_rows += "\n" + _chart_row("Não reconhecida", extra_severity_count, severity_maximum)

    pauses = diagnostics.get("pauses")
    recoveries = diagnostics.get("recoveries")
    pause_counts = [value for value in (pauses, recoveries) if value is not None]
    pause_maximum = max(pause_counts, default=0)
    recovery_rows = "\n".join((
        _chart_row("Pausas", pauses, pause_maximum),
        _chart_row("Recuperações", recoveries, pause_maximum),
    ))

    severity = html.escape(severity_labels[diagnostics["severity"]], quote=True)
    latest_state_value = diagnostics.get("latest_state")
    latest_state = html.escape(latest_state_value or "Desconhecido", quote=True)
    source = html.escape(snapshot["source"], quote=True)
    schema = html.escape(snapshot["schema"], quote=True)
    generated_at = snapshot["generated_at"]
    generated_label = f"{generated_at:.3f}" if isinstance(generated_at, float) else str(generated_at)
    recovery_ratio = html.escape(_recovery_ratio_label(diagnostics), quote=True)
    open_pause_value = diagnostics.get("open_pause")
    open_pause = "Desconhecido" if open_pause_value is None else "Sim" if open_pause_value else "Não"
    average_pause_value = diagnostics.get("avg_pause_seconds")
    average_pause = (
        "Desconhecido" if average_pause_value is None
        else html.escape(f"{average_pause_value:.3f} s", quote=True)
    )
    events_analyzed = html.escape(str(diagnostics["events_analyzed"]), quote=True)
    snapshot_events = html.escape(str(len(snapshot["events"])), quote=True)
    critical_events_value = diagnostics.get("critical_events")
    critical_events = "Desconhecido" if critical_events_value is None else html.escape(str(critical_events_value), quote=True)
    if "window_seconds" not in diagnostics:
        window_label = "Desconhecido"
    elif diagnostics["window_seconds"] is None:
        window_label = "Sem filtro temporal solicitado"
    else:
        window_label = f"{diagnostics['window_seconds']:.3f} s"
    window_label = html.escape(window_label, quote=True)
    event_trace_section = _render_event_trace_section(snapshot)
    function_evidence_section = _render_function_evidence_section(snapshot)

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nexus — Painel de recuperação</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; color: #17212b; background: #f3f6f8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-width: 280px; }}
    main {{ width: min(100% - 2rem, 1080px); margin: 0 auto; padding: 2rem 0 3rem; }}
    h1, h2, p {{ margin-top: 0; }}
    h1 {{ margin-bottom: .5rem; font-size: clamp(1.7rem, 4vw, 2.5rem); }}
    h2 {{ margin-bottom: .4rem; font-size: 1.15rem; }}
    .intro {{ color: #405262; max-width: 70ch; line-height: 1.5; }}
    .eyebrow {{ color: #365e78; font-size: .8rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }}
    .metrics, .charts {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin-top: 1.25rem; }}
    .card {{ background: #fff; border: 1px solid #d5dfe6; border-radius: .8rem; padding: 1.1rem; box-shadow: 0 2px 8px #17212b0a; }}
    .metric dt {{ color: #526474; font-size: .88rem; }}
    .metric dd {{ margin: .2rem 0 0; font-size: 1.35rem; font-weight: 700; overflow-wrap: anywhere; }}
    .metric .small {{ font-size: .95rem; }}
    figure {{ margin: 0; }}
    figcaption p {{ margin-bottom: .8rem; color: #526474; font-size: .9rem; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{ padding: .55rem .35rem; border-bottom: 1px solid #e5ebef; text-align: left; }}
    thead th {{ color: #526474; font-size: .8rem; }}
    tbody th {{ width: 30%; font-weight: 600; }}
    .count {{ width: 20%; text-align: right; font-variant-numeric: tabular-nums; }}
    .bar-track {{ display: block; width: 100%; height: .8rem; overflow: hidden; border-radius: 999px; background: #e6edf2; }}
    .bar-fill {{ display: block; height: 100%; min-width: 0; border-radius: inherit; background: #176b87; }}
    .note {{ margin-top: 1.25rem; padding: .9rem 1rem; border-left: 4px solid #176b87; background: #e8f2f6; line-height: 1.5; }}
    .meta {{ margin-top: 1rem; color: #526474; font-size: .82rem; line-height: 1.6; overflow-wrap: anywhere; }}
    nav {{ margin: 1rem 0; }}
    nav a {{ color: #145a75; font-weight: 650; }}
    .section-heading {{ margin-top: 2rem; }}
    .catalog-summary {{ margin: 1rem 0; line-height: 1.5; }}
    .catalog-summary h3 {{ margin: .75rem 0 .25rem; font-size: 1rem; }}
    .catalog-summary p {{ margin-bottom: .55rem; }}
    .table-scroll {{ overflow-x: auto; margin-top: 1rem; }}
    .catalog-table {{ table-layout: auto; font-size: .84rem; }}
    .catalog-table th, .catalog-table td {{ vertical-align: top; overflow-wrap: anywhere; }}
    .catalog-table caption {{ padding: .6rem 0; text-align: left; color: #405262; }}
    .catalog-table th[scope="row"] {{ min-width: 9rem; }}
    .refs {{ margin: 0; padding-left: 1.2rem; }}
    .refs li {{ margin-bottom: .25rem; }}
    .refs code, .catalog-table code {{ overflow-wrap: anywhere; }}
    .trace-status {{ margin: 1rem 0 .5rem; padding: .75rem; background: #e8f2f6; line-height: 1.5; }}
    .event-trace {{ margin: .5rem 0 0; padding-left: 1.5rem; }}
    .event-trace li {{ margin: .65rem 0; padding-left: .3rem; line-height: 1.5; }}
    .event-timestamp {{ display: block; color: #526474; font-size: .9rem; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
    .event-label {{ font-weight: 650; }}
    @media (max-width: 640px) {{ main {{ width: min(100% - 1.25rem, 1080px); padding-top: 1.25rem; }} .metrics, .charts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <p class="eyebrow">Snapshot de recuperação</p>
      <h1>Nexus — Painel de recuperação</h1>
      <p class="intro">Resumo dos indicadores observados neste snapshot; os gráficos mostram contagens, não uma avaliação geral do sistema.</p>
    </header>
    <nav aria-label="Navegação do painel"><a href="#event-trace">Rastro de eventos</a> · <a href="#functions-and-evidence">Funções e evidências</a></nav>
    <section class="metrics" aria-label="Resumo dos indicadores">
      <dl class="card metric"><dt>Severidade geral</dt><dd>{severity}</dd></dl>
      <dl class="card metric"><dt>Eventos analisados</dt><dd>{events_analyzed}</dd></dl>
      <dl class="card metric"><dt>Estado mais recente</dt><dd>{latest_state}</dd></dl>
      <dl class="card metric"><dt>Recuperações / pausas</dt><dd>{recovery_ratio}</dd></dl>
      <dl class="card metric"><dt>Pausa em aberto</dt><dd>{open_pause}</dd></dl>
      <dl class="card metric"><dt>Duração média da pausa</dt><dd>{average_pause}</dd></dl>
      <dl class="card metric"><dt>Eventos críticos</dt><dd>{critical_events}</dd></dl>
      <dl class="card metric"><dt>Eventos incluídos no snapshot</dt><dd>{snapshot_events}</dd></dl>
    </section>
    <section class="charts" aria-label="Gráficos do snapshot">
      <figure class="card">
        <figcaption><h2>Eventos por severidade</h2><p>Contagem por classe registrada no snapshot.</p></figcaption>
        <table>
          <thead><tr><th scope="col">Classe</th><th scope="col">Gráfico</th><th scope="col" class="count">Eventos</th></tr></thead>
          <tbody>{severity_rows}</tbody>
        </table>
      </figure>
      <figure class="card">
        <figcaption><h2>Pausas e recuperações</h2><p>Contagens de transições reconhecidas pelo diagnóstico.</p></figcaption>
        <table>
          <thead><tr><th scope="col">Tipo</th><th scope="col">Gráfico</th><th scope="col" class="count">Eventos</th></tr></thead>
          <tbody>{recovery_rows}</tbody>
        </table>
      </figure>
    </section>
    {event_trace_section}
    {function_evidence_section}
    <p class="note">Este MVP visualiza somente o snapshot de recuperação disponível; não é um fluxo ao vivo de todo o Nexus. Contagens zero não significam que a função esteja inativa.</p>
    <p class="meta">Schema: {schema} · Fonte: {source} · Snapshot gerado (Unix): {html.escape(generated_label, quote=True)} · Janela: {window_label} · Limite de exportação: não informado no schema v1 · Eventos incluídos: {snapshot_events}; eventos analisados podem diferir quando há janela temporal.</p>
  </main>
</body>
</html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="JSON snapshot path, or '-' for stdin")
    parser.add_argument(
        "--format", choices=("text", "html"), default="text",
        help="output format (default: text)",
    )
    args = parser.parse_args(argv)
    try:
        snapshot = load_snapshot(args.snapshot)
        rendered = render_html_dashboard(snapshot) if args.format == "html" else render_dashboard(snapshot)
        print(rendered)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
        print(f"dashboard error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
