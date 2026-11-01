"""Renderizador determinístico para snapshots de recuperação do Nexus.

O dashboard é deliberadamente operacional: apresenta métricas observáveis do
contrato versionado e não infere capacidades cognitivas a partir delas.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from vita.nexus_constitutional_bridge_v3 import NexusConstitutionalBridge


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="JSON snapshot path, or '-' for stdin")
    args = parser.parse_args(argv)
    try:
        print(render_dashboard(load_snapshot(args.snapshot)))
    except (OSError, json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
        print(f"dashboard error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
