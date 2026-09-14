"""Governança de memória e defesa proativa para a roadmap Nexus v4.0.

O módulo é opcional: funciona mesmo quando PyTorch ou GPU não estão disponíveis.
Ele separa telemetria, política de orçamento e resposta defensiva para permitir
integração posterior com o CentralRouter sem acoplamento ao monólito.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover - ambiente sem torch
    torch = None


@dataclass
class VramSnapshot:
    available: bool
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    total_bytes: int = 0
    timestamp: float = field(default_factory=time)

    @property
    def utilization(self) -> float:
        if not self.total_bytes:
            return 0.0
        return min(1.0, self.reserved_bytes / self.total_bytes)


@dataclass
class DefenseDecision:
    action: str
    severity: str
    reason: str
    snapshot: VramSnapshot


class VramDefenseGuard:
    """Observa pressão de memória e aplica respostas graduais e auditáveis."""

    def __init__(self, soft_limit: float = 0.80, hard_limit: float = 0.92) -> None:
        if not 0 < soft_limit < hard_limit <= 1:
            raise ValueError("limites devem satisfazer 0 < soft < hard <= 1")
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self.history: List[DefenseDecision] = []

    def snapshot(self) -> VramSnapshot:
        if torch is None or not torch.cuda.is_available():
            return VramSnapshot(available=False)
        device = torch.cuda.current_device()
        return VramSnapshot(
            available=True,
            allocated_bytes=int(torch.cuda.memory_allocated(device)),
            reserved_bytes=int(torch.cuda.memory_reserved(device)),
            total_bytes=int(torch.cuda.get_device_properties(device).total_memory),
        )

    def evaluate(self, snapshot: Optional[VramSnapshot] = None) -> DefenseDecision:
        current = snapshot or self.snapshot()
        if not current.available:
            decision = DefenseDecision("observe", "info", "GPU indisponível; execução permanece em modo CPU", current)
        elif current.utilization >= self.hard_limit:
            decision = DefenseDecision("emergency_release", "critical", "pressão de VRAM acima do limite rígido", current)
        elif current.utilization >= self.soft_limit:
            decision = DefenseDecision("defer_and_release", "warning", "pressão de VRAM acima do limite suave", current)
        else:
            decision = DefenseDecision("continue", "normal", "pressão de VRAM dentro do orçamento", current)
        self.history.append(decision)
        return decision

    def mitigation_plan(self, decision: Optional[DefenseDecision] = None) -> Dict[str, Any]:
        current = decision or self.evaluate()
        actions = {
            "continue": ["admit_next_task"],
            "defer_and_release": ["defer_next_task", "release_cached_tensors", "gc_collect"],
            "emergency_release": ["stop_noncritical_tasks", "release_cached_tensors", "gc_collect", "require_recheck"],
            "observe": ["use_cpu_fallback"],
        }[current.action]
        return {"action": current.action, "severity": current.severity, "steps": actions, "utilization": current.snapshot.utilization}

    def statistics(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for item in self.history:
            counts[item.action] = counts.get(item.action, 0) + 1
        return {"samples": len(self.history), "actions": counts, "soft_limit": self.soft_limit, "hard_limit": self.hard_limit}
