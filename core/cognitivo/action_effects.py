"""Tabela única de efeitos de ação (R1), compartilhada por RealMCTSPlanner e RealWorldModelSimulator.

Cada ação tem duas visões do mesmo efeito:
- ``metrics``: deltas numéricos em [0, 1] usados pelo planejador MCTS (valores idênticos aos anteriores);
- ``world``: atribuições simbólicas ao estado do modelo de mundo (aplicadas em ordem).

MONITOR e ESCALATE são observação/encaminhamento: não alteram o estado do sistema (``world`` vazio),
mas são ações conhecidas e, portanto, não geram aviso.
"""

from typing import Any, Dict

ACTION_EFFECTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "APPLY_PATCH": {"metrics": {"security": +0.9, "stability": +0.3, "cost": -0.2},
                    "world": {"security_patch_applied": True, "alert_level": "low", "system_status": "secure"}},
    "IGNORE_ALERT": {"metrics": {"security": -0.8, "stability": -0.5, "cost": 0.0},
                     "world": {"alert_level": "critical", "system_status": "compromised"}},
    "ROLLBACK": {"metrics": {"security": +0.4, "stability": +0.6, "cost": -0.4},
                 "world": {"security_patch_applied": False, "alert_level": "medium", "system_status": "stable"}},
    "ESCALATE": {"metrics": {"security": +0.6, "stability": +0.2, "cost": -0.3},
                 "world": {}},
    "MONITOR": {"metrics": {"security": +0.2, "stability": +0.1, "cost": -0.1},
                "world": {}},
    "ISOLATE_SYSTEM": {"metrics": {"security": +0.7, "stability": -0.2, "cost": -0.3},
                       "world": {"alert_level": "medium", "system_status": "isolated"}},
    "RESTORE_BACKUP": {"metrics": {"security": +0.5, "stability": +0.8, "cost": -0.5},
                       "world": {"alert_level": "medium", "system_status": "stable"}},
    "DEPLOY_UPDATE": {"metrics": {"security": +0.4, "stability": +0.5, "cost": -0.4},
                      "world": {"security_patch_applied": True, "alert_level": "low", "system_status": "secure"}},
    "INCREASE_LOGGING": {"metrics": {"security": +0.3, "stability": +0.0, "cost": -0.1},
                         "world": {"logging_level": "verbose"}},
    "SHUTDOWN_SERVICE": {"metrics": {"security": +0.8, "stability": -0.7, "cost": -0.2},
                         "world": {"alert_level": "low", "system_status": "offline"}},
}


def planner_deltas() -> Dict[str, Dict[str, float]]:
    """Cópia independente dos deltas numéricos (o planejador pode mutá-la sem afetar a tabela)."""
    return {name: dict(entry["metrics"]) for name, entry in ACTION_EFFECTS.items()}
