"""Tabela única de efeitos de ação (R1), compartilhada por RealMCTSPlanner e RealWorldModelSimulator.

Cada ação tem duas visões do mesmo efeito:
- ``metrics``: deltas numéricos em [0, 1] usados pelo planejador MCTS (valores idênticos aos anteriores);
- ``world``: atribuições simbólicas ao estado do modelo de mundo (aplicadas em ordem).

MONITOR e ESCALATE são observação/encaminhamento: não alteram o estado do sistema (``world`` vazio),
mas são ações conhecidas e, portanto, não geram aviso.
"""

import re
import unicodedata
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = ["ACTION_EFFECTS", "planner_deltas", "veto_key", "veto_index", "veto_lookup"]

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


# --- Veto por NOME canônico (follow-up da #156; revisão da Pearl) -----------------------------------------------
# A chave canônica serve só ao VETO (sobre-aproxima de propósito). O vocabulário permitido e a tabela de efeitos
# continuam com nome exato: uma variante não declarada segue sem modelo (falha fechado).
_SEPARATORS = re.compile(r"[\s\-.:/\\,\u2010-\u2015\u2212\u00b7]+")
_CALL_SYNTAX = re.compile(r"(?:\(_*\))?[;!?]*\Z")
_ASCII_KEY = re.compile(r"[A-Za-z0-9_]+")


def veto_key(name: Any) -> Optional[str]:
    """Chave canônica ``[A-Z0-9_]+`` de um nome de ação para comparar vetos, ou ``None`` (não canonicalizável).

    Ordem: NFKC; remove Cf/Cc que não sejam espaço (``\\t`` e ``\\n`` são separadores); separadores
    (espaço, ``- . : / \\ ,``, U+2010–U+2015, U+2212, ``·``) viram ``_``; tira a sintaxe de chamada final
    (``()``, ``;``, ``!``, ``?``); exige ASCII **antes** de ``upper()`` (cirílico, ``ı`` e ``ſ`` nunca viram ASCII);
    ``upper()``. Idempotente. Sufixos ``_V2``, ``.exec``, plural e prefixo ``do_`` continuam identidades distintas.
    Não-``str`` e vazio devolvem ``None`` (nunca chama método do chamador: usa ``str.__str__``).
    """
    if not isinstance(name, str):
        return None
    text = unicodedata.normalize("NFKC", str.__str__(name))
    text = "".join(c for c in text if c.isspace() or unicodedata.category(c) not in ("Cf", "Cc"))
    text = _CALL_SYNTAX.sub("", _SEPARATORS.sub("_", text.strip()))
    text = re.sub(r"_+", "_", text).strip("_")
    if not text or not _ASCII_KEY.fullmatch(text):
        return None
    return text.upper()


def veto_index(vetoes: Mapping[Any, Any]) -> Dict[str, Any]:
    """Índice ``chave canônica -> motivo`` de um mapa de vetos (a primeira chave na ordem do mapa vence).

    Chave que não canonicaliza fica de fora (continua valendo só pelo nome exato).
    """
    index: Dict[str, Any] = {}
    for name, reason in vetoes.items():
        key = veto_key(name)
        if key is not None:
            index.setdefault(key, reason)
    return index


def veto_lookup(action: Any, vetoes: Mapping[Any, Any],
                index: Optional[Mapping[str, Any]] = None) -> Tuple[bool, Any]:
    """``(vetada, motivo)`` de ``action``: nome exato primeiro, depois a chave canônica.

    Devolve o par (e não só o motivo) para que um veto cujo motivo é ``None`` continue sendo veto.
    """
    if isinstance(action, str) and action in vetoes:
        return True, vetoes[action]
    key = veto_key(action)
    if key is None:
        return False, None
    table = index if index is not None else veto_index(vetoes)
    if key in table:
        return True, table[key]
    return False, None
