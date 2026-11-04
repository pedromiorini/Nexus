"""Ciclo de decisão do roteador (Laird, 2012): propor → rejeitar → selecionar → aplicar.

Módulo puro e isolado (não importa o monólito). Quando não há uma escolha única e segura
o ciclo falha fechado: devolve um *impasse* tipado (NO_CHANGE, TIE, CONFLICT,
CONSTRAINT_FAILURE, PROPOSER_FAILURE) com um *subgoal* de registro, e nunca aplica nada.
Pré-registro DC1–DC10: docs/cognitivo/AUDITORIA_2026-10-22.md.

Limitações declaradas: o subgoal é apenas um registro (o ciclo não o resolve) e a ponte
com o ``CentralRouter`` não faz parte deste módulo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from core.cognitivo.counterfactual import _normalize_action_key
from core.cognitivo.mcts_planner import DEFAULT_VETOES

__all__ = [
    "Decision",
    "DecisionInputError",
    "Impasse",
    "Subgoal",
    "run_decision_cycle",
    "select_operator",
]

_IMPASSE_CODES = {
    "NO_CHANGE": "DC_IMPASSE_NO_CHANGE",
    "TIE": "DC_IMPASSE_TIE",
    "CONFLICT": "DC_IMPASSE_CONFLICT",
    "CONSTRAINT_FAILURE": "DC_IMPASSE_CONSTRAINT",
    "PROPOSER_FAILURE": "DC_IMPASSE_PROPOSER",
}

_FROZEN_DEFAULT_VETOES: Mapping[str, str] = MappingProxyType(
    {str.__str__(k): str.__str__(v) for k, v in DEFAULT_VETOES.items()}
)


class DecisionInputError(ValueError):
    """Entrada inválida do ciclo de decisão. ``code`` identifica o motivo."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class Impasse:
    kind: str
    code: str
    candidates: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Subgoal:
    """Registro do que precisaria ser resolvido; o ciclo não o executa."""

    kind: str
    goal: str
    candidates: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    selected: Optional[str] = None
    impasse: Optional[Impasse] = None
    subgoal: Optional[Subgoal] = None
    candidates: Tuple[str, ...] = ()
    rejected: Dict[str, str] = field(default_factory=dict)
    applied: bool = False
    result: Any = None
    error: Optional[str] = None
    phases: Tuple[str, ...] = ("propose", "select")


def _text(value: Any, code: str, where: str) -> Tuple[str, str]:
    """(grafia original como ``str`` exato, chave normalizada); nunca chama métodos do chamador."""
    if not isinstance(value, str):
        raise DecisionInputError(code, f"{where} deve ser str")
    plain = str.__str__(value)
    key = _normalize_action_key(plain)
    if not key:
        raise DecisionInputError(code, f"{where} vazio após normalizar")
    return plain, key


def _proposals(items: Any) -> List[Tuple[str, str]]:
    if items is None:
        raise DecisionInputError("DC_NO_INPUT", "proposals ausente")
    if not isinstance(items, (list, tuple)):
        raise DecisionInputError("DC_BAD_INPUT", "proposals deve ser lista ou tupla")
    seen: Dict[str, str] = {}
    for item in items:
        plain, key = _text(item, "DC_BAD_PROPOSAL", "operador")
        seen.setdefault(key, plain)
    return list(seen.items())


def _rejects(rejects: Any) -> Dict[str, str]:
    if rejects is None:
        source: Mapping = _FROZEN_DEFAULT_VETOES
    elif isinstance(rejects, Mapping):
        source = rejects
    else:
        raise DecisionInputError("DC_BAD_REJECTS", "rejects deve ser um mapa operador -> motivo")
    out: Dict[str, str] = {}
    for name, reason in dict(source).items():
        _, key = _text(name, "DC_BAD_REJECTS", "operador rejeitado")
        if not isinstance(reason, str):
            raise DecisionInputError("DC_BAD_REJECTS", "motivo da rejeição deve ser str")
        out.setdefault(key, str.__str__(reason))
    return out


def _better(better: Any) -> List[Tuple[str, str]]:
    if better is None:
        return []
    if not isinstance(better, (list, tuple)):
        raise DecisionInputError("DC_BAD_PREFERENCE", "better deve ser lista de pares")
    pairs: List[Tuple[str, str]] = []
    for pair in better:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise DecisionInputError("DC_BAD_PREFERENCE", "cada preferência é um par (melhor, pior)")
        _, best = _text(pair[0], "DC_BAD_PREFERENCE", "preferência")
        _, worse = _text(pair[1], "DC_BAD_PREFERENCE", "preferência")
        if best == worse:
            raise DecisionInputError("DC_BAD_PREFERENCE", "operador não pode ser melhor que ele mesmo")
        pairs.append((best, worse))
    return pairs


def _impasse(kind: str, candidates: Tuple[str, ...], rejected: Dict[str, str],
             error: Optional[str] = None) -> Decision:
    return Decision(
        selected=None,
        impasse=Impasse(kind, _IMPASSE_CODES[kind], candidates),
        subgoal=Subgoal(kind, "resolve:" + kind, candidates),
        candidates=candidates,
        rejected=rejected,
        error=error,
    )


def _decide(proposals: List[Tuple[str, str]], rejects: Dict[str, str],
            better: Sequence[Tuple[str, str]]) -> Decision:
    if not proposals:
        return _impasse("NO_CHANGE", (), {})
    spelling = dict(proposals)
    rejected = {spelling[key]: rejects[key] for key in spelling if key in rejects}
    alive = [key for key in spelling if key not in rejects]
    if not alive:
        return _impasse("CONSTRAINT_FAILURE", tuple(spelling.values()), rejected)
    members = set(alive)
    edges = {(a, b) for a, b in better if a in members and b in members}
    indegree = {key: 0 for key in alive}
    for _, worse in edges:
        indegree[worse] += 1
    # Kahn: sobra nó sem remover se, e somente se, há ciclo entre os operadores restantes.
    pending = dict(indegree)
    ready = [key for key, degree in pending.items() if degree == 0]
    removed = 0
    while ready:
        node = ready.pop()
        removed += 1
        for a, b in edges:
            if a == node:
                pending[b] -= 1
                if pending[b] == 0:
                    ready.append(b)
    if removed != len(alive):
        return _impasse("CONFLICT", tuple(spelling[key] for key in alive), rejected)
    free = tuple(spelling[key] for key in alive if indegree[key] == 0)
    if len(free) > 1:
        return _impasse("TIE", free, rejected)
    return Decision(selected=free[0], candidates=free, rejected=rejected)


def select_operator(
    proposals: Sequence[str],
    *,
    rejects: Optional[Mapping[str, str]] = None,
    better: Optional[Sequence[Tuple[str, str]]] = None,
) -> Decision:
    """Seleciona um operador ou devolve um impasse. Puro: não aplica nada."""
    items = _proposals(proposals)
    return _decide(items, _rejects(rejects), _better(better))


def run_decision_cycle(
    state: Any,
    proposers: Sequence[Callable[[Any], Sequence[str]]],
    *,
    apply: Optional[Callable[[str, Any], Any]] = None,
    rejects: Optional[Mapping[str, str]] = None,
    better: Optional[Sequence[Tuple[str, str]]] = None,
) -> Decision:
    """Propõe (em ordem), rejeita, seleciona e, só com seleção única, aplica."""
    if proposers is None:
        raise DecisionInputError("DC_NO_INPUT", "proposers ausente")
    if not isinstance(proposers, (list, tuple)) or not all(callable(p) for p in proposers):
        raise DecisionInputError("DC_BAD_INPUT", "proposers deve ser lista de chamáveis")
    if apply is not None and not callable(apply):
        raise DecisionInputError("DC_BAD_INPUT", "apply deve ser chamável")
    veto = _rejects(rejects)
    prefs = _better(better)

    gathered: List[Any] = []
    for index, proposer in enumerate(proposers):
        try:
            produced = proposer(state)
            if not isinstance(produced, (list, tuple)):
                raise TypeError("proposer deve devolver lista ou tupla")
            gathered.extend(_proposals(list(produced)))
        except Exception as exc:  # noqa: BLE001 - qualquer falha de proposer vira impasse
            return _impasse("PROPOSER_FAILURE", (), {},
                            f"DC_PROPOSER_FAILED: proposer[{index}] {type(exc).__name__}")
    merged: Dict[str, str] = {}
    for key, plain in gathered:
        merged.setdefault(key, plain)
    decision = _decide(list(merged.items()), veto, prefs)
    if decision.selected is None or apply is None:
        return decision
    phases = ("propose", "select", "apply")
    try:
        outcome = apply(decision.selected, state)
    except Exception as exc:  # noqa: BLE001 - falha de apply não esconde a seleção
        return Decision(selected=decision.selected, candidates=decision.candidates,
                        rejected=decision.rejected, applied=False, result=None,
                        error=f"DC_APPLY_FAILED: {type(exc).__name__}", phases=phases)
    return Decision(selected=decision.selected, candidates=decision.candidates,
                    rejected=decision.rejected, applied=True, result=outcome, phases=phases)
