"""Ciclo de decisão do roteador (Laird, 2012): propor → rejeitar → selecionar → aplicar.

Módulo puro e isolado (não importa o monólito). Quando não há uma escolha única e segura
o ciclo falha fechado: devolve um *impasse* tipado (NO_CHANGE, TIE, CONFLICT,
CONSTRAINT_FAILURE, PROPOSER_FAILURE, OPERATOR_NO_CHANGE) com um *subgoal* de registro.
Pré-registro DC1–DC16: docs/cognitivo/AUDITORIA_2026-10-22.md.

Veto aditivo (errata E-DC5; chave canônica ``veto_key``, ver abaixo): o conjunto efetivo é o veto padrão (``DEFAULT_VETOES``,
congelado no import) somado a ``rejects``; ``rejects=None`` e ``{}`` mantêm o padrão e só
``use_default_rejects=False`` o desliga. ``allowed`` fecha o vocabulário: operador cuja
chave normalizada não esteja nele é rejeitado (cobre grafias com caracteres invisíveis,
largura total, pontuação etc.). O veto também compara a chave canônica ``veto_key`` (caixa, separadores,
invisíveis, ``()`` final), então a grafia alternativa de um operador vetado é rejeitada com o motivo do veto,
não com o do vocabulário. O motivo do padrão/``rejects`` vence o do vocabulário.

``OPERATOR_NO_CHANGE`` (errata E-DC8): se ``apply`` levanta, o operador selecionado é mantido
em ``selected`` com ``applied=False``, mas o impasse e o subgoal registram que a aplicação não
mudou o estado. Invariante: todo impasse significa que nada foi aplicado; ``selected`` é
``None`` em todo impasse exceto ``OPERATOR_NO_CHANGE``.

Diferenças em relação ao Soar (documentadas, não bugs): (1) CONFLICT é mais estrito que o
Soar: um ciclo em ``better`` impede a escolha mesmo com um operador solto, enquanto o Soar
escolheria o operador solto C; aqui ele fica de fora por segurança. (2) «Tudo rejeitado»
(CONSTRAINT_FAILURE) corresponde ao state no-change do Soar. (3) Empate é impasse, sem
preferência *indifferent*. (4) O subgoal é só registro; nada o resolve.

Erros de entrada: ``rejects``, ``better``, ``allowed``, proposers e ``apply`` inválidos
levantam ``DecisionInputError`` (com ``code``) antes de qualquer efeito; a ponte com o
``CentralRouter`` deve tratar essa exceção como bloqueio, não como impasse. A interface de
ponte ``rejectors(state)``/``required`` (rejeitadores e operadores obrigatórios dependentes
do estado) **não** é implementada aqui; fica para a ponte.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from core.cognitivo.action_effects import veto_index, veto_key
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
    "OPERATOR_NO_CHANGE": "DC_IMPASSE_OPERATOR_NO_CHANGE",
}

_OUT_OF_VOCABULARY = "operador fora do vocabulário permitido"



def _freeze_default_vetoes() -> Mapping[str, str]:
    frozen: Dict[str, str] = {}
    for name, reason in DEFAULT_VETOES.items():
        frozen.setdefault(_normalize_action_key(str.__str__(name)), str.__str__(reason))
    return MappingProxyType(frozen)


_FROZEN_DEFAULT_VETOES: Mapping[str, str] = _freeze_default_vetoes()
# Chaves canônicas (veto_key) do veto padrão, congeladas no import como o mapa exato.
def _freeze_default_canon() -> Mapping[str, str]:
    index = veto_index({str.__str__(name): reason for name, reason in DEFAULT_VETOES.items()})
    return MappingProxyType({key: str.__str__(reason) for key, reason in index.items()})


_FROZEN_DEFAULT_CANON: Mapping[str, str] = _freeze_default_canon()


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
    rejected: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
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


def _rejects(rejects: Any, use_default: Any) -> Dict[str, str]:
    """Padrão ∪ ``rejects`` (aditivo); na colisão de chave vence o primeiro, ou seja, o padrão."""
    if not isinstance(use_default, bool):
        raise DecisionInputError("DC_BAD_INPUT", "use_default_rejects deve ser bool")
    if rejects is None:
        source: Mapping = {}
    elif isinstance(rejects, Mapping):
        source = rejects
    else:
        raise DecisionInputError("DC_BAD_REJECTS", "rejects deve ser um mapa operador -> motivo")
    out: Dict[str, str] = dict(_FROZEN_DEFAULT_VETOES) if use_default else {}
    canon: Dict[str, str] = dict(_FROZEN_DEFAULT_CANON) if use_default else {}
    for name, reason in dict(source).items():
        plain, key = _text(name, "DC_BAD_REJECTS", "operador rejeitado")
        if not isinstance(reason, str):
            raise DecisionInputError("DC_BAD_REJECTS", "motivo da rejeição deve ser str")
        out.setdefault(key, str.__str__(reason))
        canonical = veto_key(plain)
        if canonical is not None:
            canon.setdefault(canonical, str.__str__(reason))
    # Chaves canônicas entram DEPOIS das exatas: o nome exato (e o padrão) vence na colisão.
    for key, reason in canon.items():
        out.setdefault(key, reason)
    return out


def _allowed(allowed: Any) -> Optional[frozenset]:
    """``None`` = sem restrição; senão o vocabulário fechado (chaves normalizadas)."""
    if allowed is None:
        return None
    if not isinstance(allowed, (list, tuple)) or not allowed:
        raise DecisionInputError("DC_BAD_ALLOWED", "allowed deve ser lista ou tupla não vazia")
    return frozenset(_text(item, "DC_BAD_ALLOWED", "operador permitido")[1] for item in allowed)


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
        rejected=MappingProxyType(rejected),
        error=error,
    )


def _decide(proposals: List[Tuple[str, str]], rejects: Dict[str, str],
            better: Sequence[Tuple[str, str]], allowed: Optional[frozenset]) -> Decision:
    if not proposals:
        return _impasse("NO_CHANGE", (), {})
    spelling = dict(proposals)
    blocked: Dict[str, str] = {}
    for key in spelling:
        canonical = veto_key(spelling[key])
        if key in rejects:
            blocked[key] = rejects[key]
        elif canonical is not None and canonical in rejects:  # veto por chave canônica (vence o vocabulário)
            blocked[key] = rejects[canonical]
        elif allowed is not None and key not in allowed:
            blocked[key] = _OUT_OF_VOCABULARY
    rejected = {spelling[key]: blocked[key] for key in spelling if key in blocked}
    alive = [key for key in spelling if key not in blocked]
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
    return Decision(selected=free[0], candidates=free, rejected=MappingProxyType(rejected))


def select_operator(
    proposals: Sequence[str],
    *,
    rejects: Optional[Mapping[str, str]] = None,
    better: Optional[Sequence[Tuple[str, str]]] = None,
    allowed: Optional[Sequence[str]] = None,
    use_default_rejects: bool = True,
) -> Decision:
    """Seleciona um operador ou devolve um impasse. Puro: não aplica nada.

    ``rejects`` soma ao veto padrão (aditivo); ``use_default_rejects=False`` desliga o padrão.
    ``allowed`` (lista/tupla não vazia) fecha o vocabulário de operadores; ``None`` não restringe.
    """
    items = _proposals(proposals)
    return _decide(items, _rejects(rejects, use_default_rejects), _better(better), _allowed(allowed))


def run_decision_cycle(
    state: Any,
    proposers: Sequence[Callable[[Any], Sequence[str]]],
    *,
    apply: Optional[Callable[[str, Any], Any]] = None,
    rejects: Optional[Mapping[str, str]] = None,
    better: Optional[Sequence[Tuple[str, str]]] = None,
    allowed: Optional[Sequence[str]] = None,
    use_default_rejects: bool = True,
) -> Decision:
    """Propõe (em ordem), rejeita, seleciona e, só com seleção única, aplica.

    Mesmas regras de ``select_operator`` para ``rejects`` (aditivo), ``allowed`` e
    ``use_default_rejects``. Falha de ``apply`` vira o impasse ``OPERATOR_NO_CHANGE``.
    """
    if proposers is None:
        raise DecisionInputError("DC_NO_INPUT", "proposers ausente")
    if not isinstance(proposers, (list, tuple)) or not all(callable(p) for p in proposers):
        raise DecisionInputError("DC_BAD_INPUT", "proposers deve ser lista de chamáveis")
    if apply is not None and not callable(apply):
        raise DecisionInputError("DC_BAD_INPUT", "apply deve ser chamável")
    veto = _rejects(rejects, use_default_rejects)
    prefs = _better(better)
    vocabulary = _allowed(allowed)

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
    decision = _decide(list(merged.items()), veto, prefs, vocabulary)
    if decision.selected is None or apply is None:
        return decision
    phases = ("propose", "select", "apply")
    try:
        outcome = apply(decision.selected, state)
    except Exception as exc:  # noqa: BLE001 - falha de apply não esconde a seleção
        kind = "OPERATOR_NO_CHANGE"
        chosen = (decision.selected,)
        return Decision(selected=decision.selected,
                        impasse=Impasse(kind, _IMPASSE_CODES[kind], chosen),
                        subgoal=Subgoal(kind, "resolve:" + kind, chosen),
                        candidates=decision.candidates, rejected=decision.rejected,
                        applied=False, result=None,
                        error=f"DC_APPLY_FAILED: {type(exc).__name__}", phases=phases)
    return Decision(selected=decision.selected, candidates=decision.candidates,
                    rejected=decision.rejected, applied=True, result=outcome, phases=phases)
