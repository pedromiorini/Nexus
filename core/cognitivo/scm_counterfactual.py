"""Contrafactual SCM por abdução-ação-predição (Pearl, 2009, cap. 7) sobre o mundo simbólico.

Modelo: variáveis endógenas = chaves do WORLD_SCHEMA; mecanismo de cada ação = atribuição
determinística (efeito da tabela ACTION_EFFECTS e intervenção declarada; texto livre NÃO é
mecanismo); ruído exógeno U = estado inicial, desconhecido. Evidência = estado final observado
após a sequência factual. Os três passos:

1. abdução: U[k] = observado[k] para toda chave k que a sequência factual não atribui; para chave
   que ela atribui, U[k] não é identificável (o valor final não depende de U[k]);
2. ação: troca a sequência factual pela contrafactual (do);
3. predição: o valor de k é conhecido se a contrafactual o atribui ou se U[k] foi abduzido;
   caso contrário a chave fica em ``unidentified`` — nunca é adivinhada; ``possible`` devolve o
   domínio (valores possíveis) de cada uma dessas chaves.

Limites do modelo (declarados): o modelo é sem arestas entre variáveis — cada ação só atribui
valores às chaves do ``WORLD_SCHEMA``, e nenhuma chave é função de outra; a única operação
causal aceita é do(ação). Os nomes de ação exatos são exigidos (``apply_patch`` não vira
``APPLY_PATCH``): a tabela de efeitos e a intervenção declarada continuam por nome exato.
Uma subclasse de ``str`` na observação ou no nome da ação vira ``str`` exato
(``str.__str__``) antes de validar: ``strip``, ``lower``, ``__eq__``, ``__hash__``
e ``__str__`` do chamador não são chamados. Espaço nas pontas do nome não é removido.

Veto por nome canônico: o veto de ``rank_alternatives`` compara a chave canônica
``core.cognitivo.action_effects.veto_key`` (NFKC, sem caracteres invisíveis, separadores e sintaxe de
chamada final viram ``_``, ASCII, maiúsculas), então ``shutdown_service``, ``SHUTDOWN SERVICE`` e
``SHUTDOWN_SERVICE()`` são vetadas como ``SHUTDOWN_SERVICE``. ``declare_intervention`` recusa o nome
que não canonicaliza. O nome exato do ``vetoes`` continua valendo e vence na colisão.

Limitação conhecida (veto por nome, não por efeito): um alias declarado com ``declare_intervention``
cujo nome canônico seja outro (por exemplo ``SHUTDOWN_SERVICE_V2``, ``SHUTDOWN_SERVICES``,
``do_shutdown_service`` ou ``SHUTDOWN_SERVICE.exec``: o sufixo ou prefixo faz outra identidade) com o
mesmo efeito de ``SHUTDOWN_SERVICE`` não é vetado pelo ``DEFAULT_VETOES``; para vetá-lo, passe o nome do
alias em ``vetoes``. Não há veto por efeito. O ``DEFAULT_VETOES`` é copiado no import: alterá-lo
depois não muda este módulo.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.cognitivo.action_effects import ACTION_EFFECTS, veto_index, veto_lookup
from core.cognitivo.causal_world_model import STATUS_UTILITY, WORLD_SCHEMA, _schema_pair
from core.cognitivo.mcts_planner import DEFAULT_VETOES

# Cópia congelada no import: um ``DEFAULT_VETOES.clear()`` externo não desliga o veto em silêncio.
_FROZEN_DEFAULT_VETOES: Mapping = MappingProxyType(dict(DEFAULT_VETOES))


def _freeze_vetoes(vetoes: Optional[Mapping]) -> Mapping:
    """Valida (mapa str -> str) e copia ``vetoes``; ``None`` usa a cópia congelada do padrão."""
    if vetoes is None:
        return _FROZEN_DEFAULT_VETOES
    if not isinstance(vetoes, Mapping):
        raise TypeError("vetoes deve ser um mapa ação -> motivo")
    frozen: Dict[str, str] = {}
    for action, reason in dict(vetoes).items():
        try:  # str.__str__ só aceita str e devolve um str exato (sem chamar __str__ do chamador)
            frozen[str.__str__(action)] = str.__str__(reason)
        except TypeError:
            raise TypeError("vetoes: ação e motivo devem ser str") from None
    return MappingProxyType(frozen)


class ScmInputError(ValueError):
    """Entrada inválida do contrafactual SCM. ``code`` identifica o motivo."""

    def __init__(self, code: str, message: str, actions: Tuple[str, ...] = ()) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.actions = actions


@dataclass(frozen=True)
class ScmCounterfactual:
    consistent: bool
    state: Dict[str, Any] = field(default_factory=dict)
    unidentified: Tuple[str, ...] = ()
    conflicts: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    backdrop: Dict[str, Any] = field(default_factory=dict)
    verified_status: Optional[str] = None
    utility: Optional[int] = None
    possible: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)


def _canonical_observation(observed: object) -> Dict[str, Any]:
    if observed is None:
        raise ScmInputError("SCM_NO_OBSERVATION", "observed ausente")
    if not isinstance(observed, dict):
        raise TypeError("observed deve ser dict")
    if not observed:
        raise ScmInputError("SCM_NO_OBSERVATION", "observed vazio")
    canon: Dict[str, Any] = {}
    for key, value in observed.items():
        if isinstance(key, str):
            key = str.__str__(key)
        if isinstance(value, str):
            value = str.__str__(value)
        pair = _schema_pair(key, value)
        if pair is None:
            raise ScmInputError("SCM_BAD_OBSERVATION", f"{key!r}={value!r} fora do WORLD_SCHEMA")
        if pair[0] in canon and canon[pair[0]] != pair[1]:
            raise ScmInputError("SCM_BAD_OBSERVATION", f"{pair[0]!r} observado com dois valores")
        canon[pair[0]] = pair[1]
    return canon


def _actions(actions: object, *, name: str) -> List[str]:
    if actions is None:
        raise ScmInputError("SCM_NO_ACTIONS", f"{name} ausente")
    if isinstance(actions, (str, bytes)) or not isinstance(actions, (list, tuple)):
        raise TypeError(f"{name} deve ser lista de str")
    out: List[str] = []
    for action in actions:
        if not isinstance(action, str):
            raise TypeError(f"{name} deve conter só str")
        action = str.__str__(action)
        if not action.strip():
            raise ScmInputError("SCM_BAD_ACTION", f"{name} contém ação em branco")
        out.append(action)
    return out


def _mechanism(world_model: object, action: str) -> Optional[Dict[str, Any]]:
    """Efeitos verificados da ação (tabela, depois declarada); None se a ação não é modelada."""
    entry = ACTION_EFFECTS.get(action)
    effects: Dict[str, Any] = dict(entry["world"]) if entry is not None else {}
    reasoning = getattr(world_model, "causal_reasoning", None)
    declared_all = getattr(reasoning, "declared_interventions", None)
    declared: Dict[str, Any] = {}
    if isinstance(declared_all, dict):
        raw = declared_all.get(action)
        for key, value in (raw.items() if isinstance(raw, dict) else ()):
            pair = _schema_pair(key, value)
            if pair is not None:
                declared[pair[0]] = pair[1]
    if entry is None and not declared:
        return None
    effects.update(declared)
    return effects


def _assignments(world_model: object, actions: Sequence[str]) -> Dict[str, Any]:
    unmodeled = tuple(dict.fromkeys(a for a in actions if _mechanism(world_model, a) is None))
    if unmodeled:
        raise ScmInputError("SCM_UNMODELED", f"ação sem efeito verificado: {', '.join(unmodeled)}", unmodeled)
    assigned: Dict[str, Any] = {}
    for action in actions:
        assigned.update(_mechanism(world_model, action) or {})
    return assigned


def counterfactual_outcome(
    observed: Dict[str, Any],
    factual_actions: Sequence[str],
    counterfactual_actions: Sequence[str],
    world_model: object = None,
) -> ScmCounterfactual:
    """Abdução-ação-predição. Não muta nenhum argumento nem o modelo de mundo."""
    evidence = _canonical_observation(observed)
    factual = _actions(factual_actions, name="factual_actions")
    counter = _actions(counterfactual_actions, name="counterfactual_actions")
    assigned_f = _assignments(world_model, factual)

    conflicts = {
        key: (evidence[key], value)
        for key, value in sorted(assigned_f.items())
        if key in evidence and evidence[key] != value
    }
    if conflicts:
        return ScmCounterfactual(False, {}, tuple(sorted(WORLD_SCHEMA)), conflicts)

    assigned_c = _assignments(world_model, counter)
    backdrop = {key: value for key, value in sorted(evidence.items()) if key not in assigned_f}
    state = dict(backdrop)
    state.update(assigned_c)
    state = dict(sorted(state.items()))
    unidentified = tuple(sorted(set(WORLD_SCHEMA) - set(state)))
    status = state.get("system_status")
    return ScmCounterfactual(
        True, state, unidentified, {}, backdrop, status,
        STATUS_UTILITY.get(status) if status is not None else None,
        {key: tuple(WORLD_SCHEMA[key]) for key in unidentified},
    )


def rank_alternatives(
    observed: Dict[str, Any],
    factual_actions: Sequence[str],
    alternatives: Sequence[Sequence[str]],
    world_model: object = None,
    vetoes: Optional[Mapping] = None,
) -> Dict[str, Any]:
    """Ordena sequências alternativas pelo status contrafactual identificado.

    O veto marca, não exclui: toda alternativa identificada continua em ``ranking`` (cada item
    traz ``vetoed``, lista de ``{"action", "reason"}``, vazia se não vetada) e ``vetoed`` lista as
    vetadas. ``best`` é a primeira não vetada; ``tied_best`` reúne os itens com a utilidade de
    ``best`` (os vetados aparecem marcados). ``vetoes=None`` usa ``DEFAULT_VETOES``; ``{}`` desliga.

    A decisão é ``best``, não ``ranking[0]``: o primeiro do ranking pode estar vetado. ``best=None``
    significa todas vetadas (ou nenhuma alternativa identificada); então não há recomendação.
    ``vetoes`` é validado (mapa str -> str, senão ``TypeError`` antes de rankear) e copiado na entrada.
    """
    evidence = _canonical_observation(observed)
    factual = _actions(factual_actions, name="factual_actions")
    vetoes = _freeze_vetoes(vetoes)
    veto_canon = veto_index(vetoes)
    if alternatives is None or (isinstance(alternatives, (list, tuple)) and not alternatives):
        raise ScmInputError("SCM_NO_ALTERNATIVES", "alternatives ausente ou vazia")
    if isinstance(alternatives, (str, bytes)) or not isinstance(alternatives, (list, tuple)):
        raise TypeError("alternatives deve ser lista de listas de ação")
    sequences = [_actions(seq, name="alternatives[i]") for seq in alternatives]
    _assignments(world_model, factual)
    factual_status = evidence.get("system_status")
    factual_utility = STATUS_UTILITY.get(factual_status) if factual_status is not None else None

    ranking: List[Dict[str, Any]] = []
    unidentified: List[List[str]] = []
    unmodeled: List[List[str]] = []
    conflicts: Dict[str, Tuple[Any, Any]] = {}
    for seq in sequences:
        try:
            result = counterfactual_outcome(evidence, factual, seq, world_model)
        except ScmInputError as exc:
            if exc.code != "SCM_UNMODELED":
                raise
            unmodeled.append(list(seq))
            continue
        if not result.consistent:
            conflicts = result.conflicts
            break
        if result.utility is None:
            unidentified.append(list(seq))
            continue
        ranking.append({
            "actions": list(seq),
            "verified_status": result.verified_status,
            "utility": result.utility,
            "gain": None if factual_utility is None else result.utility - factual_utility,
            "state": dict(result.state),
            "unidentified": list(result.unidentified),
            "vetoed": [{"action": a, "reason": str(reason)} for a, (hit, reason) in
                       ((a, veto_lookup(a, vetoes, veto_canon)) for a in dict.fromkeys(seq)) if hit],
        })
    if conflicts:
        return {"consistent": False, "conflicts": conflicts, "factual_utility": factual_utility,
                "ranking": [], "best": None, "tied_best": [], "vetoed": [],
                "unidentified": [], "unmodeled": []}
    ranking.sort(key=lambda r: (-r["utility"], len(r["actions"]), tuple(r["actions"])))
    unidentified.sort(key=lambda s: (len(s), tuple(s)))
    unmodeled.sort(key=lambda s: (len(s), tuple(s)))
    allowed = [item for item in ranking if not item["vetoed"]]
    best = allowed[0] if allowed else None
    tied_best = [item for item in ranking if best is not None and item["utility"] == best["utility"]]
    vetoed = [{"actions": item["actions"], "vetoes": item["vetoed"]} for item in ranking if item["vetoed"]]
    return {"consistent": True, "conflicts": {}, "factual_utility": factual_utility,
            "ranking": ranking, "best": best, "tied_best": tied_best, "vetoed": vetoed,
            "unidentified": unidentified, "unmodeled": unmodeled}
