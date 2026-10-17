"""Livro-razão de fatos temporais (bi-temporal enxuto). Somente biblioteca padrão.

Modelo: cada fato tem tempo de validade [valid_from, valid_to) e tempo de registro
(recorded_at). Nada é apagado: invalidar é anexar um evento ``close`` a um log
encadeado por SHA-256 (adulteração detectável). O estado atual é uma projeção do
log; ``known_at`` reconstrói o que o sistema acreditava em um instante passado.
Predicados funcionais (um valor por vez, ex.: "mora_em") substituem o valor
anterior; os demais são multivalorados e apenas acumulam.

Limites: a substituição é determinística ("o mais recente vence"), sem arbitragem
por confiança; a cadeia de hash detecta edição e remoção no meio, mas truncamento
final só é detectado com ``expected_head`` guardado fora do arquivo; as consultas
com ``known_at`` reconstroem a projeção (O(eventos)).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

SCHEMA = "nexus.temporal_facts.v1"
GENESIS = "0" * 64
_FACT_KEYS = frozenset({"fact_id", "subject", "predicate", "object", "valid_from", "valid_to",
                        "recorded_at", "confidence", "source", "superseded_by", "metadata"})


@dataclass(frozen=True)
class TemporalFact:
    fact_id: str
    subject: str
    predicate: str
    object: str
    valid_from: float
    valid_to: Optional[float]  # None = sem fim conhecido; intervalo semiaberto
    recorded_at: float
    confidence: float
    source: str
    superseded_by: Optional[str] = None
    closed_at: Optional[float] = None  # recorded_at do evento que fixou valid_to
    corroborated_by: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


def _digest(event: Dict[str, Any]) -> str:
    body = json.dumps({k: v for k, v in event.items() if k != "hash"},
                      sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _reject_constant(name: str) -> Any:
    raise ValueError(f"constante JSON não permitida: {name}")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} deve ser um número finito")
    return float(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} deve ser uma string não vazia")
    return value


def _json_value(value: Any) -> Any:
    """Return a detached JSON value, rejecting lossy or non-finite payloads."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("metadata deve conter somente números finitos")
        return value
    if type(value) is list:
        return [_json_value(item) for item in value]
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError("chaves de metadata devem ser strings")
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError("metadata deve conter somente valores JSON")


class TemporalFactStore:
    def __init__(self, *, functional_predicates: Iterable[str] = (),
                 clock: Callable[[], float] = time.time) -> None:
        self._functional = frozenset(functional_predicates)
        self._clock = clock
        self._lock = threading.RLock()
        self._facts: Dict[str, TemporalFact] = {}
        self._index: Dict[Tuple[str, str], List[str]] = {}
        self._events: List[Dict[str, Any]] = []
        self._head = GENESIS
        self._seq = 0
        self._last_recorded = -math.inf

    @staticmethod
    def _covers(fact: TemporalFact, t: float) -> bool:
        return fact.valid_from <= t and (fact.valid_to is None or t < fact.valid_to)

    def _now(self) -> float:
        now = max(_number(self._clock(), "clock"), self._last_recorded)
        self._last_recorded = now
        return now

    # ── log → projeção ───────────────────────────────────────────────────
    def _apply(self, event: Dict[str, Any]) -> None:
        kind = event["kind"]
        if kind == "assert":
            fact = TemporalFact(**copy.deepcopy(event["fact"]))
            self._facts[fact.fact_id] = fact
            self._index.setdefault((fact.subject, fact.predicate), []).append(fact.fact_id)
        elif kind == "close":
            old = self._facts[event["fact_id"]]
            self._facts[old.fact_id] = replace(old, valid_to=event["valid_to"],
                                               superseded_by=event["superseded_by"],
                                               closed_at=event["recorded_at"])
        elif kind == "corroborate":
            old = self._facts[event["fact_id"]]
            self._facts[old.fact_id] = replace(
                old, corroborated_by=old.corroborated_by + (event["source"],))
        else:  # pragma: no cover - log só é escrito/validado por este módulo
            raise ValueError("evento desconhecido")

    def _emit(self, event: Dict[str, Any]) -> None:
        event["seq"] = len(self._events) + 1
        event["prev"] = self._head
        event["hash"] = _digest(event)
        self._events.append(copy.deepcopy(event))
        self._head = event["hash"]
        self._apply(event)

    def _project(self, known_at: float) -> Dict[str, TemporalFact]:
        tmp = TemporalFactStore(functional_predicates=self._functional, clock=self._clock)
        for event in self._events:
            if event["recorded_at"] <= known_at:
                tmp._apply(event)
        return tmp._facts

    def _versions(self, subject: str, predicate: str) -> List[TemporalFact]:
        return [self._facts[i] for i in self._index.get((subject, predicate), ())]

    # ── escrita ──────────────────────────────────────────────────────────
    def assert_fact(self, subject: str, predicate: str, obj: str, *,
                    valid_from: Optional[float] = None, confidence: float = 0.9,
                    source: str = "observation",
                    metadata: Optional[Dict[str, Any]] = None) -> TemporalFact:
        _text(subject, "subject"), _text(predicate, "predicate"), _text(obj, "object")
        _text(source, "source")
        conf = _number(confidence, "confidence")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence deve estar em [0, 1]")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata deve ser dict")
        try:
            metadata_value = _json_value(metadata) if metadata is not None else {}
            # A árvore é nativa e sem perdas; o preflight também captura limites
            # do serializador antes de qualquer evento close/assert ser anexado.
            json.dumps(metadata_value, allow_nan=False)
        except (OverflowError, RecursionError, TypeError, ValueError) as exc:
            raise ValueError("metadata não é serializável como JSON estrito") from exc
        requested_valid_from = None if valid_from is None else _number(valid_from, "valid_from")
        with self._lock:
            now = self._now()
            vf = now if requested_valid_from is None else requested_valid_from
            same = self._versions(subject, predicate)
            for fact in same:  # reafirmação: idempotente; fonte nova vira corroboração
                if fact.object == obj and self._covers(fact, vf):
                    if source not in (fact.source,) + fact.corroborated_by:
                        self._emit({"kind": "corroborate", "fact_id": fact.fact_id,
                                    "source": source, "recorded_at": now})
                    return copy.deepcopy(self._facts[fact.fact_id])
            new_to: Optional[float] = None
            prev: Optional[TemporalFact] = None
            if predicate in self._functional:
                later: Optional[TemporalFact] = None
                for fact in same:
                    if self._covers(fact, vf):
                        prev = fact
                    elif fact.valid_from > vf and (later is None or fact.valid_from < later.valid_from):
                        later = fact
                new_to = prev.valid_to if prev else (later.valid_from if later else None)
            self._seq += 1
            fact_id = f"tf-{self._seq:06d}"
            if prev is not None:
                self._emit({"kind": "close", "fact_id": prev.fact_id, "valid_to": vf,
                            "superseded_by": fact_id, "recorded_at": now})
            self._emit({"kind": "assert", "recorded_at": now, "fact": {
                "fact_id": fact_id, "subject": subject, "predicate": predicate,
                "object": obj, "valid_from": vf, "valid_to": new_to, "recorded_at": now,
                "confidence": conf, "source": source, "superseded_by": None,
                "metadata": metadata_value}})
            return copy.deepcopy(self._facts[fact_id])

    def retract(self, subject: str, predicate: str, obj: Optional[str] = None, *,
                valid_to: Optional[float] = None) -> List[TemporalFact]:
        """Encerra fatos válidos em ``valid_to`` (padrão: agora) sem substituí-los."""
        _text(subject, "subject"), _text(predicate, "predicate")
        if obj is not None:
            _text(obj, "object")
        requested_valid_to = None if valid_to is None else _number(valid_to, "valid_to")
        with self._lock:
            now = self._now()
            vt = now if requested_valid_to is None else requested_valid_to
            closed: List[TemporalFact] = []
            for fact in self._versions(subject, predicate):
                if (obj is None or fact.object == obj) and self._covers(fact, vt):
                    self._emit({"kind": "close", "fact_id": fact.fact_id, "valid_to": vt,
                                "superseded_by": None, "recorded_at": now})
                    closed.append(self._facts[fact.fact_id])
            return copy.deepcopy(closed)

    # ── leitura ──────────────────────────────────────────────────────────
    def facts_as_of(self, valid_at: float, *, known_at: Optional[float] = None,
                    subject: Optional[str] = None,
                    predicate: Optional[str] = None) -> List[TemporalFact]:
        t = _number(valid_at, "valid_at")
        with self._lock:
            source = self._facts if known_at is None else self._project(_number(known_at, "known_at"))
            hits = [f for f in source.values() if self._covers(f, t)
                    and (subject is None or f.subject == subject)
                    and (predicate is None or f.predicate == predicate)]
            return copy.deepcopy(sorted(hits, key=lambda f: (f.valid_from, f.fact_id)))

    def get_current_facts(self, subject: Optional[str] = None,
                          predicate: Optional[str] = None) -> List[TemporalFact]:
        with self._lock:
            return self.facts_as_of(self._now(), subject=subject, predicate=predicate)

    def history(self, subject: str, predicate: str) -> List[TemporalFact]:
        with self._lock:
            hits = sorted(self._versions(subject, predicate), key=lambda f: (f.valid_from, f.fact_id))
            return copy.deepcopy(hits)

    @property
    def head_hash(self) -> str:
        return self._head

    def check_invariants(self) -> List[str]:
        """Violações de consistência: janelas invertidas ou sobrepostas em predicados funcionais."""
        with self._lock:
            problems: List[str] = []
            for (subject, predicate), ids in self._index.items():
                facts = [self._facts[i] for i in ids]
                for f in facts:
                    if f.valid_to is not None and f.valid_to < f.valid_from:
                        problems.append(f"{f.fact_id}: valid_to anterior a valid_from")
                if predicate not in self._functional:
                    continue
                live = sorted((f for f in facts if f.valid_to != f.valid_from),
                              key=lambda f: (f.valid_from, f.fact_id))
                for a, b in zip(live, live[1:]):
                    if a.valid_to is None or a.valid_to > b.valid_from:
                        problems.append(f"{a.fact_id}/{b.fact_id}: janelas sobrepostas em "
                                        f"({subject}, {predicate})")
            return problems

    # ── snapshot encadeado ───────────────────────────────────────────────
    def export_snapshot(self, as_json: bool = False) -> Union[Dict[str, Any], str]:
        with self._lock:
            snap = {"schema": SCHEMA, "functional_predicates": sorted(self._functional),
                    "head": self._head, "events": copy.deepcopy(self._events)}
        return json.dumps(snap, sort_keys=True, allow_nan=False) if as_json else snap

    @classmethod
    def from_snapshot(cls, snapshot: Union[Dict[str, Any], str], *,
                      clock: Callable[[], float] = time.time,
                      expected_head: Optional[str] = None) -> "TemporalFactStore":
        """Valida todo o payload (schema, cadeia, tipos, referências, invariantes) antes de devolver."""
        if isinstance(snapshot, str):
            try:
                snapshot = json.loads(snapshot, parse_constant=_reject_constant)
            except ValueError as exc:
                raise ValueError("snapshot JSON inválido") from exc
        if not isinstance(snapshot, dict) or snapshot.get("schema") != SCHEMA:
            raise ValueError("schema incompatível")
        funcs, events = snapshot.get("functional_predicates"), snapshot.get("events")
        if not isinstance(funcs, list) or not all(isinstance(x, str) for x in funcs):
            raise ValueError("functional_predicates inválido")
        if not isinstance(events, list):
            raise ValueError("events inválido")
        store = cls(functional_predicates=funcs, clock=clock)
        for n, raw in enumerate(events, 1):
            if not isinstance(raw, dict):
                raise ValueError(f"evento {n} inválido")
            try:
                chain_ok = raw.get("seq") == n and raw.get("prev") == store._head \
                    and raw.get("hash") == _digest(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"evento {n} não serializável") from exc
            if not chain_ok:
                raise ValueError(f"cadeia de hash inválida no evento {n}")
            recorded = _number(raw.get("recorded_at"), "recorded_at")
            if recorded < store._last_recorded:
                raise ValueError(f"recorded_at regride no evento {n}")
            kind = raw.get("kind")
            if kind == "assert":
                fact = raw.get("fact")
                if not isinstance(fact, dict) or set(fact) != _FACT_KEYS:
                    raise ValueError(f"fato inválido no evento {n}")
                for name in ("subject", "predicate", "object", "source"):
                    _text(fact[name], name)
                vf = _number(fact["valid_from"], "valid_from")
                if fact["valid_to"] is not None and _number(fact["valid_to"], "valid_to") < vf:
                    raise ValueError(f"janela invertida no evento {n}")
                if not 0.0 <= _number(fact["confidence"], "confidence") <= 1.0:
                    raise ValueError(f"confidence fora de [0, 1] no evento {n}")
                if _number(fact["recorded_at"], "recorded_at") != recorded:
                    raise ValueError(f"recorded_at divergente no evento {n}")
                if not isinstance(fact["metadata"], dict):
                    raise ValueError(f"metadata inválido no evento {n}")
                try:
                    _json_value(fact["metadata"])
                except (RecursionError, ValueError) as exc:
                    raise ValueError(f"metadata não é JSON estrito no evento {n}") from exc
                expected_id = f"tf-{store._seq + 1:06d}"
                if fact["fact_id"] != expected_id or fact["superseded_by"] is not None:
                    raise ValueError(f"fact_id ou superseded_by inválido no evento {n}")
                store._seq += 1
            elif kind in ("close", "corroborate"):
                if raw.get("fact_id") not in store._facts:
                    raise ValueError(f"referência inexistente no evento {n}")
                if kind == "close":
                    close_at = _number(raw.get("valid_to"), "valid_to")
                    fact = store._facts[raw["fact_id"]]
                    if close_at < fact.valid_from:
                        raise ValueError(f"close anterior ao início do fato no evento {n}")
                    if fact.valid_to is not None and close_at > fact.valid_to:
                        raise ValueError(f"close não pode estender janela encerrada no evento {n}")
                    if fact.closed_at is not None and close_at == fact.valid_to:
                        raise ValueError(f"close duplicado não pode reescrever o fechamento no evento {n}")
                    target = raw.get("superseded_by")
                    if target is not None and (
                        not isinstance(target, str) or target == raw["fact_id"]
                    ):
                        raise ValueError(f"superseded_by inválido no evento {n}")
                else:
                    _text(raw.get("source"), "source")
            else:
                raise ValueError(f"tipo de evento desconhecido no evento {n}")
            store._events.append(copy.deepcopy(raw))
            store._head = raw["hash"]
            store._last_recorded = recorded
            store._apply(raw)
        for event in store._events:
            target_id = event.get("superseded_by") if event["kind"] == "close" else None
            if target_id is None:
                continue
            prior = store._facts[event["fact_id"]]
            successor = store._facts.get(target_id)
            if (successor is None or prior.predicate not in store._functional
                    or prior.subject != successor.subject
                    or prior.predicate != successor.predicate
                    or successor.valid_from != event["valid_to"]
                    or successor.recorded_at != event["recorded_at"]):
                raise ValueError("superseded_by não referencia o sucessor correspondente")
        if expected_head is not None and store._head != expected_head:
            raise ValueError("head divergente: possível truncamento")
        if snapshot.get("head") != store._head:
            raise ValueError("head do snapshot inconsistente com a cadeia")
        problems = store.check_invariants()
        if problems:
            raise ValueError("invariantes violadas: " + "; ".join(problems[:3]))
        return store
