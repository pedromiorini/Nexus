"""Fila priorizada e limitada para tarefas adiadas pelo sistema de defesa."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import heapq
import json
import math
import threading
from time import time
from typing import Any, Dict, List, Optional
import uuid


@dataclass(order=True)
class DeferredTask:
    sort_key: tuple = field(init=False, repr=False)
    priority: int
    created_at: float
    task_id: str = field(compare=False)
    prompt: str = field(compare=False)
    context: Dict[str, Any] = field(compare=False, default_factory=dict)
    attempts: int = field(compare=False, default=0)
    max_attempts: int = field(compare=False, default=3)
    last_reason: str = field(compare=False, default="")

    def __post_init__(self) -> None:
        self.sort_key = (-self.priority, self.created_at)


class DeferredTaskQueue:
    """Fila em memória; snapshots transportam tarefas pendentes, não tarefas em voo."""

    SNAPSHOT_SCHEMA = "nexus.deferred_task_queue.v1"

    def __init__(self, max_attempts: int = 3, max_size: int = 256) -> None:
        if max_attempts < 1 or max_size < 1:
            raise ValueError("max_attempts e max_size devem ser positivos")
        self.max_attempts = max_attempts
        self.max_size = max_size
        self._heap: List[DeferredTask] = []
        # _tasks indexa tanto tarefas pendentes quanto tarefas entregues a consumidores.
        self._tasks: Dict[str, DeferredTask] = {}
        self._in_flight: set[str] = set()
        self._stats = {"enqueued": 0, "retried": 0, "completed": 0, "discarded": 0}
        self._lock = threading.RLock()

    def enqueue(self, prompt: str, context: Optional[Dict[str, Any]] = None, priority: int = 0, reason: str = "") -> DeferredTask:
        with self._lock:
            if len(self._heap) >= self.max_size:
                self._stats["discarded"] += 1
                raise OverflowError("fila de tarefas adiadas está cheia")
            task_id = str(uuid.uuid4())
            while task_id in self._tasks:
                task_id = str(uuid.uuid4())
            task = DeferredTask(priority, time(), task_id, prompt, context or {}, 0, self.max_attempts, reason)
            heapq.heappush(self._heap, task)
            self._tasks[task.task_id] = task
            self._stats["enqueued"] += 1
            return task

    def recheck_and_pop(self, pressure_critical: bool) -> Optional[DeferredTask]:
        with self._lock:
            if pressure_critical or not self._heap:
                return None
            task = heapq.heappop(self._heap)
            self._in_flight.add(task.task_id)
            task.attempts += 1
            self._stats["retried"] += 1
            return task

    def _requeue_status(self, task: DeferredTask, reason: str = "") -> str:
        """Transiciona apenas uma tarefa em voo; devolve queued/discarded/ignored."""
        with self._lock:
            task_id = task.task_id
            if self._tasks.get(task_id) is not task or task_id not in self._in_flight:
                return "ignored"
            task.last_reason = reason or task.last_reason
            if task.attempts >= task.max_attempts or len(self._heap) >= self.max_size:
                self._in_flight.remove(task_id)
                self._tasks.pop(task_id, None)
                self._stats["discarded"] += 1
                return "discarded"
            self._in_flight.remove(task_id)
            heapq.heappush(self._heap, task)
            return "queued"

    def requeue(self, task: DeferredTask, reason: str = "") -> bool:
        """Reinsere uma tarefa em voo no máximo uma vez e sem exceder a capacidade."""
        return self._requeue_status(task, reason) == "queued"

    def consume(self, processor, max_batch: int = 1, pressure_critical: bool = False, on_success=None, on_failure=None, on_discard=None) -> Dict[str, Any]:
        """Processa um lote finito. Erros de callback são reportados, sem reverter estados."""
        if max_batch < 1:
            raise ValueError("max_batch deve ser positivo")
        if pressure_critical:
            return {
                "processed": 0, "completed": 0, "retried": 0,
                "discarded": 0, "superseded": 0, "callback_errors": [], "blocked": True,
            }

        processed = completed = retried = discarded = superseded = 0
        callback_errors: List[Dict[str, str]] = []
        with self._lock:
            initial_depth = min(len(self._heap), max_batch)

        def notify(name: str, callback, task: DeferredTask) -> None:
            if callback is None:
                return
            try:
                callback(task)
            except Exception as exc:
                # Não inclui a mensagem, que pode conter conteúdo sensível da tarefa.
                callback_errors.append({
                    "callback": name,
                    "task_id": task.task_id,
                    "exception": type(exc).__name__,
                })

        for _ in range(initial_depth):
            task = self.recheck_and_pop(False)
            if task is None:
                break
            processed += 1
            try:
                ok = bool(processor(task))
            except Exception as exc:
                ok = False
                task.last_reason = f"{type(exc).__name__}: {exc}"

            if ok:
                if self.complete(task.task_id):
                    completed += 1
                    notify("on_success", on_success, task)
                else:
                    superseded += 1
                continue

            transition = self._requeue_status(task)
            if transition == "queued":
                retried += 1
                notify("on_failure", on_failure, task)
            elif transition == "discarded":
                discarded += 1
                notify("on_failure", on_failure, task)
                notify("on_discard", on_discard, task)
            else:
                superseded += 1

        return {
            "processed": processed,
            "completed": completed,
            "retried": retried,
            "discarded": discarded,
            "superseded": superseded,
            "callback_errors": callback_errors,
            "blocked": False,
        }

    def _remove_pending_locked(self, task_id: str) -> None:
        if task_id in self._in_flight:
            self._in_flight.remove(task_id)
            return
        self._heap = [task for task in self._heap if task.task_id != task_id]
        heapq.heapify(self._heap)

    def complete(self, task_id: str) -> bool:
        """Conclui uma tarefa ativa; é idempotente e remove pendências do heap."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            self._tasks.pop(task_id, None)
            self._remove_pending_locked(task_id)
            self._stats["completed"] += 1
            return True

    def discard(self, task_id: str) -> bool:
        """Descarta uma tarefa ativa; é idempotente e remove pendências do heap."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            self._tasks.pop(task_id, None)
            self._remove_pending_locked(task_id)
            self._stats["discarded"] += 1
            return True

    def statistics(self) -> Dict[str, Any]:
        with self._lock:
            return {"depth": len(self._heap), "max_attempts": self.max_attempts, "max_size": self.max_size, **self._stats}

    def export_snapshot(self, as_json: bool = False) -> Any:
        """Exporta somente tarefas pendentes; nenhum processamento é disparado."""
        with self._lock:
            payload = {
                "schema": self.SNAPSHOT_SCHEMA,
                "generated_at": time(),
                "max_attempts": self.max_attempts,
                "max_size": self.max_size,
                "statistics": dict(self._stats),
                "tasks": [self._task_to_dict(task) for task in sorted(self._heap, key=lambda item: item.sort_key)],
            }
            return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) if as_json else payload

    def restore_snapshot(self, snapshot: Any, replace: bool = False) -> Dict[str, Any]:
        """Valida o snapshot inteiro e restaura-o atomicamente, sem invalidar tarefas em voo."""
        payload = copy.deepcopy(self._decode_snapshot(snapshot))
        with self._lock:
            if replace and self._in_flight:
                raise ValueError("não é possível substituir a fila enquanto há tarefas em voo")
            tasks = self._validate_snapshot(payload, replace=replace)
            if not replace and len(self._heap) + len(tasks) > self.max_size:
                raise ValueError("snapshot excede a capacidade disponível da fila")
            if replace and len(tasks) > self.max_size:
                raise ValueError("snapshot excede max_size da fila")

            restored = []
            for task in tasks:
                detached = dict(task)
                detached["context"] = copy.deepcopy(task["context"])
                restored.append(DeferredTask(**detached))
            if replace:
                self._heap.clear()
                self._tasks.clear()
                self._in_flight.clear()
                self._stats = dict(payload["statistics"])
            for task in restored:
                # Todas as colisões foram verificadas antes de modificar o estado.
                heapq.heappush(self._heap, task)
                self._tasks[task.task_id] = task
            return {"restored": len(restored), "depth": len(self._heap), "replaced": replace}

    @staticmethod
    def _task_to_dict(task: DeferredTask) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "prompt": task.prompt,
            "context": copy.deepcopy(task.context),
            "priority": task.priority,
            "created_at": task.created_at,
            "attempts": task.attempts,
            "max_attempts": task.max_attempts,
            "last_reason": task.last_reason,
        }

    @staticmethod
    def _decode_snapshot(snapshot: Any) -> Dict[str, Any]:
        if isinstance(snapshot, str):
            try:
                snapshot = json.loads(snapshot)
            except json.JSONDecodeError as exc:
                raise ValueError("snapshot não é JSON válido") from exc
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot deve ser um objeto")
        return snapshot

    def _validate_snapshot(self, payload: Dict[str, Any], replace: bool = False) -> List[Dict[str, Any]]:
        if payload.get("schema") != self.SNAPSHOT_SCHEMA:
            raise ValueError("schema de snapshot incompatível")
        required = ("generated_at", "max_attempts", "max_size", "statistics", "tasks")
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(f"campos ausentes no snapshot: {', '.join(missing)}")
        if not self._is_finite_number(payload["generated_at"]):
            raise ValueError("generated_at inválido no snapshot")
        if (
            type(payload["max_attempts"]) is not int
            or type(payload["max_size"]) is not int
            or payload["max_attempts"] != self.max_attempts
            or payload["max_size"] != self.max_size
        ):
            raise ValueError("limites do snapshot não correspondem à fila")
        if not isinstance(payload["statistics"], dict) or any(
            type(payload["statistics"].get(key)) is not int or payload["statistics"][key] < 0
            for key in self._stats
        ):
            raise ValueError("estatísticas inválidas no snapshot")
        if not isinstance(payload["tasks"], list):
            raise ValueError("tasks deve ser um array")
        if len(payload["tasks"]) > self.max_size:
            raise ValueError("snapshot excede max_size da fila")

        validated: List[Dict[str, Any]] = []
        ids = set()
        fields = ("task_id", "prompt", "context", "priority", "created_at", "attempts", "max_attempts", "last_reason")
        for raw in payload["tasks"]:
            if not isinstance(raw, dict) or any(field not in raw for field in fields):
                raise ValueError("tarefa inválida no snapshot")
            if not isinstance(raw["task_id"], str) or not raw["task_id"] or raw["task_id"] in ids:
                raise ValueError("task_id ausente ou duplicado no snapshot")
            if not replace and raw["task_id"] in self._tasks:
                raise ValueError(f"task_id já existente na fila: {raw['task_id']}")
            if not isinstance(raw["prompt"], str) or not isinstance(raw["context"], dict):
                raise ValueError("prompt ou context inválido no snapshot")
            if not isinstance(raw["priority"], int) or isinstance(raw["priority"], bool):
                raise ValueError("priority inválida no snapshot")
            if not self._is_finite_number(raw["created_at"]):
                raise ValueError("created_at inválido no snapshot")
            if not isinstance(raw["attempts"], int) or isinstance(raw["attempts"], bool) or raw["attempts"] < 0:
                raise ValueError("attempts inválido no snapshot")
            if (
                type(raw["max_attempts"]) is not int
                or raw["max_attempts"] < 1
                or raw["max_attempts"] != self.max_attempts
                or raw["attempts"] > raw["max_attempts"]
            ):
                raise ValueError("max_attempts inválido ou incompatível com a fila no snapshot")
            if not isinstance(raw["last_reason"], str):
                raise ValueError("last_reason inválido no snapshot")
            try:
                json.dumps(raw["context"], allow_nan=False)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError("context não é serializável") from exc
            ids.add(raw["task_id"])
            validated.append({field: raw[field] for field in fields})
        return validated

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
