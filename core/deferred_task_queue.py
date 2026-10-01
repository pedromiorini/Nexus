"""Fila priorizada e limitada para tarefas adiadas pelo sistema de defesa."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional
import heapq
import json
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
    """Fila em memória; snapshots são contratos de transporte, não persistência física."""

    SNAPSHOT_SCHEMA = "nexus.deferred_task_queue.v1"

    def __init__(self, max_attempts: int = 3, max_size: int = 256) -> None:
        if max_attempts < 1 or max_size < 1:
            raise ValueError("max_attempts e max_size devem ser positivos")
        self.max_attempts = max_attempts
        self.max_size = max_size
        self._heap: List[DeferredTask] = []
        self._tasks: Dict[str, DeferredTask] = {}
        self._stats = {"enqueued": 0, "retried": 0, "completed": 0, "discarded": 0}

    def enqueue(self, prompt: str, context: Optional[Dict[str, Any]] = None, priority: int = 0, reason: str = "") -> DeferredTask:
        if len(self._heap) >= self.max_size:
            self._stats["discarded"] += 1
            raise OverflowError("fila de tarefas adiadas está cheia")
        task = DeferredTask(priority, time(), str(uuid.uuid4()), prompt, context or {}, 0, self.max_attempts, reason)
        heapq.heappush(self._heap, task)
        self._tasks[task.task_id] = task
        self._stats["enqueued"] += 1
        return task

    def recheck_and_pop(self, pressure_critical: bool) -> Optional[DeferredTask]:
        if pressure_critical or not self._heap:
            return None
        task = heapq.heappop(self._heap)
        self._tasks.pop(task.task_id, None)
        task.attempts += 1
        self._stats["retried"] += 1
        return task

    def requeue(self, task: DeferredTask, reason: str = "") -> bool:
        task.last_reason = reason or task.last_reason
        if task.attempts >= task.max_attempts:
            self._stats["discarded"] += 1
            return False
        heapq.heappush(self._heap, task)
        self._tasks[task.task_id] = task
        return True

    def consume(self, processor, max_batch: int = 1, pressure_critical: bool = False, on_success=None, on_failure=None, on_discard=None) -> Dict[str, Any]:
        """Processa um lote finito; callbacks são opcionais e nunca controlam o retry."""
        if max_batch < 1:
            raise ValueError("max_batch deve ser positivo")
        if pressure_critical:
            return {"processed": 0, "completed": 0, "retried": 0, "discarded": 0, "blocked": True}
        processed = completed = retried = discarded = 0
        initial_depth = min(len(self._heap), max_batch)
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
                self.complete(task.task_id)
                completed += 1
                if on_success is not None:
                    on_success(task)
            elif self.requeue(task):
                retried += 1
                if on_failure is not None:
                    on_failure(task)
            else:
                discarded += 1
                if on_failure is not None:
                    on_failure(task)
                if on_discard is not None:
                    on_discard(task)
        return {"processed": processed, "completed": completed, "retried": retried, "discarded": discarded, "blocked": False}

    def complete(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)
        self._stats["completed"] += 1

    def discard(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)
        self._stats["discarded"] += 1

    def statistics(self) -> Dict[str, Any]:
        return {"depth": len(self._heap), "max_attempts": self.max_attempts, "max_size": self.max_size, **self._stats}


    def export_snapshot(self, as_json: bool = False) -> Any:
        """Exporta somente tarefas pendentes; nenhum processamento é disparado."""
        payload = {
            "schema": self.SNAPSHOT_SCHEMA,
            "generated_at": time(),
            "max_attempts": self.max_attempts,
            "max_size": self.max_size,
            "statistics": dict(self._stats),
            "tasks": [self._task_to_dict(task) for task in sorted(self._heap, key=lambda item: item.sort_key)],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) if as_json else payload

    def restore_snapshot(self, snapshot: Any, replace: bool = False) -> Dict[str, Any]:
        """Restaura tarefas pendentes após validar o snapshot inteiro de forma atômica."""
        payload = self._decode_snapshot(snapshot)
        tasks = self._validate_snapshot(payload, replace=replace)
        if not replace and len(self._heap) + len(tasks) > self.max_size:
            raise ValueError("snapshot excede a capacidade disponível da fila")
        if replace and len(tasks) > self.max_size:
            raise ValueError("snapshot excede max_size da fila")

        restored = [DeferredTask(**task) for task in tasks]
        if replace:
            self._heap.clear()
            self._tasks.clear()
            self._stats = dict(payload["statistics"])
        for task in restored:
            if task.task_id in self._tasks:
                raise ValueError(f"task_id duplicado na fila: {task.task_id}")
            heapq.heappush(self._heap, task)
            self._tasks[task.task_id] = task
        return {"restored": len(restored), "depth": len(self._heap), "replaced": replace}

    @staticmethod
    def _task_to_dict(task: DeferredTask) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "prompt": task.prompt,
            "context": task.context,
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
        if not isinstance(payload["generated_at"], (int, float)) or isinstance(payload["generated_at"], bool):
            raise ValueError("generated_at inválido no snapshot")
        if payload["max_attempts"] != self.max_attempts or payload["max_size"] != self.max_size:
            raise ValueError("limites do snapshot não correspondem à fila")
        if not isinstance(payload["statistics"], dict) or any(
            not isinstance(payload["statistics"].get(key), int) or payload["statistics"][key] < 0
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
            if not isinstance(raw["created_at"], (int, float)) or isinstance(raw["created_at"], bool):
                raise ValueError("created_at inválido no snapshot")
            if not isinstance(raw["attempts"], int) or isinstance(raw["attempts"], bool) or raw["attempts"] < 0:
                raise ValueError("attempts inválido no snapshot")
            if not isinstance(raw["max_attempts"], int) or raw["max_attempts"] < 1 or raw["attempts"] > raw["max_attempts"]:
                raise ValueError("max_attempts inválido no snapshot")
            if not isinstance(raw["last_reason"], str):
                raise ValueError("last_reason inválido no snapshot")
            try:
                json.dumps(raw["context"])
            except (TypeError, ValueError) as exc:
                raise ValueError("context não é serializável") from exc
            ids.add(raw["task_id"])
            validated.append({field: raw[field] for field in fields})
        return validated
