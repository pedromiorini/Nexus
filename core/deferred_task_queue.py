"""Fila priorizada e limitada para tarefas adiadas pelo sistema de defesa."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional
import heapq
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
    """Fila em memória; não executa tarefas automaticamente nem perde contexto silenciosamente."""

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
