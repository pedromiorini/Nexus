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

    def complete(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)
        self._stats["completed"] += 1

    def discard(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)
        self._stats["discarded"] += 1

    def statistics(self) -> Dict[str, Any]:
        return {"depth": len(self._heap), "max_attempts": self.max_attempts, "max_size": self.max_size, **self._stats}
