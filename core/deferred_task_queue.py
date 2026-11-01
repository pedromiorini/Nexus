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
    _STATISTICS_FIELDS = ("enqueued", "retried", "completed", "discarded")

    def __init__(self, max_attempts: int = 3, max_size: int = 256) -> None:
        if (
            type(max_attempts) is not int
            or max_attempts < 1
            or type(max_size) is not int
            or max_size < 1
        ):
            raise ValueError("max_attempts e max_size devem ser inteiros positivos")
        self.max_attempts = max_attempts
        self.max_size = max_size
        self._heap: List[DeferredTask] = []
        # _tasks indexa tanto tarefas pendentes quanto tarefas entregues a consumidores.
        self._tasks: Dict[str, DeferredTask] = {}
        self._in_flight: set[str] = set()
        self._stats = {key: 0 for key in self._STATISTICS_FIELDS}
        self._lock = threading.RLock()

    def enqueue(self, prompt: str, context: Optional[Dict[str, Any]] = None, priority: int = 0, reason: str = "") -> DeferredTask:
        """Exige prioridade int built-in; contexto mantém deepcopy arbitrário em memória."""
        if type(priority) is not int:
            raise ValueError("priority deve ser um inteiro built-in")
        with self._lock:
            if len(self._heap) >= self.max_size:
                self._stats["discarded"] += 1
                raise OverflowError("fila de tarefas adiadas está cheia")
            try:
                copied_context = copy.deepcopy(context) if context is not None else {}
            except Exception as exc:
                raise ValueError("context must be deep-copyable") from exc
            if not isinstance(copied_context, dict):
                raise ValueError("context must be a dictionary")
            task_id = str(uuid.uuid4())
            while task_id in self._tasks:
                task_id = str(uuid.uuid4())
            task = DeferredTask(priority, time(), task_id, prompt, copied_context, 0, self.max_attempts, reason)
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
        if type(max_batch) is not int or max_batch < 1:
            raise ValueError("max_batch deve ser um inteiro positivo")
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
                task.last_reason = f"{type(exc).__name__}: processor raised an exception"

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
        """Exporta pendências; capacidade JSON excedida gera ValueError em ambos os modos."""
        with self._lock:
            payload = {
                "schema": self.SNAPSHOT_SCHEMA,
                "generated_at": time(),
                "max_attempts": self.max_attempts,
                "max_size": self.max_size,
                "statistics": dict(self._stats),
                "tasks": [self._task_to_dict(task) for task in sorted(self._heap, key=lambda item: item.sort_key)],
            }
            validated_tasks, _ = self._validate_snapshot(payload, replace=True)
            payload["tasks"] = validated_tasks
            encoded = self._encode_snapshot(payload)
            return encoded if as_json else payload

    def restore_snapshot(self, snapshot: Any, replace: bool = False) -> Dict[str, Any]:
        """Valida e testa JSON antes de mutar; falha de capacidade gera ValueError.

        Sem teto fixo de profundidade: vale a capacidade do encoder neste ambiente.
        Tarefas em voo não são invalidadas e extras ignorados não são serializados.
        """
        payload = self._decode_snapshot(snapshot)
        with self._lock:
            if replace and self._in_flight:
                raise ValueError("não é possível substituir a fila enquanto há tarefas em voo")
            tasks, snapshot_statistics = self._validate_snapshot(payload, replace=replace)
            if not replace and len(self._heap) + len(tasks) > self.max_size:
                raise ValueError("snapshot excede a capacidade disponível da fila")
            if replace and len(tasks) > self.max_size:
                raise ValueError("snapshot excede max_size da fila")

            # Serializa somente dados capturados/validados, no mesmo envelope do export.
            # Não percorre extensões ignoradas nem publica tarefas antes do preflight.
            self._encode_snapshot({
                "schema": self.SNAPSHOT_SCHEMA,
                "generated_at": 0.0,
                "max_attempts": self.max_attempts,
                "max_size": self.max_size,
                "statistics": snapshot_statistics,
                "tasks": tasks,
            })
            restored = [DeferredTask(**task) for task in tasks]
            staged_heap = [] if replace else list(self._heap)
            staged_tasks = {} if replace else dict(self._tasks)
            staged_in_flight = set() if replace else set(self._in_flight)
            staged_stats = (
                dict(snapshot_statistics)
                if replace
                else {key: self._stats[key] for key in self._STATISTICS_FIELDS}
            )
            for task in restored:
                heapq.heappush(staged_heap, task)
                staged_tasks[task.task_id] = task

            # O heap, o índice, os IDs em voo e os contadores só se tornam visíveis
            # depois que toda a construção temporária terminou com sucesso.
            self._heap, self._tasks, self._in_flight, self._stats = (
                staged_heap, staged_tasks, staged_in_flight, staged_stats
            )
            return {"restored": len(restored), "depth": len(self._heap), "replaced": replace}

    @staticmethod
    def _encode_snapshot(payload: Dict[str, Any]) -> str:
        """Usa o encoder real, sem orçamento arbitrário de profundidade."""
        try:
            return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except RecursionError as exc:
            raise ValueError("snapshot excede a capacidade de serialização JSON") from exc

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
            def reject_constant(value: str) -> None:
                raise ValueError(f"constante JSON não permitida: {value}")

            def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
                result: Dict[str, Any] = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"chave JSON duplicada: {key}")
                    result[key] = value
                return result

            try:
                snapshot = json.loads(
                    snapshot,
                    parse_constant=reject_constant,
                    object_pairs_hook=reject_duplicate_keys,
                )
            except (json.JSONDecodeError, RecursionError, ValueError) as exc:
                raise ValueError("snapshot não é JSON válido") from exc
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot deve ser um objeto")
        return snapshot

    def _validate_snapshot(
        self, payload: Dict[str, Any], replace: bool = False
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        if not isinstance(payload, dict):
            raise ValueError("snapshot deve ser um objeto")
        try:
            payload = dict.copy(payload)
        except RuntimeError as exc:
            raise ValueError("snapshot foi modificado durante a leitura") from exc
        if any(type(key) is not str for key in payload):
            raise ValueError("chaves do snapshot devem ser strings")

        schema = payload.get("schema")
        if type(schema) is not str or schema != self.SNAPSHOT_SCHEMA:
            raise ValueError("schema de snapshot incompatível")
        required = ("generated_at", "max_attempts", "max_size", "statistics", "tasks")
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(f"campos ausentes no snapshot: {', '.join(missing)}")
        generated_at = payload["generated_at"]
        snapshot_max_attempts = payload["max_attempts"]
        snapshot_max_size = payload["max_size"]
        if not self._is_finite_number(generated_at):
            raise ValueError("generated_at inválido no snapshot")
        if (
            type(snapshot_max_attempts) is not int
            or type(snapshot_max_size) is not int
            or snapshot_max_attempts != self.max_attempts
            or snapshot_max_size != self.max_size
        ):
            raise ValueError("limites do snapshot não correspondem à fila")
        statistics = payload["statistics"]
        if not isinstance(statistics, dict):
            raise ValueError("estatísticas inválidas no snapshot")
        try:
            statistics = dict.copy(statistics)
        except RuntimeError as exc:
            raise ValueError("estatísticas foram modificadas durante a leitura") from exc
        expected_statistics = self._STATISTICS_FIELDS
        if (
            any(type(key) is not str for key in statistics)
            or set(statistics) != set(expected_statistics)
            or any(type(statistics[key]) is not int or statistics[key] < 0 for key in expected_statistics)
        ):
            raise ValueError("estatísticas inválidas no snapshot")
        validated_statistics = {key: statistics[key] for key in expected_statistics}

        tasks = payload["tasks"]
        if not isinstance(tasks, list):
            raise ValueError("tasks deve ser um array")
        try:
            tasks = list.copy(tasks)
        except RuntimeError as exc:
            raise ValueError("tasks foi modificado durante a leitura") from exc
        if len(tasks) > self.max_size:
            raise ValueError("snapshot excede max_size da fila")

        validated: List[Dict[str, Any]] = []
        ids = set()
        fields = ("task_id", "prompt", "context", "priority", "created_at", "attempts", "max_attempts", "last_reason")
        for raw in tasks:
            if not isinstance(raw, dict):
                raise ValueError("tarefa inválida no snapshot")
            try:
                raw = dict.copy(raw)
            except RuntimeError as exc:
                raise ValueError("tarefa foi modificada durante a leitura") from exc
            if any(type(key) is not str for key in raw):
                raise ValueError("chaves da tarefa devem ser strings")
            if any(field not in raw for field in fields):
                raise ValueError("tarefa inválida no snapshot")

            task_id = raw["task_id"]
            prompt = raw["prompt"]
            context_value = raw["context"]
            priority = raw["priority"]
            created_at = raw["created_at"]
            attempts = raw["attempts"]
            max_attempts = raw["max_attempts"]
            last_reason = raw["last_reason"]

            if type(task_id) is not str or not task_id or task_id in ids:
                raise ValueError("task_id ausente ou duplicado no snapshot")
            if not replace and task_id in self._tasks:
                raise ValueError(f"task_id já existente na fila: {task_id}")
            if type(prompt) is not str or type(context_value) is not dict:
                raise ValueError("prompt ou context inválido no snapshot")
            if type(priority) is not int:
                raise ValueError("priority inválida no snapshot")
            if not self._is_finite_number(created_at):
                raise ValueError("created_at inválido no snapshot")
            if type(attempts) is not int or attempts < 0:
                raise ValueError("attempts inválido no snapshot")
            if (
                type(max_attempts) is not int
                or max_attempts < 1
                or max_attempts != self.max_attempts
                or attempts >= max_attempts
            ):
                raise ValueError("max_attempts inválido ou incompatível com a fila no snapshot")
            if type(last_reason) is not str:
                raise ValueError("last_reason inválido no snapshot")
            context = self._copy_strict_json_context(context_value)
            ids.add(task_id)
            validated.append({
                "task_id": task_id,
                "prompt": prompt,
                "context": context,
                "priority": priority,
                "created_at": created_at,
                "attempts": attempts,
                "max_attempts": max_attempts,
                "last_reason": last_reason,
            })
        return validated, validated_statistics

    @staticmethod
    def _copy_strict_json_context(context: Any) -> Dict[str, Any]:
        """Valida e copia uma árvore JSON nativa sem executar hooks de objetos externos."""
        if type(context) is not dict:
            raise ValueError("context deve ser um objeto JSON")

        copied: Dict[str, Any] = {}
        active_containers = {id(context)}
        stack = [(context, copied, iter(context.items()), False)]
        while stack:
            source, target, iterator, is_list = stack[-1]
            try:
                if is_list:
                    value = next(iterator)
                    key = None
                else:
                    key, value = next(iterator)
                    if type(key) is not str:
                        raise ValueError("chaves de context devem ser strings")
            except StopIteration:
                active_containers.remove(id(source))
                stack.pop()
                continue

            value_type = type(value)
            if value_type is dict or value_type is list:
                value_id = id(value)
                if value_id in active_containers:
                    raise ValueError("context não pode conter ciclos")
                child = {} if value_type is dict else []
                if is_list:
                    target.append(child)
                else:
                    target[key] = child
                active_containers.add(value_id)
                child_iterator = iter(value.items()) if value_type is dict else iter(value)
                stack.append((value, child, child_iterator, value_type is list))
            elif value_type in (str, int, bool, type(None)):
                if is_list:
                    target.append(value)
                else:
                    target[key] = value
            elif value_type is float and math.isfinite(value):
                if is_list:
                    target.append(value)
                else:
                    target[key] = value
            else:
                raise ValueError("context deve conter somente valores JSON nativos e finitos")
        return copied

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
