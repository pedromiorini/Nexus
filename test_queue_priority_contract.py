"""Contrato A06: prioridade escalar estrita, contexto em memória compatível."""
import copy
from enum import IntEnum
import unittest
from unittest.mock import patch

from core.deferred_task_queue import DeferredTaskQueue


class IntegerPriority(int):
    pass


class FloatPriority(float):
    pass


class EnumPriority(IntEnum):
    HIGH = 10


class QueuePriorityContractTests(unittest.TestCase):
    @staticmethod
    def capture(queue):
        return (
            tuple(id(task) for task in queue._heap),
            tuple((key, id(task), copy.deepcopy(vars(task))) for key, task in queue._tasks.items()),
            set(queue._in_flight),
            queue.statistics(),
        )

    def test_invalid_priority_preserves_empty_pending_full_and_in_flight_state(self):
        invalid_values = (
            True, False, 1.5, 1.0, float("nan"), float("inf"),
            float("-inf"), IntegerPriority(3), FloatPriority(3),
            EnumPriority.HIGH, "3", None, object(),
        )
        for state in ("empty", "pending", "full", "in_flight"):
            for invalid in invalid_values:
                with self.subTest(state=state, priority_type=type(invalid).__name__):
                    queue = DeferredTaskQueue(max_size=2)
                    if state != "empty":
                        queue.enqueue("keep", {"values": [1]}, priority=2)
                    if state == "full":
                        queue.enqueue("also keep", priority=-2)
                    elif state == "in_flight":
                        queue.recheck_and_pop(False)
                        queue.enqueue("pending", priority=0)
                    references = (queue._heap, queue._tasks, queue._in_flight, queue._stats)
                    before = self.capture(queue)
                    with self.assertRaisesRegex(ValueError, "priority"):
                        queue.enqueue("invalid", {"values": [2]}, priority=invalid)
                    self.assertEqual(self.capture(queue), before)
                    for actual, original in zip(
                        (queue._heap, queue._tasks, queue._in_flight, queue._stats), references
                    ):
                        self.assertIs(actual, original)
                    if state != "full":
                        queue.enqueue("usable", priority=1)

    def test_invalid_priority_precedes_copy_allocation_clock_constructor_and_heap(self):
        queue = DeferredTaskQueue()
        for target in ("copy.deepcopy", "uuid.uuid4", "time", "DeferredTask", "heapq.heappush"):
            with self.subTest(target=target):
                before = self.capture(queue)
                with patch("core.deferred_task_queue." + target) as operation:
                    with self.assertRaisesRegex(ValueError, "priority"):
                        queue.enqueue("invalid", {"anything": object()}, priority=True)
                    operation.assert_not_called()
                self.assertEqual(self.capture(queue), before)

    def test_priority_subclass_hooks_are_not_executed(self):
        calls = []

        def hook(*_args, **_kwargs):
            calls.append("called")
            raise AssertionError("priority hook executed")

        class HookedInt(int):
            __neg__ = __int__ = __index__ = __float__ = hook
            __lt__ = __gt__ = __eq__ = __bool__ = __repr__ = __str__ = hook
            __getattribute__ = hook

        class HookedFloat(float):
            __neg__ = __int__ = __float__ = hook
            __lt__ = __gt__ = __eq__ = __bool__ = __repr__ = __str__ = hook
            __getattribute__ = hook

        for invalid in (HookedInt(3), HookedFloat(3)):
            for full in (False, True):
                with self.subTest(priority_type=type(invalid).__name__, full=full):
                    queue = DeferredTaskQueue(max_size=1)
                    if full:
                        queue.enqueue("keep")
                    before = self.capture(queue)
                    with self.assertRaisesRegex(ValueError, "priority"):
                        queue.enqueue("invalid", priority=invalid)
                    self.assertEqual(calls, [])
                    self.assertEqual(self.capture(queue), before)

    def test_valid_integer_priorities_keep_order_and_snapshot_round_trip(self):
        priorities = [0, -1, 1, -(10 ** 100), 10 ** 100]
        queue = DeferredTaskQueue(max_size=len(priorities))
        for priority in priorities:
            task = queue.enqueue(str(priority), priority=priority)
            self.assertIs(type(task.priority), int)
            self.assertEqual(task.priority, priority)
        for as_json in (False, True):
            restored = DeferredTaskQueue(max_size=len(priorities))
            restored.restore_snapshot(queue.export_snapshot(as_json=as_json), replace=True)
            self.assertEqual(
                [restored.recheck_and_pop(False).priority for _ in priorities],
                sorted(priorities, reverse=True),
            )
        self.assertEqual(queue.statistics()["enqueued"], len(priorities))
        self.assertEqual(queue.statistics()["discarded"], 0)
        self.assertEqual(
            [queue.recheck_and_pop(False).priority for _ in priorities],
            sorted(priorities, reverse=True),
        )

    def test_default_priority_and_equal_priority_fifo_remain_unchanged(self):
        queue = DeferredTaskQueue()
        with patch("core.deferred_task_queue.time", side_effect=[1.0, 2.0]):
            first = queue.enqueue("first")
            second = queue.enqueue("second", priority=0)
        self.assertIs(type(first.priority), int)
        self.assertEqual(first.priority, 0)
        self.assertIs(queue.recheck_and_pop(False), first)
        self.assertIs(queue.recheck_and_pop(False), second)

    def test_valid_priority_on_full_queue_retains_overflow_statistics(self):
        queue = DeferredTaskQueue(max_size=1)
        existing = queue.enqueue("keep")
        with self.assertRaises(OverflowError):
            queue.enqueue("overflow", priority=1)
        self.assertEqual(queue._heap, [existing])
        self.assertEqual(queue.statistics(), {
            "depth": 1, "max_attempts": 3, "max_size": 1,
            "enqueued": 1, "retried": 0, "completed": 0, "discarded": 1,
        })

    def test_arbitrary_deepcopyable_context_remains_accepted_in_memory_only(self):
        class Payload:
            def __init__(self):
                self.values = [1]

        queue = DeferredTaskQueue()
        payload = Payload()
        task = queue.enqueue("in memory", {"payload": payload}, priority=1)
        self.assertIsInstance(task.context["payload"], Payload)
        self.assertIsNot(task.context["payload"], payload)
        payload.values.append(2)
        self.assertEqual(task.context["payload"].values, [1])
        with self.assertRaises(ValueError):
            queue.export_snapshot()
        self.assertIs(queue.recheck_and_pop(False), task)
        self.assertTrue(queue.complete(task.task_id))


if __name__ == "__main__":
    unittest.main()
