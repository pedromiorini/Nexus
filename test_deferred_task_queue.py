import unittest
from core.deferred_task_queue import DeferredTaskQueue
from core.constitutional_brain import CentralRouter


class DeferredTaskQueueTests(unittest.TestCase):
    def test_priority_order(self):
        queue = DeferredTaskQueue()
        low = queue.enqueue("low", priority=1)
        high = queue.enqueue("high", priority=10)
        first = queue.recheck_and_pop(False)
        self.assertEqual(first.task_id, high.task_id)
        self.assertEqual(first.prompt, "high")
        self.assertEqual(first.attempts, 1)

    def test_critical_pressure_does_not_pop(self):
        queue = DeferredTaskQueue()
        queue.enqueue("blocked", priority=5)
        self.assertIsNone(queue.recheck_and_pop(True))
        self.assertEqual(queue.statistics()["depth"], 1)

    def test_capacity_limit(self):
        queue = DeferredTaskQueue(max_size=1)
        queue.enqueue("first")
        with self.assertRaises(OverflowError):
            queue.enqueue("second")
        self.assertEqual(queue.statistics()["discarded"], 1)

    def test_completion_and_stats(self):
        queue = DeferredTaskQueue()
        task = queue.enqueue("work", priority=3)
        ready = queue.recheck_and_pop(False)
        queue.complete(task.task_id)
        self.assertEqual(ready.task_id, task.task_id)
        self.assertEqual(queue.statistics()["retried"], 1)
        self.assertEqual(queue.statistics()["completed"], 1)

    def test_consume_batch_completes_tasks(self):
        queue = DeferredTaskQueue()
        queue.enqueue("a", priority=1)
        queue.enqueue("b", priority=2)
        seen = []
        result = queue.consume(lambda task: seen.append(task.prompt) or True, max_batch=2)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(seen, ["b", "a"])

    def test_consume_retries_then_discards(self):
        queue = DeferredTaskQueue(max_attempts=2)
        queue.enqueue("unstable")
        first = queue.consume(lambda task: False)
        second = queue.consume(lambda task: False)
        self.assertEqual(first["retried"], 1)
        self.assertEqual(second["discarded"], 1)
        self.assertEqual(queue.statistics()["depth"], 0)

    def test_consume_blocked_by_critical_pressure(self):
        queue = DeferredTaskQueue()
        queue.enqueue("blocked")
        result = queue.consume(lambda task: True, pressure_critical=True)
        self.assertTrue(result["blocked"])
        self.assertEqual(result["processed"], 0)

    def test_lifecycle_callbacks(self):
        queue = DeferredTaskQueue(max_attempts=1)
        queue.enqueue("ok")
        queue.enqueue("bad")
        success, failure, discarded = [], [], []
        result = queue.consume(
            lambda task: task.prompt == "ok",
            max_batch=2,
            on_success=lambda task: success.append(task.prompt),
            on_failure=lambda task: failure.append(task.prompt),
            on_discard=lambda task: discarded.append(task.prompt),
        )
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["discarded"], 1)
        self.assertEqual(success, ["ok"])
        self.assertEqual(failure, ["bad"])
        self.assertEqual(discarded, ["bad"])

    def test_router_reprocess_preserves_context_and_metrics(self):
        router = CentralRouter.__new__(CentralRouter)
        router.deferred_queue = DeferredTaskQueue(max_attempts=1)
        router.stats = {"deferred_reprocessing": {
            "attempts": 0, "completed": 0, "failed": 0,
            "discarded": 0, "total_latency_ms": 0.0,
        }}
        router.vram_guard = type("Guard", (), {
            "evaluate": lambda self: type("Decision", (), {"action": "normal"})()
        })()
        observed = []
        router.route = lambda prompt, context: observed.append((prompt, context)) or {"success": True}
        router.deferred_queue.enqueue("recover", {"trace_id": "t-1"})
        callbacks = []
        result = router.reprocess_deferred_tasks(
            on_success=lambda task: callbacks.append(task.task_id)
        )
        self.assertEqual(result["completed"], 1)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0], "recover")
        self.assertEqual(observed[0][1]["trace_id"], "t-1")
        self.assertTrue(observed[0][1]["_deferred_reprocess"])
        self.assertEqual(router.stats["deferred_reprocessing"]["attempts"], 1)
        self.assertEqual(router.stats["deferred_reprocessing"]["completed"], 1)
        self.assertEqual(len(callbacks), 1)


if __name__ == "__main__":
    unittest.main()
