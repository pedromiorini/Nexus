import unittest

from core.deferred_task_queue import DeferredTaskQueue


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


if __name__ == "__main__":
    unittest.main()
