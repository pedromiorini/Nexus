import unittest

from hypothesis import given, settings, strategies as st

from core.deferred_task_queue import DeferredTaskQueue


class DeferredTaskPropertyTests(unittest.TestCase):
    @settings(max_examples=40, deadline=None)
    @given(st.lists(st.integers(min_value=-10, max_value=10), min_size=1, max_size=12))
    def test_priority_order_is_non_increasing(self, priorities):
        queue = DeferredTaskQueue(max_attempts=2, max_size=max(16, len(priorities)))
        for index, priority in enumerate(priorities):
            queue.enqueue(f"prompt-{index}", context={"index": index}, priority=priority)
        observed = []
        while True:
            task = queue.recheck_and_pop(False)
            if task is None:
                break
            observed.append(task.priority)
        self.assertEqual(observed, sorted(priorities, reverse=True))

    @settings(max_examples=30, deadline=None)
    @given(st.integers(min_value=1, max_value=12))
    def test_critical_pressure_preserves_depth(self, size):
        queue = DeferredTaskQueue(max_attempts=2, max_size=size)
        for index in range(size):
            queue.enqueue(f"prompt-{index}")
        before = queue.statistics()
        result = queue.consume(lambda task: True, max_batch=size, pressure_critical=True)
        after = queue.statistics()
        self.assertTrue(result["blocked"])
        self.assertEqual(before["depth"], after["depth"])
        self.assertEqual(before["enqueued"], after["enqueued"])

    @settings(max_examples=25, deadline=None)
    @given(st.integers(min_value=1, max_value=4))
    def test_retry_policy_never_exceeds_max_attempts(self, max_attempts):
        queue = DeferredTaskQueue(max_attempts=max_attempts, max_size=2)
        task = queue.enqueue("unstable", context={"source": "property"})
        for _ in range(max_attempts):
            result = queue.consume(lambda current: False, max_batch=1)
            self.assertEqual(result["processed"], 1)
            self.assertLessEqual(task.attempts, max_attempts)
        self.assertEqual(queue.statistics()["depth"], 0)
        self.assertEqual(queue.statistics()["discarded"], 1)

    @settings(max_examples=30, deadline=None)
    @given(st.dictionaries(st.text(min_size=1, max_size=8), st.integers(), max_size=5))
    def test_context_is_preserved(self, context):
        queue = DeferredTaskQueue(max_attempts=1, max_size=2)
        task = queue.enqueue("context", context=context)
        popped = queue.recheck_and_pop(False)
        self.assertIsNotNone(popped)
        self.assertEqual(popped.task_id, task.task_id)
        self.assertEqual(popped.context, context)


if __name__ == "__main__":
    unittest.main()
