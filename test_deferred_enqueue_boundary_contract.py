"""Contrato da fronteira de enqueue: tipos inválidos não entram na fila.

O snapshot já recusa bool em priority e não-str em prompt/last_reason.
enqueue, na base medida, ainda grava esses valores e, com a fila cheia,
conta o descarte antes de olhar o tipo.
"""
import math
import unittest

from hypothesis import given, settings, strategies as st

from core.deferred_task_queue import DeferredTaskQueue


class _IndexInt(int):
    """Inteiro real, não bool. O snapshot aceita subclasse de int."""


def _stats(queue):
    return queue.statistics()


class DeferredEnqueueBoundaryContractTests(unittest.TestCase):
    def test_boolean_and_non_integer_priorities_do_not_enter_the_queue(self):
        invalid = (
            True,
            False,
            1.5,
            "1",
            None,
            float("nan"),
            float("inf"),
            float("-inf"),
        )
        for priority in invalid:
            queue = DeferredTaskQueue()
            before = _stats(queue)
            with self.subTest(priority=priority):
                with self.assertRaisesRegex(ValueError, "priority deve ser inteiro"):
                    queue.enqueue("pending", priority=priority)
                self.assertEqual(_stats(queue), before)
                self.assertEqual(queue.export_snapshot()["tasks"], [])

    def test_non_string_prompt_and_reason_do_not_enter_the_queue(self):
        cases = (
            ("prompt", {"prompt": None}),
            ("prompt", {"prompt": b"pending"}),
            ("prompt", {"prompt": 1}),
            ("prompt", {"prompt": True}),
            ("reason", {"reason": None}),
            ("reason", {"reason": 1}),
            ("reason", {"reason": True}),
            ("reason", {"reason": b"wait"}),
        )
        for field, kwargs in cases:
            queue = DeferredTaskQueue()
            before = _stats(queue)
            with self.subTest(field=field, value=kwargs[field]):
                with self.assertRaisesRegex(ValueError, f"{field} deve ser string"):
                    queue.enqueue(**{"prompt": "pending", "priority": 1, **kwargs})
                self.assertEqual(_stats(queue), before)
                self.assertEqual(queue.export_snapshot()["tasks"], [])

    def test_invalid_priority_on_a_full_queue_does_not_count_as_discard(self):
        queue = DeferredTaskQueue(max_size=1)
        queue.enqueue("keep", priority=1)
        before = _stats(queue)
        with self.assertRaisesRegex(ValueError, "priority deve ser inteiro"):
            queue.enqueue("extra", priority=True)
        self.assertEqual(_stats(queue), before)
        self.assertEqual(queue.export_snapshot()["tasks"][0]["prompt"], "keep")

    def test_full_queue_still_discards_a_valid_enqueue(self):
        queue = DeferredTaskQueue(max_size=1)
        queue.enqueue("keep", priority=1)
        with self.assertRaisesRegex(OverflowError, "cheia"):
            queue.enqueue("extra", priority=2, reason="wait")
        stats = _stats(queue)
        self.assertEqual(stats["discarded"], 1)
        self.assertEqual(stats["enqueued"], 1)
        self.assertEqual(stats["depth"], 1)
        self.assertEqual(queue.export_snapshot()["tasks"][0]["prompt"], "keep")

    def test_integer_priority_and_strings_keep_current_order(self):
        queue = DeferredTaskQueue()
        low = queue.enqueue("", priority=-3, reason="")
        high = queue.enqueue("high", priority=_IndexInt(4), reason="wait")
        zero = queue.enqueue("zero", priority=0)
        first = queue.recheck_and_pop(False)
        self.assertEqual(first.task_id, high.task_id)
        self.assertEqual(first.priority, 4)
        self.assertIsInstance(first.priority, int)
        self.assertNotIsInstance(first.priority, bool)
        self.assertEqual(low.prompt, "")
        self.assertEqual(low.last_reason, "")
        self.assertEqual(zero.priority, 0)
        self.assertEqual(_stats(queue)["enqueued"], 3)

    @settings(max_examples=30, deadline=None)
    @given(st.integers(min_value=-20, max_value=20), st.text(max_size=12), st.text(max_size=12))
    def test_integers_and_text_round_trip_without_coercion(self, priority, prompt, reason):
        queue = DeferredTaskQueue(max_size=1)
        task = queue.enqueue(prompt, priority=priority, reason=reason)
        self.assertEqual(task.priority, priority)
        self.assertEqual(task.prompt, prompt)
        self.assertEqual(task.last_reason, reason)
        exported = queue.export_snapshot()["tasks"][0]
        self.assertEqual(exported["priority"], priority)
        self.assertEqual(exported["prompt"], prompt)
        self.assertEqual(exported["last_reason"], reason)
        self.assertTrue(math.isfinite(exported["created_at"]))


if __name__ == "__main__":
    unittest.main()
