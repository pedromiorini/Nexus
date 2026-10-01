import json
import unittest

from core.deferred_task_queue import DeferredTaskQueue


class DeferredTaskSnapshotTests(unittest.TestCase):
    def test_round_trip_preserves_order_and_task_state(self):
        source = DeferredTaskQueue(max_attempts=4, max_size=8)
        first = source.enqueue("low", {"trace": "l"}, priority=1, reason="wait")
        second = source.enqueue("high", {"trace": "h"}, priority=9)
        popped = source.recheck_and_pop(False)
        self.assertEqual(popped.task_id, second.task_id)
        source.requeue(popped, "retry")

        restored = DeferredTaskQueue(max_attempts=4, max_size=8)
        result = restored.restore_snapshot(source.export_snapshot(as_json=True))
        self.assertEqual(result["restored"], 2)
        self.assertEqual(restored.recheck_and_pop(False).task_id, second.task_id)
        recovered = restored.recheck_and_pop(False)
        self.assertEqual(recovered.task_id, first.task_id)
        self.assertEqual(recovered.context, {"trace": "l"})

    def test_invalid_snapshot_does_not_mutate_queue(self):
        queue = DeferredTaskQueue(max_size=4)
        existing = queue.enqueue("existing")
        before = queue.statistics().copy()
        payload = queue.export_snapshot()
        payload["tasks"][0]["attempts"] = payload["tasks"][0]["max_attempts"] + 1
        with self.assertRaises(ValueError):
            queue.restore_snapshot(payload)
        self.assertEqual(queue.statistics(), before)
        self.assertEqual(queue.recheck_and_pop(False).task_id, existing.task_id)

    def test_schema_duplicate_and_capacity_are_rejected(self):
        queue = DeferredTaskQueue(max_attempts=2, max_size=2)
        payload = queue.export_snapshot()
        payload["tasks"] = [{
            "task_id": "same", "prompt": "a", "context": {}, "priority": 0,
            "created_at": 1.0, "attempts": 0, "max_attempts": 2, "last_reason": "",
        }, {
            "task_id": "same", "prompt": "b", "context": {}, "priority": 0,
            "created_at": 2.0, "attempts": 0, "max_attempts": 2, "last_reason": "",
        }]
        with self.assertRaises(ValueError):
            queue.restore_snapshot(payload)
        payload["tasks"][1]["task_id"] = "other"
        queue.enqueue("occupied")
        with self.assertRaises(ValueError):
            queue.restore_snapshot(payload)
        payload["schema"] = "nexus.deferred_task_queue.v9"
        with self.assertRaises(ValueError):
            DeferredTaskQueue(max_size=2).restore_snapshot(payload)

    def test_restore_never_executes_processor_or_callbacks(self):
        source = DeferredTaskQueue()
        source.enqueue("safe")
        target = DeferredTaskQueue()
        called = []
        result = target.restore_snapshot(source.export_snapshot())
        self.assertEqual(result["restored"], 1)
        self.assertEqual(called, [])
        self.assertEqual(target.statistics()["depth"], 1)

    def test_replace_restores_snapshot_statistics(self):
        source = DeferredTaskQueue()
        source.enqueue("recover")
        snapshot = source.export_snapshot()
        target = DeferredTaskQueue()
        target.enqueue("stale")
        result = target.restore_snapshot(json.dumps(snapshot), replace=True)
        self.assertTrue(result["replaced"])
        self.assertEqual(target.statistics()["depth"], 1)
        self.assertEqual(target.recheck_and_pop(False).prompt, "recover")


    def test_snapshot_validation_rejects_malformed_envelopes(self):
        queue = DeferredTaskQueue(max_attempts=2, max_size=2)
        base = queue.export_snapshot()
        invalids = [
            {},
            {"generated_at": "now"},
            {"generated_at": 1.0, "max_attempts": 9, "max_size": 2},
            {"generated_at": 1.0, "max_attempts": 2, "max_size": 2, "statistics": {}, "tasks": []},
            {"generated_at": 1.0, "max_attempts": 2, "max_size": 2, "statistics": {"enqueued": 0, "retried": 0, "completed": 0, "discarded": 0}, "tasks": "nope"},
        ]
        for changes in invalids:
            payload = {} if not changes else dict(base)
            payload.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                queue.restore_snapshot(payload)
        with self.assertRaises(ValueError):
            queue.restore_snapshot("not-json")
        with self.assertRaises(ValueError):
            queue.restore_snapshot([])
        malformed_task = dict(base)
        malformed_task["tasks"] = [{}]
        with self.assertRaises(ValueError):
            queue.restore_snapshot(malformed_task)

    def test_snapshot_validation_rejects_invalid_task_fields(self):
        queue = DeferredTaskQueue(max_attempts=2, max_size=2)
        base_task = {
            "task_id": "id", "prompt": "p", "context": {}, "priority": 0,
            "created_at": 1.0, "attempts": 0, "max_attempts": 2, "last_reason": "",
        }
        cases = [
            {"task_id": ""}, {"prompt": 3}, {"context": []}, {"priority": True},
            {"created_at": True}, {"attempts": -1}, {"max_attempts": 0},
            {"attempts": 2, "max_attempts": 1}, {"last_reason": None},
            {"context": {"bad": object()}},
        ]
        for change in cases:
            payload = queue.export_snapshot()
            task = dict(base_task)
            task.update(change)
            payload["tasks"] = [task]
            with self.subTest(change=change), self.assertRaises(ValueError):
                queue.restore_snapshot(payload)

    def test_restore_capacity_and_discard_contracts(self):
        queue = DeferredTaskQueue(max_attempts=2, max_size=1)
        source = DeferredTaskQueue(max_attempts=2, max_size=1)
        source.enqueue("one")
        oversized = source.export_snapshot()
        oversized["tasks"].append(dict(oversized["tasks"][0], task_id="two"))
        with self.assertRaises(ValueError):
            queue.restore_snapshot(oversized, replace=True)
        queue.enqueue("occupied")
        with self.assertRaises(ValueError):
            queue.restore_snapshot(source.export_snapshot())
        discarded = DeferredTaskQueue()
        task = discarded.enqueue("drop")
        discarded.discard(task.task_id)
        self.assertEqual(discarded.statistics()["discarded"], 1)

    def test_processor_exception_is_recorded_and_retried(self):
        queue = DeferredTaskQueue(max_attempts=2)
        queue.enqueue("boom")
        failures = []
        result = queue.consume(lambda task: (_ for _ in ()).throw(RuntimeError("bad")), on_failure=failures.append)
        self.assertEqual(result["retried"], 1)
        self.assertEqual(failures[0].last_reason, "RuntimeError: bad")

    def test_consume_breaks_if_pop_returns_none(self):
        queue = DeferredTaskQueue()
        queue.enqueue("one")
        original = queue.recheck_and_pop
        queue.recheck_and_pop = lambda pressure: None
        try:
            self.assertEqual(queue.consume(lambda task: True)["processed"], 0)
        finally:
            queue.recheck_and_pop = original


if __name__ == "__main__":
    unittest.main()
