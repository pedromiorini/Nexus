"""F-04: capacidade real do JSON, sem teto fixo e sem publicar restore parcial."""
import json
import unittest
from unittest.mock import patch

from core.deferred_task_queue import DeferredTaskQueue


def nested_context(depth, kind="dict"):
    value = "leaf"
    for level in range(depth - 1):
        value = {"k": value} if kind == "dict" or (kind == "mixed" and level % 2) else [value]
    return {"k": value}


def incoming_snapshot(context):
    source = DeferredTaskQueue(max_size=8)
    source.enqueue("first", {"safe": True})
    source.enqueue("last")
    snapshot = source.export_snapshot()
    snapshot["tasks"][-1]["context"] = context
    return snapshot


class QueueJsonPreflightContract(unittest.TestCase):
    def capture(self, queue):
        return (
            (queue._heap, queue._tasks, queue._in_flight, queue._stats),
            list(queue._heap), dict(queue._tasks), set(queue._in_flight),
            queue.statistics(),
            {key: (task.context, task.attempts, task.last_reason, task.sort_key)
             for key, task in queue._tasks.items()},
        )

    def assert_unchanged(self, queue, before):
        refs, heap, tasks, in_flight, stats, fields = before
        for current, original in zip(
            (queue._heap, queue._tasks, queue._in_flight, queue._stats), refs
        ):
            self.assertIs(current, original)
        self.assertEqual([id(task) for task in queue._heap], [id(task) for task in heap])
        self.assertEqual(set(queue._tasks), set(tasks))
        for key, task in tasks.items():
            self.assertIs(queue._tasks[key], task)
            context, attempts, reason, sort_key = fields[key]
            self.assertIs(task.context, context)
            self.assertEqual((task.attempts, task.last_reason, task.sort_key),
                             (attempts, reason, sort_key))
        self.assertEqual(queue._in_flight, in_flight)
        self.assertEqual(queue.statistics(), stats)

    def test_deep_restore_rejects_before_any_staging_or_mutation(self):
        for kind in ("dict", "list", "mixed"):
            snapshot = incoming_snapshot(nested_context(10000, kind))
            for replace, in_flight in ((True, False), (False, False), (False, True)):
                with self.subTest(kind=kind, replace=replace, in_flight=in_flight):
                    target = DeferredTaskQueue(max_size=8)
                    target.enqueue("keep", {"preserve": [1, 2]})
                    if in_flight:
                        target.recheck_and_pop(False)
                        target.enqueue("pending")
                    before = self.capture(target)
                    with patch("core.deferred_task_queue.heapq.heappush") as push:
                        with self.assertRaisesRegex(ValueError, "capacidade.*JSON"):
                            target.restore_snapshot(snapshot, replace=replace)
                        push.assert_not_called()
                    self.assert_unchanged(target, before)
                    self.assertEqual(target.enqueue("usable").prompt, "usable")

    def test_deep_export_is_controlled_and_does_not_mutate_queue(self):
        for kind in ("dict", "list", "mixed"):
            source = DeferredTaskQueue(max_size=8)
            task = source.enqueue("mutable API remains supported")
            # enqueue/task objects remain mutable; export must defend this boundary too.
            task.context = nested_context(10000, kind)
            before = self.capture(source)
            for as_json in (False, True):
                with self.subTest(kind=kind, as_json=as_json):
                    with self.assertRaisesRegex(ValueError, "capacidade.*JSON"):
                        source.export_snapshot(as_json=as_json)
                    self.assert_unchanged(source, before)

    def test_original_5000_level_case_follows_actual_encoder_capacity(self):
        # CPython 3.11 rejects 5000 here; 3.12 can encode it. Neither gains a ceiling.
        for kind in ("dict", "list", "mixed"):
            snapshot = incoming_snapshot(nested_context(5000, kind))
            try:
                json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
            except RecursionError:
                exportable = False
            else:
                exportable = True
            for replace in (False, True):
                with self.subTest(kind=kind, replace=replace, exportable=exportable):
                    target = DeferredTaskQueue(max_size=8)
                    target.enqueue("keep")
                    before = self.capture(target)
                    if exportable:
                        self.assertEqual(target.restore_snapshot(snapshot, replace=replace)["restored"], 2)
                        payload = target.export_snapshot()
                        encoded = target.export_snapshot(as_json=True)
                        self.assertIn('"last"', encoded)
                        self.assertEqual(len(payload["tasks"]), 2 if replace else 3)
                    else:
                        with self.assertRaisesRegex(ValueError, "capacidade.*JSON"):
                            target.restore_snapshot(snapshot, replace=replace)
                        self.assert_unchanged(target, before)

    def test_exportable_depths_above_64_remain_compatible(self):
        for depth in (65, 100, 300):
            for kind in ("dict", "list", "mixed"):
                context = nested_context(depth, kind)
                snapshot = incoming_snapshot(context)
                for replace in (False, True):
                    with self.subTest(depth=depth, kind=kind, replace=replace):
                        target = DeferredTaskQueue(max_size=8)
                        target.restore_snapshot(snapshot, replace=replace)
                        encoded = target.export_snapshot(as_json=True)
                        decoded = target.export_snapshot()
                        # generated_at is intentionally not part of the stable contract.
                        decoded["generated_at"] = json.loads(encoded)["generated_at"]
                        self.assertEqual(json.dumps(decoded, sort_keys=True, separators=(",", ":"),
                                                    allow_nan=False), encoded)
                        replay = DeferredTaskQueue(max_size=8)
                        replay.restore_snapshot(encoded, replace=True)
                        copied = replay.export_snapshot()["tasks"][-1]["context"]
                        self.assertEqual(json.dumps(copied), json.dumps(context))
                        self.assertIsNot(copied, context)

    def test_encoder_recursion_is_normalized_before_restore_constructs_tasks(self):
        snapshot = incoming_snapshot({"safe": True})
        for replace in (False, True):
            target = DeferredTaskQueue(max_size=8)
            target.enqueue("keep")
            before = self.capture(target)
            with self.subTest(replace=replace), patch(
                "core.deferred_task_queue.json.dumps", side_effect=RecursionError("injected")
            ), patch("core.deferred_task_queue.DeferredTask") as constructor:
                with self.assertRaisesRegex(ValueError, "capacidade.*JSON") as caught:
                    target.restore_snapshot(snapshot, replace=replace)
                self.assertIsInstance(caught.exception.__cause__, RecursionError)
                constructor.assert_not_called()
            self.assert_unchanged(target, before)

    def test_deep_json_text_rejection_preserves_target(self):
        # Construct text iteratively, without needing the recursive encoder.
        text = '{"k":' * 5000 + 'null' + '}' * 5000
        for replace in (False, True):
            target = DeferredTaskQueue(max_size=8)
            target.enqueue("keep")
            before = self.capture(target)
            with self.subTest(replace=replace), self.assertRaises(ValueError):
                target.restore_snapshot(text, replace=replace)
            self.assert_unchanged(target, before)


if __name__ == "__main__":
    unittest.main()
