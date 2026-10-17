import json
import unittest
from unittest.mock import patch

import core.deferred_task_queue as deferred_queue_module

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

    def test_snapshot_cannot_raise_task_attempt_limit_and_rejects_atomically(self):
        source = DeferredTaskQueue(max_attempts=1, max_size=2)
        source.enqueue("imported")
        payload = source.export_snapshot()
        payload["tasks"][0]["max_attempts"] = 3

        target = DeferredTaskQueue(max_attempts=1, max_size=2)
        existing = target.enqueue("keep")
        before = target.statistics().copy()

        with self.assertRaises(ValueError):
            target.restore_snapshot(payload, replace=True)

        self.assertEqual(target.statistics(), before)
        self.assertEqual(target._tasks[existing.task_id].prompt, "keep")
        self.assertEqual(target.statistics()["depth"], 1)

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

    def test_restore_statistics_semantics_follow_replace_flag(self):
        source = DeferredTaskQueue(max_attempts=3, max_size=6)
        source.enqueue("recover")
        completed = source.enqueue("already completed")
        source.complete(completed.task_id)
        snapshot = source.export_snapshot()
        source_counters = dict(snapshot["statistics"])

        merged = DeferredTaskQueue(max_attempts=3, max_size=6)
        merged.enqueue("keep")
        destination_counters = {
            key: merged.statistics()[key]
            for key in ("enqueued", "retried", "completed", "discarded")
        }
        merged.restore_snapshot(snapshot, replace=False)
        self.assertEqual(
            {key: merged.statistics()[key] for key in destination_counters},
            destination_counters,
        )
        self.assertEqual(merged.statistics()["depth"], 2)

        replaced = DeferredTaskQueue(max_attempts=3, max_size=6)
        replaced.enqueue("stale")
        replaced.restore_snapshot(snapshot, replace=True)
        self.assertEqual(
            {key: replaced.statistics()[key] for key in source_counters},
            source_counters,
        )
        self.assertEqual(replaced.statistics()["depth"], 1)

    def test_statistics_keys_are_exact_and_extra_counters_are_rejected(self):
        source = DeferredTaskQueue(max_attempts=2, max_size=4)
        source.enqueue("imported")
        snapshot = source.export_snapshot()
        snapshot["statistics"]["uncontracted"] = 17

        for replace in (False, True):
            target = DeferredTaskQueue(max_attempts=2, max_size=4)
            kept = target.enqueue("keep")
            before = target.statistics().copy()
            with self.subTest(replace=replace), self.assertRaises(ValueError):
                target.restore_snapshot(snapshot, replace=replace)
            self.assertEqual(target.statistics(), before)
            self.assertEqual(target._tasks[kept.task_id].prompt, "keep")

    def test_task_at_max_attempts_cannot_be_restored_as_pending(self):
        source = DeferredTaskQueue(max_attempts=2, max_size=4)
        source.enqueue("exhausted")
        snapshot = source.export_snapshot()
        snapshot["tasks"][0]["attempts"] = snapshot["tasks"][0]["max_attempts"]

        target = DeferredTaskQueue(max_attempts=2, max_size=4)
        kept = target.enqueue("keep")
        before = target.statistics().copy()
        with self.assertRaises(ValueError):
            target.restore_snapshot(snapshot, replace=True)
        self.assertEqual(target.statistics(), before)
        self.assertEqual(target._tasks[kept.task_id].prompt, "keep")

    def test_contexts_reject_non_json_trees_on_export_and_restore(self):
        dict_subclass = type("DictSubclass", (dict,), {})
        list_subclass = type("ListSubclass", (list,), {})
        invalid_contexts = [
            {"tuple": (1, 2)},
            {1: "non-string key"},
            {"nested": {1: "numeric", "1": "text"}},
            {"nested": {"value": float("nan")}},
            {"nested": {"value": float("inf")}},
            {"set": {1, 2}},
            {"subclass": dict_subclass({"value": 1})},
            {"subclass": list_subclass([1])},
        ]

        for context in invalid_contexts:
            source = DeferredTaskQueue(max_attempts=2, max_size=4)
            task = source.enqueue("source")
            task.context = context
            with self.subTest(stage="export-object", context=context):
                with self.assertRaises(ValueError):
                    source.export_snapshot()
            with self.subTest(stage="export-json", context=context):
                with self.assertRaises(ValueError):
                    source.export_snapshot(as_json=True)

            payload = DeferredTaskQueue(max_attempts=2, max_size=4).export_snapshot()
            payload["tasks"] = [{
                "task_id": "incoming",
                "prompt": "incoming",
                "context": context,
                "priority": 0,
                "created_at": 1.0,
                "attempts": 0,
                "max_attempts": 2,
                "last_reason": "",
            }]
            target = DeferredTaskQueue(max_attempts=2, max_size=4)
            kept = target.enqueue("keep")
            before = target.statistics().copy()
            with self.subTest(stage="restore", context=context):
                with self.assertRaises(ValueError):
                    target.restore_snapshot(payload, replace=True)
                self.assertEqual(target.statistics(), before)
                self.assertEqual(target._tasks[kept.task_id].prompt, "keep")

    def test_contexts_are_validated_before_custom_deepcopy_can_run(self):
        calls = []

        class CustomCopy:
            def __deepcopy__(self, memo):
                calls.append("called")
                return "silently converted"

        source = DeferredTaskQueue(max_attempts=2, max_size=4)
        task = source.enqueue("source")
        task.context = {"nested": [CustomCopy()]}
        with self.subTest(stage="export"):
            with self.assertRaises(ValueError):
                source.export_snapshot()
        self.assertEqual(calls, [])

        payload = DeferredTaskQueue(max_attempts=2, max_size=4).export_snapshot()
        payload["tasks"] = [{
            "task_id": "incoming",
            "prompt": "incoming",
            "context": {"nested": [CustomCopy()]},
            "priority": 0,
            "created_at": 1.0,
            "attempts": 0,
            "max_attempts": 2,
            "last_reason": "",
        }]
        target = DeferredTaskQueue(max_attempts=2, max_size=4)
        kept = target.enqueue("keep")
        before = target.statistics().copy()
        with self.subTest(stage="restore"):
            with self.assertRaises(ValueError):
                target.restore_snapshot(payload, replace=True)
        self.assertEqual(calls, [])
        self.assertEqual(target.statistics(), before)
        self.assertEqual(target._tasks[kept.task_id].prompt, "keep")

    def test_restore_uses_captured_task_id_when_input_changes_during_context_copy(self):
        source = DeferredTaskQueue(max_attempts=2, max_size=4)
        incoming = source.enqueue("incoming", {"items": [1, 2, 3]})
        payload = source.export_snapshot()
        raw_task = payload["tasks"][0]

        target = DeferredTaskQueue(max_attempts=2, max_size=4)
        existing = target.enqueue("existing")
        original_copy = DeferredTaskQueue._copy_strict_json_context

        def change_task_id_after_context_copy(context):
            copied = original_copy(context)
            raw_task["task_id"] = existing.task_id
            return copied

        with patch.object(
            DeferredTaskQueue,
            "_copy_strict_json_context",
            side_effect=change_task_id_after_context_copy,
        ):
            target.restore_snapshot(payload, replace=False)

        self.assertIs(target._tasks[existing.task_id], existing)
        self.assertEqual(target._tasks[incoming.task_id].prompt, "incoming")
        self.assertEqual(target.statistics()["depth"], 2)

    def test_replace_uses_captured_statistics_if_input_changes_during_context_copy(self):
        source = DeferredTaskQueue(max_attempts=2, max_size=4)
        source.enqueue("incoming", {"items": [1]})
        payload = source.export_snapshot()
        target = DeferredTaskQueue(max_attempts=2, max_size=4)
        target.enqueue("stale")
        original_copy = DeferredTaskQueue._copy_strict_json_context

        def change_statistics_after_context_copy(context):
            copied = original_copy(context)
            payload["statistics"]["enqueued"] = True
            return copied

        with patch.object(
            DeferredTaskQueue,
            "_copy_strict_json_context",
            side_effect=change_statistics_after_context_copy,
        ):
            target.restore_snapshot(payload, replace=True)

        self.assertEqual(target.statistics()["enqueued"], 1)
        self.assertIs(type(target.statistics()["enqueued"]), int)
        self.assertEqual(target.statistics()["depth"], 1)

    def test_restore_staging_failure_is_atomic_for_both_modes(self):
        source = DeferredTaskQueue(max_attempts=2, max_size=6)
        source.enqueue("incoming-1")
        source.enqueue("incoming-2")
        snapshot = source.export_snapshot()

        for replace in (False, True):
            with self.subTest(replace=replace):
                target = DeferredTaskQueue(max_attempts=2, max_size=6)
                kept = target.enqueue("keep", {"nested": {"value": "same"}})
                before_stats = target.statistics().copy()
                before_heap = target._heap
                before_tasks = target._tasks
                before_in_flight = target._in_flight
                before_counters = target._stats
                before_task_map = dict(target._tasks)
                push_count = 0
                real_heappush = deferred_queue_module.heapq.heappush

                def fail_on_second_push(heap, item):
                    nonlocal push_count
                    push_count += 1
                    if push_count == 2:
                        raise MemoryError("injected staging failure")
                    return real_heappush(heap, item)

                with patch(
                    "core.deferred_task_queue.heapq.heappush",
                    side_effect=fail_on_second_push,
                ):
                    with self.assertRaises(MemoryError):
                        target.restore_snapshot(snapshot, replace=replace)

                self.assertEqual(push_count, 2)
                self.assertIs(target._heap, before_heap)
                self.assertIs(target._tasks, before_tasks)
                self.assertIs(target._in_flight, before_in_flight)
                self.assertIs(target._stats, before_counters)
                self.assertEqual(target._tasks, before_task_map)
                self.assertIs(target._tasks[kept.task_id], kept)
                self.assertEqual(target.statistics(), before_stats)

    def test_replace_is_rejected_without_mutation_while_task_is_in_flight(self):
        source = DeferredTaskQueue()
        source.enqueue("replacement")
        snapshot = source.export_snapshot()
        target = DeferredTaskQueue()
        active = target.enqueue("active")
        self.assertEqual(target.recheck_and_pop(False).task_id, active.task_id)
        before = target.statistics().copy()

        with self.assertRaises(ValueError):
            target.restore_snapshot(snapshot, replace=True)

        self.assertEqual(target.statistics(), before)
        self.assertTrue(target.complete(active.task_id))

    def test_export_and_restore_contexts_are_detached(self):
        source = DeferredTaskQueue()
        task = source.enqueue("isolated", {"nested": {"value": "original"}})
        exported = source.export_snapshot()
        exported["tasks"][0]["context"]["nested"]["value"] = "export mutation"
        self.assertEqual(task.context["nested"]["value"], "original")

        payload = source.export_snapshot()
        target = DeferredTaskQueue()
        target.restore_snapshot(payload)
        payload["tasks"][0]["context"]["nested"]["value"] = "input mutation"
        restored = target.recheck_and_pop(False)
        self.assertEqual(restored.context["nested"]["value"], "original")
        restored.context["nested"]["value"] = "restored mutation"
        self.assertEqual(payload["tasks"][0]["context"]["nested"]["value"], "input mutation")

    def test_nonfinite_values_are_rejected_atomically_with_replace(self):
        queue = DeferredTaskQueue(max_attempts=2, max_size=4)
        kept = queue.enqueue("keep", {"nested": {"value": "unchanged"}})
        before = queue.statistics().copy()
        invalid_snapshots = []

        for field, value in (
            ("generated_at", float("nan")),
            ("generated_at", float("inf")),
            ("generated_at", float("-inf")),
            ("created_at", float("nan")),
            ("created_at", float("inf")),
            ("created_at", float("-inf")),
            ("context", {"bad": float("nan")}),
        ):
            payload = queue.export_snapshot()
            if field == "context":
                payload["tasks"][0][field] = value
            elif field == "created_at":
                payload["tasks"][0][field] = value
            else:
                payload[field] = value
            invalid_snapshots.append(payload)

        for payload in invalid_snapshots:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                queue.restore_snapshot(payload, replace=True)
            self.assertEqual(queue.statistics(), before)
            self.assertEqual(queue._tasks[kept.task_id].context["nested"]["value"], "unchanged")

        invalid_context = DeferredTaskQueue()
        invalid_context.enqueue("nonfinite", {"value": float("inf")})
        with self.assertRaises(ValueError):
            invalid_context.export_snapshot(as_json=True)

    def test_json_text_rejects_nonstandard_constants_even_in_extensions(self):
        source = DeferredTaskQueue(max_attempts=2, max_size=4)
        source.enqueue("incoming")
        payload = source.export_snapshot()
        payload["extension"] = float("nan")
        encoded = json.dumps(payload)

        target = DeferredTaskQueue(max_attempts=2, max_size=4)
        kept = target.enqueue("keep")
        before = target.statistics().copy()
        with self.assertRaises(ValueError):
            target.restore_snapshot(encoded, replace=True)
        self.assertEqual(target.statistics(), before)
        self.assertEqual(target._tasks[kept.task_id].prompt, "keep")

    def test_snapshot_preserves_builtin_subclass_compatibility_outside_context(self):
        class DictSubclass(dict):
            pass

        class ListSubclass(list):
            pass

        class TextSubclass(str):
            pass

        class IntegerSubclass(int):
            pass

        source = DeferredTaskQueue(max_attempts=2, max_size=4)
        source.enqueue("original")
        payload = DictSubclass(source.export_snapshot())
        payload["schema"] = TextSubclass(payload["schema"])
        payload["generated_at"] = IntegerSubclass(1)
        payload["statistics"] = DictSubclass(payload["statistics"])
        payload[object()] = "ignored root extension"

        task = DictSubclass(payload["tasks"][0])
        task.update({
            "task_id": TextSubclass("legacy-id"),
            "prompt": TextSubclass("legacy prompt"),
            "priority": IntegerSubclass(3),
            "created_at": IntegerSubclass(1),
            "attempts": IntegerSubclass(0),
            "last_reason": TextSubclass("legacy reason"),
        })
        task[object()] = "ignored task extension"
        payload["tasks"] = ListSubclass([task])

        target = DeferredTaskQueue(max_attempts=2, max_size=4)
        result = target.restore_snapshot(payload, replace=True)
        restored = target._tasks["legacy-id"]

        self.assertEqual(result["restored"], 1)
        self.assertIsInstance(restored.task_id, TextSubclass)
        self.assertIsInstance(restored.prompt, TextSubclass)
        self.assertIsInstance(restored.priority, IntegerSubclass)
        self.assertIsInstance(restored.attempts, IntegerSubclass)
        self.assertEqual(restored.context, {})
        self.assertEqual(restored.last_reason, "legacy reason")

    def test_boolean_numbers_are_rejected_by_snapshot_contract(self):
        source = DeferredTaskQueue(max_attempts=1, max_size=1)
        source.enqueue("pending")
        target = DeferredTaskQueue(max_attempts=1, max_size=1)
        target.enqueue("keep")
        before = target.statistics().copy()
        cases = [
            ("max_attempts", True),
            ("max_size", True),
            ("statistics", {"enqueued": True}),
            ("task_max_attempts", True),
        ]
        for case, value in cases:
            payload = source.export_snapshot()
            if case == "statistics":
                payload["statistics"].update(value)
            elif case == "task_max_attempts":
                payload["tasks"][0]["max_attempts"] = value
            else:
                payload[case] = value
            with self.subTest(case=case), self.assertRaises(ValueError):
                target.restore_snapshot(payload, replace=True)
            self.assertEqual(target.statistics(), before)

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
            {"created_at": True}, {"attempts": -1}, {"attempts": True}, {"max_attempts": 0},
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
