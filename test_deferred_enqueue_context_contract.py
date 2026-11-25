"""Contrato do contexto no enqueue: tipo inválido não entra e não conta descarte.

Na base medida, a checagem de capacidade incrementa discarded antes de
copiar o contexto. Um não-dict cujo deepcopy devolve dict é gravado.
"""
import unittest

from hypothesis import given, settings, strategies as st

from core.deferred_task_queue import DeferredTaskQueue


class _CopiesToDict:
    def __init__(self):
        self.copied = False

    def __deepcopy__(self, memo):
        self.copied = True
        return {}


class _CopiesToList(dict):
    def __deepcopy__(self, memo):
        return ["nope"]


class _CopyFails(dict):
    def __deepcopy__(self, memo):
        raise RuntimeError("boom")


class _DictSubclass(dict):
    pass


def _stats(queue):
    return queue.statistics()


class DeferredEnqueueContextContractTests(unittest.TestCase):
    def test_non_dict_context_on_a_full_queue_does_not_count_as_discard(self):
        invalid = (
            True,
            False,
            ["nope"],
            "ctx",
            1,
            (1,),
            b"ctx",
            {1},
            _CopiesToDict(),
            _CopiesToList(),
        )
        for context in invalid:
            queue = DeferredTaskQueue(max_size=1)
            queue.enqueue("keep", {"ok": True}, priority=1, reason="")
            before = _stats(queue)
            with self.subTest(context=type(context).__name__):
                with self.assertRaisesRegex(ValueError, "context must be a dictionary"):
                    queue.enqueue("extra", context, priority=2, reason="wait")
                self.assertEqual(_stats(queue), before)
                self.assertEqual(queue.export_snapshot()["tasks"][0]["prompt"], "keep")

    def test_non_dict_is_rejected_before_deepcopy_normalizes_it(self):
        queue = DeferredTaskQueue()
        context = _CopiesToDict()
        before = _stats(queue)
        with self.assertRaisesRegex(ValueError, "context must be a dictionary"):
            queue.enqueue("pending", context)
        self.assertFalse(context.copied)
        self.assertEqual(_stats(queue), before)
        self.assertEqual(queue.export_snapshot()["tasks"], [])

    def test_deepcopy_that_returns_a_non_dict_does_not_enter(self):
        for max_size in (1, 4):
            queue = DeferredTaskQueue(max_size=max_size)
            if max_size == 1:
                queue.enqueue("keep", {"ok": True})
            before = _stats(queue)
            with self.subTest(max_size=max_size):
                with self.assertRaisesRegex(ValueError, "context must be a dictionary"):
                    queue.enqueue("extra", _CopiesToList())
                self.assertEqual(_stats(queue), before)

    def test_deepcopy_failure_on_a_full_queue_does_not_count_as_discard(self):
        queue = DeferredTaskQueue(max_size=1)
        queue.enqueue("keep", {"ok": True})
        before = _stats(queue)
        with self.assertRaisesRegex(ValueError, "context must be deep-copyable") as caught:
            queue.enqueue("extra", _CopyFails())
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(_stats(queue), before)
        self.assertEqual(queue.export_snapshot()["tasks"][0]["prompt"], "keep")

    def test_none_and_dict_contexts_stay_detached(self):
        queue = DeferredTaskQueue()
        missing = queue.enqueue("missing")
        empty = queue.enqueue("empty", {}, reason="")
        nested = {"trace": {"values": ["before"]}}
        task = queue.enqueue("nested", nested, priority=-1, reason="wait")
        nested["trace"]["values"].append("after")
        missing.context["later"] = 1
        self.assertEqual(empty.context, {})
        self.assertEqual(task.context, {"trace": {"values": ["before"]}})
        self.assertEqual(task.last_reason, "wait")
        self.assertEqual(_stats(queue)["enqueued"], 3)
        self.assertEqual(_stats(queue)["discarded"], 0)

    def test_dict_subclass_and_full_valid_enqueue_keep_current_behavior(self):
        queue = DeferredTaskQueue(max_size=2)
        context = _DictSubclass(kind="sub")
        task = queue.enqueue("sub", context, priority=0, reason="")
        context["kind"] = "mutated"
        self.assertEqual(task.context, {"kind": "sub"})
        self.assertIsInstance(task.context, dict)
        queue.enqueue("second", {"ok": True})
        before = _stats(queue)
        with self.assertRaisesRegex(OverflowError, "cheia"):
            queue.enqueue("third", {"ok": True}, priority=3, reason="wait")
        self.assertEqual(_stats(queue)["discarded"], before["discarded"] + 1)
        self.assertEqual(_stats(queue)["enqueued"], before["enqueued"])
        self.assertEqual(_stats(queue)["depth"], 2)

    def test_nonfinite_context_values_still_enter_and_fail_json_export(self):
        queue = DeferredTaskQueue()
        task = queue.enqueue("nonfinite", {"value": float("inf")})
        self.assertEqual(task.context["value"], float("inf"))
        self.assertEqual(queue.export_snapshot()["tasks"][0]["context"]["value"], float("inf"))
        with self.assertRaises(ValueError):
            queue.export_snapshot(as_json=True)

    @settings(max_examples=25, deadline=None)
    @given(
        st.dictionaries(
            st.text(max_size=8),
            st.one_of(
                st.none(),
                st.booleans(),
                st.integers(min_value=-5, max_value=5),
                st.text(max_size=8),
            ),
            max_size=4,
        )
    )
    def test_json_scalars_in_a_dict_round_trip_without_coercion(self, context):
        queue = DeferredTaskQueue(max_size=1)
        task = queue.enqueue("pending", context, priority=1, reason="")
        self.assertEqual(task.context, context)
        self.assertIs(type(task.context), dict)
        self.assertEqual(queue.export_snapshot()["tasks"][0]["context"], context)


if __name__ == "__main__":
    unittest.main()
