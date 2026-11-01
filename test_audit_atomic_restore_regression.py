"""Sondas independentes de atomicidade e rejeição de eventos inesperados."""
import copy
import unittest
from unittest.mock import patch

from core.temporal_facts import TemporalFactStore
from test_snapshot_depth_atomicity_contract import ledger, nested_dict, reseal


def state(store):
    return (store.export_snapshot(as_json=True), store._seq, store._last_recorded,
            copy.deepcopy(store._facts), copy.deepcopy(store._index))


def nested_dict_depth(value):
    """Conta iterativamente o formato usado por nested_dict, sem recursão."""
    depth = 0
    while type(value) is dict:
        depth += 1
        if not value:
            return depth
        if tuple(value) != ("k",):
            return -1
        value = value["k"]
    return -1


class IndependentAtomicityTests(unittest.TestCase):
    def test_apply_failure_after_projection_change_restores_all_state(self):
        store = ledger()
        before = state(store)
        original = store._apply

        def fail(event):
            original(event)
            if event["kind"] == "assert":
                raise RuntimeError("after projection mutation")

        with patch.object(store, "_apply", side_effect=fail):
            with self.assertRaises(RuntimeError):
                store.assert_fact("ana", "lives_in", "B", valid_from=10.)
        self.assertEqual(state(store), before)

    def test_return_copy_failure_restores_timestamp_sequence_and_events(self):
        store = ledger()
        before = state(store)
        store._clock = lambda: 200.
        original = copy.deepcopy

        def fail(value, *args, **kwargs):
            if getattr(value, "fact_id", None) == "tf-000002":
                raise RuntimeError("return copy")
            return original(value, *args, **kwargs)

        with patch("core.temporal_facts.copy.deepcopy", side_effect=fail):
            with self.assertRaises(RuntimeError):
                store.assert_fact("ana", "lives_in", "B", valid_from=10.)
        self.assertEqual(state(store), before)
        self.assertEqual(store.assert_fact("ana", "lives_in", "B", valid_from=10.).fact_id,
                         "tf-000002")

    def test_corroboration_failure_after_append_rolls_back(self):
        store = ledger()
        before = state(store)
        original = store._emit

        def fail(event):
            original(event)
            raise RuntimeError("after append")

        with patch.object(store, "_emit", side_effect=fail):
            with self.assertRaises(RuntimeError):
                store.assert_fact("ana", "lives_in", "A", valid_from=0., source="second")
        self.assertEqual(state(store), before)

    def test_base_exception_rolls_back_and_store_remains_usable(self):
        store = ledger()
        before = state(store)
        original = store._emit

        def fail(event):
            original(event)
            raise KeyboardInterrupt()

        with patch.object(store, "_emit", side_effect=fail):
            with self.assertRaises(KeyboardInterrupt):
                store.retract("ana", "lives_in", valid_to=10.)
        self.assertEqual(state(store), before)
        self.assertEqual(len(store.retract("ana", "lives_in", valid_to=10.)), 1)


class IndependentBoundaryTests(unittest.TestCase):
    def test_runtime_depth_failure_does_not_leave_a_partial_ledger(self):
        for levels in (100, 450, 490, 600, 900):
            with self.subTest(levels=levels):
                store = ledger()
                before = state(store)
                before_head = store.head_hash
                before_seq = store._seq
                before_event_count = len(store._events)
                metadata = nested_dict(levels)
                try:
                    fact = store.assert_fact("ana", "lives_in", "B", valid_from=10.,
                                             metadata=metadata)
                except (ValueError, RecursionError):
                    self.assertEqual(state(store), before)
                else:
                    self.assertEqual(fact.fact_id, f"tf-{before_seq + 1:06d}")
                    self.assertNotEqual(store.head_hash, before_head)
                    self.assertEqual(store._seq, before_seq + 1)
                    self.assertEqual(len(store._events), before_event_count + 2)
                    self.assertEqual(store._events[-1]["kind"], "assert")
                    self.assertEqual(store._index[("ana", "lives_in")],
                                     ["tf-000001", fact.fact_id])
                    self.assertEqual(nested_dict_depth(fact.metadata), levels)
                    self.assertEqual(nested_dict_depth(
                        store._events[-1]["fact"]["metadata"]), levels)
                    self.assertEqual(nested_dict_depth(
                        store._facts[fact.fact_id].metadata), levels)
                    self.assertEqual(store.check_invariants(), [])

    def test_restore_copy_recursion_is_reported_as_value_error(self):
        snapshot = ledger().export_snapshot()
        original = copy.deepcopy

        def fail(value, *args, **kwargs):
            if isinstance(value, dict) and value.get("kind") == "assert":
                raise RecursionError("event copy")
            return original(value, *args, **kwargs)

        with patch("core.temporal_facts.copy.deepcopy", side_effect=fail):
            with self.assertRaises(ValueError):
                TemporalFactStore.from_snapshot(snapshot)

    def test_restore_replay_recursion_is_reported_as_value_error(self):
        snapshot = ledger().export_snapshot()
        with patch.object(TemporalFactStore, "_apply", side_effect=RecursionError("replay")):
            with self.assertRaises(ValueError):
                TemporalFactStore.from_snapshot(snapshot)

    def test_unknown_fields_rejected_for_every_event_kind(self):
        store = ledger()
        store.assert_fact("ana", "lives_in", "B", valid_from=10.)
        store.assert_fact("ana", "lives_in", "B", valid_from=10., source="second")
        for index in range(4):
            snapshot = store.export_snapshot()
            snapshot["events"][index]["extension"] = "unexpected"
            with self.subTest(kind=snapshot["events"][index]["kind"]):
                with self.assertRaises(ValueError):
                    TemporalFactStore.from_snapshot(reseal(snapshot))

    def test_cycles_rejected_without_state_change(self):
        store = ledger()
        before = state(store)
        cycle = {}
        cycle["cycle"] = cycle
        with self.assertRaises(ValueError):
            store.assert_fact("ana", "lives_in", "B", metadata=cycle)
        self.assertEqual(state(store), before)


if __name__ == "__main__":
    unittest.main()
