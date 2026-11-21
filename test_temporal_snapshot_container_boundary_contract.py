"""Contratos da fronteira de containers externos em TemporalFactStore.from_snapshot."""
import copy
import unittest

from core.temporal_facts import TemporalFactStore


class SnapshotGetTrapDict(dict):
    override_calls = 0

    def get(self, key, default=None):
        type(self).override_calls += 1
        raise RuntimeError("snapshot dict get override must not run")


class EventsIterTrapList(list):
    override_calls = 0

    def __iter__(self):
        type(self).override_calls += 1
        raise RuntimeError("events list iterator override must not run")


class FunctionalPredicatesIterTrapList(list):
    override_calls = 0

    def __iter__(self):
        type(self).override_calls += 1
        raise RuntimeError("functional_predicates list iterator override must not run")


def valid_snapshot():
    store = TemporalFactStore(functional_predicates=("lives_in",), clock=lambda: 100.0)
    store.assert_fact("ana", "lives_in", "Curitiba", valid_from=0.0,
                      metadata={"source": "test"})
    return store.export_snapshot()


class TemporalSnapshotContainerBoundaryContract(unittest.TestCase):
    def assert_restores_without_override(self, payload, trap_type):
        trap_type.override_calls = 0
        restored = TemporalFactStore.from_snapshot(payload, clock=lambda: 200.0)
        self.assertEqual(restored.head_hash, valid_snapshot()["head"])
        self.assertEqual(
            [(fact.subject, fact.predicate, fact.object)
             for fact in restored.history("ana", "lives_in")],
            [("ana", "lives_in", "Curitiba")],
        )
        self.assertEqual(trap_type.override_calls, 0)

    def test_snapshot_dict_subclass_get_override_is_not_called(self):
        payload = SnapshotGetTrapDict(copy.deepcopy(valid_snapshot()))
        self.assert_restores_without_override(payload, SnapshotGetTrapDict)

    def test_events_list_subclass_iterator_is_not_called(self):
        payload = copy.deepcopy(valid_snapshot())
        payload["events"] = EventsIterTrapList(payload["events"])
        self.assert_restores_without_override(payload, EventsIterTrapList)

    def test_functional_predicates_list_subclass_iterator_is_not_called(self):
        payload = copy.deepcopy(valid_snapshot())
        payload["functional_predicates"] = FunctionalPredicatesIterTrapList(
            payload["functional_predicates"]
        )
        self.assert_restores_without_override(payload, FunctionalPredicatesIterTrapList)

    def test_native_valid_snapshot_is_not_false_rejected(self):
        payload = valid_snapshot()
        restored = TemporalFactStore.from_snapshot(payload, clock=lambda: 200.0)
        self.assertEqual(restored.head_hash, payload["head"])
        self.assertEqual(restored.export_snapshot(), payload)

    def test_invalid_subclassed_snapshot_is_still_rejected_after_safe_copy(self):
        payload = SnapshotGetTrapDict(copy.deepcopy(valid_snapshot()))
        dict.__setitem__(payload, "schema", "not-the-temporal-schema")
        SnapshotGetTrapDict.override_calls = 0

        with self.assertRaisesRegex(ValueError, "schema incompatível"):
            TemporalFactStore.from_snapshot(payload)

        self.assertEqual(SnapshotGetTrapDict.override_calls, 0)


if __name__ == "__main__":
    unittest.main()
