"""Contrato para profundidade JSON segura nos metadados do ledger temporal."""
import json
import unittest

from core.temporal_facts import TemporalFactStore


SUPPORTED_DEPTH = 128


def nested_dict(levels):
    value = {}
    for _ in range(levels - 1):
        value = {"k": value}
    return value


def alternating_containers(levels):
    value = None
    for index in range(levels):
        value = {"k": value} if index % 2 == 0 else [value]
    return value


def ledger():
    store = TemporalFactStore(functional_predicates={"lives_in"}, clock=lambda: 100.0)
    store.assert_fact("ana", "lives_in", "A", valid_from=0.0)
    return store


def state(store):
    return (store.export_snapshot(as_json=True), store._seq, store._last_recorded,
            dict(store._facts), {key: list(ids) for key, ids in store._index.items()})


class TemporalMetadataDepthLimitContract(unittest.TestCase):
    def test_supported_boundary_round_trips(self):
        store = ledger()
        store.assert_fact("ana", "lives_in", "B", valid_from=10.0,
                          metadata=nested_dict(SUPPORTED_DEPTH))

        snapshot = store.export_snapshot()
        inputs = (
            ("dict", snapshot),
            ("json", json.dumps(snapshot, sort_keys=True, allow_nan=False)),
        )
        for snapshot_type, payload in inputs:
            with self.subTest(snapshot_type=snapshot_type):
                replay = TemporalFactStore.from_snapshot(payload, expected_head=store.head_hash)
                self.assertEqual(replay.head_hash, store.head_hash)
                self.assertEqual(replay.history("ana", "lives_in"),
                                 store.history("ana", "lives_in"))

    def test_depth_above_boundary_is_rejected_atomically_for_dicts_and_mixed_containers(self):
        payloads = (nested_dict(SUPPORTED_DEPTH + 1),
                    alternating_containers(SUPPORTED_DEPTH + 1))
        for metadata in payloads:
            with self.subTest(metadata_type=type(metadata).__name__):
                store = ledger()
                before = state(store)

                with self.assertRaisesRegex(ValueError, "profundidade"):
                    store.assert_fact("ana", "lives_in", "B", valid_from=10.0,
                                      metadata=metadata)

                self.assertEqual(state(store), before)
                self.assertEqual(store.check_invariants(), [])


if __name__ == "__main__":
    unittest.main()
