"""Regressões da auditoria 2026-10-19: atomicidade do ledger e profundidade em snapshots.

Cenários reproduzidos antes da correção (Python 3.12.3):
* ``assert_fact`` com metadata aninhado ~600 níveis levantava RecursionError DEPOIS de anexar o
  evento ``close``: o fato anterior ficava encerrado sem sucessor e o próprio snapshot do store
  deixava de ser restaurável.
* ``TemporalFactStore.from_snapshot`` deixava vazar RecursionError (JSON profundo ou metadata
  profundo com cadeia de hash forjada).
"""
import copy
import unittest
from unittest.mock import patch

from core.temporal_facts import GENESIS, TemporalFactStore, _digest

# Sem teto novo: preservar profundidade previamente aceita pelo candidato.
ROUND_TRIP_DEPTH = 100


def nested_dict(levels):
    """Dict com exatamente ``levels`` níveis de contêiner (o mais interno é vazio)."""
    value = {}
    for _ in range(levels - 1):
        value = {"k": value}
    return value


def ledger():
    store = TemporalFactStore(functional_predicates={"lives_in"}, clock=lambda: 100.0)
    store.assert_fact("ana", "lives_in", "A", valid_from=0.0)
    return store


def reseal(snapshot):
    """Recalcula a cadeia de hash (um atacante pode forjá-la: ela não é autenticada)."""
    prev = GENESIS
    for n, event in enumerate(snapshot["events"], 1):
        event["seq"], event["prev"] = n, prev
        event["hash"] = _digest(event)
        prev = event["hash"]
    snapshot["head"] = prev
    return snapshot


class LedgerAtomicityContract(unittest.TestCase):
    def assert_untouched(self, store, before_json, before_head):
        self.assertEqual(store.head_hash, before_head)
        self.assertEqual(store.export_snapshot(as_json=True), before_json)
        self.assertEqual(store.check_invariants(), [])
        restored = TemporalFactStore.from_snapshot(store.export_snapshot(as_json=True))
        self.assertEqual(restored.history("ana", "lives_in"), store.history("ana", "lives_in"))

    def test_previously_supported_metadata_depth_round_trips(self):
        store = ledger()
        store.assert_fact("ana", "lives_in", "B", valid_from=10.0,
                          metadata=nested_dict(ROUND_TRIP_DEPTH))
        replay = TemporalFactStore.from_snapshot(store.export_snapshot(as_json=True),
                                                 expected_head=store.head_hash)
        self.assertEqual(replay.history("ana", "lives_in"), store.history("ana", "lives_in"))

    def test_failure_between_close_and_assert_rolls_back_the_close(self):
        store = ledger()
        before, head = store.export_snapshot(as_json=True), store.head_hash
        original = TemporalFactStore._emit
        calls = []

        def fail_on_second(self, event):
            calls.append(event["kind"])
            if len(calls) == 2:
                raise RecursionError("injected failure after close")
            return original(self, event)

        with patch.object(TemporalFactStore, "_emit", fail_on_second):
            with self.assertRaises(RecursionError):
                store.assert_fact("ana", "lives_in", "B", valid_from=10.0)
        self.assertEqual(calls, ["close", "assert"])
        self.assert_untouched(store, before, head)
        # O store continua utilizável e o id do fato não foi consumido.
        fact = store.assert_fact("ana", "lives_in", "B", valid_from=10.0)
        self.assertEqual(fact.fact_id, "tf-000002")
        self.assertEqual(store.check_invariants(), [])

    def test_failure_in_the_middle_of_retract_rolls_back_every_close(self):
        store = TemporalFactStore(clock=lambda: 100.0)
        store.assert_fact("ana", "likes", "x", valid_from=0.0)
        store.assert_fact("ana", "likes", "y", valid_from=0.0)
        before, head = store.export_snapshot(as_json=True), store.head_hash
        original = TemporalFactStore._emit
        calls = []

        def fail_on_second(self, event):
            calls.append(event["kind"])
            if len(calls) == 2:
                raise RuntimeError("injected failure")
            return original(self, event)

        with patch.object(TemporalFactStore, "_emit", fail_on_second):
            with self.assertRaises(RuntimeError):
                store.retract("ana", "likes", valid_to=50.0)
        self.assertEqual(store.head_hash, head)
        self.assertEqual(store.export_snapshot(as_json=True), before)
        self.assertEqual([f.valid_to for f in store.history("ana", "likes")], [None, None])


class LedgerRestoreContract(unittest.TestCase):
    def test_deep_json_text_is_a_value_error(self):
        for payload in ("[" * 200000 + "]" * 200000,
                        '{"schema":"nexus.temporal_facts.v1","x":' + "[" * 200000 + "]" * 200000 + "}"):
            with self.subTest(prefix=payload[:12]):
                with self.assertRaises(ValueError):
                    TemporalFactStore.from_snapshot(payload)

    def test_deep_in_memory_event_is_a_value_error(self):
        snapshot = ledger().export_snapshot()
        deep = []
        for _ in range(200000):
            deep = [deep]
        snapshot["events"][0]["fact"] = deep
        with self.assertRaises(ValueError):
            TemporalFactStore.from_snapshot(snapshot)

    def test_unknown_event_fields_cannot_smuggle_payload_into_the_log(self):
        snapshot = ledger().export_snapshot()
        snapshot["events"][0]["extra"] = nested_dict(500)
        with self.assertRaises(ValueError):
            TemporalFactStore.from_snapshot(reseal(snapshot))
        snapshot = ledger().export_snapshot()
        snapshot["events"][0]["extra"] = "x"
        with self.assertRaisesRegex(ValueError, "campos do evento 1"):
            TemporalFactStore.from_snapshot(reseal(snapshot))

    def test_unhashable_event_kind_is_a_value_error(self):
        snapshot = ledger().export_snapshot()
        snapshot["events"][0]["kind"] = ["assert"]
        with self.assertRaises(ValueError):
            TemporalFactStore.from_snapshot(reseal(snapshot))

    def test_untouched_export_still_restores(self):
        store = ledger()
        store.assert_fact("ana", "lives_in", "B", valid_from=10.0, metadata={"a": [1, {"b": None}]})
        store.assert_fact("ana", "lives_in", "B", valid_from=10.0, source="other")
        replay = TemporalFactStore.from_snapshot(copy.deepcopy(store.export_snapshot()),
                                                 expected_head=store.head_hash)
        self.assertEqual(replay.history("ana", "lives_in"), store.history("ana", "lives_in"))


if __name__ == "__main__":
    unittest.main()
