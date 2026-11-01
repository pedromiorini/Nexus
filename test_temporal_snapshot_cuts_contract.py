"""Restore deve validar cortes observáveis, não estados intermediários do timestamp."""
import copy
import hashlib
import json
import random
import unittest
from unittest.mock import patch

from core.temporal_facts import GENESIS, SCHEMA, TemporalFactStore


def assertion(number, recorded, start, end=None, *, subject="ana", predicate="mora_em"):
    return {"kind": "assert", "recorded_at": recorded, "fact": {
        "fact_id": f"tf-{number:06d}", "subject": subject, "predicate": predicate,
        "object": str(number), "valid_from": start, "valid_to": end,
        "recorded_at": recorded, "confidence": 0.9, "source": "fixture",
        "superseded_by": None, "metadata": {"nested": [number]}}}


def close(number, recorded, end):
    return {"kind": "close", "recorded_at": recorded,
            "fact_id": f"tf-{number:06d}", "valid_to": end, "superseded_by": None}


def snapshot(events):
    """Selagem independente do digest de produção; não altera os eventos fornecidos."""
    events = copy.deepcopy(events)
    previous = GENESIS
    for number, event in enumerate(events, 1):
        event.update(seq=number, prev=previous)
        body = json.dumps(event, sort_keys=True, allow_nan=False, separators=(",", ":"))
        previous = hashlib.sha256(body.encode("utf-8")).hexdigest()
        event["hash"] = previous
    return {"schema": SCHEMA, "functional_predicates": ["mora_em"],
            "head": previous, "events": events}


class SnapshotCutsContract(unittest.TestCase):
    def restore(self, events, *, rejected=False):
        payload = snapshot(events)
        before = copy.deepcopy(payload)
        for as_text in (False, True):
            with self.subTest(as_text=as_text):
                value = json.dumps(payload) if as_text else payload
                if rejected:
                    with self.assertRaisesRegex(ValueError, "sobrepost"):
                        TemporalFactStore.from_snapshot(value, expected_head=payload["head"])
                else:
                    restored = TemporalFactStore.from_snapshot(value, expected_head=payload["head"])
                    self.assertEqual(restored.export_snapshot(), payload)
                self.assertEqual(payload, before)
        return None if rejected else restored

    def test_later_close_cannot_repair_observable_functional_overlap(self):
        self.restore([assertion(1, 10, 0), assertion(2, 20, 5), close(1, 30, 5)],
                     rejected=True)

    def test_later_close_cannot_hide_overlap_with_right_neighbor(self):
        self.restore([assertion(1, 10, 10), assertion(2, 20, 5), close(2, 30, 10)],
                     rejected=True)

    def test_invalid_middle_cut_is_not_hidden_by_unrelated_events(self):
        self.restore([assertion(1, 10, 0), assertion(2, 20, 5),
                      assertion(3, 20, 0, subject="bia"), close(1, 30, 5)], rejected=True)

    def test_terminal_cut_also_rejects_overlap(self):
        self.restore([assertion(1, 10, 0), assertion(2, 20, 5)], rejected=True)

    def test_same_timestamp_close_and_assert_are_one_cut_in_either_order(self):
        for tail in ([close(1, 20, 5), assertion(2, 20, 5)],
                     [assertion(2, 20, 5), close(1, 20, 5)]):
            with self.subTest(order=[event["kind"] for event in tail]):
                restored = self.restore([assertion(1, 10, 0)] + tail)
                self.assertEqual([f.object for f in restored.facts_as_of(7, known_at=19)], ["1"])
                self.assertEqual([f.object for f in restored.facts_as_of(7, known_at=20)], ["2"])
                self.assertEqual(restored.facts_as_of(7, known_at=9), [])

    def test_adjacent_half_open_intervals_are_accepted(self):
        restored = self.restore([assertion(1, 10, 0, 5), assertion(2, 20, 5)])
        self.assertEqual([f.object for f in restored.facts_as_of(5, known_at=20)], ["2"])

    def test_nonfunctional_overlap_and_separate_functional_keys_are_accepted(self):
        restored = self.restore([assertion(1, 10, 0, predicate="conhece"),
                                 assertion(2, 20, 5, predicate="conhece"),
                                 assertion(3, 30, 0), assertion(4, 40, 0, subject="bia")])
        self.assertEqual(len(restored.facts_as_of(7, known_at=20)), 2)

    def test_empty_intervals_do_not_overlap_and_shrink_can_empty_existing(self):
        self.restore([assertion(1, 10, 0), assertion(2, 20, 0), close(1, 20, 0),
                      assertion(3, 30, 5, 5), close(2, 30, 0), assertion(4, 30, 0)])

    def test_multiple_dirty_intervals_are_removed_before_reinsertion(self):
        self.restore([assertion(1, 10, 0, 10), assertion(2, 10, 10, 20),
                      assertion(3, 20, 5, 15), close(1, 20, 5), close(2, 20, 10)])

    def test_out_of_order_valid_times_check_both_neighbors(self):
        for middle in ((5, 11), (-1, 1), (5, 9), (20, None)):
            start, end = middle
            with self.subTest(window=middle):
                self.restore([assertion(1, 10, 0, 5), assertion(2, 20, 10, 20),
                              assertion(3, 30, start, end)],
                             rejected=middle in ((5, 11), (-1, 1)))

    def test_corroboration_and_repeated_shrinks_preserve_cut_consistency(self):
        self.restore([assertion(1, 10, 0, 10),
                      {"kind": "corroborate", "recorded_at": 20,
                       "fact_id": "tf-000001", "source": "second"},
                      close(1, 30, 8), close(1, 40, 5), assertion(2, 40, 5)])

    def test_restore_does_not_call_global_checker_per_cut(self):
        events = [assertion(i, i, 0, subject=str(i)) for i in range(1, 21)]
        original = TemporalFactStore.check_invariants
        calls = []

        def counted(store):
            calls.append(len(store._events))
            return original(store)

        with patch.object(TemporalFactStore, "check_invariants", counted):
            self.restore(events)
        # Um checker terminal por restore é permitido, nunca um por corte/evento.
        self.assertLessEqual(len(calls), 2)

    def test_empty_snapshot_is_accepted(self):
        self.restore([])

    def test_seeded_ledgers_match_independent_pairwise_cut_oracle(self):
        rng = random.Random(20261019)
        outcomes = set()
        for case in range(80):
            events = []
            closed = set()
            for number in range(1, 13):
                recorded = number // 3
                start = rng.randrange(8)
                events.append(assertion(number, recorded, start, start + rng.randrange(4),
                                        subject=rng.choice(("ana", "bia")),
                                        predicate=rng.choice(("mora_em", "conhece"))))
                if number > 1 and rng.random() < 0.5:
                    prior = rng.randrange(1, number)
                    fact = next(e["fact"] for e in events if e["kind"] == "assert"
                                and e["fact"]["fact_id"] == f"tf-{prior:06d}")
                    if prior not in closed and fact["valid_to"] > fact["valid_from"]:
                        events.append(close(prior, recorded, fact["valid_from"]))
                        closed.add(prior)
            invalid = False
            for cut in sorted({e["recorded_at"] for e in events}):
                facts = {}
                for event in events:
                    if event["recorded_at"] > cut:
                        continue
                    if event["kind"] == "assert":
                        fact = copy.deepcopy(event["fact"])
                        facts[fact["fact_id"]] = fact
                    else:
                        facts[event["fact_id"]]["valid_to"] = event["valid_to"]
                live = [f for f in facts.values() if f["predicate"] == "mora_em"
                        and f["valid_from"] < f["valid_to"]]
                for i, a in enumerate(live):
                    for b in live[i + 1:]:
                        if (a["subject"] == b["subject"]
                                and max(a["valid_from"], b["valid_from"])
                                < min(a["valid_to"], b["valid_to"])):
                            invalid = True
            outcomes.add(invalid)
            with self.subTest(case=case, invalid=invalid):
                self.restore(events, rejected=invalid)
        self.assertEqual(outcomes, {False, True})


class ExactTemporalNumbersContract(unittest.TestCase):
    bad_numbers = (2**53 + 1, -(2**53 + 1), 10**400, True)

    def state(self, store):
        return copy.deepcopy((store._events, store._head, store._seq,
                              store._facts, store._index, store._last_recorded))

    def populated(self):
        store = TemporalFactStore(functional_predicates=["mora_em"], clock=lambda: 10)
        store.assert_fact("ana", "mora_em", "A", valid_from=0)
        return store

    def test_invalid_write_times_leave_all_state_unchanged(self):
        for bad in self.bad_numbers:
            for operation in ("assert", "retract"):
                with self.subTest(value=bad, operation=operation):
                    store = self.populated()
                    before = self.state(store)
                    with self.assertRaisesRegex(ValueError, "valid_from|valid_to"):
                        if operation == "assert":
                            store.assert_fact("ana", "mora_em", "B", valid_from=bad)
                        else:
                            store.retract("ana", "mora_em", valid_to=bad)
                    self.assertEqual(self.state(store), before)

    def test_invalid_clock_leaves_all_state_unchanged(self):
        for bad in self.bad_numbers:
            for operation in ("assert", "corroborate", "retract", "current"):
                with self.subTest(value=bad, operation=operation):
                    store = self.populated()
                    store._clock = lambda: bad
                    before = self.state(store)
                    with self.assertRaisesRegex(ValueError, "clock"):
                        if operation == "assert":
                            store.assert_fact("ana", "mora_em", "B")
                        elif operation == "corroborate":
                            store.assert_fact("ana", "mora_em", "A", source="second")
                        elif operation == "retract":
                            store.retract("ana", "mora_em")
                        else:
                            store.get_current_facts()
                    self.assertEqual(self.state(store), before)

    def test_invalid_query_cutoffs_raise_valueerror_without_mutation(self):
        store = self.populated()
        before = self.state(store)
        for bad in self.bad_numbers:
            for field in ("valid_at", "known_at"):
                with self.subTest(value=bad, field=field):
                    kwargs = {"valid_at": 0, "known_at": 10, field: bad}
                    with self.assertRaisesRegex(ValueError, field):
                        store.facts_as_of(**kwargs)
                    self.assertEqual(self.state(store), before)

    def test_snapshot_rejects_each_numeric_field_in_object_and_json(self):
        for bad in self.bad_numbers:
            for field in ("event_recorded", "fact_recorded", "valid_from",
                          "assert_valid_to", "close_valid_to"):
                events = [assertion(1, 10, -(2**54)), close(1, 20, 30)]
                if field == "event_recorded":
                    events[0]["recorded_at"] = bad
                elif field == "fact_recorded":
                    events[0]["fact"]["recorded_at"] = bad
                elif field == "assert_valid_to":
                    events[0]["fact"]["valid_to"] = bad
                elif field == "close_valid_to":
                    events[1]["valid_to"] = bad
                else:
                    events[0]["fact"][field] = bad
                payload = snapshot(events)
                before = copy.deepcopy(payload)
                name = "recorded_at" if "recorded" in field else (
                    "valid_to" if "valid_to" in field else field)
                for as_text in (False, True):
                    with self.subTest(value=bad, field=field, as_text=as_text):
                        with self.assertRaisesRegex(ValueError, name):
                            TemporalFactStore.from_snapshot(
                                json.dumps(payload) if as_text else payload,
                                expected_head=payload["head"])
                        self.assertEqual(payload, before)

    def test_adversarial_collapsed_recorded_cut_is_rejected_before_query(self):
        payload = snapshot([assertion(1, 2**53 + 1, 0), close(1, 2**53, 5)])
        for value in (payload, json.dumps(payload)):
            with self.subTest(as_text=isinstance(value, str)):
                with self.assertRaisesRegex(ValueError, "recorded_at"):
                    # Previously restore succeeded and this cutoff raised KeyError.
                    store = TemporalFactStore.from_snapshot(value, expected_head=payload["head"])
                    store.facts_as_of(0, known_at=2**53)

    def test_exact_large_ints_are_accepted_not_capped_at_2_to_53(self):
        for exact in (2**53, 2**53 + 2, -(2**53), 2**100):
            with self.subTest(value=exact):
                store = TemporalFactStore(clock=lambda: exact)
                fact = store.assert_fact("ana", "p", "A", valid_from=exact, confidence=1)
                self.assertEqual(fact.recorded_at, exact)
                self.assertEqual(fact.valid_from, exact)
                self.assertEqual(store.facts_as_of(exact, known_at=exact), [fact])
                store.retract("ana", "p", valid_to=exact)
                self.assertEqual(store.facts_as_of(exact, known_at=exact), [])
                # Independently sealed external ints must retain their values too.
                payload = snapshot([assertion(1, exact, exact, exact)])
                restored = TemporalFactStore.from_snapshot(payload, expected_head=payload["head"])
                self.assertEqual(restored.export_snapshot(), payload)

    def test_valid_confidence_domain_is_preserved(self):
        for confidence in (0, 1, 0.0, 0.25, 1.0):
            with self.subTest(confidence=confidence):
                store = TemporalFactStore(clock=lambda: 0)
                fact = store.assert_fact("ana", "p", "A", confidence=confidence)
                self.assertEqual(fact.confidence, confidence)


if __name__ == "__main__":
    unittest.main()
