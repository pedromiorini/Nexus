import copy
import json
import random
import unittest

from core.temporal_facts import TemporalFactStore


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def store(clock=None, functional=("mora_em",)):
    return TemporalFactStore(functional_predicates=functional, clock=clock or Clock())


class TemporalFactContract(unittest.TestCase):
    def test_new_fact_is_current_with_provenance(self):
        s = store()
        f = s.assert_fact("ana", "mora_em", "Curitiba", confidence=0.7,
                          source="user", metadata={"episode": "e1"})
        cur = s.get_current_facts("ana", "mora_em")
        self.assertEqual([x.fact_id for x in cur], [f.fact_id])
        self.assertEqual((cur[0].source, cur[0].confidence, cur[0].recorded_at, cur[0].metadata),
                         ("user", 0.7, 100.0, {"episode": "e1"}))
        self.assertIsNone(cur[0].valid_to)

    def test_conflict_closes_old_and_keeps_history(self):
        c = Clock(); s = store(c)
        old = s.assert_fact("ana", "mora_em", "Curitiba")
        c.t = 200.0
        new = s.assert_fact("ana", "mora_em", "Lisboa")
        hist = s.history("ana", "mora_em")
        self.assertEqual([h.object for h in hist], ["Curitiba", "Lisboa"])
        self.assertEqual(hist[0].valid_to, 200.0)
        self.assertEqual(hist[0].superseded_by, new.fact_id)
        self.assertEqual([x.object for x in s.get_current_facts("ana")], ["Lisboa"])
        self.assertEqual(old.fact_id, hist[0].fact_id)
        self.assertEqual((hist[0].closed_at, hist[1].closed_at), (200.0, None))

    def test_non_functional_predicate_accumulates(self):
        s = store()
        s.assert_fact("ana", "conhece", "bia")
        s.assert_fact("ana", "conhece", "caio")
        self.assertEqual(len(s.get_current_facts("ana", "conhece")), 2)

    def test_reassertion_is_idempotent(self):
        s = store()
        a = s.assert_fact("ana", "mora_em", "Curitiba")
        n = len(s.export_snapshot()["events"])
        b = s.assert_fact("ana", "mora_em", "Curitiba")
        self.assertEqual(a.fact_id, b.fact_id)
        self.assertEqual(len(s.export_snapshot()["events"]), n)

    def test_as_of_window_is_half_open(self):
        c = Clock(0.0); s = store(c)
        s.assert_fact("ana", "mora_em", "A", valid_from=10)
        s.assert_fact("ana", "mora_em", "B", valid_from=20)
        at = lambda t: [f.object for f in s.facts_as_of(t)]  # noqa: E731
        self.assertEqual(at(5), [])
        self.assertEqual(at(10), ["A"])
        self.assertEqual(at(19.9), ["A"])
        self.assertEqual(at(20), ["B"])

    def test_known_at_separates_belief_from_truth(self):
        c = Clock(10.0); s = store(c)
        s.assert_fact("ana", "mora_em", "A", valid_from=0)
        c.t = 20.0
        s.assert_fact("ana", "mora_em", "B", valid_from=5)  # correção retroativa
        objs = lambda t, k: [f.object for f in s.facts_as_of(t, known_at=k)]  # noqa: E731
        self.assertEqual(objs(7, 15), ["A"])
        self.assertEqual(objs(7, 25), ["B"])
        self.assertEqual(objs(3, 25), ["A"])
        self.assertEqual(objs(7, 5), [])

    def test_out_of_order_assertion_does_not_overlap(self):
        s = store(Clock(100.0))
        s.assert_fact("ana", "mora_em", "B", valid_from=10)
        s.assert_fact("ana", "mora_em", "A", valid_from=5)
        a, b = s.history("ana", "mora_em")
        self.assertEqual((a.object, a.valid_to, b.object, b.valid_to), ("A", 10.0, "B", None))
        self.assertEqual([f.object for f in s.get_current_facts("ana")], ["B"])

    def test_insert_inside_closed_interval_splits_it(self):
        s = store(Clock(100.0))
        s.assert_fact("ana", "mora_em", "A", valid_from=0)
        s.assert_fact("ana", "mora_em", "B", valid_from=10)
        s.assert_fact("ana", "mora_em", "C", valid_from=5)
        got = [(f.object, f.valid_from, f.valid_to) for f in s.history("ana", "mora_em")]
        self.assertEqual(got, [("A", 0.0, 5.0), ("C", 5.0, 10.0), ("B", 10.0, None)])

    def test_returned_facts_are_isolated_copies(self):
        s = store()
        f = s.assert_fact("ana", "mora_em", "A", metadata={"k": [1]})
        f.metadata["k"].append(2)
        self.assertEqual(s.history("ana", "mora_em")[0].metadata, {"k": [1]})

    def test_invalid_inputs_rejected_without_state_change(self):
        s = store()
        bad = [dict(subject="", predicate="p", obj="o"), dict(subject="a", predicate=" ", obj="o"),
               dict(subject="a", predicate="p", obj="o", valid_from=True),
               dict(subject="a", predicate="p", obj="o", valid_from=float("nan")),
               dict(subject="a", predicate="p", obj="o", confidence=1.5),
               dict(subject="a", predicate="p", obj="o", confidence=float("inf")),
               dict(subject="a", predicate="p", obj="o", source=3),
               dict(subject="a", predicate="p", obj="o", metadata=[1])]
        for kw in bad:
            with self.assertRaises(ValueError, msg=kw):
                s.assert_fact(**kw)
        self.assertEqual(s.export_snapshot()["events"], [])

    def test_non_json_metadata_is_rejected_atomically(self):
        c = Clock(100.0)
        s = store(c)
        old = s.assert_fact("ana", "mora_em", "Curitiba")
        before = s.export_snapshot()

        for metadata in ({"bad": object()}, {"bad": float("nan")}, {1: "not a string key"},
                         {"bad": (1, 2)}):
            with self.assertRaises(ValueError):
                s.assert_fact("ana", "mora_em", "Lisboa", metadata=metadata)
            self.assertEqual(s.export_snapshot(), before)

        self.assertIsNone(s.history("ana", "mora_em")[0].valid_to)
        new = s.assert_fact("ana", "mora_em", "Lisboa")
        self.assertEqual(new.fact_id, "tf-000002")
        restored = TemporalFactStore.from_snapshot(s.export_snapshot(as_json=True), clock=c)
        self.assertEqual(restored.history("ana", "mora_em"), s.history("ana", "mora_em"))
        self.assertEqual(old.fact_id, "tf-000001")

    def test_clock_never_goes_backwards(self):
        c = Clock(50.0); s = store(c)
        s.assert_fact("a", "p", "1")
        c.t = 10.0
        f = s.assert_fact("a", "p", "2")
        self.assertEqual(f.recorded_at, 50.0)

    def test_rejected_timestamps_do_not_advance_recording_watermark(self):
        c = Clock(50.0); s = store(c)
        with self.assertRaises(ValueError):
            s.assert_fact("ana", "mora_em", "Curitiba", valid_from=float("nan"))
        c.t = 1.0
        first = s.assert_fact("ana", "mora_em", "Curitiba")
        self.assertEqual(first.recorded_at, 1.0)

        c.t = 2.0
        with self.assertRaises(ValueError):
            s.retract("bia", "mora_em", valid_to=float("nan"))
        c.t = 0.0
        second = s.assert_fact("bia", "mora_em", "Lisboa")
        self.assertEqual(second.recorded_at, 1.0)

    def test_random_invariants_no_overlap_and_replay_matches(self):
        rng = random.Random(20261017)
        c = Clock(0.0); s = store(c)
        for _ in range(300):
            c.t += rng.random()
            s.assert_fact(rng.choice("xy"), "mora_em", rng.choice("ABCD"),
                          valid_from=rng.randint(0, 40))
        for subj in "xy":
            hist = s.history(subj, "mora_em")
            for a, b in zip(hist, hist[1:]):
                self.assertIsNotNone(a.valid_to)
                self.assertLessEqual(a.valid_to, b.valid_from)
            for t in range(0, 45):
                self.assertLessEqual(len(s.facts_as_of(t, subject=subj)), 1)
        for t in range(0, 45):
            self.assertEqual(s.facts_as_of(t), s.facts_as_of(t, known_at=c.t))
        self.assertEqual(s.check_invariants(), [])
        again = TemporalFactStore.from_snapshot(s.export_snapshot(as_json=True), clock=c,
                                                expected_head=s.head_hash)
        for t in range(0, 45):
            self.assertEqual(s.facts_as_of(t), again.facts_as_of(t))


class RetractionAndCorroboration(unittest.TestCase):
    def test_retract_closes_without_replacement_and_keeps_history(self):
        c = Clock(100.0); s = store(c)
        s.assert_fact("ana", "mora_em", "A", valid_from=0)
        c.t = 150.0
        closed = s.retract("ana", "mora_em", valid_to=120)
        self.assertEqual([(f.valid_to, f.superseded_by, f.closed_at) for f in closed],
                         [(120.0, None, 150.0)])
        self.assertEqual(s.facts_as_of(119)[0].object, "A")
        self.assertEqual(s.facts_as_of(120), [])
        self.assertEqual(s.get_current_facts("ana"), [])
        self.assertEqual(len(s.history("ana", "mora_em")), 1)

    def test_retract_filters_by_object_and_ignores_missing(self):
        s = store()
        s.assert_fact("ana", "conhece", "bia")
        s.assert_fact("ana", "conhece", "caio")
        self.assertEqual(s.retract("ana", "conhece", "zed"), [])
        s.retract("ana", "conhece", "bia")
        self.assertEqual([f.object for f in s.get_current_facts("ana")], ["caio"])

    def test_new_source_corroborates_without_new_fact(self):
        s = store()
        a = s.assert_fact("ana", "mora_em", "A", source="user")
        b = s.assert_fact("ana", "mora_em", "A", source="tool:crm")
        self.assertEqual(a.fact_id, b.fact_id)
        self.assertEqual(b.corroborated_by, ("tool:crm",))
        n = len(s.export_snapshot()["events"])
        s.assert_fact("ana", "mora_em", "A", source="tool:crm")
        s.assert_fact("ana", "mora_em", "A", source="user")
        self.assertEqual(len(s.export_snapshot()["events"]), n)
        self.assertEqual(len(s.history("ana", "mora_em")), 1)


class SnapshotContract(unittest.TestCase):
    def build(self):
        c = Clock(10.0); s = store(c)
        s.assert_fact("ana", "mora_em", "A", valid_from=0, metadata={"k": 1})
        c.t = 20.0
        s.assert_fact("ana", "mora_em", "B", valid_from=5, source="user")
        s.assert_fact("ana", "mora_em", "B", source="tool:x")
        s.retract("ana", "mora_em", valid_to=30)
        return s, c

    def test_round_trip_preserves_facts_history_and_head(self):
        s, c = self.build()
        again = TemporalFactStore.from_snapshot(s.export_snapshot(as_json=True), clock=c,
                                                expected_head=s.head_hash)
        self.assertEqual(again.history("ana", "mora_em"), s.history("ana", "mora_em"))
        self.assertEqual(again.head_hash, s.head_hash)
        self.assertEqual(again.facts_as_of(7, known_at=15), s.facts_as_of(7, known_at=15))
        c.t = 40.0
        nxt = again.assert_fact("ana", "mora_em", "C", valid_from=35)
        self.assertEqual(nxt.fact_id, "tf-000003")

    def test_tampering_removal_and_reordering_are_rejected(self):
        s, c = self.build()
        snap = s.export_snapshot()
        edited = copy.deepcopy(snap)
        edited["events"][0]["fact"]["object"] = "Z"
        removed = copy.deepcopy(snap)
        del removed["events"][1]
        swapped = copy.deepcopy(snap)
        swapped["events"][0], swapped["events"][1] = swapped["events"][1], swapped["events"][0]
        for bad in (edited, removed, swapped):
            with self.assertRaises(ValueError):
                TemporalFactStore.from_snapshot(bad, clock=c)

    def test_tail_truncation_needs_expected_head(self):
        s, c = self.build()
        snap = s.export_snapshot()
        snap["events"] = snap["events"][:-1]
        snap["head"] = snap["events"][-1]["hash"]
        TemporalFactStore.from_snapshot(snap, clock=c)  # cadeia ainda consistente
        with self.assertRaises(ValueError):
            TemporalFactStore.from_snapshot(snap, clock=c, expected_head=s.head_hash)

    def test_malformed_payloads_are_rejected(self):
        s, c = self.build()
        good = s.export_snapshot()
        mutate = [
            lambda d: d.update(schema="outro"),
            lambda d: d.update(events="x"),
            lambda d: d.update(functional_predicates=[1]),
            lambda d: d.update(head="0" * 64),
        ]
        for fn in mutate:
            bad = copy.deepcopy(good)
            fn(bad)
            with self.assertRaises(ValueError):
                TemporalFactStore.from_snapshot(bad, clock=c)
        for text in ("{", '{"schema": NaN}', "[]", "null"):
            with self.assertRaises(ValueError):
                TemporalFactStore.from_snapshot(text, clock=c)

    def test_semantically_invalid_but_well_chained_events_are_rejected(self):
        from core import temporal_facts as tf

        def rechain(events):
            prev = tf.GENESIS
            for n, ev in enumerate(events, 1):
                ev["seq"], ev["prev"] = n, prev
                ev["hash"] = tf._digest(ev)
                prev = ev["hash"]
            return prev

        s, c = self.build()
        base = s.export_snapshot()

        def forge(edit):
            snap = copy.deepcopy(base)
            edit(snap["events"])
            snap["head"] = rechain(snap["events"])
            return snap

        cases = [
            lambda ev: ev[0]["fact"].update(confidence=7),
            lambda ev: ev[0]["fact"].update(valid_to=-5.0),
            lambda ev: ev[0]["fact"].update(subject=""),
            lambda ev: ev[0].update(kind="mystery"),
            lambda ev: ev[1].update(fact_id="tf-999999"),
            lambda ev: ev[0].update(recorded_at=999.0),
            lambda ev: ev.append({"kind": "corroborate", "fact_id": "nope", "source": "x",
                                  "recorded_at": 99.0}),
            lambda ev: ev[2]["fact"].update(valid_from=-100.0),  # sobrepõe A em predicado funcional
        ]
        for edit in cases:
            with self.assertRaises(ValueError):
                TemporalFactStore.from_snapshot(forge(edit), clock=c)

    def test_wrong_prev_link_on_last_event_is_rejected_even_if_self_hash_is_valid(self):
        from core import temporal_facts as tf

        s, c = self.build()
        snap = s.export_snapshot()
        last = snap["events"][-1]
        last["prev"] = "f" * 64
        last["hash"] = tf._digest(last)
        snap["head"] = last["hash"]
        with self.assertRaises(ValueError):
            TemporalFactStore.from_snapshot(snap, clock=c)

    def test_superseded_by_must_reference_a_matching_successor(self):
        from core import temporal_facts as tf

        s, c = self.build()
        s.assert_fact("bia", "conhece", "caio")
        snap = s.export_snapshot()
        close = snap["events"][1]
        close["superseded_by"] = "tf-000003"
        prev = tf.GENESIS
        for n, event in enumerate(snap["events"], 1):
            event["seq"], event["prev"] = n, prev
            event["hash"] = tf._digest(event)
            prev = event["hash"]
        snap["head"] = prev

        with self.assertRaises(ValueError):
            TemporalFactStore.from_snapshot(snap, clock=c)

    def test_snapshot_is_strict_json_and_detached(self):
        s, c = self.build()
        text = s.export_snapshot(as_json=True)
        self.assertEqual(json.loads(text)["schema"], "nexus.temporal_facts.v1")
        snap = s.export_snapshot()
        snap["events"].clear()
        self.assertGreater(len(s.export_snapshot()["events"]), 0)


if __name__ == "__main__":
    unittest.main()
