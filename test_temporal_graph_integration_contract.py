"""Contrato da injeção opcional de TemporalFactStore em RealKnowledgeGraph.

T1–T4 foram pré-registrados na rodada de CONTINUATION.md antes deste arquivo.
Sem store, o grafo permanece o de main. Com store, a projeção temporal sai do
livro-razão; as arestas não são reescritas nem apagadas.
"""
import importlib.util
import math
import unittest
from unittest.mock import patch

from hypothesis import given, settings, strategies as st

from core.constitutional_brain import RealKnowledgeGraph
from core.temporal_facts import TemporalFactStore


class Clock:
    def __init__(self, t=10.0):
        self.t = t

    def __call__(self):
        return self.t


def _objects(facts):
    return [fact.object for fact in facts]


class TemporalGraphIntegrationContract(unittest.TestCase):
    def test_t1_as_of_comes_from_the_store_not_the_graph(self):
        clock = Clock(10.0)
        store = TemporalFactStore(functional_predicates={"mora_em"}, clock=clock)
        graph = RealKnowledgeGraph(temporal_store=store)

        graph.add_relation("Maria", "mora_em", "Lisboa", valid_from=1.0, source="obs")
        graph.add_relation("Maria", "mora_em", "Porto", valid_from=5.0, source="obs")

        at_three = graph.relations_as_of(3.0, subject="Maria", predicate="mora_em")
        present = graph.current_relations(subject="Maria", predicate="mora_em")

        self.assertEqual(_objects(at_three), ["Lisboa"])
        self.assertEqual(_objects(present), ["Porto"])
        self.assertEqual(at_three, store.facts_as_of(3.0, subject="Maria", predicate="mora_em"))
        self.assertEqual(present, store.get_current_facts("Maria", "mora_em"))
        self.assertEqual(
            sorted(rel.target_id for rel in graph.find_relations(source_id="Maria")),
            ["Lisboa", "Porto"],
        )
        self.assertEqual(graph.total_relations, 2)
        self.assertEqual(store.check_invariants(), [])

    def test_t2_explicit_none_matches_bare_constructor(self):
        injected = RealKnowledgeGraph(temporal_store=None)
        bare = RealKnowledgeGraph()
        self.assertIsNone(injected.temporal_store)
        self.assertIsNone(bare.temporal_store)

        for graph in (injected, bare):
            graph.add_relation(
                "ana", "mora_em", "curitiba",
                properties={"note": "legacy"}, confidence=0.7,
            )
            graph.add_relation("ana", "trabalha_em", "curitiba")

        def fingerprint(graph):
            relations = [
                (rel.source_id, rel.relation_type, rel.target_id, rel.properties, rel.confidence)
                for rel in graph.find_relations(source_id="ana")
            ]
            stats = graph.get_statistics()
            return relations, stats["total_entities"], stats["total_relations"], stats["relation_types"]

        self.assertEqual(fingerprint(injected), fingerprint(bare))
        self.assertEqual(
            {rel.relation_type for rel in injected.find_relations(source_id="ana")},
            {"mora_em", "trabalha_em"},
        )
        self.assertEqual(injected.find_relations(source_id="ana")[0].properties, {"note": "legacy"})
        with self.assertRaisesRegex(RuntimeError, "temporal_store ausente"):
            injected.relations_as_of(3.0, subject="ana", predicate="mora_em")
        with self.assertRaisesRegex(RuntimeError, "temporal_store ausente"):
            injected.current_relations(subject="ana", predicate="mora_em")
        self.assertEqual(len(injected.find_relations(source_id="ana")), 2)

    def test_t2_non_store_types_are_rejected(self):
        for bad in (True, False, 1, "store", object()):
            with self.assertRaisesRegex(TypeError, "TemporalFactStore"):
                RealKnowledgeGraph(temporal_store=bad)

    def test_t3_each_opening_add_appends_one_event_and_snapshot_roundtrips(self):
        clock = Clock(50.0)
        store = TemporalFactStore(clock=clock)
        graph = RealKnowledgeGraph(temporal_store=store)
        adds = (
            ("ana", "conhece", "bia", 1.0),
            ("ana", "conhece", "caio", 2.0),
            ("bia", "conhece", "ana", 3.0),
        )
        for index, (subject, predicate, obj, valid_from) in enumerate(adds, start=1):
            graph.add_relation(
                subject, predicate, obj,
                valid_from=valid_from, confidence=0.4, source="test:t3",
                properties={"note": "nao-entra-no-fato"},
            )
            events = store.export_snapshot()["events"]
            self.assertEqual(len(events), index)
            self.assertEqual(events[-1]["kind"], "assert")
            fact = events[-1]["fact"]
            self.assertEqual(fact["object"], obj)
            self.assertEqual(fact["confidence"], 0.4)
            self.assertEqual(fact["source"], "test:t3")
            self.assertEqual(fact["metadata"], {})
            self.assertEqual(fact["valid_from"], valid_from)

        self.assertEqual(store.check_invariants(), [])
        head = store.head_hash
        snapshot = store.export_snapshot()
        restored = TemporalFactStore.from_snapshot(snapshot, clock=clock, expected_head=head)
        for instant in (1.0, 2.0, 3.0, 50.0):
            self.assertEqual(
                [(f.subject, f.predicate, f.object) for f in restored.facts_as_of(instant)],
                [(f.subject, f.predicate, f.object) for f in store.facts_as_of(instant)],
            )
        with self.assertRaisesRegex(ValueError, "head divergente"):
            TemporalFactStore.from_snapshot(snapshot, clock=clock, expected_head="0" * 64)

    def test_t4_same_instant_replacement_keeps_superseded_trace(self):
        clock = Clock(20.0)
        store = TemporalFactStore(functional_predicates={"mora_em"}, clock=clock)
        graph = RealKnowledgeGraph(temporal_store=store)

        graph.add_relation("Maria", "mora_em", "Lisboa", valid_from=5.0, source="obs-1")
        graph.add_relation("Maria", "mora_em", "Porto", valid_from=5.0, source="obs-2")

        history = store.history("Maria", "mora_em")
        self.assertEqual(_objects(history), ["Lisboa", "Porto"])
        old, new = history
        self.assertEqual(old.superseded_by, new.fact_id)
        self.assertEqual(old.valid_to, 5.0)
        self.assertIsNone(new.superseded_by)
        self.assertEqual(store.check_invariants(), [])

        events = store.export_snapshot()["events"]
        self.assertEqual([event["kind"] for event in events], ["assert", "close", "assert"])
        self.assertEqual(events[1]["fact_id"], old.fact_id)
        self.assertEqual(events[1]["superseded_by"], new.fact_id)
        self.assertEqual(
            sorted(rel.target_id for rel in graph.find_relations(source_id="Maria")),
            ["Lisboa", "Porto"],
        )
        self.assertEqual(
            _objects(graph.relations_as_of(5.0, subject="Maria", predicate="mora_em")),
            ["Porto"],
        )

    def test_invalid_temporal_input_does_not_mutate_the_graph(self):
        store = TemporalFactStore(functional_predicates={"mora_em"}, clock=Clock(10.0))
        graph = RealKnowledgeGraph(temporal_store=store)
        before = store.export_snapshot()
        cases = (
            {"confidence": True},
            {"confidence": False},
            {"confidence": math.nan},
            {"confidence": math.inf},
            {"confidence": -math.inf},
            {"confidence": -0.1},
            {"confidence": 1.1},
            {"valid_from": math.nan},
            {"valid_from": math.inf},
            {"valid_from": True},
            {"source": ""},
            {"source": "   "},
            {"source": None},
        )
        for kwargs in cases:
            with self.assertRaises(ValueError):
                graph.add_relation("Maria", "mora_em", "Lisboa", **kwargs)
        with self.assertRaises(ValueError):
            graph.add_relation("", "mora_em", "Lisboa")
        with self.assertRaises(ValueError):
            graph.add_relation("Maria", "", "Lisboa")
        with self.assertRaises(ValueError):
            graph.add_relation("Maria", "mora_em", "")

        self.assertEqual(store.export_snapshot(), before)
        self.assertEqual(graph.entity_index, {})
        self.assertEqual(graph.relation_index, {})
        self.assertEqual(graph.total_entities, 0)
        self.assertEqual(graph.total_relations, 0)
        if graph.nx:
            self.assertEqual(graph.graph.number_of_nodes(), 0)
            self.assertEqual(graph.graph.number_of_edges(), 0)
        else:
            self.assertEqual(graph.graph["nodes"], {})
            self.assertEqual(graph.graph["edges"], {})

    def test_negative_finite_valid_from_is_kept(self):
        store = TemporalFactStore(clock=Clock(10.0))
        graph = RealKnowledgeGraph(temporal_store=store)
        graph.add_relation("Maria", "conhece", "Ana", valid_from=-5.0, source="obs")
        found = graph.relations_as_of(-5.0, subject="Maria", predicate="conhece")
        self.assertEqual(_objects(found), ["Ana"])
        self.assertEqual(found[0].valid_from, -5.0)

    def test_opaque_property_stays_on_the_relation_only(self):
        opaque = object()
        store = TemporalFactStore(clock=Clock(10.0))
        graph = RealKnowledgeGraph(temporal_store=store)
        relation = graph.add_relation(
            "ana", "observa", "evento",
            properties={"opaque": opaque},
            source="test:opaque",
        )
        self.assertIs(relation.properties["opaque"], opaque)
        self.assertEqual(store.history("ana", "observa")[0].metadata, {})

    def test_fallback_without_networkx_still_records_the_fact(self):
        store = TemporalFactStore(clock=Clock(10.0))
        with patch.dict("sys.modules", {"networkx": None}):
            graph = RealKnowledgeGraph(temporal_store=store)
            graph.add_entity("bia", "person")
            relation = graph.add_relation(
                "ana", "conhece", "bia", source="test:fallback", valid_from=4.0,
            )
        self.assertIsNone(graph.nx)
        self.assertEqual(graph.graph["edges"]["ana"], [("bia", "conhece")])
        self.assertEqual(store.history("ana", "conhece")[0].object, relation.target_id)
        self.assertEqual(
            _objects(graph.relations_as_of(4.0, subject="ana", predicate="conhece")),
            ["bia"],
        )

    @unittest.skipUnless(importlib.util.find_spec("networkx"), "networkx opcional ausente")
    def test_networkx_edge_still_accumulates_when_store_is_on(self):
        store = TemporalFactStore(clock=Clock(10.0))
        graph = RealKnowledgeGraph(temporal_store=store)
        graph.add_relation("ana", "mora_em", "curitiba", valid_from=1.0)
        graph.add_relation("ana", "trabalha_em", "curitiba", valid_from=2.0)
        edge = graph.graph["ana"]["curitiba"]
        self.assertEqual(edge["relation"], "trabalha_em")
        self.assertEqual(edge["relations"], ["mora_em", "trabalha_em"])
        self.assertEqual(len(store.export_snapshot()["events"]), 2)

    @settings(max_examples=20, deadline=None, derandomize=True)
    @given(st.lists(
        st.integers(min_value=0, max_value=10_000),
        min_size=1, max_size=6, unique=True,
    ))
    def test_t3_property_opening_adds_append_one_event_each(self, valid_froms):
        clock = Clock(float(max(valid_froms) + 10))
        store = TemporalFactStore(clock=clock)
        graph = RealKnowledgeGraph(temporal_store=store)
        for index, valid_from in enumerate(valid_froms, start=1):
            graph.add_relation(
                "ana", "conhece", f"p{index}",
                valid_from=float(valid_from), source="prop",
            )
            self.assertEqual(len(store.export_snapshot()["events"]), index)
        self.assertEqual(store.check_invariants(), [])
        head = store.head_hash
        restored = TemporalFactStore.from_snapshot(
            store.export_snapshot(), clock=clock, expected_head=head,
        )
        live = store.facts_as_of(clock.t, subject="ana", predicate="conhece")
        self.assertEqual(
            [(f.object, f.valid_from) for f in restored.facts_as_of(
                clock.t, subject="ana", predicate="conhece")],
            [(f.object, f.valid_from) for f in live],
        )
        self.assertEqual(
            [f.valid_from for f in live],
            sorted(float(v) for v in valid_froms),
        )


if __name__ == "__main__":
    unittest.main()
