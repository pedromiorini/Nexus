import unittest
from unittest.mock import patch

from core.constitutional_brain import RealKnowledgeGraph
from core.temporal_facts import TemporalFactStore


class Clock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value


class TemporalKnowledgeGraphIntegrationContract(unittest.TestCase):
    def test_default_graph_keeps_temporal_store_disabled(self):
        graph = RealKnowledgeGraph()
        relation = graph.add_relation(
            "ana",
            "conhece",
            "bia",
            properties={"note": "legacy"},
            confidence=0.7,
        )

        self.assertIsNone(getattr(graph, "temporal_store", None))
        self.assertEqual(relation.relation_type, "conhece")
        self.assertEqual(relation.properties, {"note": "legacy"})
        self.assertEqual(graph.find_relations(source_id="ana"), [relation])
        self.assertEqual(graph.total_relations, 1)

    def test_injected_store_records_mapping_and_preserves_graph_projection(self):
        clock = Clock(200.0)
        store = TemporalFactStore(
            functional_predicates={"mora_em"},
            clock=clock,
        )
        graph = RealKnowledgeGraph(temporal_store=store)

        relation = graph.add_relation(
            "ana",
            "mora_em",
            "curitiba",
            properties={"episode": "e1"},
            confidence=0.75,
            source="test:knowledge-graph",
            valid_from=150.0,
        )

        facts = store.history("ana", "mora_em")
        self.assertEqual(len(facts), 1)
        self.assertEqual(
            (facts[0].subject, facts[0].predicate, facts[0].object),
            (relation.source_id, relation.relation_type, relation.target_id),
        )
        self.assertEqual(facts[0].valid_from, 150.0)
        self.assertEqual(facts[0].recorded_at, 200.0)
        self.assertEqual(facts[0].confidence, 0.75)
        self.assertEqual(facts[0].source, "test:knowledge-graph")
        self.assertEqual(facts[0].metadata, {})
        self.assertIs(graph.temporal_store, store)
        self.assertEqual(graph.find_relations(source_id="ana"), [relation])
        self.assertEqual(graph.total_relations, 1)

    def test_functionality_is_explicit_and_does_not_remove_graph_history(self):
        clock = Clock(100.0)
        store = TemporalFactStore(
            functional_predicates={"mora_em"},
            clock=clock,
        )
        graph = RealKnowledgeGraph(temporal_store=store)

        graph.add_relation("ana", "mora_em", "A", valid_from=0.0)
        clock.value = 200.0
        graph.add_relation("ana", "mora_em", "B", valid_from=10.0)
        graph.add_relation("ana", "conhece", "bia")
        graph.add_relation("ana", "conhece", "caio")

        history = store.history("ana", "mora_em")
        self.assertEqual(
            [(fact.object, fact.valid_from, fact.valid_to) for fact in history],
            [("A", 0.0, 10.0), ("B", 10.0, None)],
        )
        self.assertEqual(
            [fact.object for fact in store.get_current_facts("ana", "conhece")],
            ["bia", "caio"],
        )
        self.assertEqual(len(graph.find_relations(source_id="ana")), 4)
        self.assertEqual(graph.total_relations, 4)

    def test_invalid_temporal_assertion_has_no_partial_graph_changes(self):
        store = TemporalFactStore(functional_predicates={"mora_em"}, clock=Clock())
        graph = RealKnowledgeGraph(temporal_store=store)
        before = store.export_snapshot()

        with self.assertRaises(ValueError):
            graph.add_relation("new-source", "mora_em", "new-target", source="")
        with self.assertRaises(ValueError):
            graph.add_relation("new-source", "mora_em", "new-target", confidence=2.0)

        self.assertEqual(store.export_snapshot(), before)
        self.assertEqual(graph.entity_index, {})
        self.assertEqual(graph.relation_index, {})
        self.assertEqual(graph.total_entities, 0)
        self.assertEqual(graph.total_relations, 0)
        if graph.nx:
            self.assertEqual(graph.graph.number_of_nodes(), 0)
            self.assertEqual(graph.graph.number_of_edges(), 0)

    def test_properties_are_not_implicitly_serialized_as_temporal_metadata(self):
        opaque = object()
        store = TemporalFactStore(clock=Clock())
        graph = RealKnowledgeGraph(temporal_store=store)

        relation = graph.add_relation(
            "ana",
            "observa",
            "evento",
            properties={"opaque": opaque},
            source="test:opaque-property",
        )

        self.assertIs(relation.properties["opaque"], opaque)
        fact = store.history("ana", "observa")[0]
        self.assertEqual(fact.metadata, {})

    def test_fallback_graph_records_temporal_fact_without_networkx(self):
        store = TemporalFactStore(clock=Clock())
        with patch.dict("sys.modules", {"networkx": None}):
            graph = RealKnowledgeGraph(temporal_store=store)
            graph.add_entity("bia", "person")
            relation = graph.add_relation(
                "ana",
                "conhece",
                "bia",
                source="test:fallback",
                valid_from=10.0,
            )

        self.assertIsNone(graph.nx)
        self.assertEqual(graph.graph["edges"]["ana"], [("bia", "conhece")])
        self.assertEqual(
            store.history("ana", "conhece")[0].object,
            relation.target_id,
        )


if __name__ == "__main__":
    unittest.main()
