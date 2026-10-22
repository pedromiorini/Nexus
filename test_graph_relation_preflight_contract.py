"""Contratos de preflight de atributos NetworkX antes de alterar projeção/ledger."""
import copy
import unittest
from unittest.mock import patch

from core.constitutional_brain import RealKnowledgeGraph
from core.temporal_facts import TemporalFactStore


class GraphRelationPreflightContract(unittest.TestCase):
    def snapshot(self, graph):
        return (
            copy.deepcopy(graph.entity_index),
            copy.deepcopy(graph.type_index),
            copy.deepcopy(graph.relation_index),
            graph.total_entities, graph.total_relations,
            copy.deepcopy(dict(graph.graph.nodes(data=True))),
            copy.deepcopy(list(graph.graph.edges(data=True))),
            graph.temporal_store.export_snapshot() if graph.temporal_store else None,
        )

    def test_invalid_keyword_keys_leave_new_and_populated_graph_unchanged(self):
        for temporal in (False, True):
            for populated in (False, True):
                for key in (1, True, None, ("key",)):
                    with self.subTest(temporal=temporal, populated=populated, key=key):
                        store = TemporalFactStore(
                            functional_predicates={"lives_in"}, clock=lambda: 100.0,
                        ) if temporal else None
                        graph = RealKnowledgeGraph(temporal_store=store)
                        self.assertIsNotNone(graph.nx, "CI profile must include NetworkX")
                        if populated:
                            graph.add_relation("ana", "lives_in", "A", valid_from=0.0)
                        before = self.snapshot(graph)
                        with self.assertRaises(TypeError):
                            graph.add_relation(
                                "ana", "lives_in", "B", properties={key: "x"}, valid_from=10.0,
                            )
                        self.assertEqual(self.snapshot(graph), before)

    def test_keyword_parameter_collisions_are_rejected_before_mutation(self):
        for key in ("u_of_edge", "v_of_edge"):
            with self.subTest(key=key):
                graph = RealKnowledgeGraph(temporal_store=TemporalFactStore(clock=lambda: 100.0))
                before = self.snapshot(graph)
                with self.assertRaises(TypeError):
                    graph.add_relation("ana", "knows", "bia", properties={key: "x"})
                self.assertEqual(self.snapshot(graph), before)

    def test_self_collision_preserves_empty_and_populated_graph_with_or_without_store(self):
        for temporal in (False, True):
            for populated in (False, True):
                with self.subTest(temporal=temporal, populated=populated):
                    store = TemporalFactStore(clock=lambda: 100.0) if temporal else None
                    graph = RealKnowledgeGraph(temporal_store=store)
                    self.assertIsNotNone(graph.nx, "CI profile must include NetworkX")
                    if populated:
                        graph.add_relation("ana", "knows", "bia", valid_from=0.0)
                    before = self.snapshot(graph)
                    properties = {"self": "opaque"}
                    with self.assertRaises(TypeError):
                        graph.add_relation(
                            "ana", "knows", "caio", properties=properties, valid_from=10.0,
                        )
                    self.assertEqual(self.snapshot(graph), before)
                    self.assertEqual(properties, {"self": "opaque"})

    def test_self_collision_cannot_close_functional_fact_or_call_mutating_backends(self):
        store = TemporalFactStore(functional_predicates={"lives_in"}, clock=lambda: 100.0)
        graph = RealKnowledgeGraph(temporal_store=store)
        graph.add_relation("ana", "lives_in", "A", valid_from=0.0)
        graph.add_entity("B", "city")
        before = self.snapshot(graph)
        with patch.object(store, "assert_fact", wraps=store.assert_fact) as assert_fact:
            with patch.object(graph.graph, "add_edge", wraps=graph.graph.add_edge) as add_edge:
                with self.assertRaises(TypeError):
                    graph.add_relation(
                        "ana", "lives_in", "B", properties={"self": "opaque"}, valid_from=10.0,
                    )
                self.assertEqual(self.snapshot(graph), before)
                assert_fact.assert_not_called()
                add_edge.assert_not_called()
        self.assertEqual(self.snapshot(graph), before)
        history = store.history("ana", "lives_in")
        self.assertEqual([(fact.object, fact.valid_to) for fact in history], [("A", None)])
        self.assertEqual(
            [fact.object for fact in store.get_current_facts("ana", "lives_in")], ["A"],
        )

    def test_fallback_accepts_self_property_without_networkx_keyword_call(self):
        with patch.dict("sys.modules", {"networkx": None}):
            graph = RealKnowledgeGraph(temporal_store=TemporalFactStore(clock=lambda: 100.0))
        relation = graph.add_relation("ana", "knows", "bia", properties={"self": "opaque"})
        self.assertEqual(relation.properties, {"self": "opaque"})
        self.assertEqual(graph.graph["edges"]["ana"], [("bia", "knows")])
        self.assertEqual(len(graph.temporal_store.history("ana", "knows")), 1)

    def test_valid_attributes_preserve_opaque_values_and_reserved_projection_fields(self):
        opaque = object()
        graph = RealKnowledgeGraph(temporal_store=TemporalFactStore(clock=lambda: 100.0))
        properties = {"": "empty-key", "note": opaque, "relation": "caller-value"}
        relation = graph.add_relation("ana", "knows", "bia", properties=properties)
        self.assertEqual(relation.properties, properties)
        self.assertIs(relation.properties["note"], opaque)
        self.assertEqual(graph.graph["ana"]["bia"]["relation"], "knows")
        self.assertEqual(graph.total_relations, 1)
        self.assertEqual(len(graph.temporal_store.history("ana", "knows")), 1)

    def test_fallback_retains_legacy_non_keyword_attributes(self):
        with patch.dict("sys.modules", {"networkx": None}):
            graph = RealKnowledgeGraph(temporal_store=TemporalFactStore(clock=lambda: 100.0))
        relation = graph.add_relation("ana", "knows", "bia", properties={1: "x"})
        self.assertEqual(relation.properties, {1: "x"})
        self.assertEqual(graph.graph["edges"]["ana"], [("bia", "knows")])
        self.assertEqual(graph.total_relations, 1)


if __name__ == "__main__":
    unittest.main()
