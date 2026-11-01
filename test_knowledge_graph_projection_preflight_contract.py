"""A07: preflight da projeção NetworkX, não uma transação grafo/ledger."""
import copy
import unittest
from unittest.mock import patch

from core.constitutional_brain import RealKnowledgeGraph
from core.temporal_facts import TemporalFactStore


class StringKey(str):
    pass


class KnowledgeGraphProjectionPreflightContract(unittest.TestCase):
    def make_graph(self, temporal, fallback=False):
        store = TemporalFactStore(functional_predicates={"lives"}, clock=lambda: 100.0) if temporal else None
        if fallback:
            with patch.dict("sys.modules", {"networkx": None}):
                graph = RealKnowledgeGraph(temporal_store=store)
            self.assertIsNone(graph.nx)
        else:
            graph = RealKnowledgeGraph(temporal_store=store)
            self.assertIsNotNone(graph.nx, "Este contrato exige o backend NetworkX do perfil CI")
        return graph

    def state(self, graph):
        return copy.deepcopy((
            graph.entity_index, graph.type_index, graph.relation_index,
            graph.total_entities, graph.total_relations,
            graph.total_queries, graph.total_multihop_queries,
            list(graph.graph.nodes(data=True)), list(graph.graph.edges(data=True)),
            graph.temporal_store.export_snapshot() if graph.temporal_store is not None else None,
        ))

    def test_invalid_kwargs_leave_empty_and_populated_state_untouched(self):
        for temporal in (False, True):
            for populated in (False, True):
                for endpoints in (("new-source", "new-target"), ("ana", "A")):
                    for key in (7, None, ("tuple",), "u_of_edge", "v_of_edge"):
                        with self.subTest(temporal=temporal, populated=populated, endpoints=endpoints, key=key):
                            graph = self.make_graph(temporal)
                            if populated:
                                graph.add_relation("ana", "lives", "A", valid_from=0.0)
                            before = self.state(graph)
                            entity_index, type_index, relation_index = graph.entity_index, graph.type_index, graph.relation_index
                            entities = dict(graph.entity_index)
                            relation_lists = dict(graph.relation_index)
                            old_relations = list(graph.find_relations())
                            properties = {key: "bad-key"}
                            with patch.object(graph, "add_entity", wraps=graph.add_entity) as add_entity, patch.object(graph.graph, "add_edge", wraps=graph.graph.add_edge) as add_edge:
                                if temporal:
                                    with patch.object(graph.temporal_store, "assert_fact", wraps=graph.temporal_store.assert_fact) as append:
                                        with self.assertRaises(TypeError):
                                            graph.add_relation(endpoints[0], "lives", endpoints[1], properties=properties, valid_from=10.0)
                                        self.assertEqual(self.state(graph), before)
                                        append.assert_not_called()
                                else:
                                    with self.assertRaises(TypeError):
                                        graph.add_relation(endpoints[0], "lives", endpoints[1], properties=properties)
                                self.assertEqual(self.state(graph), before)
                                add_entity.assert_not_called()
                                add_edge.assert_not_called()
                            self.assertEqual(self.state(graph), before)
                            self.assertIs(graph.entity_index, entity_index)
                            self.assertIs(graph.type_index, type_index)
                            self.assertIs(graph.relation_index, relation_index)
                            for name, entity in entities.items():
                                self.assertIs(graph.entity_index[name], entity)
                            for name, relations in relation_lists.items():
                                self.assertIs(graph.relation_index[name], relations)
                            for old, current in zip(old_relations, graph.find_relations()):
                                self.assertIs(old, current)
                            self.assertEqual(properties, {key: "bad-key"})

    def test_valid_projection_preserves_reserved_fields_opaque_values_and_history(self):
        for temporal in (False, True):
            with self.subTest(temporal=temporal):
                graph = self.make_graph(temporal)
                opaque = object()
                properties = {StringKey("note"): opaque, "relation": "caller", "relations": ["caller"], "timestamp": -1, "confidence": -1}
                first = graph.add_relation("ana", "lives", "A", properties, confidence=0.7, valid_from=0.0)
                first_edge = graph.graph["ana"]["A"]
                self.assertEqual(first_edge["relation"], "lives")
                self.assertEqual(first_edge["relations"], ["lives"])
                self.assertEqual(first_edge["timestamp"], first.timestamp)
                self.assertEqual(first_edge["confidence"], 0.7)
                second = graph.add_relation("ana", "visits", "A", {"extra": "ok"}, confidence=0.8)
                edge = graph.graph["ana"]["A"]
                self.assertEqual(edge["relations"], ["lives", "visits"])
                self.assertEqual(edge["relation"], "visits")
                self.assertEqual(edge["timestamp"], second.timestamp)
                self.assertEqual(edge["confidence"], 0.8)
                self.assertIs(edge["note"], opaque)
                self.assertIs(first.properties["note"], opaque)
                self.assertEqual(properties["relations"], ["caller"])
                self.assertEqual(graph.find_relations(), [first, second])
                self.assertEqual((graph.total_entities, graph.total_relations), (2, 2))
                if temporal:
                    facts = graph.temporal_store.history("ana", "lives")
                    self.assertEqual(len(facts), 1)
                    self.assertEqual(facts[0].metadata, {})
                    self.assertEqual(facts[0].object, "A")
                    self.assertIsNot(first.properties, properties)
                else:
                    self.assertIs(first.properties, properties)

    def test_default_and_empty_properties_remain_valid(self):
        for temporal in (False, True):
            with self.subTest(temporal=temporal):
                graph = self.make_graph(temporal)
                for properties in (None, {}):
                    relation = graph.add_relation("a", "links", "b", properties)
                    self.assertEqual(relation.properties, {})
                self.assertEqual(graph.graph["a"]["b"]["relations"], ["links", "links"])

    def test_fallback_keeps_non_string_keys_without_kwargs_restrictions(self):
        for temporal in (False, True):
            with self.subTest(temporal=temporal):
                graph = self.make_graph(temporal, fallback=True)
                properties = {7: "allowed-in-fallback", "u_of_edge": "also-allowed"}
                relation = graph.add_relation("a", "links", "b", properties)
                self.assertIs(relation.properties, properties)
                self.assertEqual(graph.graph["edges"]["a"], [("b", "links")])
                self.assertEqual((graph.total_entities, graph.total_relations), (2, 1))
                if temporal:
                    self.assertEqual(graph.temporal_store.history("a", "links")[0].object, "b")


if __name__ == "__main__":
    unittest.main()
