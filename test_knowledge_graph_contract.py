import importlib.util
import sys
import unittest
from unittest.mock import patch

from core.constitutional_brain import RealKnowledgeGraph


class KnowledgeGraphContractTests(unittest.TestCase):
    def test_duplicate_entity_updates_in_place_without_inflating_statistics(self):
        graph = RealKnowledgeGraph()
        graph.add_entity("n1", "person", confidence=0.9)
        graph.add_entity("n1", "project", confidence=0.3)

        statistics = graph.get_statistics()
        self.assertEqual(statistics["total_entities"], 1)
        self.assertAlmostEqual(statistics["avg_entity_confidence"], 0.3)
        self.assertEqual(graph.find_entity("n1").type, "project")
        self.assertEqual(graph.find_entities_by_type("person"), [])
        self.assertEqual([entity.id for entity in graph.find_entities_by_type("project")], ["n1"])

    def test_distinct_entities_are_counted_and_averaged(self):
        graph = RealKnowledgeGraph()
        graph.add_entity("n1", "person", confidence=0.8)
        graph.add_entity("n2", "person", confidence=0.4)

        statistics = graph.get_statistics()
        self.assertEqual(statistics["total_entities"], 2)
        self.assertAlmostEqual(statistics["avg_entity_confidence"], 0.6)


class KnowledgeGraphMultiRelationContract(unittest.TestCase):
    def test_relation_index_counts_every_relation_between_the_same_pair(self):
        graph = RealKnowledgeGraph()
        graph.add_relation("ana", "mora_em", "curitiba")
        graph.add_relation("ana", "trabalha_em", "curitiba")
        indexed = {r.relation_type for r in graph.find_relations(source_id="ana")}
        self.assertEqual(indexed, {"mora_em", "trabalha_em"})

    @unittest.skipUnless(importlib.util.find_spec("networkx"), "networkx opcional ausente")
    def test_networkx_edge_keeps_all_relation_types_for_the_same_pair(self):
        graph = RealKnowledgeGraph()
        graph.add_relation("ana", "mora_em", "curitiba")
        graph.add_relation("ana", "trabalha_em", "curitiba")
        graph.add_relation("ana", "trabalha_em", "curitiba")
        edge = graph.graph["ana"]["curitiba"]
        self.assertEqual(edge["relation"], "trabalha_em")
        self.assertEqual(edge["relations"], ["mora_em", "trabalha_em", "trabalha_em"])
        self.assertEqual(graph.graph.number_of_edges(), 1)

    @unittest.skipUnless(importlib.util.find_spec("networkx"), "networkx opcional ausente")
    def test_reserved_relations_property_does_not_break_networkx_edge(self):
        graph = RealKnowledgeGraph()
        graph.add_relation("ana", "mora_em", "curitiba",
                           properties={"relations": ["caller-value"], "note": "preserved"})
        edge = graph.graph["ana"]["curitiba"]
        self.assertEqual(edge["relations"], ["mora_em"])
        self.assertEqual(edge["note"], "preserved")


class MissingNodesContract:
    def test_missing_endpoints_return_empty_result(self):
        for start, end in (
            ("missing", "known"),
            ("known", "missing"),
            ("missing", "other-missing"),
            ("missing", "missing"),
        ):
            for max_hops in (0, 3):
                with self.subTest(start=start, end=end, max_hops=max_hops):
                    graph = self.make_graph()
                    graph.add_entity("known", "person")
                    result = graph.multihop_query(start, end, max_hops=max_hops)
                    self.assertEqual(result.entities, [])
                    self.assertEqual(result.paths, [])
                    self.assertEqual(result.context, "")
                    self.assertEqual(result.confidence, 0.0)
                    self.assertEqual(result.hops, 0)
                    self.assertEqual(graph.total_queries, 1)
                    self.assertEqual(graph.total_multihop_queries, 1)
                    self.assertEqual(set(graph.entity_index), {"known"})
                    self.assertEqual(graph.total_entities, 1)
                    self.assertEqual(graph.total_relations, 0)

    def test_existing_self_path_is_preserved(self):
        graph = self.make_graph()
        graph.add_entity("known", "person", confidence=0.9)
        result = graph.multihop_query("known", "known", max_hops=0)
        self.assertEqual([entity.id for entity in result.entities], ["known"])
        self.assertEqual(result.paths, [["known"]])
        self.assertEqual(result.hops, 0)
        self.assertGreater(result.confidence, 0.0)

    def test_existing_path_and_hop_limit_are_preserved(self):
        graph = self.make_graph()
        graph.add_relation("a", "links", "b")
        graph.add_relation("b", "links", "c")
        result = graph.multihop_query("a", "c", max_hops=2)
        self.assertEqual([entity.id for entity in result.entities], ["a", "b", "c"])
        self.assertEqual(result.paths, [["a", "b", "c"]])
        self.assertEqual(result.hops, 2)
        self.assertTrue(result.context)
        self.assertGreater(result.confidence, 0.0)
        limited = graph.multihop_query("a", "c", max_hops=1)
        self.assertEqual(limited.paths, [])
        self.assertEqual(limited.confidence, 0.0)

    def test_existing_disconnected_endpoints_return_empty_result(self):
        graph = self.make_graph()
        graph.add_entity("a", "person")
        graph.add_entity("b", "person")
        result = graph.multihop_query("a", "b")
        self.assertEqual(result.entities, [])
        self.assertEqual(result.paths, [])
        self.assertEqual(result.context, "")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.hops, 0)


@unittest.skipUnless(importlib.util.find_spec("networkx"), "networkx opcional ausente")
class KnowledgeGraphMissingNodesNetworkXContract(MissingNodesContract, unittest.TestCase):
    def make_graph(self):
        graph = RealKnowledgeGraph()
        self.assertIsNotNone(graph.nx)
        return graph


class KnowledgeGraphMissingNodesFallbackContract(MissingNodesContract, unittest.TestCase):
    def make_graph(self):
        with patch.dict(sys.modules, {"networkx": None}):
            graph = RealKnowledgeGraph()
        self.assertIsNone(graph.nx)
        return graph


if __name__ == "__main__":
    unittest.main()
