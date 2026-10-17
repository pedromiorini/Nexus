import importlib.util
import unittest

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


if __name__ == "__main__":
    unittest.main()
