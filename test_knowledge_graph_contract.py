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


if __name__ == "__main__":
    unittest.main()
