import sys
import unittest
from unittest.mock import patch

from core.constitutional_brain import QueryResult, RealKnowledgeGraph


class KnowledgeGraphFallbackHopsContractTests(unittest.TestCase):
    def make_graph(self):
        with patch.dict(sys.modules, {"networkx": None}):
            graph = RealKnowledgeGraph()
        self.assertIsNone(graph.nx)
        return graph

    def test_over_budget_path_returns_empty_result_with_shortest_distance(self):
        graph = self.make_graph()

        # Add the longer route first; BFS must still report the shorter 4-hop path.
        for source_id, target_id in zip("axyzw", "xyzwe"):
            graph.add_relation(source_id, "next", target_id)
        for source_id, target_id in zip("abcd", "bcde"):
            graph.add_relation(source_id, "next", target_id)

        result = graph.multihop_query("a", "e", max_hops=3)

        self.assertEqual(result, QueryResult([], [], "", 0.0, 4))

    def test_missing_endpoints_keep_the_standard_empty_result(self):
        graph = self.make_graph()
        graph.add_entity("known", "person")

        for start_id, end_id in (
            ("missing", "known"),
            ("known", "missing"),
            ("missing", "other-missing"),
        ):
            with self.subTest(start_id=start_id, end_id=end_id):
                result = graph.multihop_query(start_id, end_id, max_hops=3)
                self.assertEqual(result, QueryResult([], [], "", 0.0, 0))

        self.assertEqual(set(graph.entity_index), {"known"})


if __name__ == "__main__":
    unittest.main()
