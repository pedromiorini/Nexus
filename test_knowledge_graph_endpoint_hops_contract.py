import importlib.util
import sys
import unittest
from unittest.mock import patch

from core.constitutional_brain import QueryResult, RealKnowledgeGraph


class KnowledgeGraphPathContractMixin:
    def make_graph(self):
        raise NotImplementedError

    def test_missing_endpoint_returns_the_standard_empty_result(self):
        graph = self.make_graph()
        graph.add_entity("known", "person")

        for start_id, end_id in (
            ("missing", "known"),
            ("known", "missing"),
            ("missing", "other-missing"),
            ("missing", "missing"),
        ):
            for max_hops in (0, 3):
                with self.subTest(start_id=start_id, end_id=end_id, max_hops=max_hops):
                    result = graph.multihop_query(start_id, end_id, max_hops=max_hops)
                    self.assertEqual(result, QueryResult([], [], "", 0.0, 0))

        self.assertEqual(graph.total_queries, 8)
        self.assertEqual(graph.total_multihop_queries, 8)
        self.assertEqual(set(graph.entity_index), {"known"})
        self.assertEqual(graph.total_entities, 1)
        self.assertEqual(graph.total_relations, 0)

    def test_rejected_over_budget_path_reports_zero_hops(self):
        graph = self.make_graph()
        for source_id, target_id in zip("abcd", "bcde"):
            graph.add_relation(source_id, "next", target_id)

        result = graph.multihop_query("a", "e", max_hops=3)
        self.assertEqual(result, QueryResult([], [], "", 0.0, 0))

    def test_path_at_hop_limit_is_preserved(self):
        graph = self.make_graph()
        graph.add_relation("a", "next", "b")

        result = graph.multihop_query("a", "b", max_hops=1)
        self.assertEqual([entity.id for entity in result.entities], ["a", "b"])
        self.assertEqual(result.paths, [["a", "b"]])
        self.assertEqual(result.context, "a --[next]--> b")
        self.assertEqual(result.hops, 1)
        self.assertGreater(result.confidence, 0.0)

    def test_existing_self_path_with_zero_budget_is_preserved(self):
        graph = self.make_graph()
        graph.add_entity("known", "person", confidence=0.9)

        result = graph.multihop_query("known", "known", max_hops=0)
        self.assertEqual([entity.id for entity in result.entities], ["known"])
        self.assertEqual(result.paths, [["known"]])
        self.assertEqual(result.hops, 0)
        self.assertGreater(result.confidence, 0.0)

    def test_existing_two_hop_path_is_preserved(self):
        graph = self.make_graph()
        graph.add_relation("a", "next", "b")
        graph.add_relation("b", "next", "c")

        result = graph.multihop_query("a", "c", max_hops=2)

        self.assertEqual([entity.id for entity in result.entities], ["a", "b", "c"])
        self.assertEqual(result.paths, [["a", "b", "c"]])
        self.assertEqual(result.context, "a --[next]--> b | b --[next]--> c")
        self.assertEqual(result.hops, 2)
        self.assertGreater(result.confidence, 0.0)

    def test_existing_disconnected_endpoints_return_standard_empty_result(self):
        graph = self.make_graph()
        graph.add_entity("a", "person")
        graph.add_entity("b", "person")

        result = graph.multihop_query("a", "b")

        self.assertEqual(set(graph.entity_index), {"a", "b"})
        self.assertEqual(result, QueryResult([], [], "", 0.0, 0))


class KnowledgeGraphFallbackPathContractTests(
    KnowledgeGraphPathContractMixin, unittest.TestCase
):
    def make_graph(self):
        with patch.dict(sys.modules, {"networkx": None}):
            graph = RealKnowledgeGraph()
        self.assertIsNone(graph.nx)
        return graph


@unittest.skipUnless(
    importlib.util.find_spec("networkx"), "networkx opcional ausente"
)
class KnowledgeGraphNetworkXPathContractTests(
    KnowledgeGraphPathContractMixin, unittest.TestCase
):
    def make_graph(self):
        graph = RealKnowledgeGraph()
        self.assertIsNotNone(graph.nx)
        return graph


if __name__ == "__main__":
    unittest.main()
