import importlib.util
import unittest
from unittest.mock import patch

from core.constitutional_brain import RealKnowledgeGraph


class MultihopContractMixin:
    use_networkx = False

    def setUp(self):
        if self.use_networkx:
            self.graph = RealKnowledgeGraph()
            self.assertIsNotNone(self.graph.nx)
        else:
            with patch.dict("sys.modules", {"networkx": None}):
                self.graph = RealKnowledgeGraph()
            self.assertIsNone(self.graph.nx)
        self.original_graph = self.graph.graph
        self.original_nx = self.graph.nx

    def tearDown(self):
        self.assertIs(self.graph.graph, self.original_graph)
        self.assertIs(self.graph.nx, self.original_nx)

    def add_chain(self):
        for source, target in zip("abcd", "bcde"):
            self.graph.add_relation(source, "next", target)

    def assert_empty_result(self, result, hops):
        self.assertEqual(result.entities, [])
        self.assertEqual(result.paths, [])
        self.assertEqual(result.context, "")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.hops, hops)

    def test_over_budget_reports_real_distance_without_returning_path(self):
        self.add_chain()
        for budget in (0, 1, 3):
            with self.subTest(max_hops=budget):
                result = self.graph.multihop_query("a", "e", max_hops=budget)
                self.assert_empty_result(result, 4)
        self.assertEqual(self.graph.total_queries, 3)
        self.assertEqual(self.graph.total_multihop_queries, 3)

    def test_over_budget_reports_shortest_distance_with_cycle(self):
        self.add_chain()
        self.graph.add_relation("c", "next", "a")
        self.graph.add_relation("a", "shortcut", "d")
        result = self.graph.multihop_query("a", "e", max_hops=1)
        self.assert_empty_result(result, 2)

    def test_path_within_and_at_limit_preserves_context_and_confidence(self):
        self.graph.add_entity("a", "concept", confidence=0.8)
        self.graph.add_entity("b", "concept", confidence=0.4)
        self.graph.add_relation("a", "next", "b")
        for budget in (1, 3):
            with self.subTest(max_hops=budget):
                result = self.graph.multihop_query("a", "b", max_hops=budget)
                self.assertEqual([entity.id for entity in result.entities], ["a", "b"])
                self.assertEqual(result.paths, [["a", "b"]])
                self.assertEqual(result.context, "a --[next]--> b")
                self.assertEqual(result.hops, 1)
                self.assertAlmostEqual(result.confidence, 0.6 if self.use_networkx else 0.7)

    def test_same_existing_node_with_zero_budget(self):
        self.graph.add_entity("a", "concept", confidence=0.8)
        result = self.graph.multihop_query("a", "a", max_hops=0)
        self.assertEqual([entity.id for entity in result.entities], ["a"])
        self.assertEqual(result.paths, [["a"]])
        self.assertEqual(result.context, "")
        self.assertEqual(result.hops, 0)
        self.assertEqual(result.confidence, 0.8 if self.use_networkx else 0.7)

    def test_disconnected_and_reverse_paths_still_report_zero(self):
        self.add_chain()
        self.graph.add_entity("isolated", "concept")
        for target, start in (("isolated", "a"), ("a", "e")):
            with self.subTest(start=start, target=target):
                self.assert_empty_result(self.graph.multihop_query(start, target), 0)

    def test_cycle_without_path_still_reports_zero(self):
        self.graph.add_relation("a", "next", "b")
        self.graph.add_relation("b", "next", "a")
        self.graph.add_entity("isolated", "concept")
        self.assert_empty_result(self.graph.multihop_query("a", "isolated"), 0)

    def test_missing_node_contract_matches_explicit_empty_result(self):
        self.graph.add_entity("a", "concept")
        for start, end in (("missing", "a"), ("a", "missing"), ("missing", "missing")):
            with self.subTest(start=start, end=end):
                self.assert_empty_result(self.graph.multihop_query(start, end), 0)
        self.assertEqual(self.graph.total_queries, 3)
        self.assertEqual(self.graph.total_multihop_queries, 3)
        self.assertEqual(set(self.graph.entity_index), {"a"})


class FallbackMultihopContractTests(MultihopContractMixin, unittest.TestCase):
    pass


@unittest.skipUnless(importlib.util.find_spec("networkx"), "networkx opcional ausente")
class NetworkXMultihopContractTests(MultihopContractMixin, unittest.TestCase):
    use_networkx = True


if __name__ == "__main__":
    unittest.main()
