"""Contratos das interseções #38/#44, #39/#40 e #43/#44."""
import unittest
from unittest.mock import patch

from core.constitutional_brain import RealKnowledgeGraph
from core.deferred_task_queue import DeferredTaskQueue
from core.temporal_facts import TemporalFactStore


class ConsolidatedPRContract(unittest.TestCase):
    def test_replayed_temporal_store_preserves_graph_preflight_in_both_backends(self):
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                store = TemporalFactStore(functional_predicates={"lives_in"}, clock=lambda: 100.0)
                store.assert_fact("ana", "lives_in", "A", valid_from=0.0)
                restored = TemporalFactStore.from_snapshot(
                    store.export_snapshot(as_json=True), expected_head=store.head_hash,
                    clock=lambda: 200.0,
                )
                with patch.dict("sys.modules", {"networkx": None}) if fallback else patch.dict("sys.modules", {}):
                    graph = RealKnowledgeGraph(temporal_store=restored)
                self.assertEqual(graph.nx is None, fallback)
                before = restored.export_snapshot()
                with self.assertRaises(ValueError):
                    graph.add_relation("ana", "lives_in", "B", valid_from=float("nan"))
                self.assertEqual(restored.export_snapshot(), before)
                self.assertEqual(graph.entity_index, {})
                self.assertEqual(graph.relation_index, {})
                graph.add_relation("ana", "lives_in", "B", valid_from=10.0, source="test:composition")
                replay = TemporalFactStore.from_snapshot(
                    restored.export_snapshot(as_json=True), expected_head=restored.head_hash,
                )
                self.assertEqual(replay.history("ana", "lives_in"), restored.history("ana", "lives_in"))
                self.assertEqual(replay.check_invariants(), [])
                self.assertEqual([f.object for f in replay.get_current_facts("ana")], ["B"])

    def test_retry_snapshot_round_trip_preserves_redaction_and_detached_context(self):
        queue = DeferredTaskQueue(max_attempts=3)
        task = queue.enqueue("recover", {"nested": {"value": 1}})

        def fail(_task):
            raise RuntimeError("private-token-do-not-export")

        result = queue.consume(fail)
        self.assertEqual(result["retried"], 1)
        snapshot = queue.export_snapshot(as_json=True)
        self.assertNotIn("private-token-do-not-export", snapshot)
        restored = DeferredTaskQueue(max_attempts=3)
        restored.restore_snapshot(snapshot, replace=True)
        exported = restored.export_snapshot()
        self.assertEqual(exported["statistics"], queue.export_snapshot()["statistics"])
        pending = exported["tasks"][0]
        self.assertEqual(pending["task_id"], task.task_id)
        self.assertEqual(pending["attempts"], 1)
        self.assertEqual(pending["last_reason"], "RuntimeError: processor raised an exception")
        pending["context"]["nested"]["value"] = 999
        self.assertEqual(restored.export_snapshot()["tasks"][0]["context"]["nested"]["value"], 1)

    def test_missing_nodes_and_over_budget_queries_do_not_write_temporal_ledger(self):
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                store = TemporalFactStore(clock=lambda: 100.0)
                with patch.dict("sys.modules", {"networkx": None}) if fallback else patch.dict("sys.modules", {}):
                    graph = RealKnowledgeGraph(temporal_store=store)
                graph.add_relation("a", "next", "b")
                graph.add_relation("b", "next", "c")
                before = store.export_snapshot()
                self.assertEqual(graph.multihop_query("a", "c", max_hops=1).hops, 2)
                self.assertEqual(graph.multihop_query("missing", "missing").paths, [])
                self.assertEqual(store.export_snapshot(), before)
                self.assertEqual(graph.total_relations, 2)
                self.assertEqual(graph.total_queries, 2)


if __name__ == "__main__":
    unittest.main()
