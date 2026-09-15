import json
import time
import unittest
from core.deferred_task_queue import DeferredTaskQueue
from core.constitutional_brain import CentralRouter
from vita.nexus_constitutional_bridge_v3 import NexusConstitutionalBridge


class DeferredTaskQueueTests(unittest.TestCase):
    def test_priority_order(self):
        queue = DeferredTaskQueue()
        low = queue.enqueue("low", priority=1)
        high = queue.enqueue("high", priority=10)
        first = queue.recheck_and_pop(False)
        self.assertEqual(first.task_id, high.task_id)
        self.assertEqual(first.prompt, "high")
        self.assertEqual(first.attempts, 1)

    def test_critical_pressure_does_not_pop(self):
        queue = DeferredTaskQueue()
        queue.enqueue("blocked", priority=5)
        self.assertIsNone(queue.recheck_and_pop(True))
        self.assertEqual(queue.statistics()["depth"], 1)

    def test_capacity_limit(self):
        queue = DeferredTaskQueue(max_size=1)
        queue.enqueue("first")
        with self.assertRaises(OverflowError):
            queue.enqueue("second")
        self.assertEqual(queue.statistics()["discarded"], 1)

    def test_completion_and_stats(self):
        queue = DeferredTaskQueue()
        task = queue.enqueue("work", priority=3)
        ready = queue.recheck_and_pop(False)
        queue.complete(task.task_id)
        self.assertEqual(ready.task_id, task.task_id)
        self.assertEqual(queue.statistics()["retried"], 1)
        self.assertEqual(queue.statistics()["completed"], 1)

    def test_consume_batch_completes_tasks(self):
        queue = DeferredTaskQueue()
        queue.enqueue("a", priority=1)
        queue.enqueue("b", priority=2)
        seen = []
        result = queue.consume(lambda task: seen.append(task.prompt) or True, max_batch=2)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(seen, ["b", "a"])

    def test_consume_retries_then_discards(self):
        queue = DeferredTaskQueue(max_attempts=2)
        queue.enqueue("unstable")
        first = queue.consume(lambda task: False)
        second = queue.consume(lambda task: False)
        self.assertEqual(first["retried"], 1)
        self.assertEqual(second["discarded"], 1)
        self.assertEqual(queue.statistics()["depth"], 0)

    def test_consume_blocked_by_critical_pressure(self):
        queue = DeferredTaskQueue()
        queue.enqueue("blocked")
        result = queue.consume(lambda task: True, pressure_critical=True)
        self.assertTrue(result["blocked"])
        self.assertEqual(result["processed"], 0)

    def test_lifecycle_callbacks(self):
        queue = DeferredTaskQueue(max_attempts=1)
        queue.enqueue("ok")
        queue.enqueue("bad")
        success, failure, discarded = [], [], []
        result = queue.consume(
            lambda task: task.prompt == "ok",
            max_batch=2,
            on_success=lambda task: success.append(task.prompt),
            on_failure=lambda task: failure.append(task.prompt),
            on_discard=lambda task: discarded.append(task.prompt),
        )
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["discarded"], 1)
        self.assertEqual(success, ["ok"])
        self.assertEqual(failure, ["bad"])
        self.assertEqual(discarded, ["bad"])

    def test_router_reprocess_preserves_context_and_metrics(self):
        router = CentralRouter.__new__(CentralRouter)
        router.deferred_queue = DeferredTaskQueue(max_attempts=1)
        router.stats = {"deferred_reprocessing": {
            "attempts": 0, "completed": 0, "failed": 0,
            "discarded": 0, "total_latency_ms": 0.0,
        }, "reprocessing_policy": {
            "state": "active", "paused_until": 0.0,
            "cooldown_seconds": 30.0, "last_reason": "",
        }}
        router.vram_guard = type("Guard", (), {
            "evaluate": lambda self: type("Decision", (), {"action": "normal"})()
        })()
        router.vita_bridge = NexusConstitutionalBridge(":memory:")
        observed = []
        router.route = lambda prompt, context: observed.append((prompt, context)) or {"success": True}
        router.deferred_queue.enqueue("recover", {"trace_id": "t-1"})
        callbacks = []
        result = router.reprocess_deferred_tasks(
            on_success=lambda task: callbacks.append(task.task_id)
        )
        self.assertEqual(result["completed"], 1)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0], "recover")
        self.assertEqual(observed[0][1]["trace_id"], "t-1")
        self.assertTrue(observed[0][1]["_deferred_reprocess"])
        self.assertEqual(router.stats["deferred_reprocessing"]["attempts"], 1)
        self.assertEqual(router.stats["deferred_reprocessing"]["completed"], 1)
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(result["reprocessing_telemetry"]["status"], "nominal")
        diagnostics = router.get_recovery_diagnostics()
        self.assertIsNotNone(diagnostics)
        self.assertIn("severity", diagnostics)
        self.assertIn("events_analyzed", diagnostics)
        exported = router.export_recovery_diagnostics(as_json=True)
        decoded = json.loads(exported)
        self.assertEqual(decoded["schema"], "nexus.recovery.diagnostics.v1")
        self.assertIn("diagnostics", decoded)
        self.assertIn("events", decoded)

    def test_vita_telemetry_alerts_on_discard_rate(self):
        bridge = NexusConstitutionalBridge(":memory:")
        record = bridge.record_reprocessing_telemetry({
            "attempts": 2, "completed": 0, "failed": 2,
            "discarded": 1, "total_latency_ms": 2400.0,
        })
        self.assertEqual(record["status"], "alert")
        self.assertIn("reprocessing_failure_rate_high", record["alerts"])
        self.assertIn("reprocessing_discard_rate_high", record["alerts"])
        self.assertIn("reprocessing_latency_high", record["alerts"])
        self.assertEqual(len(bridge.reprocessing_telemetry), 1)

    def test_policy_transition_is_audited(self):
        bridge = NexusConstitutionalBridge(":memory:")
        event = bridge.record_policy_transition("active", "paused", "critical_vram", {"state": "paused"})
        audit = bridge.get_policy_audit()
        self.assertEqual(event["new_state"], "paused")
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["event_type"], "policy_transition")
        self.assertEqual(audit[0]["payload"]["reason"], "critical_vram")

    def test_recovery_analysis(self):
        bridge = NexusConstitutionalBridge(":memory:")
        now = 1000.0
        rows = [
            ("policy_transition", now, {"previous_state": "active", "new_state": "paused", "reason": "critical_vram", "policy": {}}),
            ("telemetry", now + 1, {"alerts": ["reprocessing_failure_rate_high"]}),
            ("policy_transition", now + 5, {"previous_state": "paused", "new_state": "active", "reason": "cooldown_expired", "policy": {}}),
        ]
        bridge._audit_db.executemany(
            "INSERT INTO policy_audit (event_type, timestamp, payload) VALUES (?, ?, ?)",
            [(kind, timestamp, __import__("json").dumps(payload)) for kind, timestamp, payload in rows],
        )
        bridge._audit_db.commit()
        analysis = bridge.get_recovery_analysis()
        self.assertEqual(analysis["pauses"], 1)
        self.assertEqual(analysis["recoveries"], 1)
        self.assertEqual(analysis["recovery_rate"], 1.0)
        self.assertEqual(analysis["critical_events"], 2)
        self.assertEqual(analysis["severity"], "critical")
        self.assertEqual(analysis["severity_counts"]["critical"], 2)
        self.assertEqual(analysis["total_pause_seconds"], 5.0)
        self.assertFalse(analysis["open_pause"])
        with self.assertRaises(ValueError):
            bridge.get_recovery_analysis(window_seconds=0)
        snapshot = bridge.export_recovery_diagnostics()
        self.assertEqual(snapshot["schema"], "nexus.recovery.diagnostics.v1")
        self.assertEqual(snapshot["diagnostics"]["severity"], "critical")
        self.assertEqual(len(snapshot["events"]), 3)

    def test_policy_blocks_critical_vram(self):
        router = CentralRouter.__new__(CentralRouter)
        router.deferred_queue = DeferredTaskQueue()
        router.stats = {"deferred_reprocessing": {
            "attempts": 0, "completed": 0, "failed": 0,
            "discarded": 0, "total_latency_ms": 0.0,
        }, "reprocessing_policy": {
            "state": "active", "paused_until": 0.0,
            "cooldown_seconds": 30.0, "last_reason": "",
        }}
        router.vram_guard = type("Guard", (), {
            "evaluate": lambda self: type("Decision", (), {"action": "emergency_release", "reason": "critical_vram"})()
        })()
        router.deferred_queue.enqueue("must_wait")
        result = router.reprocess_deferred_tasks()
        self.assertTrue(result["blocked"])
        self.assertEqual(result["reprocessing_policy"]["state"], "paused")
        self.assertEqual(router.deferred_queue.statistics()["depth"], 1)

    def test_degraded_policy_limits_batch(self):
        router = CentralRouter.__new__(CentralRouter)
        router.deferred_queue = DeferredTaskQueue()
        router.stats = {"deferred_reprocessing": {
            "attempts": 0, "completed": 0, "failed": 0,
            "discarded": 0, "total_latency_ms": 0.0,
        }, "reprocessing_policy": {
            "state": "degraded", "paused_until": 0.0,
            "cooldown_seconds": 30.0, "last_reason": "latency",
        }}
        router.vram_guard = type("Guard", (), {
            "evaluate": lambda self: type("Decision", (), {"action": "normal"})()
        })()
        router.vita_bridge = NexusConstitutionalBridge(":memory:")
        router.route = lambda prompt, context: {"success": True}
        router.deferred_queue.enqueue("one")
        router.deferred_queue.enqueue("two")
        result = router.reprocess_deferred_tasks(max_batch=2)
        self.assertEqual(result["processed"], 1)
        self.assertEqual(router.deferred_queue.statistics()["depth"], 1)


if __name__ == "__main__":
    unittest.main()
