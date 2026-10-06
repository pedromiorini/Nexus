import concurrent.futures
import unittest

from vita.nexus_constitutional_bridge_v3 import NexusConstitutionalBridge


class VitaAuditConcurrencyContractTests(unittest.TestCase):
    def test_concurrent_audit_writes_are_serialized(self):
        bridge = NexusConstitutionalBridge(":memory:")
        workers = 24

        def record_events(index):
            bridge.record_policy_transition(
                "active", "degraded", f"worker-{index}", {"worker": index}
            )
            return bridge.record_reprocessing_telemetry(
                {"attempts": 2, "failed": 0, "discarded": 0, "total_latency_ms": index}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(record_events, range(workers)))

        self.assertEqual(len(records), workers)
        events = bridge.get_policy_audit(limit=workers * 2)
        self.assertEqual(len(events), workers * 2)
        self.assertEqual(
            sum(event["event_type"] == "policy_transition" for event in events), workers
        )
        self.assertEqual(
            sum(event["event_type"] == "telemetry" for event in events), workers
        )
        self.assertEqual(
            {event["payload"]["reason"] for event in events if event["event_type"] == "policy_transition"},
            {f"worker-{index}" for index in range(workers)},
        )
        self.assertEqual(bridge.get_recovery_analysis(limit=workers * 2)["events_analyzed"], workers * 2)
        bridge._audit_db.close()


if __name__ == "__main__":
    unittest.main()
