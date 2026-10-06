import concurrent.futures
import tempfile
import unittest
from pathlib import Path

from vita.nexus_constitutional_bridge_v3 import NexusConstitutionalBridge


class VitaAuditCrossConnectionContractTests(unittest.TestCase):
    def test_two_bridge_instances_share_audit_file_without_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "audit.sqlite3")
            bridges = [NexusConstitutionalBridge(db_path), NexusConstitutionalBridge(db_path)]
            workers = 32

            def record_events(index):
                bridge = bridges[index % len(bridges)]
                bridge.record_policy_transition(
                    "active", "degraded", f"cross-worker-{index}", {"worker": index}
                )
                bridge.record_reprocessing_telemetry(
                    {"attempts": 1, "failed": 0, "discarded": 0, "total_latency_ms": index}
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(record_events, range(workers)))

            events = bridges[0].get_policy_audit(limit=workers * 2)
            self.assertEqual(len(events), workers * 2)
            self.assertEqual(
                sum(event["event_type"] == "policy_transition" for event in events), workers
            )
            self.assertEqual(
                sum(event["event_type"] == "telemetry" for event in events), workers
            )
            self.assertEqual(
                {
                    event["payload"]["reason"]
                    for event in events
                    if event["event_type"] == "policy_transition"
                },
                {f"cross-worker-{index}" for index in range(workers)},
            )
            for bridge in bridges:
                bridge._audit_db.close()


if __name__ == "__main__":
    unittest.main()
