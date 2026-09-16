import json
import unittest

from vita.nexus_constitutional_bridge_v3 import NexusConstitutionalBridge


class RecoveryDiagnosticsContractTests(unittest.TestCase):
    def setUp(self):
        self.bridge = NexusConstitutionalBridge(":memory:")

    def test_export_has_versioned_contract(self):
        payload = self.bridge.export_recovery_diagnostics()
        self.assertEqual(payload["schema"], "nexus.recovery.diagnostics.v1")
        self.assertIsInstance(payload["generated_at"], float)
        self.assertEqual(payload["source"], "NexusConstitutionalBridge")
        self.assertIsInstance(payload["diagnostics"], dict)
        self.assertIsInstance(payload["events"], list)

    def test_json_export_round_trips_and_validates(self):
        encoded = self.bridge.export_recovery_diagnostics(as_json=True)
        decoded = json.loads(encoded)
        result = self.bridge.validate_recovery_diagnostics(encoded)
        self.assertEqual(decoded["schema"], "nexus.recovery.diagnostics.v1")
        self.assertTrue(result["valid"])
        self.assertTrue(result["compatible"])
        self.assertEqual(result["errors"], [])

    def test_required_diagnostics_invariants(self):
        diagnostics = self.bridge.export_recovery_diagnostics()["diagnostics"]
        required = {"severity", "severity_counts", "events_analyzed", "recovery_rate"}
        self.assertTrue(required.issubset(diagnostics))
        self.assertIn(diagnostics["severity"], {"info", "warning", "critical"})
        self.assertIsInstance(diagnostics["severity_counts"], dict)
        self.assertGreaterEqual(diagnostics["events_analyzed"], 0)
        self.assertGreaterEqual(diagnostics["recovery_rate"], 0.0)

    def test_incompatible_version_is_rejected(self):
        payload = self.bridge.export_recovery_diagnostics()
        payload["schema"] = "nexus.recovery.diagnostics.v2"
        result = self.bridge.validate_recovery_diagnostics(payload)
        self.assertFalse(result["valid"])
        self.assertFalse(result["compatible"])
        self.assertIn("unsupported_schema", result["errors"])

    def test_missing_fields_are_reported(self):
        result = self.bridge.validate_recovery_diagnostics({"schema": "nexus.recovery.diagnostics.v1"})
        self.assertFalse(result["valid"])
        for field in ("generated_at", "source", "diagnostics", "events"):
            self.assertIn(f"missing_{field}", result["errors"])

    def test_malformed_payloads_are_rejected(self):
        self.assertFalse(self.bridge.validate_recovery_diagnostics("not-json")["valid"])
        self.assertFalse(self.bridge.validate_recovery_diagnostics([])["valid"])
        payload = self.bridge.export_recovery_diagnostics()
        payload["diagnostics"] = []
        result = self.bridge.validate_recovery_diagnostics(payload)
        self.assertFalse(result["valid"])
        self.assertIn("diagnostics_must_be_object", result["errors"])


if __name__ == "__main__":
    unittest.main()

