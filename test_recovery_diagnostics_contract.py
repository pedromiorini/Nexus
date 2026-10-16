import json
import unittest

from hypothesis import given, settings, strategies as st

from vita.nexus_constitutional_bridge_v3 import NexusConstitutionalBridge
from tools.diagnostics_dashboard import render_dashboard


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

    def test_failure_alert_includes_exact_threshold(self):
        record = self.bridge.record_reprocessing_telemetry({
            "attempts": 2, "failed": 1, "discarded": 0, "total_latency_ms": 0.0,
        })
        self.assertIn("reprocessing_failure_rate_high", record["alerts"])

    def test_policy_audit_is_returned_in_chronological_order(self):
        self.bridge.record_policy_transition("active", "paused", "first", {})
        self.bridge.record_policy_transition("paused", "active", "second", {})
        events = self.bridge.get_policy_audit()
        self.assertEqual([event["payload"]["reason"] for event in events], ["first", "second"])

    def test_dashboard_renders_observable_recovery_metrics(self):
        snapshot = self.bridge.export_recovery_diagnostics()
        rendered = render_dashboard(snapshot)
        self.assertIn("NEXUS RECOVERY DASHBOARD", rendered)
        self.assertIn("schema: nexus.recovery.diagnostics.v1", rendered)
        self.assertIn("events_analyzed: 0", rendered)

    def test_dashboard_rejects_invalid_snapshot(self):
        with self.assertRaises(ValueError):
            render_dashboard({"schema": "nexus.recovery.diagnostics.v9"})

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

    def test_diagnostic_field_types_are_validated_without_raising(self):
        cases = (
            ("diagnostics", "severity", [], "invalid_severity"),
            (None, "generated_at", "yesterday", "invalid_generated_at"),
            ("diagnostics", "severity_counts", [], "invalid_severity_counts"),
            ("diagnostics", "events_analyzed", "1", "invalid_events_analyzed"),
            ("diagnostics", "recovery_rate", float("inf"), "invalid_recovery_rate"),
        )
        for parent, field, value, expected_error in cases:
            payload = self.bridge.export_recovery_diagnostics()
            target = payload if parent is None else payload[parent]
            target[field] = value
            with self.subTest(field=field, value=value):
                result = self.bridge.validate_recovery_diagnostics(payload)
                self.assertFalse(result["valid"])
                self.assertIn(expected_error, result["errors"])

        nonstandard_json = self.bridge.export_recovery_diagnostics()
        nonstandard_json["generated_at"] = float("nan")
        result = self.bridge.validate_recovery_diagnostics(json.dumps(nonstandard_json))
        self.assertFalse(result["valid"])
        self.assertEqual(result["errors"], ["invalid_json"])

        payload = self.bridge.export_recovery_diagnostics()
        payload["diagnostics"]["severity"] = []
        with self.assertRaisesRegex(ValueError, "invalid_severity"):
            render_dashboard(payload)

    def test_nonfinite_metrics_are_rejected_and_json_export_is_strict(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(total_latency_ms=value), self.assertRaisesRegex(ValueError, "finite"):
                self.bridge.record_reprocessing_telemetry({
                    "attempts": 1,
                    "failed": 0,
                    "discarded": 0,
                    "total_latency_ms": value,
                })

        self.assertEqual(self.bridge.get_policy_audit(), [])

        def reject_nonfinite(value):
            raise ValueError(f"non-standard JSON constant: {value}")

        encoded = self.bridge.export_recovery_diagnostics(as_json=True)
        decoded = json.loads(encoded, parse_constant=reject_nonfinite)
        self.assertTrue(self.bridge.validate_recovery_diagnostics(decoded)["valid"])

    @settings(max_examples=60, deadline=None)
    @given(
        st.recursive(
            st.none()
            | st.booleans()
            | st.integers()
            | st.floats(allow_nan=False, allow_infinity=False)
            | st.text(max_size=32),
            lambda children: st.lists(children, max_size=4)
            | st.dictionaries(st.text(max_size=16), children, max_size=4),
            max_leaves=16,
        )
    )
    def test_arbitrary_json_values_never_escape_validator_contract(self, payload):
        result = self.bridge.validate_recovery_diagnostics(payload)
        self.assertIsInstance(result, dict)
        self.assertIn("valid", result)
        self.assertIn("compatible", result)
        self.assertIsInstance(result["valid"], bool)
        self.assertIsInstance(result["compatible"], bool)
        self.assertIsInstance(result["errors"], list)


if __name__ == "__main__":
    unittest.main()
