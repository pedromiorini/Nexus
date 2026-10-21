import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings, strategies as st

from vita.nexus_constitutional_bridge_v3 import NexusConstitutionalBridge
from tools.diagnostics_dashboard import load_snapshot, main, render_dashboard


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

    def _assert_dashboard_deep_json(self, source, *, direct=False):
        # Exceed Python and C decoder limits (which differ on Python 3.12).
        # This fixture does not define an application depth policy.
        depth = max(10000, sys.getrecursionlimit() + 100)
        text = "[" * depth + "0" + "]" * depth
        if source != "-":
            Path(source).write_text(text, encoding="utf-8")
        with patch("sys.stdin", io.StringIO(text)):
            if direct:
                with self.assertRaises(RecursionError):
                    load_snapshot(source)
                return
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(main([source]), 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertTrue(stderr.getvalue().startswith("dashboard error:"))
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_dashboard_file_cli_reports_excessive_nesting(self):
        with tempfile.TemporaryDirectory() as directory:
            self._assert_dashboard_deep_json(str(Path(directory) / "snapshot.json"))

    def test_dashboard_stdin_cli_reports_excessive_nesting(self):
        self._assert_dashboard_deep_json("-")

    def test_dashboard_file_loader_preserves_recursion_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self._assert_dashboard_deep_json(str(Path(directory) / "snapshot.json"), direct=True)

    def test_dashboard_stdin_loader_preserves_recursion_error(self):
        self._assert_dashboard_deep_json("-", direct=True)

    def _assert_dashboard_shallow_json(self, source):
        snapshot = self.bridge.export_recovery_diagnostics()
        snapshot["events"] = [{
            "event_type": "probe", "timestamp": 1.0,
            "payload": {"nested": [None, True, False, 0, -7, 1.25, "ação", {"key": "value"}]},
        }]
        text = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
        if source != "-":
            Path(source).write_text(text, encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("sys.stdin", io.StringIO(text)):
            self.assertEqual(load_snapshot(source), snapshot)
        with patch("sys.stdin", io.StringIO(text)), redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(main([source]), 0)
        self.assertEqual(stdout.getvalue(), render_dashboard(snapshot) + "\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_dashboard_file_preserves_valid_shallow_json(self):
        with tempfile.TemporaryDirectory() as directory:
            self._assert_dashboard_shallow_json(str(Path(directory) / "snapshot.json"))

    def test_dashboard_stdin_preserves_valid_shallow_json(self):
        self._assert_dashboard_shallow_json("-")

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

    def _snapshot_json_with_number(self, token, depth=0):
        snapshot = self.bridge.export_recovery_diagnostics()
        nested = '"NUMBER_TOKEN"'
        for _ in range(depth):
            nested = '{"nested":[' + nested + ']}'
        snapshot["events"] = [{
            "event_type": "numeric_control", "timestamp": 1.0,
            "payload": {"value": "NESTED_TOKEN"},
        }]
        return json.dumps(snapshot).replace('"NESTED_TOKEN"', nested).replace('"NUMBER_TOKEN"', token)

    @contextmanager
    def _snapshot_source(self, encoded, stdin):
        if stdin:
            with patch("sys.stdin", io.StringIO(encoded)):
                yield "-"
        else:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "snapshot.json"
                path.write_text(encoded, encoding="utf-8")
                yield str(path)

    def _run_dashboard(self, encoded, stdin):
        with self._snapshot_source(encoded, stdin) as source:
            return subprocess.run(
                [sys.executable, "-m", "tools.diagnostics_dashboard", source],
                input=encoded if stdin else None, capture_output=True, text=True,
                cwd=Path(__file__).resolve().parent, timeout=15, check=False,
            )

    def test_dashboard_loaders_reject_nested_float_overflow(self):
        for token in ("1e400", "-1e400"):
            for depth in (0, 1, 32):
                for stdin in (False, True):
                    with self.subTest(token=token, depth=depth, stdin=stdin):
                        encoded = self._snapshot_json_with_number(token, depth)
                        with self._snapshot_source(encoded, stdin) as source:
                            with self.assertRaisesRegex(ValueError, "finite"):
                                load_snapshot(source)

    def test_dashboard_cli_rejects_nested_float_overflow_without_output(self):
        for token in ("1e400", "-1e400"):
            for depth in (0, 32):
                for stdin in (False, True):
                    with self.subTest(token=token, depth=depth, stdin=stdin):
                        result = self._run_dashboard(self._snapshot_json_with_number(token, depth), stdin)
                        self.assertEqual(result.returncode, 2)
                        self.assertEqual(result.stdout, "")
                        self.assertIn("dashboard error:", result.stderr)
                        self.assertIn("finite", result.stderr)
                        self.assertNotIn("Traceback", result.stderr)

    def test_dashboard_loaders_reject_nonstandard_numeric_constants(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            for stdin in (False, True):
                with self.subTest(token=token, stdin=stdin):
                    encoded = self._snapshot_json_with_number(token, 2)
                    with self._snapshot_source(encoded, stdin) as source:
                        with self.assertRaisesRegex(ValueError, "non-standard JSON constant"):
                            load_snapshot(source)

    def test_dashboard_cli_rejects_nonstandard_numeric_constants(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            for stdin in (False, True):
                with self.subTest(token=token, stdin=stdin):
                    result = self._run_dashboard(self._snapshot_json_with_number(token, 2), stdin)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("dashboard error:", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_dashboard_loaders_preserve_finite_numbers_and_large_integers(self):
        tokens = ("0.0", "-0.0", "1.25", "-2.5e-3", "5e-324",
                  "1.7976931348623157e308", "-1.7976931348623157e308", "9" * 1000)
        for token in tokens:
            for stdin in (False, True):
                with self.subTest(token=token, stdin=stdin):
                    encoded = self._snapshot_json_with_number(token, 2)
                    with self._snapshot_source(encoded, stdin) as source:
                        snapshot = load_snapshot(source)
                    value = snapshot["events"][0]["payload"]["value"]["nested"][0]["nested"][0]
                    expected = json.loads(token)
                    self.assertIs(type(value), type(expected))
                    if isinstance(expected, float):
                        self.assertEqual(value.hex(), expected.hex())
                    else:
                        self.assertEqual(value, expected)
                    self.assertIn("NEXUS RECOVERY DASHBOARD", render_dashboard(snapshot))

    def test_dashboard_cli_accepts_finite_boundary_and_common_numbers(self):
        for token in ("1.7976931348623157e308", "-1.7976931348623157e308", "1.25", "-2.5e-3"):
            for stdin in (False, True):
                with self.subTest(token=token, stdin=stdin):
                    result = self._run_dashboard(self._snapshot_json_with_number(token, 2), stdin)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("NEXUS RECOVERY DASHBOARD", result.stdout)
                    self.assertEqual(result.stderr, "")

    def test_dashboard_numeric_tokens_beyond_decoder_depth_fail_operationally(self):
        # Numeric hooks do not replace the native decoder's recursion boundary.
        depth = max(10000, sys.getrecursionlimit() + 100)
        for token in ("1.25", "1e400", "-1e400"):
            encoded = "[" * depth + token + "]" * depth
            for stdin in (False, True):
                with self.subTest(token=token, stdin=stdin):
                    with self._snapshot_source(encoded, stdin) as source:
                        with self.assertRaises(RecursionError):
                            load_snapshot(source)
                    result = self._run_dashboard(encoded, stdin)
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("dashboard error:", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

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
