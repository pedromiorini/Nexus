"""Characterization tests for the legacy POST 62 system-health API."""
from contextlib import redirect_stdout
from io import StringIO
import unittest

from core.constitutional_brain import (
    DiagnosticReport,
    HealthIndicator,
    RealSystemHealthDiagnosticsEngine,
    SystemAlert,
)


class SystemHealthDiagnosticsContractTests(unittest.TestCase):
    def make_engine(self):
        with redirect_stdout(StringIO()):
            return RealSystemHealthDiagnosticsEngine()

    def test_empty_metrics_are_healthy_without_alert(self):
        engine = self.make_engine()

        indicator = engine.check_component_health("empty", {})

        self.assertIsInstance(indicator, HealthIndicator)
        self.assertEqual(indicator.status, "healthy")
        self.assertEqual(indicator.health_score, 1.0)
        self.assertEqual(engine.total_checks, 1)
        self.assertEqual(engine.get_active_alerts(), [])

    def test_error_metric_inversion_and_health_thresholds(self):
        engine = self.make_engine()

        healthy = engine.check_component_health("healthy", {"error_rate": 0.2})
        warning = engine.check_component_health("warning", {"availability": 0.5})
        critical = engine.check_component_health("critical", {"availability": 0.49})

        self.assertEqual((healthy.health_score, healthy.status), (0.8, "healthy"))
        self.assertEqual((warning.health_score, warning.status), (0.5, "warning"))
        self.assertEqual(critical.status, "critical")
        alerts = engine.get_active_alerts()
        self.assertTrue(all(isinstance(alert, SystemAlert) for alert in alerts))
        self.assertEqual([alert.severity for alert in alerts], ["warning", "critical"])

    def test_full_diagnostics_preserves_report_recommendations_and_statistics(self):
        engine = self.make_engine()

        report = engine.run_full_diagnostics(
            {
                "api": {"availability": 1.0},
                "database": {"error_rate": 0.8},
            }
        )

        self.assertIsInstance(report, DiagnosticReport)
        self.assertAlmostEqual(report.overall_health, 0.6)
        self.assertEqual(
            [item.status for item in report.indicators], ["healthy", "critical"]
        )
        self.assertEqual(
            report.recommendations,
            ["URGENT: Investigate database immediately (score: 0.20)"],
        )
        self.assertEqual(engine.get_statistics()["total_health_checks"], 2)
        self.assertEqual(engine.get_statistics()["total_diagnostic_reports"], 1)
        self.assertAlmostEqual(engine.get_system_health_summary()["overall_health"], 0.6)

    def test_filtered_alerts_are_returned_as_an_independent_list(self):
        engine = self.make_engine()
        engine.check_component_health("api", {"availability": 0.5})

        filtered = engine.get_active_alerts("warning")
        filtered.clear()

        self.assertEqual(len(engine.get_active_alerts("warning")), 1)


if __name__ == "__main__":
    unittest.main()
