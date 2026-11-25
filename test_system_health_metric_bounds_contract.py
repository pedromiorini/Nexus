import unittest
from contextlib import redirect_stdout
from decimal import Decimal
from io import StringIO

from core.system_health_diagnostics import RealSystemHealthDiagnosticsEngine


class HealthMetricWithInvalidConversion(float):
    def __float__(self):
        return float("nan")


class SystemHealthMetricBoundsContractTests(unittest.TestCase):
    def make_engine(self):
        with redirect_stdout(StringIO()):
            return RealSystemHealthDiagnosticsEngine()

    def state(self, engine):
        return (
            engine.total_checks,
            engine.total_alerts,
            engine.total_reports,
            list(engine.reports),
            list(engine.indicators),
            list(engine.alerts),
            engine.avg_system_health,
        )

    def test_accepts_normalized_bounds_and_preserves_error_inversion(self):
        engine = self.make_engine()

        low = engine.check_component_health("low", {"availability": 0.0})
        high = engine.check_component_health("high", {"availability": 1})
        inverted = engine.check_component_health("errors", {"error_rate": 0.2})
        latency = engine.check_component_health("latency", {"latency_ms": 0.25})
        decimal = engine.check_component_health(
            "decimal", {"availability": Decimal("0.5")}
        )

        self.assertEqual((low.health_score, low.status), (0.0, "critical"))
        self.assertEqual((high.health_score, high.status), (1.0, "healthy"))
        self.assertEqual((inverted.health_score, inverted.status), (0.8, "healthy"))
        self.assertEqual((latency.health_score, latency.status), (0.75, "warning"))
        self.assertEqual((decimal.health_score, decimal.status), (0.5, "warning"))
        self.assertEqual(engine.total_checks, 5)

    def test_rejects_bad_values_before_mutating_component_state(self):
        invalid = (
            ("availability", True, TypeError),
            ("availability", "0.5", TypeError),
            ("availability", None, TypeError),
            ("availability", -0.01, ValueError),
            ("availability", 1.01, ValueError),
            ("availability", Decimal("1.0000000000000000000000000001"), ValueError),
            ("error_rate", -0.01, ValueError),
            ("error_rate", 1.01, ValueError),
            ("availability", float("nan"), ValueError),
            ("availability", float("inf"), ValueError),
            ("availability", float("-inf"), ValueError),
            ("availability", HealthMetricWithInvalidConversion(0.5), ValueError),
        )
        for name, value, error_type in invalid:
            with self.subTest(name=name, value=repr(value)):
                engine = self.make_engine()
                before = self.state(engine)

                with self.assertRaises(error_type):
                    engine.check_component_health("invalid", {name: value})

                self.assertEqual(self.state(engine), before)

    def test_full_diagnostics_prevalidates_all_components_before_mutating(self):
        engine = self.make_engine()
        engine.check_component_health("existing", {"availability": 0.5})
        before = self.state(engine)

        with self.assertRaises(ValueError):
            engine.run_full_diagnostics(
                {
                    "valid": {"availability": 1.0},
                    "invalid": {"availability": 1.01},
                }
            )

        self.assertEqual(self.state(engine), before)

    def test_empty_metrics_remain_healthy(self):
        indicator = self.make_engine().check_component_health("empty", {})

        self.assertEqual((indicator.health_score, indicator.status), (1.0, "healthy"))


if __name__ == "__main__":
    unittest.main()
