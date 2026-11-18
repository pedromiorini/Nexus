"""Contrato dos contadores de telemetria Vita: bool, float e texto não viram inteiro.

Observação em main 5da85e171a979eafdf944d7c5617b76ff8fdecdf (Python 3.12.3):
record_reprocessing_telemetry faz int() em attempts/failed/discarded e float()
em total_latency_ms.

- attempts=True grava a taxa com 1 tentativa e uma linha de auditoria.
- attempts=1.5 vira 1; attempts="2" vira 2.
- attempts=float("inf") levanta OverflowError, sem mensagem de contrato.
- attempts=-1 é aceito e a taxa fica negativa.
- total_latency_ms=True vira 1.0; total_latency_ms="10" vira 10.0.
- Uma subclasse de int negativa cujo __int__/__index__ devolvem 3 é aceita como 3.
- Uma subclasse de float negativa cujo __float__ devolve 9.0 é aceita como 9.0.
- Inteiro exato, omissão e latência finita não negativa continuam aceitos.
- NaN, +inf e -inf em total_latency_ms já levantam ValueError com "finite".

Hipóteses:
- H1: bool é um contador de propósito (True=1). Falsifica se True passar a
  TypeError e o produtor só incrementar inteiros com += 1.
- H2: float e str numérica são normalização aceitável. Falsifica se o
  anti-padrão constitucional e o limite da auditoria já recusam esses tipos.
- H3: OverflowError de infinito já é falha fechada. Falsifica se o tipo da
  exceção não é o do contrato.
- H4: contagem negativa é taxa negativa honesta. Falsifica se o produtor
  só usa inteiros >= 0 e a taxa negativa não é um alerta documentado.
- H5: __int__/__float__ de subclasse pode reescrever o valor. Falsifica se
  int.__index__/float.__float__ lerem o valor real, como no limite da #193.

Critérios dos três casos, escritos antes da correção:
- origem: True, 1.5, "2", inf e -1 não gravam auditoria.
- anterior: omissão, inteiros >= 0 e latência finita seguem o cálculo atual;
  NaN de latência segue ValueError com "finite".
- incompatível: subclasse que mente em __int__/__float__ não publica o valor mentido.
"""

import math
import unittest

from hypothesis import given, settings, strategies as st

from vita.nexus_constitutional_bridge_v3 import NexusConstitutionalBridge


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class LyingCount(int):
    def __int__(self):
        return 3

    def __index__(self):
        return 3


class LyingLatency(float):
    def __float__(self):
        return 9.0


class VitaTelemetryCounterContractTests(unittest.TestCase):
    def setUp(self):
        self.bridge = NexusConstitutionalBridge(":memory:")
        self.addCleanup(self.bridge.close)

    def _fresh(self):
        bridge = NexusConstitutionalBridge(":memory:")
        self.addCleanup(bridge.close)
        return bridge

    def test_exact_nonnegative_counts_and_latency_stay_recorded(self):
        record = self.bridge.record_reprocessing_telemetry({
            "attempts": 4,
            "completed": 2,
            "failed": 2,
            "discarded": 1,
            "total_latency_ms": 1000,
        })
        self.assertEqual(record["failure_rate"], 0.5)
        self.assertEqual(record["discard_rate"], 0.25)
        self.assertEqual(record["avg_latency_ms"], 250.0)
        self.assertEqual(record["status"], "alert")
        self.assertIn("reprocessing_failure_rate_high", record["alerts"])
        self.assertIn("reprocessing_discard_rate_high", record["alerts"])
        self.assertNotIn("reprocessing_latency_high", record["alerts"])
        self.assertEqual(record["completed"], 2)
        audit = self.bridge.get_policy_audit()
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["event_type"], "telemetry")
        self.assertEqual(audit[0]["payload"]["attempts"], 4)

    def test_omitted_counters_stay_zero_without_alert(self):
        record = self.bridge.record_reprocessing_telemetry({})
        self.assertEqual(record["failure_rate"], 0.0)
        self.assertEqual(record["discard_rate"], 0.0)
        self.assertEqual(record["avg_latency_ms"], 0.0)
        self.assertEqual(record["alerts"], [])
        self.assertEqual(record["status"], "nominal")
        self.assertEqual(len(self.bridge.get_policy_audit()), 1)

    def test_integer_subclass_keeps_its_exact_value(self):
        record = self.bridge.record_reprocessing_telemetry({
            "attempts": IntSubclass(4),
            "failed": IntSubclass(2),
            "discarded": IntSubclass(0),
            "total_latency_ms": FloatSubclass(1000.0),
        })
        self.assertEqual(record["failure_rate"], 0.5)
        self.assertEqual(record["discard_rate"], 0.0)
        self.assertEqual(record["avg_latency_ms"], 250.0)
        self.assertEqual(record["status"], "alert")
        self.assertEqual(record["alerts"], ["reprocessing_failure_rate_high"])

    def test_nonfinite_and_huge_latency_stay_value_error_without_audit(self):
        cases = (
            float("nan"),
            float("inf"),
            float("-inf"),
            10 ** 400,
            -1,
            -0.5,
        )
        for value in cases:
            with self.subTest(total_latency_ms=value):
                bridge = self._fresh()
                with self.assertRaisesRegex(ValueError, "finite"):
                    bridge.record_reprocessing_telemetry({
                        "attempts": 1,
                        "failed": 0,
                        "discarded": 0,
                        "total_latency_ms": value,
                    })
                self.assertEqual(bridge.get_policy_audit(), [])
                self.assertEqual(bridge.reprocessing_telemetry, [])

    def test_invalid_counters_do_not_record(self):
        cases = (
            ("attempts_true", {"attempts": True}, TypeError, "inteiro"),
            ("attempts_false", {"attempts": False}, TypeError, "inteiro"),
            ("failed_true", {"failed": True}, TypeError, "inteiro"),
            ("discarded_false", {"discarded": False}, TypeError, "inteiro"),
            ("attempts_float", {"attempts": 1.5}, TypeError, "inteiro"),
            ("failed_one_point_zero", {"failed": 1.0}, TypeError, "inteiro"),
            ("discarded_text", {"discarded": "1"}, TypeError, "inteiro"),
            ("attempts_empty", {"attempts": ""}, TypeError, "inteiro"),
            ("attempts_none", {"attempts": None}, TypeError, "inteiro"),
            ("attempts_nan", {"attempts": float("nan")}, TypeError, "inteiro"),
            ("attempts_inf", {"attempts": float("inf")}, TypeError, "inteiro"),
            ("failed_ninf", {"failed": float("-inf")}, TypeError, "inteiro"),
            ("attempts_negative", {"attempts": -1}, ValueError, "não negativo"),
            ("failed_negative", {"failed": -1}, ValueError, "não negativo"),
            ("discarded_negative", {"discarded": -1}, ValueError, "não negativo"),
        )
        for name, patch, exc_type, fragment in cases:
            with self.subTest(case=name):
                bridge = self._fresh()
                metrics = {
                    "attempts": 2,
                    "failed": 0,
                    "discarded": 0,
                    "total_latency_ms": 0.0,
                }
                metrics.update(patch)
                with self.assertRaisesRegex(exc_type, fragment):
                    bridge.record_reprocessing_telemetry(metrics)
                self.assertEqual(bridge.get_policy_audit(), [])
                self.assertEqual(bridge.reprocessing_telemetry, [])
                follow = bridge.record_reprocessing_telemetry({
                    "attempts": 1,
                    "failed": 0,
                    "discarded": 0,
                    "total_latency_ms": 0.0,
                })
                self.assertEqual(follow["failure_rate"], 0.0)
                self.assertEqual(len(bridge.get_policy_audit()), 1)

    def test_invalid_latency_does_not_record(self):
        cases = (
            ("latency_true", True, TypeError, "número"),
            ("latency_false", False, TypeError, "número"),
            ("latency_text", "10", TypeError, "número"),
            ("latency_empty", "", TypeError, "número"),
            ("latency_none", None, TypeError, "número"),
            ("latency_list", [1.0], TypeError, "número"),
        )
        for name, value, exc_type, fragment in cases:
            with self.subTest(case=name):
                bridge = self._fresh()
                with self.assertRaisesRegex(exc_type, fragment):
                    bridge.record_reprocessing_telemetry({
                        "attempts": 1,
                        "failed": 0,
                        "discarded": 0,
                        "total_latency_ms": value,
                    })
                self.assertEqual(bridge.get_policy_audit(), [])
                self.assertEqual(bridge.reprocessing_telemetry, [])

    def test_lying_count_subclass_cannot_rewrite_a_negative(self):
        with self.assertRaisesRegex(ValueError, "não negativo"):
            self.bridge.record_reprocessing_telemetry({
                "attempts": LyingCount(-2),
                "failed": 0,
                "discarded": 0,
                "total_latency_ms": 0.0,
            })
        self.assertEqual(self.bridge.get_policy_audit(), [])
        self.assertEqual(self.bridge.reprocessing_telemetry, [])

    def test_lying_latency_subclass_cannot_rewrite_a_negative(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.bridge.record_reprocessing_telemetry({
                "attempts": 1,
                "failed": 0,
                "discarded": 0,
                "total_latency_ms": LyingLatency(-1.0),
            })
        self.assertEqual(self.bridge.get_policy_audit(), [])
        self.assertEqual(self.bridge.reprocessing_telemetry, [])


class VitaTelemetryCounterPropertyTests(unittest.TestCase):
    @settings(max_examples=30, deadline=None)
    @given(
        attempts=st.integers(min_value=0, max_value=10**6),
        failed=st.integers(min_value=0, max_value=10**6),
        discarded=st.integers(min_value=0, max_value=10**6),
        latency=st.one_of(
            st.integers(min_value=0, max_value=10**6),
            st.floats(
                min_value=0.0,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
        ),
    )
    def test_nonnegative_numbers_keep_the_rates(self, attempts, failed, discarded, latency):
        bridge = NexusConstitutionalBridge(":memory:")
        self.addCleanup(bridge.close)
        record = bridge.record_reprocessing_telemetry({
            "attempts": attempts,
            "failed": failed,
            "discarded": discarded,
            "total_latency_ms": latency,
        })
        if attempts:
            self.assertEqual(record["failure_rate"], failed / attempts)
            self.assertEqual(record["discard_rate"], discarded / attempts)
            self.assertTrue(math.isfinite(record["avg_latency_ms"]))
        else:
            self.assertEqual(record["failure_rate"], 0.0)
            self.assertEqual(record["discard_rate"], 0.0)
            self.assertEqual(record["avg_latency_ms"], 0.0)
        self.assertEqual(len(bridge.get_policy_audit()), 1)


if __name__ == "__main__":
    unittest.main()
