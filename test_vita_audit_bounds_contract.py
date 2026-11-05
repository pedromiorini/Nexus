"""Contrato dos limites da auditoria Vita: bool, vazio e não finito não imitam sucesso.

Observação em main af28cc1 (Python 3.12.3, sonda sem pipe, um evento já gravado):
- get_policy_audit(limit=True) devolve 1 linha e executa SQL.
- limit 1.5, NaN e +inf executam SQL e levantam sqlite3.IntegrityError.
- limit False, 0 e negativo levantam ValueError antes do SQL; None e ""
  levantam TypeError do operador <, sem SQL.
- get_recovery_analysis(window_seconds=NaN) devolve events_analyzed=0 e
  severity=info. window True e +inf devolvem o evento. False, 0 e -inf
  levantam ValueError. "" levanta TypeError do operador <=.

Hipóteses:
- H1: bool vale como int de propósito. Falsifica se True deixa de consultar
  e não há chamador que passe bool (só omissão ou inteiro).
- H2: NaN é a janela vazia documentada. Falsifica se a ausência continua
  sendo None e NaN passa a ser ValueError, com o evento ainda legível depois.
- H3: IntegrityError do SQLite já rejeita float no limite. Falsifica se a
  rejeição passa a ser TypeError antes de qualquer SQL.
- H4: +inf é a janela ilimitada. Falsifica se +inf for ValueError e None
  continuar sendo a ausência que lê o evento.

Critérios dos três casos, escritos antes da correção:
- origem: NaN não devolve análise vazia; limit True não devolve linha.
- anterior: None e limite 1 seguem sucesso; window 0 segue ValueError.
- incompatível: "", +inf e float no limite não são sucesso.

Códigos de saída do subprocesso, sem pipe: 0 sucesso, 2 tipo inválido
(ausente, vazio, bool ou não inteiro), 3 valor irresolúvel ou não positivo,
4 mensagem inesperada, 5 outra exceção, 9 caso desconhecido.
"""

import os
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path

from hypothesis import given, settings, strategies as st

from vita.nexus_constitutional_bridge_v3 import NexusConstitutionalBridge


ROOT = Path(__file__).resolve().parent

SCRIPT = r"""
import sys
from vita.nexus_constitutional_bridge_v3 import NexusConstitutionalBridge

kind = sys.argv[1]
bridge = NexusConstitutionalBridge(":memory:")
bridge.record_policy_transition("active", "paused", "probe", {})
try:
    if kind == "limit_true":
        bridge.get_policy_audit(limit=True)
    elif kind == "limit_false":
        bridge.get_policy_audit(limit=False)
    elif kind == "limit_float":
        bridge.get_policy_audit(limit=1.5)
    elif kind == "limit_nan":
        bridge.get_policy_audit(limit=float("nan"))
    elif kind == "limit_inf":
        bridge.get_policy_audit(limit=float("inf"))
    elif kind == "limit_none":
        bridge.get_policy_audit(limit=None)
    elif kind == "limit_empty":
        bridge.get_policy_audit(limit="")
    elif kind == "limit_zero":
        bridge.get_policy_audit(limit=0)
    elif kind == "limit_neg":
        bridge.get_policy_audit(limit=-1)
    elif kind == "limit_ok":
        rows = bridge.get_policy_audit(limit=1)
        sys.exit(0 if len(rows) == 1 else 1)
    elif kind == "win_nan":
        bridge.get_recovery_analysis(window_seconds=float("nan"))
    elif kind == "win_inf":
        bridge.get_recovery_analysis(window_seconds=float("inf"))
    elif kind == "win_ninf":
        bridge.get_recovery_analysis(window_seconds=float("-inf"))
    elif kind == "win_true":
        bridge.get_recovery_analysis(window_seconds=True)
    elif kind == "win_false":
        bridge.get_recovery_analysis(window_seconds=False)
    elif kind == "win_empty":
        bridge.get_recovery_analysis(window_seconds="")
    elif kind == "win_zero":
        bridge.get_recovery_analysis(window_seconds=0)
    elif kind == "win_neg":
        bridge.get_recovery_analysis(window_seconds=-5)
    elif kind == "win_none":
        out = bridge.get_recovery_analysis(window_seconds=None)
        sys.exit(0 if out["events_analyzed"] == 1 else 1)
    elif kind == "win_ok":
        out = bridge.get_recovery_analysis(window_seconds=1.0)
        sys.exit(0 if out["events_analyzed"] == 1 else 1)
    elif kind == "export_nan":
        bridge.export_recovery_diagnostics(window_seconds=float("nan"))
    elif kind == "export_true":
        bridge.export_recovery_diagnostics(limit=True)
    else:
        sys.exit(9)
except TypeError as exc:
    text = str(exc)
    sys.exit(2 if ("inteiro" in text or "número" in text) else 4)
except ValueError as exc:
    sys.exit(3 if "positivo" in str(exc) else 4)
except Exception:
    sys.exit(5)
else:
    sys.exit(0)
"""


class VitaAuditBoundsContractTests(unittest.TestCase):
    def setUp(self):
        self.bridge = NexusConstitutionalBridge(":memory:")
        self.addCleanup(self.bridge.close)

    def _trace(self):
        statements = []
        self.bridge._audit_db.set_trace_callback(statements.append)
        return statements

    def test_nan_window_is_not_an_empty_analysis(self):
        self.bridge.record_policy_transition("active", "paused", "keep", {})
        with self.assertRaisesRegex(ValueError, "positivo"):
            self.bridge.get_recovery_analysis(window_seconds=float("nan"))
        again = self.bridge.get_recovery_analysis(window_seconds=None)
        self.assertEqual(again["events_analyzed"], 1)
        self.assertEqual(again["severity"], "critical")

    def test_invalid_bounds_do_not_query(self):
        self.bridge.record_policy_transition("active", "paused", "keep", {})
        cases = (
            ("limit_true", lambda: self.bridge.get_policy_audit(limit=True), TypeError, "inteiro"),
            ("limit_false", lambda: self.bridge.get_policy_audit(limit=False), TypeError, "inteiro"),
            ("limit_float", lambda: self.bridge.get_policy_audit(limit=1.5), TypeError, "inteiro"),
            ("limit_nan", lambda: self.bridge.get_policy_audit(limit=float("nan")), TypeError, "inteiro"),
            ("limit_inf", lambda: self.bridge.get_policy_audit(limit=float("inf")), TypeError, "inteiro"),
            ("limit_none", lambda: self.bridge.get_policy_audit(limit=None), TypeError, "inteiro"),
            ("limit_empty", lambda: self.bridge.get_policy_audit(limit=""), TypeError, "inteiro"),
            ("limit_zero", lambda: self.bridge.get_policy_audit(limit=0), ValueError, "positivo"),
            ("limit_neg", lambda: self.bridge.get_policy_audit(limit=-1), ValueError, "positivo"),
            ("win_nan", lambda: self.bridge.get_recovery_analysis(window_seconds=float("nan")), ValueError, "positivo"),
            ("win_inf", lambda: self.bridge.get_recovery_analysis(window_seconds=float("inf")), ValueError, "positivo"),
            ("win_ninf", lambda: self.bridge.get_recovery_analysis(window_seconds=float("-inf")), ValueError, "positivo"),
            ("win_true", lambda: self.bridge.get_recovery_analysis(window_seconds=True), TypeError, "número"),
            ("win_false", lambda: self.bridge.get_recovery_analysis(window_seconds=False), TypeError, "número"),
            ("win_empty", lambda: self.bridge.get_recovery_analysis(window_seconds=""), TypeError, "número"),
            ("win_zero", lambda: self.bridge.get_recovery_analysis(window_seconds=0), ValueError, "positivo"),
            ("win_neg", lambda: self.bridge.get_recovery_analysis(window_seconds=-5), ValueError, "positivo"),
            ("export_nan", lambda: self.bridge.export_recovery_diagnostics(window_seconds=float("nan")), ValueError, "positivo"),
            ("export_true", lambda: self.bridge.export_recovery_diagnostics(limit=True), TypeError, "inteiro"),
        )
        for name, call, exc_type, fragment in cases:
            with self.subTest(case=name):
                statements = self._trace()
                with self.assertRaisesRegex(exc_type, fragment):
                    call()
                self.assertEqual(statements, [])

    def test_documented_absence_and_positive_bounds_still_read(self):
        self.bridge._audit_db.execute(
            "INSERT INTO policy_audit (event_type, timestamp, payload) VALUES (?, ?, ?)",
            ("policy_transition", 1.0, '{"previous_state":"active","new_state":"paused","reason":"old","policy":{}}'),
        )
        self.bridge._audit_db.commit()
        self.bridge.record_policy_transition("active", "paused", "keep", {})
        recent = self.bridge.get_policy_audit(limit=1)
        self.assertEqual(recent[0]["payload"]["reason"], "keep")
        opened = self.bridge.get_recovery_analysis(window_seconds=None)
        self.assertEqual(opened["events_analyzed"], 2)
        filtered = self.bridge.get_recovery_analysis(limit=2, window_seconds=1)
        self.assertEqual(filtered["events_analyzed"], 1)
        self.assertEqual(filtered["latest_state"], "paused")
        exported = self.bridge.export_recovery_diagnostics(limit=1, window_seconds=1.0)
        self.assertEqual(exported["events"][0]["payload"]["reason"], "keep")
        self.assertEqual(exported["diagnostics"]["events_analyzed"], 1)

    def test_subprocess_exit_codes_are_distinct(self):
        expected = {
            "limit_true": 2,
            "limit_false": 2,
            "limit_float": 2,
            "limit_nan": 2,
            "limit_inf": 2,
            "limit_none": 2,
            "limit_empty": 2,
            "limit_zero": 3,
            "limit_neg": 3,
            "limit_ok": 0,
            "win_nan": 3,
            "win_inf": 3,
            "win_ninf": 3,
            "win_true": 2,
            "win_false": 2,
            "win_empty": 2,
            "win_zero": 3,
            "win_neg": 3,
            "win_none": 0,
            "win_ok": 0,
            "export_nan": 3,
            "export_true": 2,
            "desconhecido": 9,
        }
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        for kind, code in expected.items():
            with self.subTest(kind=kind):
                completed = subprocess.run(
                    [sys.executable, "-c", SCRIPT, kind],
                    cwd=ROOT,
                    env=env,
                    check=False,
                )
                self.assertEqual(completed.returncode, code)

    @settings(max_examples=30, deadline=None)
    @given(
        limit=st.integers(min_value=1, max_value=32),
        window=st.one_of(
            st.none(),
            st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False, width=64),
        ),
    )
    def test_finite_positive_bounds_do_not_raise(self, limit, window):
        analysis = self.bridge.get_recovery_analysis(limit=limit, window_seconds=window)
        self.assertIn(analysis["severity"], {"info", "warning", "critical"})
        self.assertGreaterEqual(analysis["events_analyzed"], 0)


class VitaAuditBoundsRegressionTests(unittest.TestCase):
    def test_sqlite_integrity_error_is_not_the_contract(self):
        bridge = NexusConstitutionalBridge(":memory:")
        self.addCleanup(bridge.close)
        with self.assertRaises(TypeError):
            bridge.get_policy_audit(limit=1.5)
        self.assertFalse(issubclass(TypeError, sqlite3.IntegrityError))


if __name__ == "__main__":
    unittest.main()
