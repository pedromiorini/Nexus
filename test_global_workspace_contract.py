"""Contrato do GlobalWorkspace no monólito (C1, C2, C4, C6, C7, C8).

Critérios pré-registrados antes da correção, na base
``dd613abde0aa689ad5f92ebfaee7eaa73ced5ef2``. C3 (stream só após o limiar)
e C5 (``attention_duration``) ficam de fora: são P-B e P-D, fora deste PR.

Margem documentada que estes critérios assumem: 0,05. Fica acima do vão de
0,01 da guarda C6 e abaixo do vão de 0,40 de C1. Não é medida de ignição.

O espião do rascunho externo fazia ``list.append`` da cópia inteira e pedia
``content_id`` em cada elemento. Isso só passa se o callback receber um item
por chamada. O efeito pedido é entregar a cópia do workspace. O espião daqui
guarda essa cópia.

Hipóteses:
- H1: o incumbente nunca é desalojado. Falsifica se B=0,9 substitui A=0,5.
- H2: qualquer saliência maior troca o incumbente. Falsifica se 0,61 não
  substitui 0,60 e 0,66 substitui.
- H3: ``broadcast`` já entregava a alguém. Falsifica se não há ``subscribe``.
- H4: ``pending`` já tinha teto. Falsifica se 10_000 submissões deixam mais
  de 1000 pendentes.
- H5: ``update_consciousness`` já rodava em ``think``. Falsifica se a AST
  não acha a chamada nesse método.
- H6: ``attention.consciousness`` já apontava para ``brain.consciousness``.
  Falsifica se o atributo é ``None`` depois de ``CompleteNexusBrain()``.
"""
from __future__ import annotations

import ast
import contextlib
import io
import math
import os
import tempfile
import unittest
from pathlib import Path

from hypothesis import given, settings, strategies as st

from core.constitutional_brain import CompleteNexusBrain, ConsciousContent, GlobalWorkspace

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "core" / "constitutional_brain.py"
# Margem de engenharia pré-registrada com C1 e C6. Não ler do código:
# o teste tem de falhar enquanto a constante não governa a troca.
HYSTERESIS = 0.05
PENDING_CAP = 1000


def mk(salience, content_id):
    return ConsciousContent(
        content_id=content_id,
        content_type="t",
        content=content_id,
        salience=salience,
        source_module="m",
        timestamp=0.0,
    )


def ids(items):
    return [item.content_id for item in items]


class CompetitionTests(unittest.TestCase):
    def test_c1_incumbent_replaced_by_much_more_salient_pending(self):
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(0.5, "A"))
        workspace.update_workspace()
        workspace.submit_content(mk(0.9, "B"))
        workspace.update_workspace()
        self.assertEqual(ids(workspace.workspace), ["B"])
        self.assertNotIn("A", ids(workspace.pending))
        self.assertIn("A", ids(workspace.consciousness_stream))

    def test_gap_just_above_hysteresis_replaces_incumbent(self):
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(0.60, "A"))
        workspace.update_workspace()
        workspace.submit_content(mk(0.60 + HYSTERESIS + 0.01, "B"))
        workspace.update_workspace()
        self.assertEqual(ids(workspace.workspace), ["B"])
        self.assertNotIn("A", ids(workspace.pending))

    def test_exact_hysteresis_boundary_does_not_replace(self):
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(0.60, "A"))
        workspace.update_workspace()
        workspace.submit_content(mk(0.60 + HYSTERESIS, "tie"))
        workspace.update_workspace()
        self.assertEqual(ids(workspace.workspace), ["A"])

    def test_near_gap_does_not_replace_incumbent(self):
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(0.60, "A"))
        workspace.update_workspace()
        workspace.submit_content(mk(0.61, "near"))
        workspace.update_workspace()
        self.assertEqual(ids(workspace.workspace), ["A"])

    def test_c6_no_flip_between_near_equal_items(self):
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(0.60, "A"))
        workspace.update_workspace()
        flips = 0
        last = "A"
        for k in range(20):
            salience = 0.61 if k % 2 == 0 else 0.60
            workspace.submit_content(mk(salience, f"x{k}"))
            workspace.update_workspace()
            current = workspace.workspace[0].content_id
            flips += current != last
            last = current
        self.assertLessEqual(flips, 1)

    def test_same_cycle_competition_keeps_most_salient(self):
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(0.2, "lo"))
        workspace.submit_content(mk(0.9, "hi"))
        workspace.update_workspace()
        self.assertEqual(ids(workspace.workspace), ["hi"])
        self.assertEqual(ids(workspace.pending), ["lo"])

    def test_equal_salience_does_not_replace(self):
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(0.8, "A"))
        workspace.update_workspace()
        workspace.submit_content(mk(0.8, "B"))
        workspace.update_workspace()
        self.assertEqual(ids(workspace.workspace), ["A"])

    def test_capacity_two_replaces_only_the_weaker_incumbent(self):
        workspace = GlobalWorkspace(capacity=2)
        workspace.submit_content(mk(0.5, "weak"))
        workspace.submit_content(mk(0.8, "strong"))
        workspace.update_workspace()
        workspace.submit_content(mk(0.9, "new"))
        workspace.update_workspace()
        self.assertEqual(set(ids(workspace.workspace)), {"strong", "new"})
        self.assertNotIn("weak", ids(workspace.pending))

    @given(
        inc=st.floats(min_value=0.31, max_value=0.9, allow_nan=False, allow_infinity=False),
        gap=st.floats(min_value=0.051, max_value=0.4, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=40, deadline=None)
    def test_property_gap_above_margin_replaces(self, inc, gap):
        challenger = inc + gap
        self.assertGreater(challenger, inc + HYSTERESIS)
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(inc, "A"))
        workspace.update_workspace()
        workspace.submit_content(mk(challenger, "B"))
        workspace.update_workspace()
        self.assertEqual(ids(workspace.workspace), ["B"])
        self.assertNotIn("A", ids(workspace.pending))

    @given(
        inc=st.floats(min_value=0.31, max_value=0.9, allow_nan=False, allow_infinity=False),
        gap=st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=40, deadline=None)
    def test_property_gap_within_margin_keeps_incumbent(self, inc, gap):
        challenger = inc + gap
        if challenger > inc + HYSTERESIS:
            challenger = inc + HYSTERESIS
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(inc, "A"))
        workspace.update_workspace()
        workspace.submit_content(mk(challenger, "B"))
        workspace.update_workspace()
        self.assertEqual(ids(workspace.workspace), ["A"])


class BroadcastTests(unittest.TestCase):
    def test_c2_broadcast_delivers_to_subscriber(self):
        workspace = GlobalWorkspace(capacity=1)
        seen = []
        self.assertTrue(hasattr(workspace, "subscribe"))
        workspace.subscribe(seen.append)
        workspace.submit_content(mk(0.9, "A"))
        workspace.update_workspace()
        returned = workspace.broadcast()
        self.assertEqual(len(seen), 1)
        self.assertEqual(ids(seen[0]), ["A"])
        self.assertEqual(ids(returned), ["A"])
        self.assertIsNot(returned, workspace.workspace)
        self.assertIsNot(seen[0], workspace.workspace)
        self.assertEqual(workspace.total_broadcasts, 1)

    def test_broadcast_without_subscribers_still_returns_copy(self):
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(0.9, "A"))
        workspace.update_workspace()
        returned = workspace.broadcast()
        self.assertEqual(ids(returned), ["A"])
        self.assertIsNot(returned, workspace.workspace)
        self.assertEqual(workspace.total_broadcasts, 1)
        returned.clear()
        self.assertEqual(ids(workspace.workspace), ["A"])

    def test_subscriber_mutation_does_not_clear_workspace(self):
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(0.9, "A"))
        workspace.update_workspace()

        def clear(snapshot):
            snapshot.clear()

        workspace.subscribe(clear)
        workspace.broadcast()
        self.assertEqual(ids(workspace.workspace), ["A"])

    def test_subscribers_run_in_registration_order(self):
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(0.9, "A"))
        workspace.update_workspace()
        order = []
        workspace.subscribe(lambda _snapshot: order.append("a"))
        workspace.subscribe(lambda _snapshot: order.append("b"))
        workspace.broadcast()
        self.assertEqual(order, ["a", "b"])
        self.assertEqual(workspace.total_broadcasts, 1)

    def test_subscribe_rejects_non_callable(self):
        workspace = GlobalWorkspace(capacity=1)
        for bad in (None, 1, "cb", True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    workspace.subscribe(bad)
        self.assertEqual(workspace.total_broadcasts, 0)

    def test_subscriber_exception_propagates_after_count(self):
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(0.9, "A"))
        workspace.update_workspace()

        def boom(_snapshot):
            raise RuntimeError("boom")

        workspace.subscribe(boom)
        with self.assertRaises(RuntimeError):
            workspace.broadcast()
        self.assertEqual(workspace.total_broadcasts, 1)
        self.assertEqual(ids(workspace.workspace), ["A"])


class PendingTests(unittest.TestCase):
    def test_c4_pending_is_bounded(self):
        workspace = GlobalWorkspace(capacity=1)
        for i in range(10000):
            workspace.submit_content(mk(0.9, str(i)))
        workspace.update_workspace()
        self.assertLessEqual(len(workspace.pending), PENDING_CAP)

    def test_cap_drops_lowest_salience_and_keeps_higher(self):
        workspace = GlobalWorkspace(capacity=1)
        workspace.submit_content(mk(0.2, "low"))
        for i in range(PENDING_CAP):
            workspace.submit_content(mk(0.8, f"h{i}"))
        self.assertLessEqual(len(workspace.pending), PENDING_CAP)
        self.assertNotIn("low", ids(workspace.pending))
        self.assertIn("h0", ids(workspace.pending))

    def test_tie_drops_oldest(self):
        workspace = GlobalWorkspace(capacity=1, max_pending=2)
        workspace.submit_content(mk(0.5, "old"))
        workspace.submit_content(mk(0.5, "mid"))
        workspace.submit_content(mk(0.5, "new"))
        self.assertEqual(ids(workspace.pending), ["mid", "new"])

    def test_invalid_max_pending_rejected(self):
        for bad in (True, False, 0, -1, 1.5, None, float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    GlobalWorkspace(max_pending=bad)

    def test_invalid_salience_and_content_rejected(self):
        workspace = GlobalWorkspace(capacity=1)
        for bad in (True, False, float("nan"), float("inf"), float("-inf"), -0.1, "0.5", None):
            with self.subTest(bad=bad):
                before = workspace.total_contents
                with self.assertRaises(ValueError):
                    workspace.submit_content(mk(bad, "x"))
                self.assertEqual(workspace.pending, [])
                self.assertEqual(workspace.total_contents, before)
        for bad in (None, "texto", 1, {"content_id": "z"}):
            with self.subTest(content=bad):
                with self.assertRaises(TypeError):
                    workspace.submit_content(bad)
        self.assertEqual(workspace.pending, [])
        self.assertTrue(math.isfinite(0.5))


class ProductionCallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    def test_c7_update_consciousness_called_outside_selftest(self):
        main_line = None
        think = None
        for node in self.tree.body:
            if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
                main_line = node.lineno
            if isinstance(node, ast.ClassDef) and node.name == "CompleteNexusBrain":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "think":
                        think = item
        self.assertIsNotNone(main_line)
        self.assertIsNotNone(think)
        calls = [
            node.lineno
            for node in ast.walk(think)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update_consciousness"
            and node.lineno < main_line
        ]
        self.assertTrue(calls, "nenhum update_consciousness dentro de think(), antes do __main__")

    def test_c8_brain_wires_attention_to_consciousness(self):
        previous = os.environ.get("NEXUS_AUDIT_DB")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["NEXUS_AUDIT_DB"] = str(Path(tmp) / "audit.sqlite3")
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    brain = CompleteNexusBrain()
            finally:
                if previous is None:
                    os.environ.pop("NEXUS_AUDIT_DB", None)
                else:
                    os.environ["NEXUS_AUDIT_DB"] = previous
        self.assertIsNotNone(brain.consciousness)
        self.assertIsNotNone(brain.attention)
        self.assertIs(brain.attention.consciousness, brain.consciousness)


if __name__ == "__main__":
    unittest.main(verbosity=2)
