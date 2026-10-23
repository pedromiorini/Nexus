"""Contrato da working memory: clamp já existente e extração strangler.

Critérios WM1–WM6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit dcfa180) antes do move e antes do eval. O harness é
tools/cognitivo_eval.py --capability working_memory.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

from core.constitutional_brain import CentralExecutive

ROOT = Path(__file__).resolve().parent
_NAMES = (
    "WorkingMemoryItem", "RehearsalEvent", "PhonologicalLoop",
    "VisuospatialSketchpad", "CentralExecutive", "RealWorkingMemorySystem",
)
_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.working_memory as mod
ex = mod.CentralExecutive()
assert ex.allocate_attention("planning", strength=2.0)["strength"] == 1.0
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    import core.cognitivo.working_memory as wm
    return wm


class WorkingMemoryContractTests(unittest.TestCase):
    def test_attention_allocation_is_recorded_and_clamped(self):
        executive = CentralExecutive()
        allocation = executive.allocate_attention("planning", strength=2.0)
        self.assertEqual(allocation, {"target": "planning", "strength": 1.0, "delegated": False})
        self.assertEqual(executive.current_task, "planning")
        self.assertEqual(executive.attention_allocations, [allocation])

    def test_external_attention_presence_is_reported_without_unknown_call(self):
        executive = CentralExecutive(attention_system=object())
        allocation = executive.allocate_attention("monitoring", strength=-1.0)
        self.assertEqual(allocation["strength"], 0.0)
        self.assertTrue(allocation["delegated"])
        self.assertEqual(executive.current_task, "monitoring")


class WorkingMemoryExtractionContractTests(unittest.TestCase):
    def test_monolith_and_package_reexport_the_same_objects(self):
        wm = _module()
        import core.cognitivo as package
        import core.constitutional_brain as brain
        for name in _NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(wm, name))
                self.assertIs(getattr(package, name), getattr(wm, name))
                self.assertEqual(getattr(wm, name).__module__, "core.cognitivo.working_memory")

    def test_module_imports_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", _BLOCKER], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_pre_registered_checks_match_the_frozen_fingerprint(self):
        from tools.cognitivo_eval import WM_GOLDEN_SHA, eval_working_memory
        checks = {item["id"]: item for item in eval_working_memory()}
        self.assertEqual(set(checks), {"WM1", "WM2", "WM3", "WM4", "WM5", "WM6"})
        for cid, item in checks.items():
            with self.subTest(cid=cid):
                self.assertTrue(item["passed"], item)
        self.assertEqual(checks["WM6"]["value"]["sha_module"], WM_GOLDEN_SHA)
        self.assertEqual(checks["WM6"]["value"]["sha_brain"], WM_GOLDEN_SHA)
        self.assertEqual(checks["WM3"]["value"]["clock"], "wall")
        self.assertLess(checks["WM3"]["value"]["kept_activation"], 1.0)
        self.assertGreater(checks["WM3"]["value"]["kept_activation"], 0.1)

    def test_rehearse_verbal_counts_only_a_hit(self):
        import contextlib
        import io
        wm = _module()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            system = wm.RealWorkingMemorySystem()
            system.rehearse_verbal("missing")
            self.assertEqual(system.rehearsal_count, 0)
            item = system.store_verbal("keep")
            system.rehearse_verbal(item.item_id)
            self.assertEqual(system.rehearsal_count, 1)

    def test_invalid_attention_strength_is_not_coerced_from_text_or_container(self):
        executive = CentralExecutive()
        with self.assertRaises(ValueError):
            executive.allocate_attention("t", strength="abc")
        for bad in (None, [], {}):
            with self.subTest(strength=bad):
                with self.assertRaises(TypeError):
                    executive.allocate_attention("t", strength=bad)


if __name__ == "__main__":
    unittest.main()
