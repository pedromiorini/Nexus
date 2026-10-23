"""Contrato da atenção seletiva: tipos inválidos e extração strangler.

Critérios AT1–AT6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit a8323fa) antes do move e antes do eval. O harness é
tools/cognitivo_eval.py --capability attention.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

from core.constitutional_brain import BottomUpAttention, StimulusItem, TopDownAttention

ROOT = Path(__file__).resolve().parent
_NAMES = (
    "StimulusItem",
    "AttentionalFocus",
    "AttentionShift",
    "BottomUpAttention",
    "TopDownAttention",
    "AttentionalSpotlight",
    "RealAttentionSystem",
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
import core.cognitivo.attention as mod
item = mod.StimulusItem("s1", "visual", "x", 0.5, 0.5, 1.0)
assert mod.BottomUpAttention()._assess_contrast(item, {}) == 0.5
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    import core.cognitivo.attention as attention
    return attention


def _stimulus(**kwargs):
    values = {
        "stimulus_id": "s1",
        "stimulus_type": "visual",
        "content": "x",
        "salience": 0.5,
        "priority": 0.5,
        "timestamp": 1.0,
    }
    values.update(kwargs)
    return StimulusItem(**values)


class AttentionInvalidInputContractTests(unittest.TestCase):
    def test_stimulus_contrast_and_motion_strings_are_not_floats(self):
        attention = BottomUpAttention()
        with self.assertRaises(TypeError):
            attention._assess_contrast(_stimulus(contrast="0.2"), {})
        with self.assertRaises(TypeError):
            attention._assess_motion(_stimulus(motion="0.2"))

    def test_context_contrast_rejects_non_numeric_text(self):
        attention = BottomUpAttention()
        with self.assertRaises(ValueError):
            attention._assess_contrast(_stimulus(), {"contrast": "abc"})

    def test_goal_priority_rejects_text_and_containers(self):
        top = TopDownAttention()
        with self.assertRaises(TypeError):
            top.update_goal_priority("g", "abc")
        for bad in (None, [], {}):
            with self.subTest(priority=bad):
                with self.assertRaises(TypeError):
                    top.update_goal_priority("g", bad)


class AttentionExtractionContractTests(unittest.TestCase):
    def test_monolith_and_package_reexport_the_same_objects(self):
        attention = _module()
        import core.cognitivo as package
        import core.constitutional_brain as brain
        for name in _NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(attention, name))
                self.assertIs(getattr(package, name), getattr(attention, name))
                self.assertEqual(getattr(attention, name).__module__, "core.cognitivo.attention")

    def test_module_imports_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run(
            [sys.executable, "-c", _BLOCKER],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_second_attend_does_not_refocus_an_item_already_in_the_spotlight(self):
        import contextlib
        import io

        attention = _module()
        with contextlib.redirect_stdout(io.StringIO()):
            system = attention.RealAttentionSystem()
        system.available_stimuli.append(
            attention.StimulusItem("z", "visual", "z", 1.0, 1.0, 1.0)
        )
        first = system.attend()
        history = len(system.spotlight.focus_history)
        second = system.attend()
        self.assertEqual([item.stimulus_id for item in first], ["z"])
        self.assertEqual([item.stimulus_id for item in second], ["z"])
        self.assertEqual(len(system.spotlight.focus_history), history)
        self.assertEqual(len(system.spotlight.spotlight), 1)
        self.assertEqual(system.total_attentional_episodes, 2)

    def test_pre_registered_checks_match_the_frozen_fingerprint(self):
        from tools.cognitivo_eval import AT_BEHAVIOR_SHA, AT_SOURCE_SHA, eval_attention
        checks = {item["id"]: item for item in eval_attention()}
        self.assertEqual(set(checks), {"AT1", "AT2", "AT3", "AT4", "AT5", "AT6"})
        for cid, item in checks.items():
            with self.subTest(cid=cid):
                self.assertTrue(item["passed"], item)
        self.assertEqual(checks["AT6"]["value"]["sha_module"], AT_BEHAVIOR_SHA)
        self.assertEqual(checks["AT6"]["value"]["sha_brain"], AT_BEHAVIOR_SHA)
        self.assertEqual(checks["AT6"]["value"]["source_sha"], AT_SOURCE_SHA)
        self.assertEqual(checks["AT6"]["value"]["clock"], "wall")
        self.assertTrue(checks["AT6"]["value"]["identity"])


if __name__ == "__main__":
    unittest.main()
