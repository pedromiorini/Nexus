"""Contrato do episodic future thinking: tipos inválidos e extração strangler.

Critérios EF1–EF6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit 3e8940f) antes do move e antes do eval. O harness é
tools/cognitivo_eval.py --capability episodic_future.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

from core.constitutional_brain import RealEpisodicFutureThinkingSystem

ROOT = Path(__file__).resolve().parent
_NAMES = ("FutureEvent", "ProspectiveMemory", "RealEpisodicFutureThinkingSystem")
_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.episodic_future_thinking as mod
event = mod.RealEpisodicFutureThinkingSystem(object(), emotion=object()).imagine_future_event("Quiet day", "tomorrow")
assert event.location == "familiar_place" and event.vividness == 0.75
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    import core.cognitivo.episodic_future_thinking as episodic_future
    return episodic_future


def _quiet():
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return RealEpisodicFutureThinkingSystem()


class EpisodicFutureInvalidInputContractTests(unittest.TestCase):
    def test_non_string_description_without_emotion_is_not_coerced(self):
        system = _quiet()
        with self.assertRaises(AttributeError):
            system.imagine_future_event(1, "tomorrow")
        with self.assertRaises(AttributeError):
            system.imagine_future_event("plain", "tomorrow", [1])
        with self.assertRaises(AttributeError):
            system.imagine_future_event("plain", "tomorrow", True)

    def test_empty_description_and_falsy_details_keep_the_neutral_event(self):
        system = _quiet()
        empty = system.imagine_future_event("", "t")
        self.assertEqual(empty.expected_emotions, ["neutral"])
        self.assertEqual(empty.location, "unknown")
        self.assertEqual(empty.plausibility, 0.5)
        self.assertEqual(empty.vividness, 0.3)
        blank = system.imagine_future_event("plain", "t", "")
        zero = system.imagine_future_event("plain", "t", 0)
        false_details = system.imagine_future_event("plain", "t", False)
        self.assertEqual(blank.location, zero.location)
        self.assertEqual(blank.location, false_details.location)
        self.assertEqual(blank.expected_emotions, ["neutral"])
        self.assertEqual(zero.plausibility, 0.5)

    def test_bad_context_raises_only_when_an_intention_is_open(self):
        empty = _quiet()
        self.assertEqual(empty.check_prospective_triggers(None), [])
        self.assertEqual(empty.check_prospective_triggers("time"), [])
        self.assertEqual(empty.check_prospective_triggers([]), [])
        system = _quiet()
        system.set_prospective_memory("keep", "later", where="room", trigger="bell")
        with self.assertRaises(AttributeError):
            system.check_prospective_triggers(None)
        with self.assertRaises(AttributeError):
            system.check_prospective_triggers("time")
        with self.assertRaises(AttributeError):
            system.check_prospective_triggers([])
        with self.assertRaises(TypeError):
            system.check_prospective_triggers()

    def test_horizon_and_preexperience_and_compare_reject_wrong_types(self):
        system = _quiet()
        with self.assertRaises(AttributeError):
            system.project_future_self(None)
        with self.assertRaises(AttributeError):
            system.project_future_self(1)
        with self.assertRaises(AttributeError):
            system.preexperience_event(None)
        with self.assertRaises(TypeError):
            system.compare_future_scenarios(
                system.imagine_future_event("a", "t", {"significance": "high"}),
                system.imagine_future_event("b", "t"),
            )

    def test_second_fulfill_counts_again_on_a_single_intention(self):
        system = _quiet()
        self.assertIsNone(system.fulfill_intention("missing"))
        self.assertEqual(system.fulfilled_intentions, 0)
        intention = system.set_prospective_memory("only", "later")
        self.assertIsNone(system.fulfill_intention(intention.intention_id))
        self.assertIsNone(system.fulfill_intention(intention.intention_id))
        stats = system.get_statistics()
        self.assertEqual(stats["fulfilled_intentions"], 2)
        self.assertEqual(stats["prospective_memories_active"], 0)
        self.assertEqual(stats["intention_fulfillment_rate"], 2.0)


class EpisodicFutureExtractionContractTests(unittest.TestCase):
    def test_monolith_and_package_reexport_the_same_objects(self):
        episodic_future = _module()
        import core.cognitivo as package
        import core.constitutional_brain as brain
        for name in _NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(episodic_future, name))
                self.assertIs(getattr(package, name), getattr(episodic_future, name))
                self.assertEqual(
                    getattr(episodic_future, name).__module__,
                    "core.cognitivo.episodic_future_thinking",
                )

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

    def test_pre_registered_checks_match_the_frozen_fingerprint(self):
        _module()
        from tools.cognitivo_eval import EF_BEHAVIOR_SHA, EF_SOURCE_SHA, eval_episodic_future
        checks = {item["id"]: item for item in eval_episodic_future()}
        self.assertEqual(set(checks), {"EF1", "EF2", "EF3", "EF4", "EF5", "EF6"})
        for cid, item in checks.items():
            with self.subTest(cid=cid):
                self.assertTrue(item["passed"], item)
        self.assertEqual(checks["EF6"]["value"]["sha_module"], EF_BEHAVIOR_SHA)
        self.assertEqual(checks["EF6"]["value"]["sha_brain"], EF_BEHAVIOR_SHA)
        self.assertEqual(checks["EF6"]["value"]["source_sha"], EF_SOURCE_SHA)
        self.assertEqual(checks["EF6"]["value"]["clock"], "wall")
        self.assertTrue(checks["EF6"]["value"]["identity"])


if __name__ == "__main__":
    unittest.main()
