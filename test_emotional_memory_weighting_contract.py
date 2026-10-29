"""Contrato da ponderação emocional de memória: entradas inválidas, harness EW e extração strangler.

Critérios EW1–EW6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit 2bc92d8, com a errata E1) antes do harness, do contrato e do move. O harness é
tools/cognitivo_eval.py --capability emotional_memory. EW3 e EW4 falham como
pré-registrado (nenhum limite sobre valência e ativação; memória sem rótulo
com prioridade-base maior que a emocional moderada): o move não os corrige.
"""

import contextlib
import functools
import io
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.constitutional_brain import EmotionalMemoryWeight, RealEmotionalMemoryWeighting

ROOT = Path(__file__).resolve().parent
_NAMES = ("EmotionalMemoryWeight", "RealEmotionalMemoryWeighting")
_ALL = ("EW1", "EW2", "EW3", "EW4", "EW5", "EW6")
_GAPS = {"EW3", "EW4"}
_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.emotional_memory_weighting as mod
system = mod.RealEmotionalMemoryWeighting()
system.tag_memory_emotionally(1, {"valence": 0.5, "arousal": 0.5})
assert system.prioritize_recall([1, 2])[0][0] == 2
assert system.retrieve_by_valence(0.5) == [1]
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    import core.cognitivo.emotional_memory_weighting as emotional_memory_weighting
    return emotional_memory_weighting


def _quiet(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealEmotionalMemoryWeighting(*args, **kwargs)


class _Boom:
    def __getattribute__(self, name):
        raise AssertionError("called " + name)


@functools.lru_cache(maxsize=1)
def _checks():
    _module()
    from tools.cognitivo_eval import eval_emotional_memory
    return {item["id"]: item for item in eval_emotional_memory()}


class EmotionalMemoryWeightingInvalidInputContractTests(unittest.TestCase):
    def test_wrong_types_are_not_coerced(self):
        for bad in (None, [], "x"):
            with self.subTest(context=bad), self.assertRaises(AttributeError):
                _quiet().tag_memory_emotionally(1, bad)
        for ctx in ({"valence": "a"}, {"valence": None}, {"arousal": "a"}):
            with self.subTest(context=ctx), self.assertRaises(TypeError):
                _quiet().tag_memory_emotionally(1, ctx)
        with self.assertRaises(TypeError):
            _quiet().tag_memory_emotionally()
        with self.assertRaises(TypeError):
            _quiet().tag_memory_emotionally(1)
        with self.assertRaises(TypeError):
            RealEmotionalMemoryWeighting(1, 2, 3)
        with self.assertRaises(TypeError):
            _quiet().prioritize_recall(None)
        with self.assertRaises(TypeError):
            _quiet().prioritize_recall([[1]])
        with self.assertRaises(TypeError):
            _quiet().consolidate_emotional_memory([1])
        with self.assertRaises(TypeError):
            _quiet().get_emotional_profile([1])
        unhashable = _quiet(emotion_system=SimpleNamespace(emotional_states={}))
        with self.assertRaises(TypeError):
            unhashable.tag_memory_emotionally(1, {"agent_id": [1]})

    def test_failed_tag_leaves_the_state_untouched(self):
        system = _quiet()
        for ctx in ({"valence": "a"}, None):
            with self.assertRaises((TypeError, AttributeError)):
                system.tag_memory_emotionally(1, ctx)
        self.assertEqual(system.emotional_weights, {})
        self.assertEqual(system.get_statistics(), {"total_emotional_memories": 0, "avg_emotional_intensity": 0.0,
                                                   "high_arousal_count": 0, "weighted_memories": 0})

    def test_bad_query_arguments_only_fail_when_there_is_something_to_compare(self):
        empty = _quiet()
        self.assertEqual(empty.prioritize_recall([1], "valence"), [(1, 1.0)])
        self.assertEqual(empty.prioritize_recall([1], 5), [(1, 1.0)])
        self.assertEqual(empty.retrieve_by_valence("a"), [])
        self.assertEqual(empty.get_emotional_profile(None), {"has_emotional_weight": False})
        self.assertEqual(empty.consolidate_emotional_memory(None), 1.0)
        with self.assertRaises(TypeError):
            empty.retrieve_by_valence(0.0, limit="a")
        tagged = _quiet()
        tagged.tag_memory_emotionally(1, {"valence": 0.1, "arousal": 0.1})
        for call in (lambda: tagged.prioritize_recall([1], "valence"), lambda: tagged.prioritize_recall([1], 5),
                     lambda: tagged.retrieve_by_valence("a"), lambda: tagged.retrieve_by_valence(0.0, threshold="a"),
                     lambda: tagged.retrieve_by_valence(0.0, limit=1.5)):
            with self.assertRaises(TypeError):
                call()
        self.assertEqual(tagged.retrieve_by_valence(0.0, limit=None), [1])

    def test_collaborators_are_stored_and_read_only_through_the_emotion_states(self):
        boom = _Boom()
        system = _quiet(memory_system=boom)
        system.tag_memory_emotionally(1, {"valence": 0.1, "arousal": 0.2})
        system.prioritize_recall([1, 2], {"valence": 0.0})
        system.retrieve_by_valence(0.0)
        system.get_emotional_profile(1)
        system.get_statistics()
        self.assertIs(system.memory, boom)

        class Falsy:
            def __bool__(self):
                return False

            @property
            def emotional_states(self):
                raise AssertionError("emotional_states")

        weight = _quiet(emotion_system=Falsy()).tag_memory_emotionally(1, {"valence": 0.2, "agent_id": "a1"})
        self.assertEqual(weight.valence, 0.2)
        emotion = SimpleNamespace(emotional_states={"a1": SimpleNamespace(pleasure=-0.5, arousal=0.9)})
        system = _quiet(emotion_system=emotion)
        weight = system.tag_memory_emotionally(7, {"valence": 0.2, "arousal": 0.1, "agent_id": "a1"})
        self.assertEqual((weight.valence, weight.arousal), (-0.5, 0.9))
        self.assertEqual(system.get_statistics()["high_arousal_count"], 1)
        weight = system.tag_memory_emotionally(8, {"valence": 0.2, "arousal": 0.1, "agent_id": "zz"})
        self.assertEqual((weight.valence, weight.arousal), (0.2, 0.1))

    def test_current_behavior_is_recorded_as_gaps(self):
        # Gaps medidos (EW3/EW4/EW5), registrados e fora desta extração.
        system = _quiet()
        over = system.tag_memory_emotionally(1, {"valence": 3.0, "arousal": 2.0})
        self.assertGreater(over.emotional_intensity, 1.0)
        self.assertEqual(over.consolidation_boost, 5.0)
        self.assertAlmostEqual(over.recall_priority, 3.824264635, places=9)
        neg = system.tag_memory_emotionally(2, {"valence": 0.0, "arousal": -1.0})
        self.assertEqual(neg.consolidation_boost, -1.0)
        nan = system.tag_memory_emotionally(3, {"valence": float("nan"), "arousal": 0.0})
        self.assertTrue(math.isnan(nan.emotional_intensity))
        system = _quiet()
        system.tag_memory_emotionally(11, {"valence": 0.4, "arousal": 0.4})
        system.tag_memory_emotionally(12, {"valence": 0.0, "arousal": 0.0})
        self.assertEqual([m for m, _s in system.prioritize_recall([10, 11, 12])], [10, 11, 12])
        retag = _quiet()
        for _ in range(2):
            retag.tag_memory_emotionally(1, {"valence": 0.9, "arousal": 0.9})
        stats = retag.get_statistics()
        self.assertEqual((stats["total_emotional_memories"], stats["weighted_memories"], stats["high_arousal_count"]),
                         (2, 1, 2))
        self.assertIsInstance(over, EmotionalMemoryWeight)
        self.assertEqual(system.emotional_decay_rate, 0.95)


class EmotionalMemoryWeightingExtractionContractTests(unittest.TestCase):
    def test_monolith_and_package_reexport_the_same_objects(self):
        emotional_memory_weighting = _module()
        import core.cognitivo as package
        import core.constitutional_brain as brain
        for name in _NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(emotional_memory_weighting, name))
                self.assertIs(getattr(package, name), getattr(emotional_memory_weighting, name))
                self.assertEqual(getattr(emotional_memory_weighting, name).__module__,
                                 "core.cognitivo.emotional_memory_weighting")

    def test_module_imports_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", _BLOCKER], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_pre_registered_checks_match_the_frozen_fingerprint(self):
        from tools.cognitivo_eval import EW_BEHAVIOR_SHA, EW_SOURCE_SHA
        checks = _checks()
        self.assertEqual(set(checks), set(_ALL))
        for cid in _ALL:
            with self.subTest(cid=cid):
                self.assertEqual(checks[cid]["passed"], cid not in _GAPS, checks[cid])
        self.assertEqual(checks["EW1"]["value"]["reference_matches"], 120)
        self.assertEqual(checks["EW2"]["value"]["rate"], 1.0)
        self.assertEqual(checks["EW3"]["value"]["rate"], 0.0)
        self.assertEqual(checks["EW3"]["value"]["control_bad"], 0)
        self.assertEqual(checks["EW4"]["value"]["rate"], 0.0)
        self.assertEqual(checks["EW6"]["value"]["sha_module"], EW_BEHAVIOR_SHA)
        self.assertEqual(checks["EW6"]["value"]["sha_brain"], EW_BEHAVIOR_SHA)
        self.assertEqual(checks["EW6"]["value"]["source_sha"], EW_SOURCE_SHA)
        self.assertEqual(checks["EW6"]["value"]["clock"], "time.time")
        self.assertTrue(checks["EW6"]["value"]["identity"])


class EmotionalMemoryWeightingHarnessNegativeContractTests(unittest.TestCase):
    """O gate falha fechado: comportamento ou fonte adulterados reprovam o harness."""

    def _run(self, argv):
        from tools import cognitivo_eval
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            code = cognitivo_eval.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_tampered_behavior_changes_the_fingerprint(self):
        _module()
        import core.constitutional_brain as brain
        from tools import cognitivo_eval

        class Tampered(brain.RealEmotionalMemoryWeighting):
            def get_statistics(self):
                stats = super().get_statistics()
                stats["weighted_memories"] += 1
                return stats

        ns = {"RealEmotionalMemoryWeighting": Tampered, "EmotionalMemoryWeight": brain.EmotionalMemoryWeight}
        tampered = cognitivo_eval._ms_sha(cognitivo_eval.emotional_memory_fingerprint(ns))
        self.assertNotEqual(tampered, cognitivo_eval.EW_BEHAVIOR_SHA)

    def test_wrong_golden_fails_ew6_and_exits_1(self):
        _module()
        from tools import cognitivo_eval
        with mock.patch.object(cognitivo_eval, "EW_BEHAVIOR_SHA", "0" * 64):
            checks = {item["id"]: item for item in cognitivo_eval.eval_emotional_memory()}
            code, out, _ = self._run(["--capability", "emotional_memory"])
        self.assertFalse(checks["EW6"]["passed"])
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL)
        self.assertIn('"EW6"', out.split('"failed"')[1])
        with mock.patch.object(cognitivo_eval, "EW_SOURCE_SHA", "0" * 64):
            self.assertFalse({i["id"]: i for i in cognitivo_eval.eval_emotional_memory()}["EW6"]["passed"])

    def test_cli_rejects_absent_empty_and_unknown_capability_with_exit_2(self):
        from tools import cognitivo_eval
        code, out, err = self._run([])
        self.assertEqual((code, out), (cognitivo_eval.EXIT_USAGE, ""))
        self.assertIn("ausente", err)
        for value in ("", "   "):
            with self.subTest(value=value):
                code, out, err = self._run(["--capability", value])
                self.assertEqual((code, out), (2, ""))
                self.assertIn("ausente ou vazio", err)
        for value in ("emotional-memory", "Emotional_Memory", "emotional", "emotional_memory2"):
            with self.subTest(value=value):
                code, out, err = self._run(["--capability", value])
                self.assertEqual((code, out), (2, ""))
                self.assertIn("desconhecida", err)

    def test_invalid_scenario_exits_3_not_1(self):
        from tools import cognitivo_eval

        def invalid(_criteria="v1"):
            return [cognitivo_eval._check("EW2", None, False, "x", scenario_invalid=True)]

        with mock.patch.dict(cognitivo_eval.EVALUATORS, {"emotional_memory": invalid}):
            code, _, _ = self._run(["--capability", "emotional_memory"])
        self.assertEqual(code, cognitivo_eval.EXIT_INVALID_SCENARIO)
        self.assertNotEqual(cognitivo_eval.EXIT_INVALID_SCENARIO, cognitivo_eval.EXIT_FAIL)

    def test_cli_reports_only_the_registered_gaps_and_exits_1(self):
        _module()
        from tools import cognitivo_eval
        code, out, _ = self._run(["--capability", "emotional_memory"])
        failed = json.loads(out)["failed"]
        self.assertEqual(set(failed), _GAPS)
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL)

    def test_source_text_requires_the_marker(self):
        from tools import cognitivo_eval
        with self.assertRaises(ValueError):
            cognitivo_eval.emotional_memory_source_text("nothing here")
        with self.assertRaises(ValueError):
            cognitivo_eval.emotional_memory_source_text("x\n" + cognitivo_eval._EW_MARKER + "\nno end")


if __name__ == "__main__":
    unittest.main()
