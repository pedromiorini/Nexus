"""Contrato da ponderação emocional de memória: entradas inválidas, harness EW e extração strangler.

Critérios EW1–EW6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md (commit 2bc92d8, com a
errata E1) antes do harness, do contrato e do move; EW7–EW9 pré-registrados no commit d034ee1, antes do
contrato RED e da correção. O harness é tools/cognitivo_eval.py --capability emotional_memory. Depois da
correção de EW3 (limite do domínio de valência e ativação) e de EW4 (memória sem rótulo pontua 0.0, não
1.0), os nove critérios passam e o harness sai com 0.
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
_ALL = ("EW1", "EW2", "EW3", "EW4", "EW5", "EW6", "EW7", "EW8", "EW9")
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
        self.assertEqual(empty.prioritize_recall([1], "valence"), [(1, 0.0)])
        self.assertEqual(empty.prioritize_recall([1], 5), [(1, 0.0)])
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

    def test_valence_and_arousal_are_clamped_to_the_valid_domain(self):
        system = _quiet()
        over = system.tag_memory_emotionally(1, {"valence": 3.0, "arousal": 2.0})
        self.assertEqual((over.valence, over.arousal, over.emotional_intensity), (1.0, 1.0, 1.0))
        self.assertEqual((over.consolidation_boost, over.recall_priority), (3.0, 1.5))
        neg = system.tag_memory_emotionally(2, {"valence": 0.0, "arousal": -1.0})
        self.assertEqual((neg.arousal, neg.consolidation_boost, neg.emotional_intensity, neg.recall_priority),
                         (0.0, 1.0, 0.0, 0.0))
        low = system.tag_memory_emotionally(3, {"valence": -5.0, "arousal": -5.0})
        self.assertEqual((low.valence, low.arousal), (-1.0, 0.0))
        self.assertAlmostEqual(low.emotional_intensity, 0.707106781, places=9)
        wide = system.tag_memory_emotionally(4, {"valence": float("inf"), "arousal": float("-inf")})
        self.assertEqual((wide.valence, wide.arousal), (1.0, 0.0))
        for weight in (over, neg, low, wide):
            self.assertTrue(-1.0 <= weight.valence <= 1.0 and 0.0 <= weight.arousal <= 1.0)
            self.assertTrue(0.0 <= weight.emotional_intensity <= 1.0 and 1.0 <= weight.consolidation_boost <= 3.0)
            self.assertLessEqual(weight.recall_priority, 1.5)
        stats = system.get_statistics()
        self.assertEqual((stats["total_emotional_memories"], stats["high_arousal_count"], stats["weighted_memories"]),
                         (4, 1, 4))
        self.assertAlmostEqual(stats["avg_emotional_intensity"], (1.0 + 0.0 + 2 * 0.707106781) / 4, places=8)
        self.assertEqual(system.consolidate_emotional_memory(1), 3.0)
        self.assertEqual(system.consolidate_emotional_memory(2), 1.0)
        self.assertEqual(system.get_emotional_profile(1)["emotional_category"], "excited_positive")
        self.assertEqual(system.get_emotional_profile(3)["emotional_category"], "sad_negative")

    def test_in_domain_values_are_returned_unchanged_with_their_types(self):
        system = _quiet()
        for mid, (v, a, kinds) in enumerate(((1, 1, (int, int)), (-1, 0, (int, int)), (True, True, (bool, bool)),
                                             (0.3, 0.4, (float, float)))):
            weight = system.tag_memory_emotionally(mid, {"valence": v, "arousal": a})
            self.assertEqual((type(weight.valence), type(weight.arousal)), kinds)
            self.assertEqual((weight.valence, weight.arousal), (v, a))
        over = system.tag_memory_emotionally(9, {"valence": 2, "arousal": 2})
        self.assertEqual((type(over.valence), over.valence, over.arousal), (float, 1.0, 1.0))
        ctx = {"valence": 3.0, "arousal": 2.0}
        system.tag_memory_emotionally(10, ctx)
        self.assertEqual(ctx, {"valence": 3.0, "arousal": 2.0})
        default = system.tag_memory_emotionally(11, {})
        self.assertEqual((default.valence, default.arousal, default.consolidation_boost), (0.0, 0.0, 1.0))

    def test_the_agent_state_override_is_clamped_too(self):
        low = SimpleNamespace(emotional_states={"a1": SimpleNamespace(pleasure=-2.0, arousal=-0.6)})
        system = _quiet(emotion_system=low)
        weight = system.tag_memory_emotionally(1, {"valence": 0.2, "arousal": 0.1, "agent_id": "a1"})
        self.assertEqual((weight.valence, weight.arousal, weight.consolidation_boost), (-1.0, 0.0, 1.0))
        self.assertEqual(system.get_statistics()["high_arousal_count"], 0)
        high = SimpleNamespace(emotional_states={"a1": SimpleNamespace(pleasure=0.4, arousal=1.7)})
        system = _quiet(emotion_system=high)
        weight = system.tag_memory_emotionally(1, {"agent_id": "a1"})
        self.assertEqual((weight.valence, weight.arousal, weight.consolidation_boost), (0.4, 1.0, 3.0))
        self.assertEqual(system.get_statistics()["high_arousal_count"], 1)

    def test_a_moderate_emotional_memory_outranks_an_unlabeled_one(self):
        system = _quiet()
        system.tag_memory_emotionally(11, {"valence": 0.4, "arousal": 0.4})
        system.tag_memory_emotionally(12, {"valence": 0.0, "arousal": 0.0})
        ranked = system.prioritize_recall([10, 11, 12])
        self.assertEqual([m for m, _s in ranked], [11, 10, 12])
        self.assertAlmostEqual(ranked[0][1], 0.6, places=9)
        self.assertEqual([s for _m, s in ranked[1:]], [0.0, 0.0])
        self.assertEqual(_quiet().prioritize_recall([5, 5]), [(5, 0.0), (5, 0.0)])
        small = _quiet()
        small.tag_memory_emotionally(31, {"valence": 0.01, "arousal": 0.01})
        self.assertEqual([m for m, _s in small.prioritize_recall([30, 31])], [31, 30])
        extreme = _quiet()
        extreme.tag_memory_emotionally(50, {"valence": 3.0, "arousal": 2.0})
        self.assertEqual(extreme.prioritize_recall([51, 50]), [(50, 1.5), (51, 0.0)])
        query = _quiet()
        query.tag_memory_emotionally(40, {"valence": -1.0, "arousal": 1.0})
        self.assertAlmostEqual(query.prioritize_recall([41, 40], {"valence": 1.0})[0][1], 0.3, places=9)
        self.assertEqual(query.prioritize_recall([41], {"valence": 0.5}), [(41, 0.0)])

    def test_prioritizing_does_not_tag_or_change_the_counters(self):
        system = _quiet()
        system.tag_memory_emotionally(1, {"valence": 0.3, "arousal": 0.3})
        before = (dict(system.emotional_weights), system.get_statistics())
        ids = [7, 8, 1]
        system.prioritize_recall(ids, {"valence": 0.2})
        self.assertEqual((dict(system.emotional_weights), system.get_statistics()), before)
        self.assertEqual(ids, [7, 8, 1])
        self.assertEqual(_quiet().prioritize_recall([]), [])

    def test_residual_gaps_are_recorded_and_were_not_widened(self):
        # Fora do escopo desta correção (pré-registro d034ee1): continuam como estão.
        nan = _quiet().tag_memory_emotionally(3, {"valence": float("nan"), "arousal": 0.0})
        self.assertTrue(math.isnan(nan.emotional_intensity))
        query = _quiet()
        query.tag_memory_emotionally(1, {"valence": -1.0, "arousal": 1.0})
        self.assertAlmostEqual(query.prioritize_recall([1], {"valence": 3.0})[0][1], -2.1, places=9)
        tie = _quiet()
        tie.tag_memory_emotionally(12, {"valence": 0.0, "arousal": 0.0})
        self.assertEqual(tie.prioritize_recall([10, 12]), [(10, 0.0), (12, 0.0)])
        self.assertEqual(tie.prioritize_recall([12, 10]), [(12, 0.0), (10, 0.0)])
        retag = _quiet()
        for _ in range(2):
            retag.tag_memory_emotionally(1, {"valence": 0.9, "arousal": 0.9})
        stats = retag.get_statistics()
        self.assertEqual((stats["total_emotional_memories"], stats["weighted_memories"], stats["high_arousal_count"]),
                         (2, 1, 2))
        self.assertEqual(_quiet().emotional_decay_rate, 0.95)
        import inspect
        self.assertEqual(inspect.getsource(RealEmotionalMemoryWeighting).count("emotional_decay_rate"), 1)
        self.assertIsInstance(nan, EmotionalMemoryWeight)

    def test_invalid_types_still_raise_without_touching_the_state(self):
        for ctx in ({"valence": "a"}, {"valence": None}, {"valence": [1]}, {"arousal": "a"}, {"arousal": None}):
            system = _quiet()
            with self.subTest(ctx=ctx), self.assertRaises(TypeError):
                system.tag_memory_emotionally(1, ctx)
            self.assertEqual((system.emotional_weights, system.total_emotional_memories), ({}, 0))


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
                self.assertTrue(checks[cid]["passed"], checks[cid])
        self.assertEqual(checks["EW1"]["value"]["reference_matches"], 120)
        self.assertEqual(checks["EW2"]["value"]["rate"], 1.0)
        self.assertEqual(checks["EW3"]["value"]["rate"], 1.0)
        self.assertEqual(checks["EW3"]["value"]["control_bad"], 0)
        self.assertEqual(checks["EW4"]["value"]["rate"], 1.0)
        self.assertEqual((checks["EW7"]["value"]["hits"], checks["EW7"]["value"]["total"]), (21, 21))
        self.assertEqual((checks["EW8"]["value"]["hits"], checks["EW8"]["value"]["total"]), (12, 12))
        self.assertEqual(checks["EW9"]["value"]["drift"], 0)
        self.assertGreaterEqual(checks["EW9"]["value"]["unaffected_sections"], 400)
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

    def test_cli_passes_all_nine_criteria_and_exits_0(self):
        _module()
        code, out, _ = self._run(["--capability", "emotional_memory"])
        report = json.loads(out)
        self.assertEqual(report["failed"], [])
        self.assertEqual([item["id"] for item in report["emotional_memory"]], list(_ALL))
        self.assertEqual(code, 0)

    def test_regressed_unlabeled_score_fails_ew8_and_exits_1(self):
        _module()
        import core.constitutional_brain as brain
        from tools import cognitivo_eval

        class Legacy(brain.RealEmotionalMemoryWeighting):
            def prioritize_recall(self, memory_ids, query=None):
                ranked = super().prioritize_recall(memory_ids, query)
                return sorted(((m, s if m in self.emotional_weights else 1.0) for m, s in ranked),
                              key=lambda item: item[1], reverse=True)

        self.assertNotEqual(cognitivo_eval._ew_rank_items(Legacy), cognitivo_eval._EW_RANK_EXPECTED)
        self.assertEqual(cognitivo_eval._ew_rank_items(brain.RealEmotionalMemoryWeighting),
                         cognitivo_eval._EW_RANK_EXPECTED)

    def test_unclamped_weights_fail_ew7(self):
        _module()
        import core.constitutional_brain as brain
        from tools import cognitivo_eval

        class Raw(brain.RealEmotionalMemoryWeighting):
            @staticmethod
            def _clamp_to_domain(value, low, high):
                return value

        self.assertEqual(cognitivo_eval._ew_clamp_items(brain.RealEmotionalMemoryWeighting),
                         cognitivo_eval._EW_CLAMP_EXPECTED)
        self.assertNotEqual(cognitivo_eval._ew_clamp_items(Raw), cognitivo_eval._EW_CLAMP_EXPECTED)
        with mock.patch.dict(cognitivo_eval._EW_CLAMP_EXPECTED, {"i1": [3.0, 2.0, 1.0, 5.0, 1.5]}):
            checks = {i["id"]: i for i in cognitivo_eval.eval_emotional_memory()}
            code, out, _ = self._run(["--capability", "emotional_memory"])
        self.assertFalse(checks["EW7"]["passed"])
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL)
        self.assertEqual(json.loads(out)["failed"], ["EW7"])

    def test_drift_in_an_unaffected_section_fails_ew9_and_exits_1(self):
        _module()
        from tools import cognitivo_eval
        rows = tuple("f" * 40 for _ in cognitivo_eval._EW_BASE_ROWS)
        sha = cognitivo_eval._ms_sha(list(rows))
        with mock.patch.object(cognitivo_eval, "_EW_BASE_ROWS", rows), \
                mock.patch.object(cognitivo_eval, "_EW_BASE_ROWS_SHA", sha):
            checks = {i["id"]: i for i in cognitivo_eval.eval_emotional_memory()}
            code, out, _ = self._run(["--capability", "emotional_memory"])
        self.assertFalse(checks["EW9"]["passed"])
        self.assertNotIn("scenario_invalid", checks["EW9"])
        self.assertGreater(checks["EW9"]["value"]["drift"], 0)
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL)
        self.assertIn("EW9", json.loads(out)["failed"])

    def test_altered_base_list_or_too_few_unaffected_sections_is_an_invalid_scenario(self):
        _module()
        from tools import cognitivo_eval
        with mock.patch.object(cognitivo_eval, "_EW_BASE_ROWS_SHA", "0" * 64):
            checks = {i["id"]: i for i in cognitivo_eval.eval_emotional_memory()}
            code, _, _ = self._run(["--capability", "emotional_memory"])
        self.assertFalse(checks["EW9"]["passed"])
        self.assertTrue(checks["EW9"]["scenario_invalid"])
        self.assertEqual(code, cognitivo_eval.EXIT_INVALID_SCENARIO)
        with mock.patch.object(cognitivo_eval, "_EW_MIN_UNAFFECTED", 10_000):
            checks = {i["id"]: i for i in cognitivo_eval.eval_emotional_memory()}
            code, _, _ = self._run(["--capability", "emotional_memory"])
        self.assertTrue(checks["EW9"]["scenario_invalid"])
        self.assertEqual(code, cognitivo_eval.EXIT_INVALID_SCENARIO)

    def test_source_text_requires_the_marker(self):
        from tools import cognitivo_eval
        with self.assertRaises(ValueError):
            cognitivo_eval.emotional_memory_source_text("nothing here")
        with self.assertRaises(ValueError):
            cognitivo_eval.emotional_memory_source_text("x\n" + cognitivo_eval._EW_MARKER + "\nno end")


if __name__ == "__main__":
    unittest.main()
