"""Contrato do motor de raciocínio temporal (POST 41): harness TR, entradas inválidas e extração strangler.

Critérios TR1–TR6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit 8f2bd90) antes do harness, do contrato e do move. O harness é
tools/cognitivo_eval.py --capability temporal_reasoning. A extração é
move-only: os quatro nomes continuam importáveis de core.constitutional_brain
(strangler) e agora vivem em core/cognitivo/temporal_reasoning.py.
"""

import contextlib
import functools
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from core.constitutional_brain import (
    RealTemporalReasoningEngine,
    TemporalEvent,
    TemporalRelation,
    TemporalSequence,
)

ROOT = Path(__file__).resolve().parent
_NAMES = ("TemporalEvent", "TemporalRelation", "TemporalSequence", "RealTemporalReasoningEngine")
_ALL = ("TR1", "TR2", "TR3", "TR4", "TR5", "TR6")
_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.temporal_reasoning as mod
engine = mod.RealTemporalReasoningEngine()
first = engine.track_event("deploy", "a", 2.0)
second = engine.track_event("alert", "b")
seq = engine.build_temporal_sequence([second.event_id, first.event_id])
assert [e.event_type for e in seq.events] == ["deploy", "alert"]
assert engine.predict_next_event([first.event_id]) == "alert"
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    import core.cognitivo.temporal_reasoning as temporal_reasoning
    return temporal_reasoning


def _quiet(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealTemporalReasoningEngine(*args, **kwargs)


@functools.lru_cache(maxsize=1)
def _checks():
    _module()
    from tools import cognitivo_eval
    return {item["id"]: item for item in cognitivo_eval.eval_temporal_reasoning()}


class TemporalReasoningInvalidInputContractTests(unittest.TestCase):
    """Entradas inválidas: o comportamento atual é preservado (TypeError ou resultado vazio)."""

    def test_wrong_arity_and_wrong_types_raise_type_error(self):
        engine = _quiet()
        engine.track_event("t", "d")
        with self.assertRaises(TypeError):
            engine.track_event()
        with self.assertRaises(TypeError):
            _quiet(1, 2, 3, 4)
        for call in (lambda: TemporalEvent(), lambda: TemporalRelation(), lambda: TemporalSequence(),
                     lambda: engine.build_temporal_sequence(), lambda: engine.build_temporal_sequence(None),
                     lambda: engine.build_temporal_sequence(5), lambda: engine.get_events_between(),
                     lambda: engine.get_events_between("a", "b"), lambda: engine.get_events_between(None, None),
                     lambda: engine.predict_next_event()):
            with self.assertRaises(TypeError):
                call()

    def test_prediction_with_invalid_sequence_requires_existing_sequences(self):
        engine = _quiet()
        self.assertIsNone(engine.predict_next_event(None))
        engine.track_event("t", "d")
        engine.build_temporal_sequence([engine.events[0].event_id])
        for bad in (None, 5):
            with self.assertRaises(TypeError):
                engine.predict_next_event(bad)

    def test_unknown_ids_and_strings_are_tolerated(self):
        engine = _quiet()
        self.assertEqual(engine.build_temporal_sequence(None).events, [])
        self.assertEqual(engine.build_temporal_sequence("abc").events, [])
        self.assertIsNone(engine.reason_about_duration(None))
        self.assertIsNone(engine.reason_about_duration([1]))
        self.assertEqual(engine.get_events_between("a", "b"), [])
        self.assertIsNone(engine.predict_next_event("ab"))

    def test_string_duration_breaks_relation_inference_after_the_event_is_stored(self):
        engine = _quiet()
        engine.track_event("x", "y", duration="5")
        with self.assertRaises(TypeError):
            engine.track_event("x", "z", duration=3.0)
        self.assertEqual((len(engine.events), engine.total_events_tracked), (2, 2))
        self.assertEqual((engine.total_relations_detected, len(engine.temporal_relations)), (0, 0))


class TemporalReasoningGapContractTests(unittest.TestCase):
    """Valores exatos dos gaps caracterizados (pré-registro, lacunas i–xii)."""

    def _pair(self, first, second):
        engine = _quiet()
        for index, (duration, stamp) in enumerate((first, second)):
            with mock.patch("time.time", return_value=stamp):
                engine.track_event("t", f"d{index}", duration)
        return [(r.relation_type, r.confidence) for r in engine.temporal_relations]

    def test_relation_labels_and_confidences(self):
        self.assertEqual(self._pair((5, 10.0), (5, 15.0)), [("before", 0.9)])
        self.assertEqual(self._pair((3, 10.0), (20, 5.0)), [("overlaps", 0.8)])
        self.assertEqual(self._pair((5, 10.0), (5, 10.0)), [("during", 0.85)])
        self.assertEqual(self._pair((0, 10.0), (5, 10.0)), [("simultaneous", 0.7)])
        self.assertEqual(self._pair((None, 10.0), (None, 10.0625)), [("simultaneous", 0.7)])
        self.assertEqual(self._pair((None, 10.0), (None, 10.125)), [("before", 0.9)])
        self.assertEqual(self._pair((None, 10.0), (None, 9.875)), [("after", 0.9)])

    def test_sequence_duration_and_average_gaps(self):
        engine = _quiet()
        with mock.patch("time.time", return_value=5.0):
            event = engine.track_event("t", "d", 100)
        self.assertEqual(engine.build_temporal_sequence([event.event_id]).total_duration, 0.0)
        empty = engine.build_temporal_sequence([])
        self.assertEqual((empty.events, empty.total_duration, engine.total_sequences_built), ([], 0.0, 2))
        fresh = _quiet()
        ids = []
        for stamp in range(4):
            with mock.patch("time.time", return_value=float(stamp)):
                ids.append(fresh.track_event("t", "d").event_id)
        averages = []
        for chosen in (ids, ids[:2], ids[:2]):
            fresh.build_temporal_sequence(chosen)
            averages.append(fresh.avg_sequence_length)
        self.assertEqual(averages, [4.0, 3.0, 2.5])

    def test_duration_window_and_prediction_gaps(self):
        engine = _quiet()
        for stamp, etype, duration in ((1.0, "x", 2.0), (2.0, "x", 4.0), (3.0, "x", None), (4.0, "x", 0.0)):
            with mock.patch("time.time", return_value=stamp):
                engine.track_event(etype, "d", duration)
        self.assertEqual([engine.reason_about_duration(e.event_id) for e in engine.events], [2.0, 4.0, 3.0, 3.0])
        spanning = _quiet()
        with mock.patch("time.time", return_value=0.0):
            spanning.track_event("t", "d", 100)
        self.assertEqual(spanning.get_events_between(10, 20), [])
        self.assertEqual(len(spanning.get_events_between(100, 100)), 1)

    def test_collaborators_are_stored_and_never_read(self):
        class Boom:
            def __getattribute__(self, name):
                raise AssertionError(name)

        c, p, w = Boom(), Boom(), Boom()
        engine = _quiet(c, p, w)
        first = engine.track_event("a", "x", 1.0)
        engine.build_temporal_sequence([first.event_id])
        engine.reason_about_duration(first.event_id)
        engine.predict_next_event([first.event_id])
        self.assertTrue(engine.causal is c and engine.planner is p and engine.world_model is w)


class TemporalReasoningExtractionContractTests(unittest.TestCase):
    def test_monolith_and_package_reexport_the_same_objects(self):
        temporal_reasoning = _module()
        import core.cognitivo as package
        import core.constitutional_brain as brain
        for name in _NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(temporal_reasoning, name))
                self.assertIs(getattr(package, name), getattr(temporal_reasoning, name))
                self.assertEqual(getattr(temporal_reasoning, name).__module__, "core.cognitivo.temporal_reasoning")

    def test_package_and_module_declare_the_four_public_names(self):
        temporal_reasoning = _module()
        import core.cognitivo as package
        self.assertEqual(sorted(temporal_reasoning.__all__), sorted(_NAMES))
        for name in _NAMES:
            with self.subTest(name=name):
                self.assertIn(name, package.__all__)

    def test_module_imports_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", _BLOCKER], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_pre_registered_checks_match_the_frozen_fingerprint(self):
        from tools.cognitivo_eval import TR_BEHAVIOR_SHA, TR_SOURCE_SHA
        checks = _checks()
        self.assertEqual(set(checks), set(_ALL))
        for cid in _ALL:
            with self.subTest(cid=cid):
                self.assertTrue(checks[cid]["passed"], checks[cid])
        self.assertEqual(checks["TR1"]["value"]["reference_matches"], 120)
        self.assertEqual(checks["TR1"]["value"]["mutated_inputs"], 0)
        self.assertEqual((checks["TR2"]["value"]["hits"], checks["TR3"]["value"]["hits"]), (20, 20))
        self.assertEqual(len(set(checks["TR5"]["value"]["digests"])), 1)
        self.assertEqual(checks["TR6"]["value"]["sha_module"], TR_BEHAVIOR_SHA)
        self.assertEqual(checks["TR6"]["value"]["sha_brain"], TR_BEHAVIOR_SHA)
        self.assertEqual(checks["TR6"]["value"]["source_sha"], TR_SOURCE_SHA)
        self.assertEqual(checks["TR6"]["value"]["clock"], ["time.time", "uuid.uuid4"])
        self.assertTrue(checks["TR6"]["value"]["identity"] and checks["TR6"]["value"]["standalone"])


class TemporalReasoningHarnessNegativeContractTests(unittest.TestCase):
    """O gate falha fechado: comportamento ou fonte adulterados reprovam o harness."""

    def _run(self, argv):
        from tools import cognitivo_eval
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            code = cognitivo_eval.main(argv)
        return code, out.getvalue(), err.getvalue()

    def _ns(self, engine_cls):
        import core.constitutional_brain as brain
        return type("Ns", (), {"TemporalEvent": brain.TemporalEvent, "TemporalRelation": brain.TemporalRelation,
                               "TemporalSequence": brain.TemporalSequence,
                               "RealTemporalReasoningEngine": engine_cls})

    def test_tampered_statistics_change_the_fingerprint(self):
        _module()
        import core.constitutional_brain as brain
        from tools import cognitivo_eval

        class Tampered(brain.RealTemporalReasoningEngine):
            def get_statistics(self):
                stats = super().get_statistics()
                stats["active_events"] += 1
                return stats

        with mock.patch.object(cognitivo_eval, "_tr_grid_digest", return_value="x"):
            tampered = cognitivo_eval._ms_sha(cognitivo_eval.tr_fingerprint(self._ns(Tampered)))
        self.assertNotEqual(tampered, cognitivo_eval.TR_BEHAVIOR_SHA)

    def test_changed_threshold_or_tie_break_breaks_the_reference_comparison(self):
        _module()
        import core.constitutional_brain as brain
        from tools import cognitivo_eval

        class Wide(brain.RealTemporalReasoningEngine):
            def _infer_temporal_relation(self, event1, event2):
                relation = super()._infer_temporal_relation(event1, event2)
                if abs(event2.timestamp - event1.timestamp) == 0.125:
                    relation.relation_type, relation.confidence = "simultaneous", 0.7
                return relation

        class Mutating(brain.RealTemporalReasoningEngine):
            def build_temporal_sequence(self, event_ids):
                event_ids.sort() if isinstance(event_ids, list) else None
                return super().build_temporal_sequence(event_ids)

        self.assertLess(cognitivo_eval._tr_grid_results(self._ns(Wide))["reference_matches"], 120)
        self.assertGreater(cognitivo_eval._tr_grid_results(self._ns(Mutating))["mutated_inputs"], 0)

    def test_wrong_golden_fails_tr6_and_exits_1(self):
        _module()
        from tools import cognitivo_eval
        digests = _checks()["TR5"]["value"]["digests"]
        with mock.patch.object(cognitivo_eval, "_tr_process_digest", side_effect=digests), \
                mock.patch.object(cognitivo_eval, "TR_BEHAVIOR_SHA", "0" * 64):
            checks = {item["id"]: item for item in cognitivo_eval.eval_temporal_reasoning()}
        self.assertFalse(checks["TR6"]["passed"])
        self.assertTrue(all(checks[i]["passed"] for i in _ALL if i != "TR6"))
        with mock.patch.object(cognitivo_eval, "_tr_process_digest", side_effect=digests * 2), \
                mock.patch.object(cognitivo_eval, "TR_BEHAVIOR_SHA", "0" * 64):
            code, out, _ = self._run(["--capability", "temporal_reasoning"])
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL)
        self.assertIn('"TR6"', out.split('"failed"')[1])
        with mock.patch.object(cognitivo_eval, "_tr_process_digest", side_effect=digests), \
                mock.patch.object(cognitivo_eval, "TR_SOURCE_SHA", "0" * 64):
            self.assertFalse({i["id"]: i for i in cognitivo_eval.eval_temporal_reasoning()}["TR6"]["passed"])

    def test_differing_hash_seed_digests_fail_tr5(self):
        _module()
        from tools import cognitivo_eval
        with mock.patch.object(cognitivo_eval, "_tr_process_digest", side_effect=["a", "a", "b"]):
            checks = {item["id"]: item for item in cognitivo_eval.eval_temporal_reasoning()}
        self.assertFalse(checks["TR5"]["passed"])

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
        for value in ("temporal-reasoning", "Temporal_Reasoning", "temporal", "temporal_reasoning2"):
            with self.subTest(value=value):
                code, out, err = self._run(["--capability", value])
                self.assertEqual((code, out), (2, ""))
                self.assertIn("desconhecida", err)

    def test_invalid_scenario_exits_3_not_1(self):
        from tools import cognitivo_eval

        def invalid(_criteria="v1"):
            return [cognitivo_eval._check("TR2", None, False, "x", scenario_invalid=True)]

        with mock.patch.dict(cognitivo_eval.EVALUATORS, {"temporal_reasoning": invalid}):
            code, _, _ = self._run(["--capability", "temporal_reasoning"])
        self.assertEqual(code, cognitivo_eval.EXIT_INVALID_SCENARIO)
        self.assertNotEqual(cognitivo_eval.EXIT_INVALID_SCENARIO, cognitivo_eval.EXIT_FAIL)

    def test_cli_passes_and_exits_0(self):
        _module()
        from tools import cognitivo_eval
        digests = _checks()["TR5"]["value"]["digests"]
        with mock.patch.object(cognitivo_eval, "_tr_process_digest", side_effect=digests):
            code, out, _ = self._run(["--capability", "temporal_reasoning"])
        self.assertEqual(json.loads(out)["failed"], [])
        self.assertEqual(code, cognitivo_eval.EXIT_OK)


if __name__ == "__main__":
    unittest.main()
