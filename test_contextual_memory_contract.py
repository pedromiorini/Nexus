"""Contrato da memória contextual: entradas inválidas, harness CM, extração strangler e correção de CM3/CM4.

Critérios CM1–CM6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit c8ca267) e CM7–CM9 no pré-registro da correção (commit 022706f), antes
do contrato e da mudança de comportamento. O harness é
tools/cognitivo_eval.py --capability contextual_memory. Depois da correção
todos os critérios passam: CM3 (seletividade com o limiar padrão: dimensões
consultadas que coincidem) e CM4 (sha256 em vez de hash()).
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
    ContextEncoder,
    ContextMatch,
    ContextualMemory,
    RealContextualMemorySystem,
)

ROOT = Path(__file__).resolve().parent
_NAMES = ("ContextualMemory", "ContextMatch", "ContextEncoder", "RealContextualMemorySystem")
_ALL = ("CM1", "CM2", "CM3", "CM4", "CM5", "CM6", "CM7", "CM8", "CM9")
_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.contextual_memory as mod
system = mod.RealContextualMemorySystem()
system.store_with_context("note", {"location": "lab"})
found = system.retrieve_by_context(context={"location": "lab"}, threshold=0.0)
assert [m.memory.content for m in found] == ["note"] and found[0].matched_features == ["location"]
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    import core.cognitivo.contextual_memory as contextual_memory
    return contextual_memory


def _quiet(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealContextualMemorySystem(*args, **kwargs)


class _Boom:
    def __getattribute__(self, name):
        raise AssertionError("called " + name)


@functools.lru_cache(maxsize=1)
def _checks():
    _module()
    from tools.cognitivo_eval import eval_contextual_memory
    return {item["id"]: item for item in eval_contextual_memory()}


class ContextualMemoryInvalidInputContractTests(unittest.TestCase):
    def test_wrong_types_are_not_coerced(self):
        for bad in (None, [], 5, "a"):
            with self.subTest(context=bad), self.assertRaises(AttributeError):
                _quiet().set_current_context(bad)
        with self.assertRaises(AttributeError):
            _quiet().store_with_context("x", 5)
        with self.assertRaises(AttributeError):
            _quiet().retrieve_by_context(context="a")
        with self.assertRaises(AttributeError):
            ContextEncoder().encode_context(None)
        with self.assertRaises(AttributeError):
            _quiet().get_context_dependent_cues(None)
        with self.assertRaises(TypeError):
            _quiet().store_with_context()
        with self.assertRaises(TypeError):
            RealContextualMemorySystem(1, 2, 3, 4)
        with self.assertRaises(TypeError):
            ContextEncoder().compute_similarity("ab", "cd")

    def test_bad_threshold_only_fails_when_there_is_something_to_compare(self):
        self.assertEqual(_quiet().retrieve_by_context(threshold="a"), [])
        system = _quiet()
        system.store_with_context("x", {})
        for bad in ("a", None):
            with self.subTest(threshold=bad), self.assertRaises(TypeError):
                system.retrieve_by_context(threshold=bad)
        self.assertEqual(system.get_statistics()["total_retrievals"], 0)

    def test_lenient_inputs_return_neutral_values(self):
        system = _quiet()
        self.assertIsNone(system.reinstate_context(None))
        self.assertIsNone(system.reinstate_context("missing"))
        self.assertEqual(system.retrieve_by_context(query=5), [])
        encoder = ContextEncoder()
        self.assertEqual(encoder.compute_similarity(None, [1.0]), 0.0)
        self.assertEqual(encoder.compute_similarity([], [1.0]), 0.0)
        self.assertEqual(encoder.compute_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_failed_set_context_leaves_the_reference_stored(self):
        system = _quiet()
        with self.assertRaises(AttributeError):
            system.set_current_context(None)
        self.assertIsNone(system.current_context)  # gap registrado em CM5, fora desta extração

    def test_collaborators_are_stored_and_only_ltm_is_written(self):
        calls = []

        class Ltm:
            def store(self, *args, **kwargs):
                calls.append((args, kwargs))

        system = _quiet(long_term_memory=Ltm(), working_memory=_Boom(), attention=_Boom())
        system.store_with_context("c", {"mood": "x"})
        system.store_with_context(5, {})
        system.retrieve_by_context(context={"mood": "x"})
        self.assertEqual(calls, [(("c", "contextual"), {"importance": 0.7}), (("5", "contextual"), {"importance": 0.7})])
        self.assertEqual(system.get_statistics()["stored_memories"], 2)

    def test_ltm_failure_leaves_the_memory_stored(self):
        class Raising:
            def store(self, *args, **kwargs):
                raise RuntimeError("ltm")

        system = _quiet(long_term_memory=Raising())
        with self.assertRaises(RuntimeError):
            system.store_with_context("x", {})
        stats = system.get_statistics()
        self.assertEqual((stats["stored_memories"], stats["total_encodings"]), (1, 1))

    def test_current_behavior_is_recorded_as_gaps(self):
        # Gaps residuais (CM5), registrados e fora da correção de CM3/CM4.
        system = _quiet()
        for i, ctx in enumerate(({"location": "a"}, {"time": "c"}, {"social": "e"})):
            system.store_with_context(f"x{i}", ctx)
        self.assertEqual(len(system.retrieve_by_context(context={"location": "zzz"})), 0)
        encoder = ContextEncoder()
        self.assertEqual(encoder.encode_context({"weather": "rain"}), encoder.encode_context({"weather": "sun"}))
        ctx = {"location": "office"}
        system = _quiet()
        system.set_current_context(ctx)
        memory = system.store_with_context("x")
        ctx["location"] = "home"
        self.assertIs(memory.context, ctx)
        self.assertEqual(memory.context_vector, encoder.encode_context({"location": "office"}))
        self.assertIsInstance(memory, ContextualMemory)
        self.assertIsInstance(system.retrieve_by_context(threshold=0.0)[0], ContextMatch)


@functools.lru_cache(maxsize=1)
def _fingerprint_grid():
    _module()
    import core.cognitivo.contextual_memory as module
    from tools import cognitivo_eval
    with mock.patch.object(cognitivo_eval, "_cm_determinism", return_value=_checks()["CM4"]["value"]):
        return cognitivo_eval.contextual_memory_fingerprint(module)["grid"]


class ContextualMemoryFixContractTests(unittest.TestCase):
    """CM3 (seletividade com o limiar padrão) e CM4 (vetor estável), pré-registro 022706f."""

    def test_default_threshold_keeps_only_matching_contexts(self):
        full = {"location": "office", "time": "morning", "mood": "calm", "task": "reading", "social": "alone"}
        swaps = (("location", "home"), ("time", "noon"), ("mood", "focused"), ("task", "review"),
                 ("social", "team"))
        system = _quiet()
        for count in range(6):
            ctx = dict(full)
            ctx.update(dict(swaps[:count]))
            system.store_with_context(f"d{count}", ctx)
        found = system.retrieve_by_context(context=full)
        self.assertEqual([m.memory.content for m in found], ["d0", "d1", "d2"])
        self.assertEqual([round(m.match_score, 6) for m in found], [1.0, 0.8, 0.6])
        self.assertEqual([len(m.matched_features) for m in found], [5, 4, 3])
        self.assertEqual(len(system.retrieve_by_context(context=full, threshold=0.0)), 6)

    def test_only_cued_dimensions_count_and_no_cue_matches_everything(self):
        system = _quiet()
        system.store_with_context("x", {"location": "office", "mood": "calm"})
        scores = {}
        for name, ctx in (("partial", {"location": "office"}), ("none", {"weather": "rain"}),
                          ("miss", {"location": "home"}), ("half", {"location": "office", "mood": "focused"})):
            scores[name] = system.retrieve_by_context(context=ctx, threshold=0.0)[0].match_score
        self.assertEqual(scores, {"partial": 1.0, "none": 1.0, "miss": 0.0, "half": 0.5})

    def test_text_query_keeps_the_additive_formula(self):
        system = _quiet()
        system.store_with_context("alpha beta", {"location": "office"})
        hit = system.retrieve_by_context(query="ALPHA", context={"location": "home"})
        miss = system.retrieve_by_context(query="zzz", context={"location": "office"})
        self.assertEqual([m.match_score for m in hit], [0.5])
        self.assertEqual([round(m.match_score, 9) for m in miss], [0.65])
        self.assertEqual(system.retrieve_by_context(query="zzz", context={"location": "home"}), [])

    def test_compute_match_edges(self):
        encoder = ContextEncoder()
        self.assertEqual(encoder.compute_match([0.1, 0.2], [0.1, 0.3]), 0.5)
        self.assertEqual(encoder.compute_match([], [1.0]), 0.0)
        self.assertEqual(encoder.compute_match([1.0], []), 0.0)
        self.assertEqual(encoder.compute_match(None, [1.0]), 0.0)
        self.assertEqual(encoder.compute_match([1.0], [1.0], []), 1.0)
        self.assertEqual(encoder.compute_match([0.1, 0.2, 0.3], [0.1]), 1.0)
        self.assertEqual(encoder.compute_match([0.1, 0.2], [0.1, 0.3], [1]), 0.0)
        self.assertEqual(encoder.compute_match([0.1, 0.2], [0.1, 0.3], [0]), 1.0)
        with self.assertRaises(IndexError):
            encoder.compute_match([1.0], [1.0], [3])
        self.assertAlmostEqual(encoder.compute_similarity([0.5, 0.5], [0.5, 0.5]), 1.0)  # cosseno intocado

    def test_encoder_is_stable_and_never_calls_builtin_hash(self):
        import builtins
        golden = {"location": "office", "time": "morning", "mood": "calm", "task": "reading", "social": "alone"}

        def boom(*_args, **_kwargs):
            raise AssertionError("hash() chamado")

        encoder = ContextEncoder()
        with mock.patch.object(builtins, "hash", boom):
            self.assertEqual(encoder.encode_context(golden), [0.2, 0.32, 0.37, 0.78, 0.2])
            self.assertEqual(encoder.encode_context({"mood": "\ud800"}), [0.04, 0.04, 0.05, 0.04, 0.04])
            self.assertEqual(encoder.encode_context({"location": 1}), encoder.encode_context({"location": "1"}))

    def test_vector_is_identical_across_hash_seeds(self):
        code = ("import json; from core.cognitivo.contextual_memory import ContextEncoder as E; "
                "print(json.dumps(E().encode_context({'location': 'lab', 'mood': 'x'})))")
        outs = set()
        for seed in ("1", "2", "random"):
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED=seed)
            proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                                  text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outs.add(proc.stdout.strip())
        self.assertEqual(len(outs), 1)

    def test_regressed_encoder_or_score_fails_cm7_and_cm8(self):
        from tools import cognitivo_eval
        import core.constitutional_brain as brain

        class HashEncoder(ContextEncoder):
            def encode_context(self, context):
                return [hash(str(context.get(dim, "unknown"))) % 100 / 100.0 for dim in self.context_dimensions]

        class WideEncoder(ContextEncoder):
            def encode_context(self, context):
                vector = super().encode_context(context)
                return [v + 0.001 for v in vector]

        class AllDimsEncoder(ContextEncoder):
            def compute_match(self, vec1, vec2, cue_indexes=None):
                return super().compute_match(vec1, vec2)

        class Cosine(ContextEncoder):
            def compute_match(self, vec1, vec2, cue_indexes=None):
                return self.compute_similarity(vec1, vec2)

        for enc_cls in (HashEncoder, WideEncoder):
            with self.subTest(encoder=enc_cls.__name__):
                stab = cognitivo_eval._cm_stability(enc_cls)
                self.assertLess(stab["golden"], stab["golden_total"])
                self.assertLess(stab["grid_ok"], stab["grid_total"])
        self.assertLess(cognitivo_eval._cm_stability(HashEncoder)["no_hash"], 7)
        for enc_cls in (AllDimsEncoder, Cosine):
            with self.subTest(match=enc_cls.__name__):
                class System(brain.RealContextualMemorySystem):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.context_encoder = enc_cls()

                with cognitivo_eval._cm_det_hash():
                    scores = cognitivo_eval._cm_scores(System, enc_cls)
                expected = {name: value for name, _m, _q, _r, value in cognitivo_eval._CM8_SCORES}
                expected["b1"] = cognitivo_eval._CM8_BANDS
                expected.update({name: value for name, _args, value in cognitivo_eval._CM8_MATCH})
                hits = sum(1 for name, value in expected.items() if scores[name] == value)
                self.assertLess(hits, 26)

    def test_scores_match_the_registered_table(self):
        scores = _checks()["CM8"]["value"]["scores"]
        self.assertEqual(len(scores), 26)
        self.assertEqual((scores["a1"], scores["a6"], scores["a15"], scores["a17"]), (1.0, 0.0, 0.65, 0.15))
        self.assertEqual(_checks()["CM8"]["value"]["hits"], [26, 26])
        self.assertEqual(_checks()["CM7"]["value"]["no_hash"], 7)
        self.assertGreaterEqual(_checks()["CM9"]["value"]["unaffected"], 12)


class ContextualMemoryExtractionContractTests(unittest.TestCase):
    def test_monolith_and_package_reexport_the_same_objects(self):
        contextual_memory = _module()
        import core.cognitivo as package
        import core.constitutional_brain as brain
        for name in _NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(contextual_memory, name))
                self.assertIs(getattr(package, name), getattr(contextual_memory, name))
                self.assertEqual(getattr(contextual_memory, name).__module__, "core.cognitivo.contextual_memory")

    def test_module_imports_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", _BLOCKER], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_pre_registered_checks_match_the_frozen_fingerprint(self):
        from tools.cognitivo_eval import CM_BEHAVIOR_SHA, CM_SOURCE_SHA
        checks = _checks()
        self.assertEqual(set(checks), set(_ALL))
        for cid in _ALL:
            with self.subTest(cid=cid):
                self.assertTrue(checks[cid]["passed"], checks[cid])
        self.assertEqual(checks["CM3"]["value"]["rate"], 1.0)
        self.assertEqual(set(checks["CM3"]["value"]["returned"]), {6})
        self.assertEqual(checks["CM3"]["value"]["recall_hits"], 20)
        self.assertEqual(checks["CM4"]["value"]["control_same"], 20)
        self.assertEqual(checks["CM4"]["value"]["cross_rate"], 1.0)
        self.assertEqual(checks["CM2"]["value"]["rate"], 1.0)
        self.assertEqual(checks["CM6"]["value"]["sha_module"], CM_BEHAVIOR_SHA)
        self.assertEqual(checks["CM6"]["value"]["sha_brain"], CM_BEHAVIOR_SHA)
        self.assertEqual(checks["CM6"]["value"]["source_sha"], CM_SOURCE_SHA)
        self.assertEqual(checks["CM6"]["value"]["clock"], "time.time")
        self.assertTrue(checks["CM6"]["value"]["identity"])


class ContextualMemoryHarnessNegativeContractTests(unittest.TestCase):
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

        class Tampered(brain.RealContextualMemorySystem):
            def get_statistics(self):
                stats = super().get_statistics()
                stats["stored_memories"] += 1
                return stats

        ns = {"RealContextualMemorySystem": Tampered, "ContextualMemory": brain.ContextualMemory,
              "ContextMatch": brain.ContextMatch, "ContextEncoder": brain.ContextEncoder}
        with mock.patch.object(cognitivo_eval, "_cm_determinism", return_value=_checks()["CM4"]["value"]):
            tampered = cognitivo_eval._ms_sha(cognitivo_eval.contextual_memory_fingerprint(ns))
        self.assertNotEqual(tampered, cognitivo_eval.CM_BEHAVIOR_SHA)

    def test_wrong_golden_fails_cm6_and_exits_1(self):
        _module()
        from tools import cognitivo_eval
        cached = _checks()["CM4"]["value"]
        with mock.patch.object(cognitivo_eval, "_cm_determinism", return_value=cached), \
                mock.patch.object(cognitivo_eval, "CM_BEHAVIOR_SHA", "0" * 64):
            checks = {item["id"]: item for item in cognitivo_eval.eval_contextual_memory()}
            code, out, _ = self._run(["--capability", "contextual_memory"])
        self.assertFalse(checks["CM6"]["passed"])
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL)
        self.assertIn('"CM6"', out.split('"failed"')[1])
        with mock.patch.object(cognitivo_eval, "_cm_determinism", return_value=cached), \
                mock.patch.object(cognitivo_eval, "CM_SOURCE_SHA", "0" * 64):
            self.assertFalse({i["id"]: i for i in cognitivo_eval.eval_contextual_memory()}["CM6"]["passed"])

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
        for value in ("contextual-memory", "Contextual_Memory", "contextual", "contextual_memory2"):
            with self.subTest(value=value):
                code, out, err = self._run(["--capability", value])
                self.assertEqual((code, out), (2, ""))
                self.assertIn("desconhecida", err)

    def test_invalid_scenario_exits_3_not_1(self):
        from tools import cognitivo_eval

        def invalid(_criteria="v1"):
            return [cognitivo_eval._check("CM2", None, False, "x", scenario_invalid=True)]

        with mock.patch.dict(cognitivo_eval.EVALUATORS, {"contextual_memory": invalid}):
            code, _, _ = self._run(["--capability", "contextual_memory"])
        self.assertEqual(code, cognitivo_eval.EXIT_INVALID_SCENARIO)
        self.assertNotEqual(cognitivo_eval.EXIT_INVALID_SCENARIO, cognitivo_eval.EXIT_FAIL)

    def test_cli_passes_and_exits_0(self):
        _module()
        from tools import cognitivo_eval
        with mock.patch.object(cognitivo_eval, "_cm_determinism", return_value=_checks()["CM4"]["value"]):
            code, out, _ = self._run(["--capability", "contextual_memory"])
        self.assertEqual(json.loads(out)["failed"], [])
        self.assertEqual(code, cognitivo_eval.EXIT_OK)

    def test_cm9_guard_and_frozen_digests_fail_closed(self):
        _module()
        from tools import cognitivo_eval
        cached = _checks()["CM4"]["value"]
        with mock.patch.object(cognitivo_eval, "_cm_determinism", return_value=cached), \
                mock.patch.object(cognitivo_eval, "_CM_MIN_UNAFFECTED", 999):
            code, out, _ = self._run(["--capability", "contextual_memory"])
        self.assertEqual(code, cognitivo_eval.EXIT_INVALID_SCENARIO)
        self.assertIn('"CM9"', out)
        with mock.patch.object(cognitivo_eval, "_cm_determinism", return_value=cached), \
                mock.patch.object(cognitivo_eval, "_CM_BASE_ROWS_SHA", "0" * 64):
            checks = {item["id"]: item for item in cognitivo_eval.eval_contextual_memory()}
        self.assertFalse(checks["CM9"]["passed"])
        self.assertFalse(checks["CM9"]["value"]["base_rows_sha_ok"])
        rows = list(cognitivo_eval._CM_BASE_ROWS)
        first = next(i for i, flag in enumerate(_fingerprint_grid()["unaffected"]) if flag)
        rows[first] = "0" * 16
        with mock.patch.object(cognitivo_eval, "_cm_determinism", return_value=cached), \
                mock.patch.object(cognitivo_eval, "_CM_BASE_ROWS", tuple(rows)), \
                mock.patch.object(cognitivo_eval, "_CM_BASE_ROWS_SHA",
                                  __import__("hashlib").sha256(json.dumps(rows).encode()).hexdigest()):
            checks = {item["id"]: item for item in cognitivo_eval.eval_contextual_memory()}
        self.assertFalse(checks["CM9"]["passed"])
        self.assertEqual(checks["CM9"]["value"]["diffs"], [first])
        self.assertNotIn("scenario_invalid", checks["CM9"])

    def test_deterministic_hash_is_restored_after_the_fingerprint(self):
        _module()
        import builtins
        import core.constitutional_brain as brain
        from tools import cognitivo_eval
        original = builtins.hash
        with mock.patch.object(cognitivo_eval, "_cm_determinism", return_value=_checks()["CM4"]["value"]):
            cognitivo_eval.contextual_memory_fingerprint(brain)
        self.assertIs(builtins.hash, original)

    def test_source_text_requires_the_marker(self):
        from tools import cognitivo_eval
        with self.assertRaises(ValueError):
            cognitivo_eval.contextual_memory_source_text("nothing here")
        with self.assertRaises(ValueError):
            cognitivo_eval.contextual_memory_source_text("x\n" + cognitivo_eval._CM_MARKER + "\nno end")


if __name__ == "__main__":
    unittest.main()
