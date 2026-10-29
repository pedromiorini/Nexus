"""Contrato da memória contextual: entradas inválidas, harness CM e extração strangler.

Critérios CM1–CM6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit c8ca267) antes do harness, do contrato e do move. O harness é
tools/cognitivo_eval.py --capability contextual_memory. CM3 e CM4 falham como
pré-registrado (limiar padrão não seletivo; hash() dependente de PYTHONHASHSEED):
o move não os corrige.
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
_ALL = ("CM1", "CM2", "CM3", "CM4", "CM5", "CM6")
_GAPS = {"CM3", "CM4"}
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
        # Gaps medidos (CM3/CM5), registrados e fora desta extração.
        system = _quiet()
        for i, ctx in enumerate(({"location": "a"}, {"time": "c"}, {"social": "e"})):
            system.store_with_context(f"x{i}", ctx)
        self.assertEqual(len(system.retrieve_by_context(context={"location": "zzz"})), 3)
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
                self.assertEqual(checks[cid]["passed"], cid not in _GAPS, checks[cid])
        self.assertEqual(checks["CM3"]["value"]["rate"], 0.0)
        self.assertEqual(set(checks["CM3"]["value"]["returned"]), {30})
        self.assertEqual(checks["CM4"]["value"]["control_same"], 20)
        self.assertEqual(checks["CM4"]["value"]["cross_rate"], 0.0)
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

    def test_cli_reports_only_the_registered_gaps_and_exits_1(self):
        _module()
        from tools import cognitivo_eval
        with mock.patch.object(cognitivo_eval, "_cm_determinism", return_value=_checks()["CM4"]["value"]):
            code, out, _ = self._run(["--capability", "contextual_memory"])
        failed = json.loads(out)["failed"]
        self.assertTrue(set(failed) <= _GAPS, failed)
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL if failed else cognitivo_eval.EXIT_OK)

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
