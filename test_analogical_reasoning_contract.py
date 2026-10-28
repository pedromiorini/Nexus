"""Contrato do analogical reasoning: entradas inválidas, harness AR e extração strangler.

Critérios AR1–AR6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit f0eacfb) antes do move e antes do eval. O harness é
tools/cognitivo_eval.py --capability analogical_reasoning.
"""

import contextlib
import io
import json
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from core.constitutional_brain import (
    AnalogyMapping,
    AnalogySituation,
    RealAnalogicalReasoningSystem,
)

ROOT = Path(__file__).resolve().parent
_NAMES = ("AnalogySituation", "AnalogyMapping", "RealAnalogicalReasoningSystem")
_ALL = ("AR1", "AR2", "AR3", "AR4", "AR5", "AR6")
_GAPS = {"AR3", "AR4"}  # gaps medidos e registrados, fora de correção nesta rodada
_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.analogical_reasoning as mod
system = mod.RealAnalogicalReasoningSystem()
system.add_situation("src", ["a", "b"], [("CAUSES", "a", "b")], "move a to b")
mapping = system.find_analogy("tgt", ["x", "y"], [("CAUSES", "x", "y")])
assert system.transfer_solution(mapping) == "move x to y" and mapping.quality_score == 1.0
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    import core.cognitivo.analogical_reasoning as analogical_reasoning
    return analogical_reasoning


def _quiet(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealAnalogicalReasoningSystem(*args, **kwargs)


def _with(known=(), **kwargs):
    system = _quiet(**kwargs)
    for domain, entities, relations, solution in known:
        system.add_situation(domain, entities, relations, solution)
    return system


_CAUSES = ("CAUSES", "a", "b")
_ONE = [("S", ["a", "b"], [_CAUSES], "move a to b")]


class _Boom:
    def __getattr__(self, name):
        raise AssertionError("called " + name)


class AnalogicalReasoningInvalidInputContractTests(unittest.TestCase):
    def test_empty_or_unresolvable_input_returns_none_without_counting(self):
        system = _with(_ONE)
        for text in ("", "   ", "Nothing relational here."):
            with self.subTest(text=text):
                self.assertIsNone(system.reason_by_analogy(text, "T"))
        self.assertIsNone(system.reason_by_analogy("Alpha causes Bravo.", "S"))  # mesmo domínio: sem fonte compatível
        self.assertIsNone(_quiet().reason_by_analogy("Alpha causes Bravo.", "T"))  # sem situações conhecidas
        stats = system.get_statistics()
        self.assertEqual((stats["total_analogies_found"], stats["successful_transfers"]), (0, 0))

    def test_wrong_types_are_not_coerced(self):
        for bad in (None, 5):
            with self.subTest(text=bad), self.assertRaises(AttributeError):
                _quiet().reason_by_analogy(bad, "T")
        with self.assertRaises(AttributeError):
            _quiet()._extract_entities(None)
        with self.assertRaises(AttributeError):
            _quiet().transfer_solution(None)
        with self.assertRaises(TypeError):
            _with(_ONE).find_analogy("T", ["x"], None)
        with self.assertRaises(TypeError):
            _with(_ONE).find_analogy("T", ["x"], [5])
        with self.assertRaises(ValueError):
            _with(_ONE).find_analogy("T", ["x"], [("CAUSES", "x")])
        with self.assertRaises(TypeError):
            _with([("S", ["a"], None, None)]).find_analogy("T", ["x", "y"], [("CAUSES", "x", "y")])
        with self.assertRaises(TypeError):
            _quiet().find_analogy()
        self.assertIsNone(_quiet().find_analogy("T", ["x"], None))  # sem fontes, o alvo nem é lido
        self.assertIsNotNone(_quiet().add_situation(None, None, None))

    def test_transfer_requires_a_string_solution_when_entities_are_mapped(self):
        source = AnalogySituation("s", "S", ["a"], [_CAUSES], 5)
        with self.assertRaises(AttributeError):
            _quiet().transfer_solution(AnalogyMapping("m", source, source, {"a": "x"}, 1.0))

    def test_no_solution_means_no_transfer_and_no_count(self):
        for solution in (None, ""):
            with self.subTest(solution=solution):
                system = _with([("S", ["a", "b"], [_CAUSES], solution)])
                mapping = system.find_analogy("T", ["x", "y"], [("CAUSES", "x", "y")])
                self.assertIsNone(system.transfer_solution(mapping))
                self.assertEqual(system.successful_transfers, 0)

    def test_collaborators_are_stored_and_never_read(self):
        system = _with(_ONE, reasoning=_Boom(), memory=_Boom(), semantic_networks=_Boom())
        mapping = system.find_analogy("T", ["x", "y"], [("CAUSES", "x", "y")])
        self.assertEqual(system.transfer_solution(mapping), "move x to y")
        self.assertEqual(system.reason_by_analogy("Storm causes Damage.", "T"), "move entity1 to entity2")
        self.assertEqual(system.get_statistics()["known_situations"], 1)
        with self.assertRaises(TypeError):
            RealAnalogicalReasoningSystem(1, 2, 3, 4)

    def test_current_behavior_is_recorded_as_gaps(self):
        # Gaps medidos (AR3, AR4, AR5), registrados e fora de correção.
        system = _with([("S", ["Rain", "Flood"], [("CAUSES", "Rain", "Flood")], "stop Rain")])
        self.assertEqual(system.reason_by_analogy("Storm causes Damage.", "T"), "stop entity1")
        self.assertEqual(_quiet()._extract_relations("dissimilar to the rest"), [("SIMILAR", "entity1", "entity2")])
        self.assertEqual(_quiet()._extract_entities("The cat causes Harm"), ["The", "Harm"])
        system = _with(_ONE)
        mapping = system.find_analogy("T", ["x", "y"], [("CAUSES", "x", "y")])
        system.transfer_solution(mapping)
        system.transfer_solution(mapping)
        self.assertEqual(system.get_statistics()["transfer_success_rate"], 2.0)
        self.assertIsNotNone(_with([("Physics", ["a", "b"], [_CAUSES], None)]).find_analogy(
            "physics", ["x", "y"], [("CAUSES", "x", "y")]))


class AnalogicalReasoningExtractionContractTests(unittest.TestCase):
    def test_monolith_and_package_reexport_the_same_objects(self):
        analogical_reasoning = _module()
        import core.cognitivo as package
        import core.constitutional_brain as brain
        for name in _NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(analogical_reasoning, name))
                self.assertIs(getattr(package, name), getattr(analogical_reasoning, name))
                self.assertEqual(getattr(analogical_reasoning, name).__module__,
                                 "core.cognitivo.analogical_reasoning")

    def test_module_imports_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", _BLOCKER], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_pre_registered_checks_match_the_frozen_fingerprint(self):
        _module()
        from tools.cognitivo_eval import AR_BEHAVIOR_SHA, AR_SOURCE_SHA, eval_analogical_reasoning
        checks = {item["id"]: item for item in eval_analogical_reasoning()}
        self.assertEqual(set(checks), set(_ALL))
        for cid in sorted(set(_ALL) - _GAPS):
            with self.subTest(cid=cid):
                self.assertTrue(checks[cid]["passed"], checks[cid])
        # AR3 e AR4: o gap está medido; os controles têm de passar para o teste não reprovar tudo.
        self.assertEqual(checks["AR3"]["value"]["control_isomorphic"]["rate"], 1.0)
        self.assertEqual(checks["AR4"]["value"]["control_plain"]["rate"], 1.0)
        for cid in sorted(_GAPS):
            for name, item in checks[cid]["value"].items():
                with self.subTest(case=name):
                    self.assertTrue(0.0 <= item["rate"] <= 1.0)
                    self.assertEqual(item["total"], 20)
        self.assertEqual(checks["AR6"]["value"]["sha_module"], AR_BEHAVIOR_SHA)
        self.assertEqual(checks["AR6"]["value"]["sha_brain"], AR_BEHAVIOR_SHA)
        self.assertEqual(checks["AR6"]["value"]["source_sha"], AR_SOURCE_SHA)
        self.assertEqual(checks["AR6"]["value"]["clock"], "none")
        self.assertTrue(checks["AR6"]["value"]["identity"])


class AnalogicalReasoningHarnessNegativeContractTests(unittest.TestCase):
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

        class Tampered(brain.RealAnalogicalReasoningSystem):
            def _map_structures(self, source, target):
                return None

        ns = {"RealAnalogicalReasoningSystem": Tampered, "AnalogySituation": brain.AnalogySituation,
              "AnalogyMapping": brain.AnalogyMapping}
        tampered = cognitivo_eval._ms_sha(cognitivo_eval.analogical_reasoning_fingerprint(ns))
        self.assertNotEqual(tampered, cognitivo_eval.AR_BEHAVIOR_SHA)

    def test_wrong_golden_fails_ar6_and_exits_1(self):
        _module()
        from tools import cognitivo_eval
        with mock.patch.object(cognitivo_eval, "AR_BEHAVIOR_SHA", "0" * 64):
            checks = {item["id"]: item for item in cognitivo_eval.eval_analogical_reasoning()}
            code, out, _ = self._run(["--capability", "analogical_reasoning"])
        self.assertFalse(checks["AR6"]["passed"])
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL)
        self.assertIn('"AR6"', out.split('"failed"')[1])
        with mock.patch.object(cognitivo_eval, "AR_SOURCE_SHA", "0" * 64):
            self.assertFalse({i["id"]: i for i in cognitivo_eval.eval_analogical_reasoning()}["AR6"]["passed"])

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
        for value in ("analogical-reasoning", "Analogical_Reasoning", "analogy", "analogical_reasoning2"):
            with self.subTest(value=value):
                code, out, err = self._run(["--capability", value])
                self.assertEqual((code, out), (2, ""))
                self.assertIn("desconhecida", err)

    def test_invalid_scenario_exits_3_not_1(self):
        from tools import cognitivo_eval

        def invalid(_criteria="v1"):
            return [cognitivo_eval._check("AR1", None, False, "x", scenario_invalid=True)]

        with mock.patch.dict(cognitivo_eval.EVALUATORS, {"analogical_reasoning": invalid}):
            code, _, _ = self._run(["--capability", "analogical_reasoning"])
        self.assertEqual(code, cognitivo_eval.EXIT_INVALID_SCENARIO)
        self.assertNotEqual(cognitivo_eval.EXIT_INVALID_SCENARIO, cognitivo_eval.EXIT_FAIL)

    def test_cli_reports_only_the_registered_gaps_and_exits_1(self):
        _module()
        from tools import cognitivo_eval
        code, out, _ = self._run(["--capability", "analogical_reasoning"])
        failed = json.loads(out)["failed"]
        self.assertTrue(set(failed) <= _GAPS, failed)
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL if failed else cognitivo_eval.EXIT_OK)

    def test_fingerprint_restores_the_global_random_state(self):
        _module()
        import core.constitutional_brain as brain
        from tools import cognitivo_eval
        random.seed(11)
        before = random.getstate()
        cognitivo_eval.analogical_reasoning_fingerprint(brain)
        self.assertEqual(random.getstate(), before)

    def test_source_text_requires_the_marker(self):
        from tools import cognitivo_eval
        with self.assertRaises(ValueError):
            cognitivo_eval.analogical_reasoning_source_text("nothing here")
        with self.assertRaises(ValueError):
            cognitivo_eval.analogical_reasoning_source_text("x\n" + cognitivo_eval._AR_MARKER + "\nno end")


if __name__ == "__main__":
    unittest.main()
