"""Contrato do analogical reasoning: entradas inválidas, harness AR e extração strangler.

Critérios AR1–AR6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit f0eacfb) antes do move; AR7–AR10 e a correção de AR3/AR4 pré-registrados
na seção «Analogical Reasoning — correção de AR3 e AR4» (commit 0e0e1d3) antes
do contrato RED e do código novo. O harness é
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
_ALL = ("AR1", "AR2", "AR3", "AR4", "AR5", "AR6", "AR7", "AR8", "AR9", "AR10")
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
        self.assertEqual(system.reason_by_analogy("Storm causes Damage.", "T"), "move Storm to Damage")
        self.assertEqual(system.get_statistics()["known_situations"], 1)
        with self.assertRaises(TypeError):
            RealAnalogicalReasoningSystem(1, 2, 3, 4)

    def test_current_behavior_is_recorded_as_gaps(self):
        # Gaps residuais medidos (AR5), registrados e fora desta correção.
        system = _with([("S", ["Rain", "Flood"], [("CAUSES", "Rain", "Flood")], "stop Rain")])
        self.assertEqual(system.reason_by_analogy("Storm causes Damage.", "T"), "stop Storm")
        self.assertEqual(system.reason_by_analogy("The cat causes Harm.", "T"), "stop The")
        self.assertEqual(system.reason_by_analogy("it causes Harm.", "T"), "stop entity1")
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
        for cid in _ALL:
            with self.subTest(cid=cid):
                self.assertTrue(checks[cid]["passed"], checks[cid])
        for cid in ("AR3", "AR4"):
            for name, item in checks[cid]["value"].items():
                with self.subTest(case=name):
                    self.assertEqual((item["hits"], item["total"], item["rate"]), (20, 20, 1.0))
        self.assertEqual(checks["AR10"]["value"]["diffs"], [])
        self.assertGreaterEqual(checks["AR10"]["value"]["unaffected"], 52)
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
            def get_statistics(self):
                stats = super().get_statistics()
                stats["known_situations"] += 1
                return stats

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

    def test_cli_passes_every_criterion_and_exits_0(self):
        _module()
        from tools import cognitivo_eval
        code, out, _ = self._run(["--capability", "analogical_reasoning"])
        report = json.loads(out)
        self.assertEqual(report["failed"], [])
        self.assertEqual([item["id"] for item in report["analogical_reasoning"]], list(_ALL))
        self.assertEqual(code, cognitivo_eval.EXIT_OK)

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


def _legacy_map_structures(self, source, target):
    """Mapeamento ingênuo do #121 (uma relação da fonte serve a várias do alvo; sem injetividade)."""
    entity_mappings, matched = {}, 0
    for t_rel, t_e1, t_e2 in target.relations:
        for s_rel, s_e1, s_e2 in source.relations:
            if t_rel == s_rel:
                entity_mappings[s_e1] = t_e1
                entity_mappings[s_e2] = t_e2
                matched += 1
                break
    if matched == 0:
        return None
    return AnalogyMapping("legacy", source, target, entity_mappings,
                          matched / max(len(source.relations), len(target.relations)))


def _legacy_transfer_solution(self, mapping):
    """Substituição em sequência e por substring do #121."""
    if not mapping.source_situation.solution:
        return None
    adapted = mapping.source_situation.solution
    for source_entity, target_entity in mapping.entity_mappings.items():
        adapted = adapted.replace(source_entity, target_entity)
    self.successful_transfers += 1
    return adapted


def _legacy_reason_by_analogy(self, problem_description, target_domain):
    """reason_by_analogy do #121: ignora as entidades extraídas."""
    mapping = self.find_analogy(target_domain, self._extract_entities(problem_description),
                                self._extract_relations(problem_description))
    return self.transfer_solution(mapping) if mapping else None


def _map(known, rels):
    system = _with(known)
    return system.find_analogy("T", [], rels)


def _transfer(mapping_spec, solution):
    source = AnalogySituation("s", "S", [], [], solution)
    system = _quiet()
    return system.transfer_solution(AnalogyMapping("m", source, source, dict(mapping_spec), 1.0)), system


_C, _P = "CAUSES", "PART_OF"


class AnalogicalReasoningMappingContractTests(unittest.TestCase):
    """AR3/AR7: um-para-um e injetivo."""

    def test_a_source_relation_supports_only_one_target_relation(self):
        mapping = _map([("S", [], [(_C, "a", "b")], None)], [(_C, "x", "y"), (_C, "u", "v")])
        self.assertEqual(mapping.entity_mappings, {"a": "x", "b": "y"})
        self.assertEqual(mapping.quality_score, 0.5)

    def test_all_relations_map_when_the_structures_are_isomorphic(self):
        mapping = _map([("S", [], [(_C, "a", "b"), (_C, "c", "d")], None)], [(_C, "x", "y"), (_C, "u", "v")])
        self.assertEqual(mapping.entity_mappings, {"a": "x", "b": "y", "c": "u", "d": "v"})
        self.assertEqual(mapping.quality_score, 1.0)

    def test_two_source_entities_never_share_a_target_entity(self):
        mapping = _map([("S", [], [(_C, "a", "b"), (_P, "c", "d")], None)], [(_C, "x", "y"), (_P, "x", "z")])
        self.assertEqual(mapping.entity_mappings, {"a": "x", "b": "y"})
        values = list(_map([("S", [], [(_C, "a", "b"), (_C, "c", "d")], None)],
                           [(_C, "x", "y"), (_C, "x", "z")]).entity_mappings.values())
        self.assertEqual(len(values), len(set(values)))

    def test_a_source_entity_keeps_its_first_target(self):
        mapping = _map([("S", [], [(_C, "a", "b"), (_P, "b", "c")], None)], [(_C, "x", "y"), (_P, "w", "z")])
        self.assertEqual(mapping.entity_mappings, {"a": "x", "b": "y"})
        self.assertEqual(mapping.quality_score, 0.5)
        chain = _map([("S", [], [(_C, "a", "b"), (_P, "b", "c")], None)], [(_C, "x", "y"), (_P, "y", "z")])
        self.assertEqual(chain.entity_mappings, {"a": "x", "b": "y", "c": "z"})

    def test_self_loops_only_match_self_loops(self):
        self.assertIsNone(_map([("S", [], [(_C, "a", "b")], None)], [(_C, "x", "x")]))
        self.assertIsNone(_map([("S", [], [(_C, "a", "a")], None)], [(_C, "x", "y")]))
        self.assertEqual(_map([("S", [], [(_C, "a", "a")], None)], [(_C, "x", "x")]).entity_mappings, {"a": "x"})

    def test_an_incompatible_source_relation_is_skipped_for_a_later_one(self):
        mapping = _map([("S", [], [(_C, "a", "b"), (_C, "b", "c"), (_C, "d", "e")], None)],
                       [(_C, "x", "y"), (_C, "p", "q")])
        self.assertEqual(mapping.entity_mappings, {"a": "x", "b": "y", "d": "p", "e": "q"})
        self.assertEqual(mapping.quality_score, 2 / 3)

    def test_the_best_source_wins_on_the_corrected_quality(self):
        mapping = _map([("S1", [], [(_C, "a", "b")], None), ("S2", [], [(_C, "c", "d"), (_C, "e", "f")], None)],
                       [(_C, "x", "y"), (_C, "u", "v")])
        self.assertEqual(mapping.source_situation.domain, "S2")
        self.assertEqual(mapping.quality_score, 1.0)

    def test_no_compatible_relation_gives_no_mapping_and_no_count(self):
        system = _with([("S", [], [(_C, "a", "b")], None)])
        self.assertIsNone(system.find_analogy("T", [], [(_C, "x", "x")]))
        self.assertEqual(system.total_analogies_found, 0)

    def test_inputs_are_not_mutated(self):
        known = [("S", ["a", "b"], [(_C, "a", "b"), (_C, "c", "d")], "go")]
        rels = [(_C, "x", "y"), (_C, "x", "z")]
        system = _with(known)
        system.find_analogy("T", ["x", "y", "z"], rels)
        self.assertEqual(rels, [(_C, "x", "y"), (_C, "x", "z")])
        self.assertEqual(system.known_situations[0].relations, [(_C, "a", "b"), (_C, "c", "d")])

    def test_malformed_relations_still_fail_loudly(self):
        with self.assertRaises(ValueError):
            _with(_ONE).find_analogy("T", [], [(_C, "x")])
        with self.assertRaises(TypeError):
            _with(_ONE).find_analogy("T", [], [5])
        with self.assertRaises(TypeError):
            _with([("S", [], None, None)]).find_analogy("T", [], [(_C, "x", "y")])


class AnalogicalReasoningTransferContractTests(unittest.TestCase):
    """AR4/AR8/AR9: transferência simultânea, por palavra inteira, com as entidades extraídas."""

    def test_substitution_is_simultaneous(self):
        self.assertEqual(_transfer({"a": "b", "b": "c"}, "a then b")[0], "b then c")
        self.assertEqual(_transfer({"a": "b", "b": "a"}, "a and b")[0], "b and a")

    def test_substitution_only_touches_whole_words(self):
        self.assertEqual(_transfer({"sun": "alpha"}, "sunny sun sun's sun.")[0], "sunny alpha alpha's alpha.")
        self.assertEqual(_transfer({"cat": "dog"}, "concatenate cat")[0], "concatenate dog")
        self.assertEqual(_transfer({"sun": "alpha"}, "Sun sun")[0], "Sun alpha")
        self.assertEqual(_transfer({"a.b": "x"}, "a.b and axb")[0], "x and axb")
        self.assertEqual(_transfer({"C++": "Rust"}, "use C++ now, C++11 later")[0], "use Rust now, C++11 later")

    def test_the_longest_key_wins_and_multiword_keys_work(self):
        self.assertEqual(_transfer({"Rain": "X", "Rainfall": "Y"}, "Rainfall and Rain")[0], "Y and X")
        self.assertEqual(_transfer({"New York": "Paris"}, "visit New York now, not New Yorker")[0],
                         "visit Paris now, not New Yorker")

    def test_identity_empty_keys_and_empty_mapping_leave_the_text(self):
        self.assertEqual(_transfer({"a": "a"}, "a b")[0], "a b")
        self.assertEqual(_transfer({"": "x", "a": "b"}, "ab a")[0], "ab b")
        text, system = _transfer({}, "keep it")
        self.assertEqual((text, system.successful_transfers), ("keep it", 1))

    def test_replacement_text_is_literal_not_a_regex_template(self):
        self.assertEqual(_transfer({"a": r"\1 \g<0>"}, "a")[0], r"\1 \g<0>")

    def test_a_non_string_solution_is_rejected_with_attribute_error_only_when_mapped(self):
        for bad in (5, ["a"], 1.5):
            with self.subTest(solution=bad), self.assertRaises(AttributeError):
                _transfer({"a": "x"}, bad)
        self.assertEqual(_transfer({}, 5)[0], 5)
        self.assertIsNone(_transfer({"a": "x"}, "")[0])
        self.assertIsNone(_transfer({"a": "x"}, None)[0])

    def test_reason_by_analogy_substitutes_the_extracted_entities(self):
        system = _with([("S", ["Rain", "Flood"], [(_C, "Rain", "Flood")], "avoid Rain before Flood")])
        self.assertEqual(system.reason_by_analogy("Alpha causes Bravo.", "T"), "avoid Alpha before Bravo")
        self.assertEqual(system.reason_by_analogy("Cc causes Dd Ee Ff", "T"), "avoid Cc before Dd")
        stats = system.get_statistics()
        self.assertEqual((stats["total_analogies_found"], stats["successful_transfers"]), (2, 2))

    def test_reason_by_analogy_keeps_placeholders_with_fewer_than_two_entities(self):
        system = _with([("S", ["Rain", "Flood"], [(_C, "Rain", "Flood")], "stop Rain")])
        self.assertEqual(system.reason_by_analogy("it causes Harm.", "T"), "stop entity1")
        self.assertEqual(system.reason_by_analogy("the thing causes it.", "T"), "stop entity1")
        self.assertEqual(system.reason_by_analogy("The cat causes Harm.", "T"), "stop The")

    def test_reason_by_analogy_without_solution_or_source_domain_returns_none(self):
        system = _with([("S", ["Rain", "Flood"], [(_C, "Rain", "Flood")], None)])
        self.assertIsNone(system.reason_by_analogy("Alpha causes Bravo.", "T"))
        self.assertEqual((system.total_analogies_found, system.successful_transfers), (1, 0))
        other = _with([("S", ["Rain", "Flood"], [(_C, "Rain", "Flood")], "stop Rain")])
        self.assertIsNone(other.reason_by_analogy("Alpha causes Bravo.", "S"))
        self.assertEqual((other.total_analogies_found, other.successful_transfers), (0, 0))


class AnalogicalReasoningRegressionContractTests(unittest.TestCase):
    """Comportamento regredido reprova o harness com código 1; cenário inválido, com código 3."""

    def _run(self, argv=("--capability", "analogical_reasoning")):
        from tools import cognitivo_eval
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            code = cognitivo_eval.main(list(argv))
        return code, json.loads(out.getvalue()) if out.getvalue().strip().startswith("{") else {}

    def _failed_with(self, name, legacy):
        with mock.patch.object(RealAnalogicalReasoningSystem, name, legacy):
            return self._run()

    def test_naive_mapping_fails_ar1_ar3_ar7(self):
        code, report = self._failed_with("_map_structures", _legacy_map_structures)
        self.assertEqual(code, 1)
        self.assertTrue({"AR1", "AR3", "AR7"} <= set(report["failed"]), report["failed"])
        self.assertNotIn("AR10", report["failed"])

    def test_sequential_substring_transfer_fails_ar1_ar4_ar8(self):
        code, report = self._failed_with("transfer_solution", _legacy_transfer_solution)
        self.assertEqual(code, 1)
        self.assertTrue({"AR4", "AR8", "AR9"} <= set(report["failed"]), report["failed"])

    def test_reason_without_extracted_entities_fails_ar4_ar9(self):
        code, report = self._failed_with("reason_by_analogy", _legacy_reason_by_analogy)
        self.assertEqual(code, 1)
        self.assertTrue({"AR4", "AR9"} <= set(report["failed"]), report["failed"])

    def test_the_guard_exits_3_when_too_few_rows_are_unaffected(self):
        from tools import cognitivo_eval
        with mock.patch.object(cognitivo_eval, "_AR_MIN_UNAFFECTED", 10 ** 6):
            code, report = self._run()
        self.assertEqual(code, cognitivo_eval.EXIT_INVALID_SCENARIO)
        ar10 = next(item for item in report["analogical_reasoning"] if item["id"] == "AR10")
        self.assertTrue(ar10["scenario_invalid"])
        self.assertFalse(ar10["passed"])

    def test_a_tampered_base_digest_list_exits_3_instead_of_trusting_it(self):
        from tools import cognitivo_eval
        rows = ("0" * 16,) + tuple(cognitivo_eval._AR_BASE_ROWS[1:])
        with mock.patch.object(cognitivo_eval, "_AR_BASE_ROWS", rows):
            code, report = self._run()
        self.assertEqual(code, cognitivo_eval.EXIT_INVALID_SCENARIO)
        ar10 = next(item for item in report["analogical_reasoning"] if item["id"] == "AR10")
        self.assertFalse(ar10["value"]["base_rows_sha_ok"])

    def test_a_changed_unaffected_row_fails_ar10_with_exit_1(self):
        import hashlib
        from tools import cognitivo_eval
        import core.cognitivo.analogical_reasoning as module
        fp = cognitivo_eval.analogical_reasoning_fingerprint(module)
        index = fp["grid"]["unaffected"].index(True)
        rows = list(cognitivo_eval._AR_BASE_ROWS)
        rows[index] = "f" * 16
        sha = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
        with mock.patch.object(cognitivo_eval, "_AR_BASE_ROWS", tuple(rows)), \
                mock.patch.object(cognitivo_eval, "_AR_BASE_ROWS_SHA", sha):
            code, report = self._run()
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL)
        self.assertEqual(report["failed"], ["AR10"])
        ar10 = next(item for item in report["analogical_reasoning"] if item["id"] == "AR10")
        self.assertEqual(ar10["value"]["diffs"], [index])

    def test_unknown_criteria_version_exits_2(self):
        code, _ = self._run(("--capability", "analogical_reasoning", "--criteria", "v9"))
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
