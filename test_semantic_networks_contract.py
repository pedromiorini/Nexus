"""Contrato das redes semânticas: entradas inválidas, harness SN e extração strangler.

Critérios SN1–SN6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit 4ba5ca1) antes do harness, do contrato e do move. O harness é
tools/cognitivo_eval.py --capability semantic_networks. SN3 e SN4 falharam como
pré-registrado no move (a ativação dependia da ordem das relações; consultas IS-A
em ciclo estouravam a recursão); a errata E1 (card t150s945) os fecha e acrescenta
SN3b/SN4b, ver a seção "Semantic Networks — fechar os gaps SN3/SN4".
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
from unittest import mock

from core.constitutional_brain import ConceptNode, RealSemanticNetworksSystem, SemanticRelation

ROOT = Path(__file__).resolve().parent
_NAMES = ("ConceptNode", "SemanticRelation", "RealSemanticNetworksSystem")
_ALL = ("SN1", "SN2", "SN3", "SN4", "SN3b", "SN4b", "SN5", "SN6")
_GAPS = set()
_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.semantic_networks as mod
system = mod.RealSemanticNetworksSystem()
system.add_concept("Animal", {"alive": True})
system.add_relation("Cat", "IS-A", "Animal")
assert system.verify_hierarchy("Cat", "animal")
assert system.inherit_properties("Cat") == {"alive": True}
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    import core.cognitivo.semantic_networks as semantic_networks
    return semantic_networks


def _quiet(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealSemanticNetworksSystem(*args, **kwargs)


class _Boom:
    def __getattribute__(self, name):
        raise AssertionError("called " + name)


@functools.lru_cache(maxsize=1)
def _checks():
    _module()
    from tools.cognitivo_eval import eval_semantic_networks
    return {item["id"]: item for item in eval_semantic_networks()}


def _acyclic_safe(base):
    class Safe(base):
        def verify_hierarchy(self, child_label, ancestor_label):
            seen, stack = set(), [child_label]
            while stack:
                current = stack.pop()
                if current.lower() == ancestor_label.lower():
                    return True
                if current.lower() in seen:
                    continue
                seen.add(current.lower())
                stack.extend(p.label for p in self.get_related_concepts(current, "IS-A"))
            return False

        def inherit_properties(self, concept_label):
            merged, seen, stack = {}, set(), [concept_label]
            while stack:
                current = stack.pop()
                if current.lower() in seen:
                    continue
                seen.add(current.lower())
                concept = self._find_or_create_concept(current)
                for key, value in concept.properties.items():
                    merged.setdefault(key, value)
                stack.extend(p.label for p in reversed(self.get_related_concepts(current, "IS-A")))
            return merged
    return Safe


def _order_independent(base):
    class Synchronous(base):
        def spread_activation(self, iterations=3):
            for _ in range(iterations):
                delta = {}
                for relation in self.relations:
                    source, target = self.concepts.get(relation.source_id), self.concepts.get(relation.target_id)
                    if source and target and source.activation > 0.1:
                        delta[target.concept_id] = (delta.get(target.concept_id, 0.0)
                                                    + source.activation * self.spread_factor * relation.strength)
                for concept_id, amount in delta.items():
                    concept = self.concepts[concept_id]
                    concept.activation = min(1.0, concept.activation + amount)
                for concept in self.concepts.values():
                    concept.activation = max(0.0, concept.activation - self.activation_decay)
            return {c.label: c.activation for c in self.concepts.values() if c.activation > 0.1}
    return Synchronous


class SemanticNetworksInvalidInputContractTests(unittest.TestCase):
    def test_wrong_types_are_not_coerced(self):
        with self.assertRaises(TypeError):
            _quiet().add_concept()
        with self.assertRaises(TypeError):
            _quiet().add_relation("A", "IS-A")
        for bad in ("a", 1.5, None):
            with self.subTest(iterations=bad), self.assertRaises(TypeError):
                _quiet().spread_activation(bad)
        with self.assertRaises(TypeError):
            _quiet(1, 2, 3)
        with self.assertRaises(AttributeError):
            _quiet().verify_hierarchy(1, "a")
        populated = _quiet()
        populated.add_concept("A")
        with self.assertRaises(AttributeError):
            populated.get_related_concepts(5)
        with self.assertRaises(AttributeError):
            populated.query_by_properties(None)
        system = _quiet()
        with self.assertRaises(TypeError):
            system.add_concept("V")
            system.activate_concept("V", "a")
        self.assertEqual((system.activations_performed, [c.label for c in system.concepts.values()]), (0, ["V"]))

    def test_failed_relation_leaves_a_partial_state(self):
        system = _quiet()
        with self.assertRaises(AttributeError):
            system.add_relation(1, "IS-A", 2)
        self.assertEqual((len(system.concepts), len(system.relations), system.total_relations), (1, 0, 0))

    def test_bad_query_arguments_only_fail_when_there_is_something_to_compare(self):
        empty = _quiet()
        self.assertEqual(empty.get_related_concepts(5), [])
        self.assertEqual(empty.query_by_properties(None), [])
        self.assertEqual(empty.get_statistics()["avg_activation"], 0.0)
        with self.assertRaises(AttributeError):
            empty.inherit_properties(5)

    def test_collaborators_are_stored_and_never_read(self):
        boom = _Boom()
        system = _quiet(knowledge_graph=boom, language=boom)
        self.assertIs(system.kg, boom)
        self.assertIs(system.language, boom)
        system.add_concept("A", {"k": 1})
        system.add_relation("A", "IS-A", "B")
        system.activate_concept("A")
        system.spread_activation(2)
        system.get_related_concepts("A")
        system.inherit_properties("A")
        system.verify_hierarchy("A", "B")
        system.query_by_properties({"k": 1})
        system.get_statistics()

    def test_lookup_ignores_case_and_reads_do_not_create_concepts(self):
        system = _quiet()
        system.add_relation("Cat", "IS-A", "animal")
        system.add_relation("cat", "HAS-A", "Fur")
        self.assertEqual([c.label for c in system.concepts.values()], ["Cat", "animal", "Fur"])
        self.assertEqual([c.label for c in system.get_related_concepts("CAT", "IS-A")], ["animal"])
        self.assertEqual(system.get_related_concepts("zzz"), [])
        self.assertFalse(system.verify_hierarchy("Q", "R"))
        self.assertEqual(len(system.concepts), 3)
        self.assertEqual(system.inherit_properties("Ghost"), {})
        self.assertEqual(len(system.concepts), 4)

    def test_the_registered_gaps_are_closed_and_the_rest_of_the_behavior_is_kept(self):
        def chain(reverse):
            system = _quiet()
            for label in "ABC":
                system.add_concept(label)
            relations = [("A", "x", "B", 1.0), ("B", "x", "C", 0.5)]
            for source, kind, target, strength in (reversed(relations) if reverse else relations):
                system.add_relation(source, kind, target, strength)
            system.activate_concept("A", 1.0)
            return system.spread_activation(3)

        for reverse in (False, True):
            self.assertAlmostEqual(chain(reverse)["C"], 0.325, places=9)
        ring = _quiet()
        for source, target in (("A", "B"), ("B", "C"), ("C", "A")):
            ring.add_relation(source, "IS-A", target)
        self.assertEqual(ring.inherit_properties("A"), {})
        self.assertFalse(ring.verify_hierarchy("A", "Z"))
        self.assertTrue(ring.verify_hierarchy("A", "C"))
        system = _quiet()
        system.activate_concept("V", -0.5)
        system.activate_concept("N", float("nan"))
        self.assertEqual(system.concepts[next(iter(system.concepts))].activation, -0.5)
        self.assertEqual(list(system.concepts.values())[1].activation, 1.0)
        system.add_relation("P", "x", "Q", strength=5.0)
        self.assertEqual(system.relations[0].strength, 5.0)
        self.assertTrue(math.isnan(_quiet().add_relation("P", "x", "Q", strength=float("nan")).strength))
        props = {"a": 1}
        self.assertIs(_quiet().add_concept("X", props).properties, props)
        self.assertIsInstance(ConceptNode("i", "l", {}, 0.0, 0.0), ConceptNode)
        self.assertEqual(SemanticRelation("i", "IS-A", "s", "t", 1.0).strength, 1.0)


class SemanticNetworksExtractionContractTests(unittest.TestCase):
    def test_monolith_and_package_reexport_the_same_objects(self):
        semantic_networks = _module()
        import core.cognitivo as package
        import core.constitutional_brain as brain
        for name in _NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(semantic_networks, name))
                self.assertIs(getattr(package, name), getattr(semantic_networks, name))
                self.assertEqual(getattr(semantic_networks, name).__module__, "core.cognitivo.semantic_networks")

    def test_module_imports_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", _BLOCKER], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_pre_registered_checks_match_the_frozen_fingerprint(self):
        from tools.cognitivo_eval import SN_BEHAVIOR_SHA, SN_SOURCE_SHA
        checks = _checks()
        self.assertEqual(set(checks), set(_ALL))
        for cid in _ALL:
            with self.subTest(cid=cid):
                self.assertEqual(checks[cid]["passed"], cid not in _GAPS, checks[cid])
                self.assertFalse(checks[cid].get("scenario_invalid", False), checks[cid])
        self.assertEqual(checks["SN1"]["value"]["reference_matches"], 120)
        self.assertEqual(checks["SN1"]["value"]["mutated_inputs"], 0)
        self.assertEqual(checks["SN2"]["value"]["rate"], 1.0)
        for cid in ("SN3", "SN3b", "SN4", "SN4b"):
            self.assertEqual(checks[cid]["value"]["rate"], 1.0, cid)
        self.assertEqual(checks["SN3"]["value"]["control_bad"], 0)
        self.assertEqual(checks["SN3b"]["value"]["control_bad"], 0)
        self.assertEqual(checks["SN4"]["value"]["errors"], {})
        self.assertEqual(checks["SN4b"]["value"]["errors"], {})
        self.assertEqual(checks["SN6"]["value"]["sha_module"], SN_BEHAVIOR_SHA)
        self.assertEqual(checks["SN6"]["value"]["sha_brain"], SN_BEHAVIOR_SHA)
        self.assertEqual(checks["SN6"]["value"]["source_sha"], SN_SOURCE_SHA)
        self.assertEqual(checks["SN6"]["value"]["clock"], "time.time")
        self.assertTrue(checks["SN6"]["value"]["identity"])
        self.assertTrue(checks["SN6"]["value"]["standalone"])


class SemanticNetworksHarnessNegativeContractTests(unittest.TestCase):
    """O gate falha fechado: comportamento ou fonte adulterados reprovam o harness."""

    def _run(self, argv):
        from tools import cognitivo_eval
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            code = cognitivo_eval.main(argv)
        return code, out.getvalue(), err.getvalue()

    def _ns(self, cls):
        return {"RealSemanticNetworksSystem": cls, "ConceptNode": ConceptNode, "SemanticRelation": SemanticRelation}

    def test_tampered_behavior_changes_the_fingerprint_and_the_grid(self):
        from tools import cognitivo_eval

        class Tampered(RealSemanticNetworksSystem):
            def get_statistics(self):
                stats = super().get_statistics()
                stats["total_relations"] += 1
                return stats

        payload = cognitivo_eval.sn_fingerprint(self._ns(Tampered))
        self.assertNotEqual(cognitivo_eval._ms_sha(payload), cognitivo_eval.SN_BEHAVIOR_SHA)
        self.assertLess(payload["grid"]["reference_matches"], 120)

    def test_tampered_semantics_are_detected_by_each_family(self):
        from tools import cognitivo_eval

        class NoDecay(RealSemanticNetworksSystem):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.activation_decay = 0.0

        class FlatInheritance(RealSemanticNetworksSystem):
            def inherit_properties(self, concept_label):
                return dict(self._find_or_create_concept(concept_label).properties)

        class NoVerify(RealSemanticNetworksSystem):
            def verify_hierarchy(self, child_label, ancestor_label):
                return child_label.lower() == ancestor_label.lower()

        self.assertLess(cognitivo_eval.sn_fingerprint(self._ns(NoDecay))["grid"]["reference_matches"], 120)
        self.assertLess(cognitivo_eval._sn_family_sn2(FlatInheritance)["rate"], 0.9)
        self.assertLess(cognitivo_eval._sn_family_sn2(NoVerify)["rate"], 0.9)

    def test_families_pass_for_a_correct_implementation_and_fail_for_the_block(self):
        from tools import cognitivo_eval
        safe = _acyclic_safe(RealSemanticNetworksSystem)
        self.assertEqual(cognitivo_eval._sn_family_sn4(safe)["rate"], 1.0)
        self.assertEqual(cognitivo_eval._sn_family_sn4(safe)["invalid"], 0)
        self.assertEqual(cognitivo_eval._sn_family_sn4(RealSemanticNetworksSystem)["rate"], 1.0)
        sync = _order_independent(RealSemanticNetworksSystem)
        self.assertEqual(cognitivo_eval._sn_family_sn3(sync)["rate"], 1.0)
        self.assertEqual(cognitivo_eval._sn_family_sn3(sync)["control_bad"], 0)
        self.assertEqual(cognitivo_eval._sn_family_sn3(RealSemanticNetworksSystem)["rate"], 1.0)

    def test_wrong_golden_fails_sn6_and_exits_1(self):
        _module()
        from tools import cognitivo_eval
        with mock.patch.object(cognitivo_eval, "SN_BEHAVIOR_SHA", "0" * 64):
            checks = {item["id"]: item for item in cognitivo_eval.eval_semantic_networks()}
            code, out, _ = self._run(["--capability", "semantic_networks"])
        self.assertFalse(checks["SN6"]["passed"])
        self.assertEqual(code, cognitivo_eval.EXIT_FAIL)
        self.assertIn('"SN6"', out.split('"failed"')[1])
        with mock.patch.object(cognitivo_eval, "SN_SOURCE_SHA", "0" * 64):
            self.assertFalse({i["id"]: i for i in cognitivo_eval.eval_semantic_networks()}["SN6"]["passed"])

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
        for value in ("semantic-networks", "Semantic_Networks", "semantic", "semantic_networks2"):
            with self.subTest(value=value):
                code, out, err = self._run(["--capability", value])
                self.assertEqual((code, out), (2, ""))
                self.assertIn("desconhecida", err)

    def test_invalid_scenario_exits_3_not_1(self):
        from tools import cognitivo_eval

        def invalid(_criteria="v1"):
            return [cognitivo_eval._check("SN2", None, False, "x", scenario_invalid=True)]

        with mock.patch.dict(cognitivo_eval.EVALUATORS, {"semantic_networks": invalid}):
            code, _, _ = self._run(["--capability", "semantic_networks"])
        self.assertEqual(code, cognitivo_eval.EXIT_INVALID_SCENARIO)
        self.assertNotEqual(cognitivo_eval.EXIT_INVALID_SCENARIO, cognitivo_eval.EXIT_FAIL)

    def test_a_changed_grid_or_poorer_coverage_makes_sn1_invalid_not_failed(self):
        _module()
        from tools import cognitivo_eval
        with mock.patch.object(cognitivo_eval, "_SN_GRID_SHA", "0" * 64):
            sn1 = {i["id"]: i for i in cognitivo_eval.eval_semantic_networks()}["SN1"]
            code, _, _ = self._run(["--capability", "semantic_networks"])
        self.assertTrue(sn1["scenario_invalid"])
        self.assertEqual(code, cognitivo_eval.EXIT_INVALID_SCENARIO)
        richer = {k: v + 1000 for k, v in cognitivo_eval._SN_RICHNESS_MIN.items()}
        with mock.patch.object(cognitivo_eval, "_SN_RICHNESS_MIN", richer):
            sn1 = {i["id"]: i for i in cognitivo_eval.eval_semantic_networks()}["SN1"]
        self.assertTrue(sn1["scenario_invalid"])
        self.assertTrue(sn1["value"]["richness_bad"])

    def test_cli_reports_no_failure_and_exits_0(self):
        _module()
        code, out, _ = self._run(["--capability", "semantic_networks"])
        failed = json.loads(out)["failed"]
        self.assertEqual(set(failed), _GAPS)
        self.assertEqual(code, 0)

    def test_source_text_requires_the_marker(self):
        from tools import cognitivo_eval
        with self.assertRaises(ValueError):
            cognitivo_eval.semantic_networks_source_text("nothing here")
        with self.assertRaises(ValueError):
            cognitivo_eval.semantic_networks_source_text("x\n" + cognitivo_eval._SN_MARKER + "\nno end")


if __name__ == "__main__":
    unittest.main()
