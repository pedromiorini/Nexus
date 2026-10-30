"""Contrato de SN3b/SN4b: ativação síncrona e consultas IS-A em ciclo nas redes semânticas.

Pré-registro e errata E1 em docs/cognitivo/AUDITORIA_2026-10-22.md, seção "Semantic Networks —
fechar os gaps SN3/SN4" (card t150s945). Os valores esperados saem da derivação à mão registrada
lá, não do código.
"""

import contextlib
import io
import itertools
import unittest

from core.constitutional_brain import RealSemanticNetworksSystem


def _quiet(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealSemanticNetworksSystem(*args, **kwargs)


def _chain(relations, iterations, start=(("A", 1.0),), labels="ABC"):
    system = _quiet()
    for label in labels:
        system.add_concept(label)
    for source, target, strength in relations:
        system.add_relation(source, "x", target, strength)
    for label, value in start:
        system.activate_concept(label, value)
    result = system.spread_activation(iterations)
    return system, result


class _InPlace(RealSemanticNetworksSystem):
    """O comportamento anterior à errata (propagação in-place, na ordem das relações)."""

    def spread_activation(self, iterations=3):
        for _ in range(iterations):
            for relation in self.relations:
                source = self.concepts.get(relation.source_id)
                target = self.concepts.get(relation.target_id)
                if source and target and source.activation > 0.1:
                    target.activation = min(1.0, target.activation
                                            + source.activation * self.spread_factor * relation.strength)
            for concept in self.concepts.values():
                concept.activation = max(0.0, concept.activation - self.activation_decay)
        return {c.label: c.activation for c in self.concepts.values() if c.activation > 0.1}


class _Recursive(RealSemanticNetworksSystem):
    """O comportamento anterior à errata (recursão sem conjunto de visitados)."""

    def inherit_properties(self, concept_label):
        concept = self._find_or_create_concept(concept_label)
        inherited = concept.properties.copy()
        for parent in self.get_related_concepts(concept_label, "IS-A"):
            for key, value in self.inherit_properties(parent.label).items():
                if key not in inherited:
                    inherited[key] = value
        return inherited

    def verify_hierarchy(self, child_label, ancestor_label):
        if child_label.lower() == ancestor_label.lower():
            return True
        return any(self.verify_hierarchy(p.label, ancestor_label)
                   for p in self.get_related_concepts(child_label, "IS-A"))


class SynchronousSpreadContractTests(unittest.TestCase):
    CHAIN = [("A", "B", 1.0), ("B", "C", 0.5)]

    def test_hand_derived_chain_values_in_both_orders(self):
        for relations in (self.CHAIN, list(reversed(self.CHAIN))):
            with self.subTest(order=relations[0][0]):
                self.assertEqual(set(_chain(relations, 1)[1]), {"A", "B"})
                one = _chain(relations, 1)[1]
                self.assertAlmostEqual(one["A"], 0.9, places=9)
                self.assertAlmostEqual(one["B"], 0.6, places=9)
                three = _chain(relations, 3)[1]
                self.assertEqual(set(three), {"A", "B", "C"})
                for label, value in (("A", 0.7), ("B", 0.9), ("C", 0.325)):
                    self.assertAlmostEqual(three[label], value, places=9)

    def test_internal_state_after_two_iterations_is_derived_from_the_start_of_each_one(self):
        system, _ = _chain(self.CHAIN, 2)
        got = {c.label: c.activation for c in system.concepts.values()}
        for label, value in (("A", 0.8), ("B", 0.9), ("C", 0.11)):
            self.assertAlmostEqual(got[label], value, places=9)

    def test_a_longer_chain_advances_one_hop_per_iteration(self):
        relations = [("A", "B", 1.0), ("B", "C", 1.0), ("C", "D", 1.0)]
        system, result = _chain(relations, 1, labels="ABCD")
        self.assertEqual(set(result), {"A", "B"})
        self.assertEqual(system.concepts[next(reversed(system.concepts))].activation, 0.0)

    def test_contributions_to_one_target_are_summed_and_capped_once(self):
        saturated = [("A", "C", 1.0), ("B", "C", 1.0)]
        for relations in (saturated, list(reversed(saturated))):
            _, result = _chain(relations, 1, start=(("A", 1.0), ("B", 1.0)))
            self.assertAlmostEqual(result["C"], 0.9, places=9)  # min(1, 0.7 + 0.7) - 0.1
        light = [("A", "C", 0.2), ("B", "C", 0.2)]
        for relations in (light, list(reversed(light))):
            _, result = _chain(relations, 1, start=(("A", 0.5), ("B", 0.5), ("C", 0.2)))
            self.assertAlmostEqual(result["C"], 0.24, places=9)  # 0.2 + 0.07 + 0.07 - 0.1

    def test_every_permutation_of_the_relations_gives_the_same_state(self):
        relations = [("A", "B", 0.9), ("B", "C", 0.8), ("C", "D", 0.7), ("A", "D", 0.4), ("D", "B", 0.3)]
        states = set()
        for order in itertools.permutations(relations):
            system, _ = _chain(order, 3, labels="ABCD")
            states.add(tuple(round(c.activation, 12) for c in system.concepts.values()))
        self.assertEqual(len(states), 1)

    def test_the_old_in_place_rule_differs_on_the_same_chain(self):
        system = _InPlace.__new__(_InPlace)
        with contextlib.redirect_stdout(io.StringIO()):
            _InPlace.__init__(system)
        for label in "ABC":
            system.add_concept(label)
        for source, target, strength in self.CHAIN:
            system.add_relation(source, "x", target, strength)
        system.activate_concept("A", 1.0)
        self.assertAlmostEqual(system.spread_activation(3)["C"], 0.645, places=9)

    def test_self_loop_and_unknown_relation_endpoints(self):
        system = _quiet()
        system.add_relation("A", "x", "A", 1.0)
        system.activate_concept("A", 1.0)
        self.assertAlmostEqual(system.spread_activation(1)["A"], 0.9, places=9)
        system.relations.append(type(system.relations[0])("r", "x", "ghost-id", "A", 1.0))
        # snapshot 0.9 + 0.63 = 1.53 limitado a 1.0, menos o decaimento: 0.9; a relação sem origem é ignorada
        self.assertAlmostEqual(system.spread_activation(1)["A"], 0.9, places=9)

    def test_empty_zero_and_negative_iterations_and_threshold_are_unchanged(self):
        self.assertEqual(_quiet().spread_activation(3), {})
        self.assertEqual(_quiet().spread_activation(0), {})
        _, result = _chain(self.CHAIN, 0)
        self.assertEqual(result, {"A": 1.0})
        _, result = _chain(self.CHAIN, -1)
        self.assertEqual(result, {"A": 1.0})
        _, result = _chain(self.CHAIN, 1, start=(("A", 0.1),))
        self.assertEqual(result, {})

    def test_invalid_iterations_and_strength_raise_without_half_applied_state(self):
        system = _quiet()
        for value in ("a", 1.5, None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                system.spread_activation(value)
        system = _quiet()
        system.add_relation("A", "x", "B", "x")
        system.add_relation("A", "x", "C", 1.0)
        system.activate_concept("A", 1.0)
        with self.assertRaises(TypeError):
            system.spread_activation(1)
        self.assertEqual([c.activation for c in system.concepts.values()], [1.0, 0.0, 0.0])


class CyclicHierarchyContractTests(unittest.TestCase):
    def _ring(self):
        system = _quiet()
        for label, props in (("A", {"a": 1}), ("B", {"b": 2}), ("C", {"c": 3})):
            system.add_concept(label, props)
        for source, target in (("A", "B"), ("B", "C"), ("C", "A")):
            system.add_relation(source, "IS-A", target)
        return system

    def test_inherit_returns_the_union_own_first_then_the_parent_without_exception(self):
        ring = self._ring()
        self.assertEqual(list(ring.inherit_properties("A").items()), [("a", 1), ("b", 2), ("c", 3)])
        self.assertEqual(list(ring.inherit_properties("B").items()), [("b", 2), ("c", 3), ("a", 1)])
        self.assertEqual(list(ring.inherit_properties("c").items()), [("c", 3), ("a", 1), ("b", 2)])

    def test_the_closest_definition_wins_inside_a_cycle(self):
        ring = _quiet()
        ring.add_concept("A", {"k": 1})
        ring.add_concept("B", {"k": 2})
        ring.add_relation("A", "IS-A", "B")
        ring.add_relation("B", "IS-A", "A")
        self.assertEqual(ring.inherit_properties("A"), {"k": 1})
        self.assertEqual(ring.inherit_properties("B"), {"k": 2})

    def test_verify_is_true_when_reachable_and_false_otherwise(self):
        ring = self._ring()
        for child in "ABC":
            for ancestor in "abcABC":
                self.assertIs(ring.verify_hierarchy(child, ancestor), True)
            self.assertIs(ring.verify_hierarchy(child, "Z"), False)
        self.assertEqual([c.label for c in ring.concepts.values()], ["A", "B", "C"])

    def test_self_loop_and_two_cycle(self):
        loop = _quiet()
        loop.add_concept("S", {"s": 1})
        loop.add_relation("S", "IS-A", "S")
        self.assertEqual(loop.inherit_properties("S"), {"s": 1})
        self.assertIs(loop.verify_hierarchy("S", "Z"), False)
        self.assertIs(loop.verify_hierarchy("S", "S"), True)
        pair = _quiet()
        pair.add_relation("X", "IS-A", "Y")
        pair.add_relation("Y", "IS-A", "X")
        self.assertIs(pair.verify_hierarchy("X", "Y"), True)
        self.assertIs(pair.verify_hierarchy("Y", "nope"), False)
        self.assertEqual(pair.inherit_properties("X"), {})

    def test_a_tail_into_a_cycle_inherits_the_whole_cycle(self):
        ring = self._ring()
        ring.add_concept("T", {"t": 0})
        ring.add_relation("T", "IS-A", "A")
        self.assertEqual(list(ring.inherit_properties("T").items()), [("t", 0), ("a", 1), ("b", 2), ("c", 3)])
        self.assertIs(ring.verify_hierarchy("T", "C"), True)
        self.assertIs(ring.verify_hierarchy("A", "T"), False)

    def test_query_by_properties_terminates_in_a_cycle(self):
        ring = self._ring()
        for key, value in (("a", 1), ("b", 2), ("c", 3)):
            self.assertEqual([c.label for c in ring.query_by_properties({key: value})], ["A", "B", "C"])
        self.assertEqual(ring.query_by_properties({"a": 2}), [])

    def test_acyclic_results_keep_order_and_shared_ancestors(self):
        diamond = _quiet()
        diamond.add_concept("D", {"d": 0})
        diamond.add_concept("B", {"b": 1})
        diamond.add_concept("C", {"c": 2})
        diamond.add_concept("A", {"a": 3, "b": 9})
        for source, target in (("D", "B"), ("D", "C"), ("B", "A"), ("C", "A")):
            diamond.add_relation(source, "IS-A", target)
        self.assertEqual(list(diamond.inherit_properties("D").items()),
                         [("d", 0), ("b", 1), ("a", 3), ("c", 2)])
        self.assertIs(diamond.verify_hierarchy("D", "A"), True)
        self.assertIs(diamond.verify_hierarchy("A", "D"), False)

    def test_absent_empty_and_unknown_labels(self):
        system = _quiet()
        self.assertIs(system.verify_hierarchy("Q", "q"), True)
        self.assertIs(system.verify_hierarchy("Q", "R"), False)
        self.assertIs(system.verify_hierarchy("", ""), True)
        self.assertIs(system.verify_hierarchy("", "x"), False)
        self.assertEqual(system.concepts, {})
        self.assertEqual(system.inherit_properties("Ghost"), {})
        self.assertEqual([c.label for c in system.concepts.values()], ["Ghost"])
        self.assertEqual(system.get_related_concepts("nobody", "IS-A"), [])

    def test_invalid_types_still_raise_attribute_error(self):
        system = _quiet()
        system.add_concept("A")
        for call in (lambda: system.verify_hierarchy(1, "a"), lambda: system.verify_hierarchy("a", 1),
                     lambda: system.inherit_properties(5), lambda: system.verify_hierarchy(None, "a")):
            with self.assertRaises(AttributeError):
                call()


class HarnessSn3bSn4bContractTests(unittest.TestCase):
    def test_real_block_passes_both_new_criteria_and_the_old_ones(self):
        from tools import cognitivo_eval
        checks = {c["id"]: c for c in cognitivo_eval.eval_semantic_networks()}
        for cid in ("SN3", "SN3b", "SN4", "SN4b"):
            with self.subTest(cid=cid):
                self.assertTrue(checks[cid]["passed"], checks[cid])
                self.assertFalse(checks[cid].get("scenario_invalid", False))
                self.assertEqual(checks[cid]["value"]["rate"], 1.0)
        self.assertEqual(checks["SN3b"]["value"]["control_bad"], 0)

    def test_the_old_behaviors_fail_the_new_criteria_as_registered(self):
        from tools import cognitivo_eval
        in_place = cognitivo_eval._sn_family_sn3b(_InPlace)
        self.assertEqual((in_place["rate"], in_place["invalid"], in_place["control_bad"]), (0.0, 0, 0))
        recursive = cognitivo_eval._sn_family_sn4b(_Recursive)
        self.assertEqual((recursive["rate"], recursive["invalid"]), (0.0, 0))
        self.assertEqual(recursive["errors"], {"RecursionError": 20})
        self.assertEqual(cognitivo_eval._sn_family_sn3(_InPlace)["rate"], 0.0)
        self.assertEqual(cognitivo_eval._sn_family_sn4(_Recursive)["rate"], 0.0)

    def test_a_wrong_but_order_independent_rule_fails_sn3b(self):
        from tools import cognitivo_eval

        class NoCap(RealSemanticNetworksSystem):
            def spread_activation(self, iterations=3):
                for _ in range(iterations):
                    for concept in self.concepts.values():
                        concept.activation = 0.0
                return {}

        self.assertEqual(cognitivo_eval._sn_family_sn3(NoCap)["rate"], 1.0)
        self.assertEqual(cognitivo_eval._sn_family_sn3b(NoCap)["rate"], 0.0)

    def test_a_partial_cycle_fix_fails_sn4b(self):
        from tools import cognitivo_eval

        class OnlyBoolean(_Recursive):
            def inherit_properties(self, concept_label):
                return {"p0": 0}

        self.assertLess(cognitivo_eval._sn_family_sn4b(OnlyBoolean)["rate"], 0.9)

    def test_reference_rules_and_the_six_changed_grid_rows(self):
        from tools import cognitivo_eval as ev
        chain = [(0, "x", 1, 1.0), (1, "x", 2, 0.5)]
        for rels in (chain, list(reversed(chain))):
            self.assertEqual([round(v, 9) for v in ev._sn_ref_spread_sync([1.0, 0.0, 0.0], rels, 3)],
                             [0.7, 0.9, 0.325])
        self.assertEqual(round(ev._sn_ref_spread([1.0, 0.0, 0.0], chain, 3)[2], 9), 0.645)
        rows = ev._sn_grid()
        sync = [ev._sn_reference(row)["spreads"] for row in rows]
        original = ev._sn_ref_spread_sync
        try:
            ev._sn_ref_spread_sync = ev._sn_ref_spread
            old = [ev._sn_reference(row)["spreads"] for row in rows]
        finally:
            ev._sn_ref_spread_sync = original
        changed = [i for i, (a, b) in enumerate(zip(old, sync))
                   if any(set(x) != set(y) or any(abs(x[k] - y[k]) > 1e-12 for k in x) for x, y in zip(a, b))]
        self.assertEqual(changed, [18, 53, 65, 83, 91, 108])

    def test_new_checks_are_scenario_invalid_when_the_instances_are_invalid(self):
        from tools import cognitivo_eval as ev
        import unittest.mock as mock
        with mock.patch.object(ev, "_SN_MIN_GAP", 10.0):
            checks = {c["id"]: c for c in ev.eval_semantic_networks()}
        self.assertTrue(checks["SN3b"]["scenario_invalid"])
        self.assertFalse(checks["SN3b"]["passed"])


if __name__ == "__main__":
    unittest.main()
