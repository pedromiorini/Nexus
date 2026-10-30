"""Contrato do seguimento de Semantic Networks (revisão do Marcus, sondas p1/p6/p9).

Pré-registro F1–F9 em docs/cognitivo/AUDITORIA_2026-10-22.md, seção "Semantic Networks —
seguimento da revisão do Marcus" (card t151s956). Os valores esperados vêm das derivações à mão
registradas lá e destes comentários, não do código.
"""

import contextlib
import io
import random
import sys
import time
import unittest

from core.constitutional_brain import RealSemanticNetworksSystem


def _quiet(cls=RealSemanticNetworksSystem):
    with contextlib.redirect_stdout(io.StringIO()):
        return cls()


def _build(relations, labels, start, cls=RealSemanticNetworksSystem):
    system = _quiet(cls)
    for label in labels:
        system.add_concept(label)
    for source, kind, target, strength in relations:
        system.add_relation(source, kind, target, strength)
    for label, value in start.items():
        system.activate_concept(label, value)
    return system


def _state(system):
    return {c.label: c.activation for c in system.concepts.values()}


class _PathBased(RealSemanticNetworksSystem):
    """Cópia congelada do algoritmo de herança do main 51ffa97 (recursão com caminho)."""

    def inherit_properties(self, concept_label):
        return self._inherit_path(concept_label, frozenset())

    def _inherit_path(self, concept_label, path):
        concept = self._find_or_create_concept(concept_label)
        path = path | {concept.concept_id}
        inherited = concept.properties.copy()
        for parent in self.get_related_concepts(concept_label, "IS-A"):
            if self._find_or_create_concept(parent.label).concept_id in path:
                continue
            for key, value in self._inherit_path(parent.label, path).items():
                if key not in inherited:
                    inherited[key] = value
        return inherited


class _CountingExpansions(RealSemanticNetworksSystem):
    def __init__(self):
        super().__init__()
        self.expansions = 0

    def get_related_concepts(self, label, relation_type=None):
        self.expansions += 1
        return super().get_related_concepts(label, relation_type)


class _SpyInherit(RealSemanticNetworksSystem):
    def __init__(self):
        super().__init__()
        self.inherit_calls = []

    def inherit_properties(self, concept_label):
        self.inherit_calls.append(concept_label)
        return super().inherit_properties(concept_label)


def _complete_graph(n, cls=RealSemanticNetworksSystem):
    system = _quiet(cls)
    for i in range(n):
        system.add_concept(f"k{i}", {f"p{i}": i})
    for i in range(n):
        for j in range(n):
            if i != j:
                system.add_relation(f"k{i}", "IS-A", f"k{j}")
    return system


def _random_graph(seed, cls=RealSemanticNetworksSystem):
    rng = random.Random(seed)
    n = rng.randint(2, 10)
    system = _quiet(cls)
    for i in range(n):
        system.add_concept(f"n{i}", {rng.choice("abcdefg"): rng.randint(0, 5) for _ in range(rng.randint(0, 3))})
    for _ in range(rng.randint(1, 30)):
        system.add_relation(f"n{rng.randrange(n)}", rng.choice(["IS-A", "IS-A", "IS-A", "part"]), f"n{rng.randrange(n)}")
    return system, n


class F1DeepChain(unittest.TestCase):
    def test_chain_of_1500_does_not_hit_recursion_limit(self):
        self.assertLess(sys.getrecursionlimit(), 1500)  # o teste só vale com limite padrão
        n = 1500
        system = _quiet()
        for i in range(n):
            props = {"own": 1} if i == 0 else {"top": "root"} if i == n - 1 else {}
            system.add_concept(f"c{i}", props)
        for i in range(n - 1):
            system.add_relation(f"c{i}", "IS-A", f"c{i + 1}")
        self.assertEqual(system.inherit_properties("c0"), {"own": 1, "top": "root"})
        self.assertTrue(system.verify_hierarchy("c0", f"c{n - 1}"))


class F2Complete(unittest.TestCase):
    def _timed(self, n):
        system = _complete_graph(n)
        start = time.perf_counter()
        result = system.inherit_properties("k0")
        elapsed = time.perf_counter() - start
        # k0 próprio primeiro; depois p1..p(n-1) na ordem do passeio em profundidade
        self.assertEqual(list(result.items()), [(f"p{i}", i) for i in range(n)])
        return elapsed

    def test_k9_under_one_second(self):
        self.assertLess(self._timed(9), 1.0)

    def test_k10_under_one_second(self):
        self.assertLess(self._timed(10), 1.0)

    def test_each_node_is_expanded_at_most_once(self):
        system = _complete_graph(9, _CountingExpansions)
        system.expansions = 0
        system.inherit_properties("k0")
        self.assertLessEqual(system.expansions, 9)


class F3PreservesCurrentResult(unittest.TestCase):
    def test_same_items_and_order_on_random_cyclic_graphs(self):
        checked = 0
        for seed in range(2000):
            new, n = _random_graph(seed)
            old, _ = _random_graph(seed, _PathBased)
            for i in range(n):
                label = f"n{i}"
                self.assertEqual(list(new.inherit_properties(label).items()),
                                 list(old.inherit_properties(label).items()), (seed, label))
                checked += 1
        self.assertGreater(checked, 8000)

    def test_unknown_label_is_still_created(self):
        system = _build([("Cat", "IS-A", "Animal", 1.0)], ["Cat", "Animal"], {})
        self.assertEqual(system.inherit_properties("Ghost"), {})
        self.assertEqual([c.label for c in system.concepts.values()], ["Cat", "Animal", "Ghost"])


class F4F5Atomic(unittest.TestCase):
    def _assert_rolled_back(self, relations, labels, start, iterations, error):
        system = _build(relations, labels, start)
        before = _state(system)
        with self.assertRaises(error):
            system.spread_activation(iterations)
        self.assertEqual(_state(system), before)

    def test_complex_strength_leaves_state_unchanged(self):
        for iterations in (1, 3):
            self._assert_rolled_back([("A", "x", "B", 0.5), ("A", "x", "C", 1j)], "ABC", {"A": 1.0}, iterations, TypeError)

    def test_error_only_in_iteration_2_rolls_back(self):
        # iteração 1: B = 0.7*0.5 = 0.35 -> 0.25; iteração 2: B (0.25 > 0.1) lê "oops" e falha
        self._assert_rolled_back([("A", "x", "B", 0.5), ("B", "x", "C", "oops")], "ABC", {"A": 1.0}, 3, TypeError)

    def test_error_only_in_iteration_3_rolls_back(self):
        # it.1: B = 0.7 -> 0.6; it.2: C = 0.6*0.7 = 0.42 -> 0.32; C (0.32 > 0.1) lê "oops" na it.3
        self._assert_rolled_back([("A", "x", "B", 1.0), ("B", "x", "C", 1.0), ("C", "x", "D", "oops")],
                                 "ABCD", {"A": 1.0}, 3, TypeError)

    def test_error_in_iteration_1_collect_phase_still_unchanged(self):
        self._assert_rolled_back([("A", "x", "B", 0.5), ("A", "x", "C", "oops")], "ABC", {"A": 1.0}, 1, TypeError)

    def test_success_path_is_unchanged(self):
        system = _build([("A", "x", "B", 1.0), ("B", "x", "C", 0.5)], "ABC", {"A": 1.0})
        result = system.spread_activation(3)
        self.assertEqual(set(result), {"A", "B", "C"})
        self.assertAlmostEqual(result["A"], 0.7, places=9)
        self.assertAlmostEqual(result["B"], 0.9, places=9)
        self.assertAlmostEqual(result["C"], 0.325, places=9)

    def test_retry_after_failure_matches_a_fresh_system(self):
        broken = _build([("A", "x", "B", 0.5), ("B", "x", "C", "oops")], "ABC", {"A": 1.0})
        with self.assertRaises(TypeError):
            broken.spread_activation(3)
        broken.relations[1].strength = 0.5
        fresh = _build([("A", "x", "B", 0.5), ("B", "x", "C", 0.5)], "ABC", {"A": 1.0})
        self.assertEqual(broken.spread_activation(3), fresh.spread_activation(3))


class F6Saturation(unittest.TestCase):
    def test_full_target_with_plus_and_minus_stays_at_point_nine(self):
        # A->B +1.0 -> +0.7; C->B -1.0 -> -0.7; B começa em 1.0: min(1, 1.0 + 0.7 - 0.7) = 1.0, menos 0.1
        system = _build([("A", "x", "B", 1.0), ("C", "x", "B", -1.0)], "ABC", {"A": 1.0, "C": 1.0, "B": 1.0})
        result = system.spread_activation(1)
        self.assertAlmostEqual(result["B"], 0.9, places=12)


class F8QueryDelegates(unittest.TestCase):
    def test_query_calls_inherit_once_per_concept(self):
        system = _build([("Cat", "IS-A", "Animal", 1.0)], ["Cat", "Animal"], {}, _SpyInherit)
        system.concepts[next(iter(system.concepts))].properties["legs"] = 4
        system.inherit_calls.clear()
        matches = system.query_by_properties({"legs": 4})
        self.assertEqual(sorted(system.inherit_calls), ["Animal", "Cat"])
        self.assertEqual([c.label for c in matches], ["Cat"])

    def test_query_matches_the_current_result_on_random_cyclic_graphs(self):
        for seed in range(300):
            new, _ = _random_graph(seed)
            old, _ = _random_graph(seed, _PathBased)
            for key in "abcd":
                for value in range(4):
                    self.assertEqual([c.label for c in new.query_by_properties({key: value})],
                                     [c.label for c in old.query_by_properties({key: value})], (seed, key, value))


class F7SpecificityKnownGaps(unittest.TestCase):
    """Comportamento desejado, mas ainda não implementado (sem mudar comportamento neste PR)."""

    @unittest.expectedFailure
    def test_most_specific_parent_wins_even_when_listed_second(self):
        system = _quiet()
        system.add_concept("Bird", {"flies": True})
        system.add_concept("Penguin", {"flies": False})
        system.add_relation("Penguin", "IS-A", "Bird")
        system.add_relation("Opus", "IS-A", "Bird")
        system.add_relation("Opus", "IS-A", "Penguin")
        self.assertEqual(system.inherit_properties("Opus"), {"flies": False})

    @unittest.expectedFailure
    def test_direct_parent_beats_grandparent(self):
        system = _quiet()
        system.add_concept("G", {"x": "avo"})
        system.add_concept("P2", {"x": "pai"})
        system.add_relation("P1", "IS-A", "G")
        system.add_relation("C", "IS-A", "P1")
        system.add_relation("C", "IS-A", "P2")
        self.assertEqual(system.inherit_properties("C"), {"x": "pai"})

    @unittest.expectedFailure
    def test_conflicting_values_in_a_cycle_are_flagged(self):
        # proposta de sinalização: um UserWarning que cita "conflito" (a forma não é compromisso)
        system = _quiet()
        system.add_concept("A", {"color": "red"})
        system.add_concept("B", {"color": "blue"})
        system.add_relation("A", "IS-A", "B")
        system.add_relation("B", "IS-A", "A")
        with self.assertWarnsRegex(UserWarning, "conflito"):
            system.inherit_properties("A")


if __name__ == "__main__":
    unittest.main()
