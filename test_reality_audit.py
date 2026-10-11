import ast
import unittest

from tools.reality_audit import (
    audit_claims,
    audit_code_claim_inventory,
    audit_code_overclaims,
    build_report,
    collect_static_evidence,
    iter_functions,
    parse_python,
    CORE,
)

# Limite do inventário de frases fortes de capacidade: só pode DIMINUIR. Ao reescrever textos de
# ênfase ("REAL <Capacidade>", "REAL e FUNCIONAL"...), reduza este número; nunca aumente.
STRONG_CLAIM_BASELINE = 51


class RealityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_report_has_version_and_scope(self):
        self.assertEqual(self.report["audit_version"], "1.1")
        self.assertEqual(self.report["scope"], "conservative reality audit")

    def test_static_evidence_is_quantified(self):
        evidence = self.report["static_evidence"]
        self.assertGreater(evidence["core_lines"], 1000)
        self.assertGreater(evidence["classes"], 0)
        self.assertGreater(evidence["functions"], 0)
        self.assertGreater(evidence["embedded_test_markers"], 0)

    def test_readme_makes_no_unqualified_capability_claims(self):
        # Antes este teste exigia um achado "unsubstantiated", que era um falso positivo da regex de
        # negação ("não constituem" não casava com "não constitui"). Agora o README é um gate honesto.
        statuses = {item["claim"]: item["status"] for item in self.report["claims"]}
        self.assertTrue(statuses, "o README deveria mencionar (e negar) AGI/consciência")
        self.assertEqual(set(statuses.values()), {"explicitly_disclaimed"}, statuses)

    def test_code_has_no_assertive_overclaims(self):
        self.assertEqual(audit_code_overclaims(), [], "o código afirma capacidades que os testes não demonstram")

    def test_documented_python_references_exist(self):
        paths = self.report["referenced_paths"]
        self.assertGreater(len(paths), 0)
        self.assertTrue(all(item["exists"] for item in paths))
        self.assertNotIn("NEXUS_CORE_v3.94.py", {item["path"] for item in paths})

    def test_known_risks_are_present(self):
        risks = " ".join(self.report["risks"])
        self.assertIn("Embedded demonstration tests", risks)
        self.assertIn("Optional dependency fallbacks", risks)

    def test_risk_about_readme_claims_is_conditional(self):
        self.assertNotIn("README contains capability claims", " ".join(self.report["risks"]))


class ClaimClassificationTests(unittest.TestCase):
    def status_of(self, text, claim="consciência/autoconsciência"):
        return {item["claim"]: item["status"] for item in audit_claims(text)}.get(claim)

    def test_unqualified_claim_is_flagged(self):
        self.assertEqual(self.status_of("O sistema possui consciência artificial."), "unsubstantiated_by_software_tests")
        self.assertEqual(self.status_of("This system is self-aware."), "unsubstantiated_by_software_tests")

    def test_disclaimers_in_portuguese_and_english_are_recognized(self):
        disclaimers = (
            "O projeto não reivindica consciência.",
            "Nomes de módulos e banners não constituem prova de inteligência geral, consciência ou prontidão.",
            "Protótipo sem reivindicar consciência.",
            "The project makes no claim of consciousness.",
            "This is not a demonstration of consciousness.",
        )
        for text in disclaimers:
            with self.subTest(text=text):
                self.assertEqual(self.status_of(text), "explicitly_disclaimed")

    def test_negation_in_an_earlier_sentence_does_not_cover_a_later_claim(self):
        text = "O texto não reivindica AGI. O módulo central possui consciência."
        self.assertEqual(self.status_of(text), "unsubstantiated_by_software_tests")


class CodeOverclaimTests(unittest.TestCase):
    def test_assertive_phrases_are_detected(self):
        samples = {
            "autoconsciência afirmada": 'print("Consciousness system is self-aware!")',
            "funcionalidade total afirmada": 'print("ALL TESTS PASSED - SYSTEM 100% FUNCTIONAL")',
            "consciência genuína afirmada": '"""Implements genuine consciousness."""',
        }
        for label, source in samples.items():
            with self.subTest(label=label):
                self.assertEqual([item["claim"] for item in audit_code_overclaims(source)], [label])

    def test_neutral_labels_are_not_flagged(self):
        self.assertEqual(audit_code_overclaims("class RealConsciousnessSystem: ...  # functional simulation"), [])


class StaticEvidenceShapeTests(unittest.TestCase):
    def test_qualified_walk_visits_every_function(self):
        tree = parse_python(CORE)
        walked = len(list(iter_functions(tree)))
        expected = len([node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))])
        self.assertEqual(walked, expected)

    def test_strict_constants_are_literals_and_dynamic_literals_are_not(self):
        evidence = collect_static_evidence()
        for item in evidence["constant_return_functions"]:
            self.assertIn("qualname", item)
            self.assertNotIn("line", item, "números de linha tornam o relatório instável a cada edição")
            ast.literal_eval(item["value"])
        self.assertGreater(len(evidence["dynamic_literal_return_functions"]), len(evidence["constant_return_functions"]))


class ClaimInventoryTests(unittest.TestCase):
    def test_levels_are_separated_and_sql_real_is_ignored(self):
        source = (
            'class RealMCTSPlanner:\n'
            '    """REAL Multimodal Perception"""\n'
            '    SQL = "score REAL NOT NULL"\n'
            '    # REAL e FUNCIONAL\n'
            '    # consciousness\n'
        )
        inventory = audit_code_claim_inventory(source)
        self.assertEqual(
            inventory["by_level"],
            {"strong_capability_claim": 2, "scientific_terminology": 1, "neutral_identifier": 1},
        )
        labels = {item["label"] for item in inventory["items"]}
        self.assertIn("classes com prefixo Real", labels)

    def test_strong_capability_claims_do_not_grow(self):
        total = audit_code_claim_inventory()["by_level"]["strong_capability_claim"]
        self.assertLessEqual(total, STRONG_CLAIM_BASELINE, "novas frases fortes de capacidade foram adicionadas ao código")

    def test_inventory_is_part_of_the_report_without_line_numbers(self):
        inventory = build_report()["code_claim_inventory"]
        self.assertEqual(set(inventory["by_level"]), {"strong_capability_claim", "scientific_terminology", "neutral_identifier"})
        for item in inventory["items"]:
            self.assertNotIn("line", item)


if __name__ == "__main__":
    unittest.main()
