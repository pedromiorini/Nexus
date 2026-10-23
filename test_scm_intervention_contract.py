"""Contrato C4: relações extraídas alteram a simulação via do(X)."""
from __future__ import annotations

import unittest

from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator


class ScmInterventionContractTests(unittest.TestCase):
    def test_extracted_relation_changes_simulation(self):
        cr = RealCausalReasoning()
        world = RealWorldModelSimulator(causal_reasoning=cr)
        self.assertEqual(world.current_state.state["system_status"], "vulnerable")
        cr.extract_causal_relations("DEPLOY_FIX causes system_status secure")
        result = world.simulate(["DEPLOY_FIX"])
        self.assertEqual(result.final_state.state.get("system_status"), "secure")
        self.assertTrue(result.success)

    def test_without_extraction_unknown_stays_and_warns(self):
        world = RealWorldModelSimulator(causal_reasoning=RealCausalReasoning())
        result = world.simulate(["DEPLOY_FIX"])
        self.assertEqual(result.final_state.state.get("system_status"), "vulnerable")
        self.assertTrue(any("DEPLOY_FIX" in w for w in result.warnings))

    def test_action_effects_table_still_applied(self):
        world = RealWorldModelSimulator(causal_reasoning=RealCausalReasoning())
        result = world.simulate(["APPLY_PATCH"])
        self.assertTrue(result.final_state.state.get("security_patch_applied"))
        self.assertFalse(result.warnings)

    def test_do_does_not_erase_table_effects(self):
        cr = RealCausalReasoning()
        world = RealWorldModelSimulator(causal_reasoning=cr)
        cr.extract_causal_relations("APPLY_PATCH causes alert_level low")
        result = world.simulate(["APPLY_PATCH"])
        self.assertTrue(result.final_state.state.get("security_patch_applied"))
        self.assertEqual(result.final_state.state.get("alert_level"), "low")

    def test_intervene_helper_parses_effect(self):
        cr = RealCausalReasoning()
        world = RealWorldModelSimulator(causal_reasoning=cr)
        cr.extract_causal_relations("X causes foo_bar baz")
        self.assertEqual(world._do("X"), {"foo_bar": "baz"})
        self.assertEqual(world._do("missing"), {})


if __name__ == "__main__":
    unittest.main()
