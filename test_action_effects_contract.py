"""Contrato R1: tabela única de efeitos de ação compartilhada por RealMCTSPlanner e
RealWorldModelSimulator, e aviso explícito para ação desconhecida (critérios C2a/C2b).

(a) origem: as duas classes consultam core.cognitivo.action_effects.ACTION_EFFECTS;
(b) não regressão: os deltas numéricos do MCTS são os mesmos de antes (fixados abaixo) e as
    transições já existentes (APPLY_PATCH/IGNORE_ALERT) não mudam;
(c) ausente/inválido: ação fora da tabela gera warning e não altera o estado.
"""

import unittest

MCTS_DELTAS = {
    "APPLY_PATCH": {"security": +0.9, "stability": +0.3, "cost": -0.2},
    "IGNORE_ALERT": {"security": -0.8, "stability": -0.5, "cost": 0.0},
    "ROLLBACK": {"security": +0.4, "stability": +0.6, "cost": -0.4},
    "ESCALATE": {"security": +0.6, "stability": +0.2, "cost": -0.3},
    "MONITOR": {"security": +0.2, "stability": +0.1, "cost": -0.1},
    "ISOLATE_SYSTEM": {"security": +0.7, "stability": -0.2, "cost": -0.3},
    "RESTORE_BACKUP": {"security": +0.5, "stability": +0.8, "cost": -0.5},
    "DEPLOY_UPDATE": {"security": +0.4, "stability": +0.5, "cost": -0.4},
    "INCREASE_LOGGING": {"security": +0.3, "stability": +0.0, "cost": -0.1},
    "SHUTDOWN_SERVICE": {"security": +0.8, "stability": -0.7, "cost": -0.2},
}
INITIAL = {"security_patch_applied": False, "alert_level": "high", "system_status": "vulnerable"}


class ActionEffectsContractTests(unittest.TestCase):
    def setUp(self):
        from core.cognitivo import action_effects
        self.table = action_effects.ACTION_EFFECTS

    def test_table_covers_planner_vocabulary_with_both_views(self):
        self.assertEqual(set(self.table), set(MCTS_DELTAS))
        for name, entry in self.table.items():
            self.assertEqual(set(entry), {"metrics", "world"}, name)

    def test_planner_deltas_come_from_table_unchanged(self):
        from core.cognitivo.mcts_planner import RealMCTSPlanner
        planner = RealMCTSPlanner()
        self.assertEqual(planner._action_effects, MCTS_DELTAS)
        self.assertEqual(planner._action_effects, {a: e["metrics"] for a, e in self.table.items()})
        planner._action_effects["APPLY_PATCH"]["security"] = 0.0
        self.assertEqual(self.table["APPLY_PATCH"]["metrics"]["security"], 0.9)

    def test_world_model_uses_table_and_keeps_existing_transitions(self):
        from core.cognitivo.causal_world_model import RealWorldModelSimulator
        wm = RealWorldModelSimulator()
        patched = wm.simulate(["APPLY_PATCH"]).final_state.state
        self.assertEqual(patched, {"security_patch_applied": True, "alert_level": "low", "system_status": "secure"})
        ignored = wm.simulate(["IGNORE_ALERT"]).final_state.state
        self.assertEqual(ignored, {"security_patch_applied": False, "alert_level": "critical",
                                   "system_status": "compromised"})
        for name, entry in self.table.items():
            result = wm.simulate([name])
            self.assertEqual(result.warnings, [], name)
            expected = dict(INITIAL, **entry["world"])
            self.assertEqual(result.final_state.state, expected, name)

    def test_at_least_80_percent_of_actions_change_world_state(self):
        changed = [a for a, e in self.table.items() if dict(INITIAL, **e["world"]) != INITIAL]
        self.assertGreaterEqual(len(changed) / len(self.table), 0.80)

    def test_unknown_action_warns_and_leaves_state(self):
        from core.cognitivo.causal_world_model import RealWorldModelSimulator
        wm = RealWorldModelSimulator()
        result = wm.simulate(["UNKNOWN_X", "APPLY_PATCH", ""])
        self.assertEqual(result.warnings, ["unknown action: 'UNKNOWN_X'", "unknown action: ''"])
        self.assertEqual(result.final_state.state["system_status"], "secure")
        self.assertEqual(wm.simulate(["UNKNOWN_X"]).final_state.state, INITIAL)


if __name__ == "__main__":
    unittest.main()
