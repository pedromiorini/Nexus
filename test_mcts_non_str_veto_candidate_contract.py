"""Contrato para candidatos padrão não-str antes da deduplicação do veto no MCTS.

A entrada pública ``available_actions`` já filtra itens não-str. Este teste cobre a
fronteira defensiva da lista interna de candidatos: um item inválido não pode ser
comparado com nomes vetados nem executar ``__eq__`` fornecido por um objeto.
"""
import unittest
from pathlib import Path

from core.cognitivo.mcts_planner import RealMCTSPlanner

VETOED = "SHUTDOWN_SERVICE"
MUTATION_NAME = "vk_mcts_non_str_candidate_vetoed"
TEST_NAME = Path(__file__).name


class MCTSNonStringVetoCandidateContractTests(unittest.TestCase):
    class HostileEquality:
        def __eq__(self, other):
            raise AssertionError("equality must not be called for a non-string candidate")

        def __hash__(self):
            # Keep this key distinct from the vetoed string during the later set filter.
            return hash(VETOED) ^ 1

    def test_non_string_default_candidate_is_skipped_before_veto_deduplication(self):
        planner = RealMCTSPlanner(vetoes={VETOED: "blocked"})
        planner._action_effects = {
            VETOED: {},
            self.HostileEquality(): {},
        }

        result = planner.plan("goal", max_iterations=0)

        self.assertEqual(result.vetoed_actions, [{"action": VETOED, "reason": "blocked"}])
        self.assertEqual(result.best_action_sequence, [])

    def test_regression_is_assigned_to_the_surviving_mutation(self):
        from tools.cognitivo_mutation import MUTATIONS

        matching = [mutation for mutation in MUTATIONS if mutation.name == MUTATION_NAME]
        self.assertEqual(len(matching), 1)
        self.assertIn(TEST_NAME, matching[0].tests)


if __name__ == "__main__":
    unittest.main()
