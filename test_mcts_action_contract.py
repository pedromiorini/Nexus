import unittest

from core.constitutional_brain import RealMCTSPlanner


class MCTSActionContractTests(unittest.TestCase):
    def test_none_uses_default_action_space(self):
        planner = RealMCTSPlanner(max_depth=1)
        result = planner.plan("secure the system", max_iterations=1)

        self.assertGreater(result.tree_nodes_created, 1)
        self.assertTrue(result.alternative_plans)

    def test_empty_action_space_is_respected(self):
        planner = RealMCTSPlanner(max_depth=1)
        result = planner.plan("secure the system", available_actions=[], max_iterations=5)

        self.assertEqual(result.best_action_sequence, [])
        self.assertEqual(result.alternative_plans, [])
        self.assertEqual(result.expected_reward, 0.0)
        self.assertEqual(result.tree_nodes_created, 1)
        self.assertEqual(result.iterations_run, 5)


if __name__ == "__main__":
    unittest.main()
