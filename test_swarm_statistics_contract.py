import unittest

from core.constitutional_brain import RealSwarmIntelligence


class SwarmStatisticsContractTests(unittest.TestCase):
    def test_statistics_are_derived_from_deliberations(self):
        swarm = RealSwarmIntelligence(num_agents=5)
        before = swarm.get_statistics()
        self.assertEqual(before["total_deliberations"], 0)
        self.assertEqual(before["avg_consensus"], 0.0)
        self.assertEqual(before["avg_diversity"], 0.0)

        first = swarm.deliberate({"risk": "low"}, [{"support": True}])
        second = swarm.deliberate({"risk": "high"}, [{"support": False}])
        statistics = swarm.get_statistics()

        self.assertEqual(statistics["total_deliberations"], 2)
        self.assertAlmostEqual(
            statistics["avg_consensus"],
            (first.consensus_strength + second.consensus_strength) / 2,
        )
        self.assertAlmostEqual(
            statistics["avg_diversity"],
            (first.diversity_score + second.diversity_score) / 2,
        )
        self.assertNotEqual(statistics["avg_consensus"], 0.63)

    def test_deliberation_without_agents_fails_explicitly(self):
        swarm = RealSwarmIntelligence(num_agents=0)
        with self.assertRaisesRegex(ValueError, "at least one swarm agent"):
            swarm.deliberate({}, [])


if __name__ == "__main__":
    unittest.main()
