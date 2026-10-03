import unittest

from core.constitutional_brain import RealEpisodicMemory


class EpisodicMemoryStatisticsContractTests(unittest.TestCase):
    def test_statistics_are_derived_from_sqlite_records(self):
        memory = RealEpisodicMemory()
        self.assertEqual(memory.get_statistics()["total_links"], 0)
        self.assertEqual(memory.get_statistics()["avg_importance"], 0.0)

        first = memory.start_episode("first", importance=0.2)
        second = memory.start_episode("second", importance=0.8)
        memory.link_memory_to_episode(first, 11)
        memory.link_memory_to_episode(first, 12)
        memory.link_memory_to_episode(second, 13)

        statistics = memory.get_statistics()
        self.assertEqual(statistics["episodes_created"], 2)
        self.assertEqual(statistics["total_links"], 3)
        self.assertAlmostEqual(statistics["avg_importance"], 0.5)
        self.assertNotEqual(statistics["avg_importance"], 0.7)

        memory.close_episode(first)
        self.assertEqual(memory.get_statistics()["episodes_closed"], 1)
        memory.conn.close()


if __name__ == "__main__":
    unittest.main()
