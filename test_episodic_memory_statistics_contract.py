import tempfile
import unittest
from pathlib import Path

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

    def test_close_count_tracks_only_real_open_to_closed_transitions(self):
        memory = RealEpisodicMemory()
        memory.close_episode("missing-episode")
        self.assertEqual(memory.get_statistics()["episodes_closed"], 0)

        episode_id = memory.start_episode("one-time-close")
        memory.close_episode(episode_id)
        memory.close_episode(episode_id)
        self.assertEqual(memory.get_statistics()["episodes_closed"], 1)
        memory.conn.close()

    def test_episode_counts_survive_reopening_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "episodic-memory.sqlite3")
            memory = RealEpisodicMemory(database)
            first = memory.start_episode("first")
            memory.start_episode("second")
            memory.close_episode(first)
            self.assertEqual(
                (memory.get_statistics()["episodes_created"], memory.get_statistics()["episodes_closed"]),
                (2, 1),
            )
            memory.close()

            reopened = RealEpisodicMemory(database)
            self.assertEqual(
                (reopened.get_statistics()["episodes_created"], reopened.get_statistics()["episodes_closed"]),
                (2, 1),
            )
            reopened.close()

    def test_unknown_episode_link_is_rejected_without_mutating_statistics(self):
        memory = RealEpisodicMemory(":memory:")

        with self.assertRaisesRegex(ValueError, "episode_id"):
            memory.link_memory_to_episode("missing-episode", 99)

        self.assertEqual(memory.get_statistics()["total_links"], 0)
        memory.close()


if __name__ == "__main__":
    unittest.main()
