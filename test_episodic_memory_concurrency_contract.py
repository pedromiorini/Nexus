import concurrent.futures
import unittest

from core.constitutional_brain import RealEpisodicMemory


class EpisodicMemoryConcurrencyContractTests(unittest.TestCase):
    def test_concurrent_episode_lifecycle_is_serialized(self):
        memory = RealEpisodicMemory(":memory:")
        workers = 16

        def create_and_close(index):
            episode_id = memory.start_episode(
                f"episode-{index}", trigger_query=f"query-{index}", importance=0.25
            )
            memory.link_memory_to_episode(episode_id, index)
            memory.close_episode(episode_id)
            return episode_id

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            episode_ids = list(pool.map(create_and_close, range(workers)))

        self.assertEqual(len(set(episode_ids)), workers)
        statistics = memory.get_statistics()
        self.assertEqual(statistics["episodes_created"], workers)
        self.assertEqual(statistics["episodes_closed"], workers)
        self.assertEqual(statistics["total_links"], workers)
        self.assertAlmostEqual(statistics["avg_importance"], 0.25)
        self.assertEqual(len(memory.search_episodes(status="closed", limit=workers)), workers)
        memory.conn.close()


if __name__ == "__main__":
    unittest.main()
