import unittest

from core.constitutional_brain import RealEpisodicMemory


class EpisodicMemorySearchLimitContractTests(unittest.TestCase):
    def setUp(self):
        self.memory = RealEpisodicMemory(":memory:")
        for title in ("needle one", "needle two"):
            episode_id = self.memory.start_episode(title)
            self.memory.close_episode(episode_id)

    def tearDown(self):
        self.memory.close()

    def test_negative_limit_is_rejected_before_database_access(self):
        statements = []
        self.memory.conn.set_trace_callback(statements.append)

        for query in (None, "needle"):
            with self.subTest(query=query):
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    self.memory.search_episodes(
                        query=query, status="closed", limit=-1
                    )

        self.memory.conn.set_trace_callback(None)
        self.assertEqual(statements, [])

    def test_boolean_and_non_integer_limits_are_rejected_before_sql(self):
        statements = []
        self.memory.conn.set_trace_callback(statements.append)

        for limit in (True, 1.5, "1", None):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(TypeError, "integer"):
                    self.memory.search_episodes(limit=limit)

        self.memory.conn.set_trace_callback(None)
        self.assertEqual(statements, [])

    def test_zero_limit_returns_empty_without_database_access(self):
        statements = []
        self.memory.conn.set_trace_callback(statements.append)

        for query in (None, "needle"):
            with self.subTest(query=query):
                self.assertEqual(
                    self.memory.search_episodes(
                        query=query, status="closed", limit=0
                    ),
                    [],
                )

        self.memory.conn.set_trace_callback(None)
        self.assertEqual(statements, [])

    def test_positive_limit_behavior_is_preserved(self):
        result = self.memory.search_episodes(
            query="needle", status="closed", limit=1
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].status, "closed")


if __name__ == "__main__":
    unittest.main()
