import unittest
from unittest.mock import Mock

from core.constitutional_brain import RealHierarchicalMemory


class MemorySqlContractTests(unittest.TestCase):
    def test_keyword_query_treats_sql_text_as_data(self):
        memory = RealHierarchicalMemory(":memory:")
        memory.store("ordinary memory")
        memory.store("quoted ' memory")
        malicious = "zzzz' UNION SELECT --"
        results = memory.retrieve(malicious, limit=5)
        self.assertEqual(results, [])
        tables = memory.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchall()
        self.assertEqual(tables, [("memories",)])

    def test_keyword_search_still_matches_parameterized_content(self):
        memory = RealHierarchicalMemory(":memory:")
        memory.store("alpha beta")
        results = memory.retrieve("alpha", limit=5)
        self.assertEqual([item["content"] for item in results], ["alpha beta"])

    def test_multiple_keywords_preserve_or_search_and_limit(self):
        memory = RealHierarchicalMemory(":memory:")
        memory.store("low beta", importance=0.2)
        memory.store("high alpha", importance=0.9)
        results = memory.retrieve("alpha beta", limit=1)
        self.assertEqual([item["content"] for item in results], ["high alpha"])

    def test_negative_limit_is_rejected_for_empty_and_keyword_queries(self):
        memory = RealHierarchicalMemory(":memory:")
        memory.store("alpha beta")

        for query in ("", "alpha"):
            with self.subTest(query=query):
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    memory.retrieve(query, limit=-1)

    def test_negative_limit_is_rejected_before_semantic_search(self):
        memory = RealHierarchicalMemory(":memory:")
        memory.fallback_mode = False
        memory.encoder = Mock()
        memory.index = Mock()
        memory.index.ntotal = 1

        with self.assertRaisesRegex(ValueError, "non-negative"):
            memory.retrieve("alpha", limit=-1)

        memory.encoder.encode.assert_not_called()
        memory.index.search.assert_not_called()

    def test_limit_must_be_a_non_boolean_integer(self):
        memory = RealHierarchicalMemory(":memory:")

        for limit in (True, 1.5, "1"):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(TypeError, "integer"):
                    memory.retrieve("", limit=limit)

    def test_zero_limit_returns_no_rows_for_empty_and_keyword_queries(self):
        memory = RealHierarchicalMemory(":memory:")
        memory.store("alpha beta")

        for query in ("", "alpha"):
            with self.subTest(query=query):
                self.assertEqual(memory.retrieve(query, limit=0), [])


if __name__ == "__main__":
    unittest.main()
