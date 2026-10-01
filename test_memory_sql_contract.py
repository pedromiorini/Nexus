import sqlite3
import unittest

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


if __name__ == "__main__":
    unittest.main()
