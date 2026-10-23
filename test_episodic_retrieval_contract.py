"""Contrato de recuperação de RealEpisodicMemory (critério EM1 de docs/cognitivo/AUDITORIA_2026-10-22.md)."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.cognitivo.episodic_memory import RealEpisodicMemory
from tools.cognitivo_eval import eval_episodic


def _closed(mem, title, **kw):
    eid = mem.start_episode(title, **kw)
    mem.close_episode(eid)
    return eid


class EpisodicRetrievalContractTests(unittest.TestCase):
    def test_em1_precision_meets_preregistered_criterion(self):
        em1 = {c["id"]: c for c in eval_episodic()}["EM1"]
        assert em1["passed"], em1


    def test_reordered_multiword_query_finds_episode(self):
        with RealEpisodicMemory() as mem:
            target = _closed(mem, "database migration rollback")
            for i in range(20):
                _closed(mem, f"unrelated note {i}")
            assert target in [e.episode_id for e in mem.search_episodes("rollback database", limit=5)]


    def test_more_matching_terms_outrank_recency(self):
        with RealEpisodicMemory() as mem:
            target = _closed(mem, "database migration rollback")
            for i in range(5):
                _closed(mem, f"database backup {i}")
            self.assertEqual(mem.search_episodes("rollback database", limit=1)[0].episode_id, target)

    def test_substring_match_still_supported(self):
        with RealEpisodicMemory() as mem:
            eid = _closed(mem, "alpha release")
            assert [e.episode_id for e in mem.search_episodes("alp")] == [eid]


    def test_status_filter_preserved(self):
        with RealEpisodicMemory() as mem:
            open_id = mem.start_episode("kernel panic")
            closed_id = _closed(mem, "kernel panic again")
            assert [e.episode_id for e in mem.search_episodes("kernel")] == [closed_id]
            assert [e.episode_id for e in mem.search_episodes("kernel", status="open")] == [open_id]


    def test_special_characters_do_not_crash(self):
        with RealEpisodicMemory() as mem:
            _closed(mem, 'quote " AND OR NOT * ( ) ^ topic')
            for q in ['"', "AND", "*", "( ^", 'topic"', "NEAR(a b)", "-"]:
                assert isinstance(mem.search_episodes(q), list)
            assert mem.search_episodes("topic AND")


    def test_legacy_database_is_backfilled(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = str(Path(tmp.name) / "legacy.sqlite3")
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE episodic_memories (episode_id TEXT PRIMARY KEY, title TEXT NOT NULL,
            start_timestamp REAL NOT NULL, end_timestamp REAL, trigger_query TEXT,
            importance REAL DEFAULT 0.5, status TEXT DEFAULT 'open')""")
        conn.execute("INSERT INTO episodic_memories VALUES ('legacy-1','network outage report',1,2,NULL,0.5,'closed')")
        conn.commit()
        conn.close()
        with RealEpisodicMemory(path) as mem:
            assert [e.episode_id for e in mem.search_episodes("outage network")] == ["legacy-1"]
            new = _closed(mem, "network outage followup")
            assert set(e.episode_id for e in mem.search_episodes("outage network")) == {"legacy-1", new}
        with RealEpisodicMemory(path) as mem:  # reabrir não duplica índice
            assert len(mem.search_episodes("outage network")) == 2


if __name__ == "__main__":
    unittest.main()
