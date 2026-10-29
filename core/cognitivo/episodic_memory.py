"""Memória episódica (POST5) extraída de core/constitutional_brain.py.

Extração strangler: o código abaixo foi movido sem alteração de comportamento;
core.constitutional_brain reexporta Episode e RealEpisodicMemory no ponto original.
"""

import re
import sqlite3
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = ["Episode", "RealEpisodicMemory"]


@dataclass
class Episode:
    episode_id: str
    title: str
    start_timestamp: float
    end_timestamp: Optional[float] = None
    trigger_query: Optional[str] = None
    importance: float = 0.5
    status: str = "open"

    @property
    def duration(self) -> Optional[float]:
        if self.end_timestamp:
            return self.end_timestamp - self.start_timestamp
        return None

class RealEpisodicMemory:
    """POST5: Episodic Memory - CONSOLIDADO"""

    def __init__(self, db_path: str = ":memory:",
                 hierarchical_memory=None,
                 constitutional_log=None):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._finalizer = weakref.finalize(self, self.conn.close)
        self.db_lock = threading.RLock()
        self.hierarchical_memory = hierarchical_memory
        self.constitutional_log = constitutional_log
        self.current_episode_id = None
        self.episodes_created = 0
        self.episodes_closed = 0
        self._create_tables()
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT COUNT(*), COALESCE(SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END), 0)
                FROM episodic_memories
            """)
            self.episodes_created, self.episodes_closed = map(int, cursor.fetchone())

    def close(self) -> None:
        """Fecha a conexão SQLite (idempotente); evita ResourceWarning no Python >= 3.13."""
        with self.db_lock:
            self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _create_tables(self):
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    episode_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    start_timestamp REAL NOT NULL,
                    end_timestamp REAL,
                    trigger_query TEXT,
                    importance REAL DEFAULT 0.5,
                    status TEXT DEFAULT 'open'
                );
                CREATE TABLE IF NOT EXISTS episodic_memory_links (
                    episode_id TEXT,
                    memory_id INTEGER,
                    relevance_score REAL DEFAULT 1.0,
                    PRIMARY KEY (episode_id, memory_id)
                );
            """)
            self.conn.commit()
            self._fts_enabled = self._create_fts_index(cursor)

    def _create_fts_index(self, cursor) -> bool:
        """Índice FTS5 (bm25) sobre title/trigger_query; preenche bancos legados. False sem FTS5."""
        try:
            cursor.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts "
                "USING fts5(episode_id UNINDEXED, title, trigger_query)"
            )
            cursor.execute("""
                INSERT INTO episodic_fts (episode_id, title, trigger_query)
                SELECT episode_id, title, COALESCE(trigger_query, '') FROM episodic_memories
                WHERE episode_id NOT IN (SELECT episode_id FROM episodic_fts)
            """)
            self.conn.commit()
            return True
        except sqlite3.OperationalError:
            self.conn.rollback()
            return False

    def start_episode(self, title: str, trigger_query: Optional[str] = None,
                     importance: float = 0.6, auto_log: bool = True) -> str:
        """Iniciar episódio"""
        episode_id = str(uuid.uuid4())
        ts = time.time()

        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO episodic_memories (episode_id, title, start_timestamp, trigger_query, importance)
                VALUES (?, ?, ?, ?, ?)
            """, (episode_id, title[:200], ts, trigger_query, importance))
            if self._fts_enabled:
                cursor.execute(
                    "INSERT INTO episodic_fts (episode_id, title, trigger_query) VALUES (?, ?, ?)",
                    (episode_id, title[:200], trigger_query or ""),
                )
            self.conn.commit()

            self.episodes_created += 1
            self.current_episode_id = episode_id
        return episode_id

    def close_episode(self, episode_id: Optional[str] = None, auto_log: bool = True):
        """Fechar episódio"""
        with self.db_lock:
            if episode_id is None:
                episode_id = self.current_episode_id

            if not episode_id:
                return

            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE episodic_memories
                SET end_timestamp = ?, status = 'closed'
                WHERE episode_id = ? AND status = 'open'
            """, (time.time(), episode_id))
            episode_closed = cursor.rowcount == 1
            self.conn.commit()

            if episode_closed:
                self.episodes_closed += 1
            if episode_closed and episode_id == self.current_episode_id:
                self.current_episode_id = None

    def link_memory_to_episode(self, episode_id: str, memory_id: int,
                               relevance_score: float = 1.0):
        """Link memory to an episode and validate it when a resolver is configured."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM episodic_memories WHERE episode_id = ?", (episode_id,))
            if cursor.fetchone() is None:
                raise ValueError("episode_id does not exist")

        memory_store = self.hierarchical_memory
        if memory_store is not None:
            get_memory = getattr(memory_store, "get_memory_by_id", None)
            if not callable(get_memory):
                raise ValueError("hierarchical_memory cannot resolve memory_id")
            if get_memory(memory_id) is None:
                raise ValueError("memory_id does not exist in hierarchical_memory")

        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO episodic_memory_links
                (episode_id, memory_id, relevance_score)
                VALUES (?, ?, ?)
            """, (episode_id, memory_id, relevance_score))
            self.conn.commit()

    def search_episodes(self, query: Optional[str] = None, 
                       status: str = "closed", limit: int = 10) -> List[Episode]:
        """Valida a query (None = ausência documentada) e delega a busca.

        A validação vem antes do atalho de ``limit == 0``: query vazia ou que não
        é ``str`` levanta mesmo com limite zero, em vez de imitar sucesso.
        """
        if query is not None and not isinstance(query, str):
            raise TypeError("query must be a string")
        if query == "":
            raise ValueError("query must be non-empty")
        return self._search_validated(query, status, limit)

    def _search_validated(self, query: Optional[str], status: str, limit: int) -> List[Episode]:
        """Buscar episódios"""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
        with self.db_lock:
            cursor = self.conn.cursor()

            if query:
                return self._ranked_search(cursor, query, status, limit)
            cursor.execute(
                "SELECT episode_id, title, start_timestamp, end_timestamp, "
                "trigger_query, importance, status FROM episodic_memories "
                "WHERE status = ? ORDER BY start_timestamp DESC LIMIT ?",
                (status, limit),
            )
            rows = cursor.fetchall()
        return [Episode(*row) for row in rows]

    def _ranked_search(self, cursor, query: str, status: str, limit: int) -> List[Episode]:
        """Termos em OR ranqueados por bm25 (FTS5), depois substring LIKE por recência."""
        rows, literal = [], query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        # A barra invertida é literal: os termos \w+ a descartariam e a união OR
        # de tokens (a, b) casaria "a_b"; com ela, só vale a substring LIKE ESCAPE.
        terms = [] if "\\" in query else re.findall(r"\w+", query)
        if self._fts_enabled and terms:
            match = " OR ".join('"' + t + '"' for t in terms)
            try:
                cursor.execute(
                    "SELECT m.episode_id, m.title, m.start_timestamp, m.end_timestamp, m.trigger_query, m.importance, m.status "
                    "FROM episodic_fts f JOIN episodic_memories m ON m.episode_id = f.episode_id "
                    "WHERE episodic_fts MATCH ? AND m.status = ? ORDER BY bm25(episodic_fts), m.start_timestamp DESC LIMIT ?",
                    (match, status, limit))
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                rows = []
        if len(rows) < limit:
            seen = {row[0] for row in rows}
            cursor.execute("""
                SELECT m.episode_id, m.title, m.start_timestamp, m.end_timestamp,
                       m.trigger_query, m.importance, m.status FROM episodic_memories m
                WHERE (m.title LIKE ? ESCAPE '\\' OR m.trigger_query LIKE ? ESCAPE '\\') AND m.status = ?
                ORDER BY m.start_timestamp DESC LIMIT ?
            """, (f"%{literal}%", f"%{literal}%", status, limit))
            rows += [row for row in cursor.fetchall() if row[0] not in seen][:limit - len(rows)]
        return [Episode(*row) for row in rows]

    def get_episode_memories(self, episode_id: str) -> List[Dict]:
        """Obter memórias de um episódio"""
        if self.hierarchical_memory is None:
            return []

        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT memory_id, relevance_score
                FROM episodic_memory_links
                WHERE episode_id = ?
            """, (episode_id,))
            links = cursor.fetchall()

        memories = []
        for mem_id, relevance in links:
            mem = self.hierarchical_memory.get_memory_by_id(mem_id)
            if mem:
                memories.append({
                    "id": mem.id,
                    "content": mem.content,
                    "importance": mem.importance,
                    "relevance_to_episode": relevance
                })

        return memories

    def get_statistics(self) -> Dict:
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM episodic_memory_links")
            total_links = int(cursor.fetchone()[0])
            cursor.execute("SELECT AVG(importance) FROM episodic_memories")
            avg_importance = cursor.fetchone()[0]
            cursor.execute("""
                SELECT COUNT(*), COALESCE(SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END), 0)
                FROM episodic_memories
            """)
            episodes_created, episodes_closed = map(int, cursor.fetchone())
        return {
            "episodes_created": episodes_created,
            "episodes_closed": episodes_closed,
            "total_links": total_links,
            "avg_importance": float(avg_importance) if avg_importance is not None else 0.0,
        }
