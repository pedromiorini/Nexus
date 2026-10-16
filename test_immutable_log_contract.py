"""Contrato do RealImmutableLog (log encadeado por hash, append-only).

Lacunas corrigidas (auditoria de 2026-10-08): a classe não tinha verificação da cadeia,
não impedia UPDATE/DELETE e recomeçava do hash gênesis a cada instância, quebrando a
cadeia de qualquer banco persistente reaberto.
"""
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from core.constitutional_brain import RealImmutableLog


def raw_rows(path):
    connection = sqlite3.connect(path)
    try:
        return connection.execute(
            "SELECT sequence, content_hash, previous_hash FROM log_entries ORDER BY sequence"
        ).fetchall()
    finally:
        connection.close()


class ImmutableLogChainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "log.sqlite3")

    def open_log(self, path=None):
        log = RealImmutableLog(path or self.path)
        self.addCleanup(log.close)
        return log

    def test_empty_log_is_valid(self):
        report = self.open_log(":memory:").verify_chain()
        self.assertTrue(report["ok"])
        self.assertEqual(report["entries"], 0)
        self.assertEqual(report["head"], RealImmutableLog.GENESIS_HASH)

    def test_chain_verifies_and_links_entries(self):
        log = self.open_log()
        hashes = [log.log_event("evento", {"n": n}) for n in range(5)]
        report = log.verify_chain()
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["entries"], 5)
        self.assertEqual(report["head"], hashes[-1])
        rows = raw_rows(self.path)
        self.assertEqual(rows[0][2], RealImmutableLog.GENESIS_HASH)
        for previous, current in zip(rows, rows[1:]):
            self.assertEqual(current[2], previous[1])

    def test_update_and_delete_are_blocked_for_regular_connections(self):
        log = self.open_log()
        log.log_event("evento", {"n": 1})
        with self.assertRaises(sqlite3.DatabaseError):
            log.conn.execute("UPDATE log_entries SET event_data = '{}' WHERE sequence = 1")
        with self.assertRaises(sqlite3.DatabaseError):
            log.conn.execute("DELETE FROM log_entries WHERE sequence = 1")
        log.conn.rollback()
        self.assertTrue(log.verify_chain()["ok"])

    def test_tampering_is_detected_even_if_triggers_are_removed(self):
        log = self.open_log()
        for n in range(3):
            log.log_event("evento", {"n": n})
        attacker = sqlite3.connect(self.path)
        attacker.execute("DROP TRIGGER log_entries_no_update")
        attacker.execute("UPDATE log_entries SET event_data = '{\"n\": 99}' WHERE sequence = 2")
        attacker.commit()
        attacker.close()
        report = log.verify_chain()
        self.assertFalse(report["ok"])
        self.assertEqual(report["first_invalid_sequence"], 2)
        self.assertIn("content_hash", report["reason"])

    def test_removed_middle_entry_is_detected(self):
        log = self.open_log()
        for n in range(4):
            log.log_event("evento", {"n": n})
        attacker = sqlite3.connect(self.path)
        attacker.execute("DROP TRIGGER log_entries_no_delete")
        attacker.execute("DELETE FROM log_entries WHERE sequence = 2")
        attacker.commit()
        attacker.close()
        report = log.verify_chain()
        self.assertFalse(report["ok"])
        self.assertEqual(report["first_invalid_sequence"], 3)
        self.assertIn("previous_hash", report["reason"])

    def test_chain_continues_after_reopening_a_persistent_log(self):
        first = RealImmutableLog(self.path)
        head = None
        for n in range(3):
            head = first.log_event("evento", {"n": n})
        first.close()
        second = self.open_log()
        self.assertEqual(second.current_hash, head)
        second.log_event("evento", {"n": 3})
        second.log_event("evento", {"n": 4})
        report = second.verify_chain()
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["entries"], 5)
        rows = raw_rows(self.path)
        self.assertEqual(rows[3][2], rows[2][1])

    def test_two_instances_on_the_same_file_share_one_chain(self):
        a, b = self.open_log(), self.open_log()
        for n in range(6):
            (a if n % 2 == 0 else b).log_event("evento", {"n": n})
        self.assertTrue(a.verify_chain()["ok"])
        self.assertEqual(b.verify_chain()["entries"], 6)

    def test_independent_connections_serialize_concurrent_writes(self):
        logs = [self.open_log() for _ in range(4)]
        barrier = threading.Barrier(len(logs))
        errors = []

        def worker(index, log):
            try:
                barrier.wait(timeout=5)
                for n in range(15):
                    log.log_event("evento", {"worker": index, "n": n})
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(index, log))
            for index, log in enumerate(logs)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        report = logs[0].verify_chain()
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["entries"], 60)

    def test_reopening_a_tampered_log_fails_closed(self):
        log = self.open_log()
        log.log_event("evento", {"n": 1})
        log.close()
        attacker = sqlite3.connect(self.path)
        attacker.execute("DROP TRIGGER log_entries_no_update")
        attacker.execute("UPDATE log_entries SET event_data = '{}' WHERE sequence = 1")
        attacker.commit()
        attacker.close()
        with self.assertRaisesRegex(ValueError, "cadeia do log inválida"):
            RealImmutableLog(self.path)

    def test_threaded_writes_keep_the_chain_valid(self):
        log = self.open_log()

        def worker(offset):
            for n in range(25):
                log.log_event("evento", {"worker": offset, "n": n})

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        report = log.verify_chain()
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["entries"], 200)

    def test_unserializable_event_does_not_corrupt_the_log(self):
        log = self.open_log()
        log.log_event("evento", {"n": 1})
        with self.assertRaises(TypeError):
            log.log_event("evento", {"bad": object()})
        log.log_event("evento", {"n": 2})
        report = log.verify_chain()
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["entries"], 2)

    def test_close_is_idempotent_and_blocks_further_writes(self):
        log = RealImmutableLog(":memory:")
        log.log_event("evento", {"n": 1})
        log.close()
        log.close()
        with self.assertRaises(sqlite3.ProgrammingError):
            log.log_event("evento", {"n": 2})

    def test_context_manager_closes_the_connection(self):
        with RealImmutableLog(":memory:") as log:
            log.log_event("evento", {"n": 1})
        with self.assertRaises(sqlite3.ProgrammingError):
            log.verify_chain()


if __name__ == "__main__":
    unittest.main()
