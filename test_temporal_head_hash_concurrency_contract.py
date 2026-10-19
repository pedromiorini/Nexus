"""Contrato de isolamento de head_hash; Events ordenam as threads sem sleeps."""
import threading
import unittest
from unittest.mock import patch

from core.temporal_facts import GENESIS, TemporalFactStore


class TemporalHeadHashConcurrencyContract(unittest.TestCase):
    def _check_transaction(self, operation, rollback):
        store = TemporalFactStore(functional_predicates=("p",), clock=lambda: 100.0)
        if operation != "assert":
            store.assert_fact("s", "p", "old", valid_from=0)
        before = store.export_snapshot()
        original_apply = store._apply
        lock = store._lock
        emitted = threading.Event()
        release_writer = threading.Event()
        reader_progress = threading.Event()
        reader_blocked = threading.Event()
        reader_done = threading.Event()
        observed = []
        transient = []
        writer_errors = []
        reader_errors = []

        class ObservedLock:
            """Sonda a tentativa real de aquisição, não a disponibilidade da thread."""
            def __enter__(self):
                if threading.current_thread() is reader:
                    if not lock.acquire(blocking=False):
                        reader_blocked.set()
                        reader_progress.set()
                        lock.acquire()
                    else:
                        reader_progress.set()
                else:
                    lock.acquire()
                return self

            def __exit__(self, *exc):
                lock.release()

        def paused_apply(event):
            # _emit já publicou _head, mas a transação ainda pode ser revertida.
            if not emitted.is_set():
                transient.append(store._head)
                emitted.set()
                if not release_writer.wait(5):
                    raise TimeoutError("escrita não liberada pelo teste")
                if rollback:
                    raise RuntimeError("rollback injetado")
            original_apply(event)

        def write():
            try:
                if operation == "retract":
                    store.retract("s", "p", valid_to=50)
                elif operation == "corroborate":
                    store.assert_fact("s", "p", "old", source="second")
                else:
                    store.assert_fact("s", "p", "new", valid_from=50)
            except BaseException as exc:
                writer_errors.append(exc)

        def read():
            try:
                observed.append(store.head_hash)
            except BaseException as exc:
                reader_errors.append(exc)
            finally:
                reader_done.set()
                reader_progress.set()

        writer = threading.Thread(target=write, daemon=True)
        reader = threading.Thread(target=read, daemon=True)
        with patch.object(store, "_lock", ObservedLock()), patch.object(
                store, "_apply", side_effect=paused_apply):
            writer.start()
            try:
                self.assertTrue(emitted.wait(5), "escrita não alcançou _apply")
                self.assertNotEqual(transient, [before["head"]])
                reader.start()
                self.assertTrue(reader_progress.wait(5), "leitor não progrediu")
                self.assertTrue(reader_blocked.is_set(), "getter não aguardou o lock da transação")
                self.assertFalse(reader_done.is_set(), "getter expôs head não confirmado")
            finally:
                release_writer.set()
                writer.join(5)
                if reader.ident is not None:
                    reader.join(5)
            self.assertFalse(writer.is_alive())
            self.assertFalse(reader.is_alive())

        self.assertEqual(reader_errors, [])
        after = store.export_snapshot()
        if rollback:
            self.assertEqual(len(writer_errors), 1)
            self.assertIs(type(writer_errors[0]), RuntimeError)
            self.assertEqual(str(writer_errors[0]), "rollback injetado")
            self.assertEqual(after, before)
        else:
            self.assertEqual(writer_errors, [])
            self.assertNotEqual(after["head"], before["head"])
            self.assertGreater(len(after["events"]), len(before["events"]))
            if operation == "replace":
                self.assertNotEqual(after["head"], transient[0])
        self.assertTrue(reader_done.is_set())
        self.assertEqual(observed, [after["head"]])
        self.assertEqual(store.head_hash, after["head"])
        self.assertEqual(store.check_invariants(), [])
        restored = TemporalFactStore.from_snapshot(after, expected_head=store.head_hash)
        self.assertEqual(restored.history("s", "p"), store.history("s", "p"))

    def test_reader_waits_for_commit(self):
        for operation in ("assert", "replace", "corroborate", "retract"):
            with self.subTest(operation=operation):
                self._check_transaction(operation, rollback=False)

    def test_reader_waits_for_rollback(self):
        for operation in ("assert", "replace", "corroborate", "retract"):
            with self.subTest(operation=operation):
                self._check_transaction(operation, rollback=True)

    def test_normal_and_reentrant_reads_preserve_value(self):
        store = TemporalFactStore(clock=lambda: 100.0)
        self.assertEqual(store.head_hash, GENESIS)
        store.assert_fact("s", "p", "o")
        with store._lock:
            self.assertIs(type(store.head_hash), str)
            self.assertEqual(store.head_hash, store.export_snapshot()["head"])


if __name__ == "__main__":
    unittest.main()
