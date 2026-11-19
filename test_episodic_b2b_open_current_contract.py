"""C1–C3 e N1–N10 da B2b: o aberto corrente só entra se a chamada pedir.

Critérios aceitos em docs/PONTE_EPISODIC_B2B_CRITERIOS_2026-11-18.md.
Este arquivo só observa. Não altera produção.
"""
import hashlib
import inspect
import subprocess
import unittest
from pathlib import Path

import test_router_episodic_memory_bridge_contract as bridge
from core.cognitivo.episodic_memory import Episode, RealEpisodicMemory

ROOT = Path(__file__).resolve().parent
BASE = "31f0aa0dff487cd7990e6fb29e63c093e30fc6a6"
LIKE_LITERAL = "test_episodic_like_literal_contract.py"
LIKE_LITERAL_SHA256 = "77ad72999806872d4c893f6a419d0635211a696747428e6145660778994c2c30"
RECALL = "remember the last time"
C1_TITLE = "Q: remember the last time"
THIS_FILE = "test_episodic_b2b_open_current_contract.py"


def _flag_available():
    return "include_current_open" in inspect.signature(RealEpisodicMemory.search_episodes).parameters


def _fields(episodes):
    return [
        (episode.episode_id, episode.title, episode.trigger_query, episode.importance, episode.status)
        for episode in episodes
    ]


class EpisodicB2bOpenCurrentContractTests(unittest.TestCase):
    def _memory(self):
        memory = RealEpisodicMemory(":memory:")
        self.addCleanup(memory.close)
        return memory

    def _open_c1(self, memory):
        return memory.start_episode(C1_TITLE, trigger_query=RECALL, importance=0.7)

    def _closed_plain(self, memory):
        episode_id = memory.start_episode(RECALL)
        memory.close_episode(episode_id)
        return episode_id

    def _like_memory(self):
        """Mesma montagem de EpisodicLikeLiteralContractTests.setUp."""
        memory = self._memory()

        def closed(title, trigger_query=None):
            episode_id = memory.start_episode(title, trigger_query=trigger_query)
            memory.close_episode(episode_id)
            return episode_id

        closed("100%")
        closed("plain")
        closed("a_b")
        closed("axb")
        closed("rate_limit")
        closed("alpha release")
        closed("note", trigger_query="save 100%")
        memory.start_episode("100% open")
        return memory

    def _router(self, memory):
        modules = bridge.harness_modules(memory)
        with bridge.silence():
            router = bridge.CentralRouter(modules, bridge.ethics_guard())
        router.vram_guard = bridge._ObserveGuard()
        return router

    def test_c1_explicit_flag_returns_the_current_open_match(self):
        memory = self._memory()
        episode_id = self._open_c1(memory)
        found = memory.search_episodes(RECALL, limit=3, include_current_open=True)
        self.assertEqual(
            _fields(found),
            [(episode_id, C1_TITLE, RECALL, 0.7, "open")],
        )
        self.assertIsInstance(found[0], Episode)
        self.assertEqual(memory.current_episode_id, episode_id)

    def test_c2_closed_stays_and_current_open_is_appended_when_asked(self):
        memory = self._memory()
        closed_id = self._closed_plain(memory)
        open_id = self._open_c1(memory)
        found = memory.search_episodes(RECALL, limit=3, include_current_open=True)
        self.assertEqual(
            _fields(found),
            [
                (closed_id, RECALL, None, 0.6, "closed"),
                (open_id, C1_TITLE, RECALL, 0.7, "open"),
            ],
        )

    def test_c3_explicit_flag_appends_after_a_full_limit(self):
        memory = self._memory()
        for index in range(3):
            episode_id = memory.start_episode(
                f"remember the last time {index}",
                trigger_query=RECALL,
            )
            memory.close_episode(episode_id)
        self.assertIsNone(memory.current_episode_id)
        prefix = [episode.episode_id for episode in memory.search_episodes(RECALL, limit=3)]
        self.assertEqual(len(prefix), 3)
        open_id = self._open_c1(memory)
        second = [
            episode.episode_id
            for episode in memory.search_episodes(RECALL, limit=3, include_current_open=True)
        ]
        self.assertEqual(second, prefix + [open_id])
        self.assertEqual(second.count(open_id), 1)

    def test_n1_like_literal_file_stays_and_passes(self):
        digest = hashlib.sha256((ROOT / LIKE_LITERAL).read_bytes()).hexdigest()
        self.assertEqual(digest, LIKE_LITERAL_SHA256)
        proc = subprocess.run(
            ["python3", "-m", "unittest", LIKE_LITERAL],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(ROOT),
                "NEXUS_AUDIT_DB": "/tmp/nexus_audit_b2b_n1.sqlite3",
                "LC_ALL": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_n2_percent_search_without_flag_keeps_the_open_out(self):
        memory = self._like_memory()
        titles = [episode.title for episode in memory.search_episodes("%")]
        self.assertCountEqual(titles, ["100%", "note"])
        self.assertNotIn("100% open", titles)
        self.assertEqual(
            [episode.title for episode in memory.search_episodes("%", status="open")],
            ["100% open"],
        )
        self.assertEqual(len(memory.search_episodes("%", limit=1)), 1)

    def test_n3_default_search_does_not_append_the_current_open(self):
        memory = self._memory()
        episode_id = self._open_c1(memory)
        self.assertEqual(memory.search_episodes(RECALL, limit=3), [])
        self.assertEqual(memory.current_episode_id, episode_id)

        other = self._memory()
        for index in range(3):
            closed_id = other.start_episode(
                f"remember the last time {index}",
                trigger_query=RECALL,
            )
            other.close_episode(closed_id)
        self.assertIsNone(other.current_episode_id)
        prefix = [episode.episode_id for episode in other.search_episodes(RECALL, limit=3)]
        self.assertEqual(len(prefix), 3)
        self._open_c1(other)
        second = [episode.episode_id for episode in other.search_episodes(RECALL, limit=3)]
        self.assertEqual(second, prefix)

    def test_n4_non_current_open_stays_out_with_or_without_the_flag(self):
        memory = self._memory()
        open_id = memory.start_episode("kernel panic")
        closed_id = memory.start_episode("kernel panic again")
        memory.close_episode(closed_id)
        self.assertIsNone(memory.current_episode_id)
        self.assertNotEqual(open_id, memory.current_episode_id)
        self.assertEqual(
            [episode.episode_id for episode in memory.search_episodes("kernel")],
            [closed_id],
        )
        self.assertEqual(
            [episode.episode_id for episode in memory.search_episodes("kernel", status="open")],
            [open_id],
        )
        if _flag_available():
            flagged = memory.search_episodes("kernel", include_current_open=True)
            self.assertEqual([episode.episode_id for episode in flagged], [closed_id])

    def test_n5_search_without_a_matching_query_does_not_append(self):
        memory = self._memory()
        alpha = memory.start_episode("alpha", trigger_query="q1", importance=0.9)
        beta = memory.start_episode("beta", importance=0.3)
        memory.close_episode(alpha)
        self.assertEqual(memory.current_episode_id, beta)
        self.assertEqual(
            [episode.episode_id for episode in memory.search_episodes(limit=10)],
            [alpha],
        )
        self.assertEqual(
            [episode.episode_id for episode in memory.search_episodes("alp", limit=5)],
            [alpha],
        )
        if _flag_available():
            flagged = memory.search_episodes("alp", limit=5, include_current_open=True)
            self.assertEqual([episode.episode_id for episode in flagged], [alpha])

    def test_n6_limit_zero_and_invalid_input_stay(self):
        memory = self._memory()
        self._open_c1(memory)
        self.assertEqual(memory.search_episodes(RECALL, limit=0), [])
        with self.assertRaises(ValueError):
            memory.search_episodes("")
        with self.assertRaises(TypeError):
            memory.search_episodes(False)
        with self.assertRaises(ValueError):
            memory.search_episodes(RECALL, limit=-1)
        with self.assertRaises(TypeError):
            memory.search_episodes(RECALL, limit=True)
        if _flag_available():
            self.assertEqual(
                memory.search_episodes(RECALL, limit=0, include_current_open=True),
                [],
            )
            with self.assertRaises(ValueError):
                memory.search_episodes("", include_current_open=True)
            with self.assertRaises(TypeError):
                memory.search_episodes(False, include_current_open=True)
            with self.assertRaises(ValueError):
                memory.search_episodes(RECALL, limit=-1, include_current_open=True)
            with self.assertRaises(TypeError):
                memory.search_episodes(RECALL, limit=True, include_current_open=True)

    def test_n7_status_open_does_not_duplicate(self):
        memory = self._memory()
        prior_id = memory.start_episode("remember the last time prior")
        current_id = self._open_c1(memory)
        found = memory.search_episodes(RECALL, status="open", limit=10)
        ids = [episode.episode_id for episode in found]
        self.assertEqual(len(ids), 2)
        self.assertEqual(set(ids), {prior_id, current_id})
        if _flag_available():
            flagged = memory.search_episodes(
                RECALL, status="open", limit=10, include_current_open=True,
            )
            flagged_ids = [episode.episode_id for episode in flagged]
            self.assertEqual(len(flagged_ids), 2)
            self.assertEqual(set(flagged_ids), {prior_id, current_id})
            self.assertEqual(len(flagged_ids), len(set(flagged_ids)))

    def test_n8_bridge_kwargs_stay_query_and_limit(self):
        source = (ROOT / "core/constitutional_brain.py").read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            source[10436],
            "            return module.search_episodes(query=prompt, limit=3)",
        )
        router, recorder = bridge.harness_h()
        with bridge.silence():
            first = router.route(bridge.RECALL, {})
            second = router.route(bridge.RECALL, {})
        self.assertEqual(
            recorder.calls,
            [("search_episodes", (), {"query": bridge.RECALL, "limit": 3})],
        )
        self.assertNotIn("include_current_open", recorder.calls[0][2])
        self.assertNotIn("status", recorder.calls[0][2])
        self.assertNotIn("episode_id", first)
        self.assertNotIn("episodic", first)
        self.assertIs(second["cached"], True)

        router, recorder = bridge.harness_h()
        context = {"request_type": "query"}
        with bridge.silence():
            router.route(bridge.RECALL, context)
            router.route(bridge.RECALL, context)
        expected = ("search_episodes", (), {"query": bridge.RECALL, "limit": 3})
        self.assertEqual(recorder.calls, [expected, expected])

        memory = RealEpisodicMemory(":memory:")
        self.addCleanup(memory.close)
        episode_id = memory.start_episode(bridge.RECALL)
        memory.close_episode(episode_id)
        seen = []
        original = memory.search_episodes

        def wrapped(*args, **kwargs):
            found = original(*args, **kwargs)
            seen.append((args, dict(kwargs), [episode.episode_id for episode in found]))
            return found

        memory.search_episodes = wrapped
        with bridge.silence():
            result = self._router(memory).route(bridge.RECALL, {})
        self.assertEqual(seen, [((), {"query": bridge.RECALL, "limit": 3}, [episode_id])])
        self.assertNotIn("include_current_open", seen[0][1])
        self.assertNotIn("episode_id", result)
        self.assertNotIn("episodic", result)

        memory = RealEpisodicMemory(":memory:")
        self.addCleanup(memory.close)
        closed_id = memory.start_episode(bridge.RECALL)
        memory.close_episode(closed_id)
        open_id = memory.start_episode("remember the last time open")
        seen = []
        original = memory.search_episodes

        def wrapped_open(*args, **kwargs):
            found = original(*args, **kwargs)
            seen.append((args, dict(kwargs)))
            return found

        memory.search_episodes = wrapped_open
        with bridge.silence():
            result = self._router(memory).route(bridge.RECALL, {})
        self.assertEqual(seen, [((), {"query": bridge.RECALL, "limit": 3})])
        self.assertEqual(memory.current_episode_id, open_id)
        self.assertNotIn("episode_id", result)
        self.assertNotIn("episodic", result)

        router, recorder = bridge.harness_h()
        router.request_type_proposers = (lambda state: ("learning",),)
        with bridge.silence():
            router.route("ping", {})
        self.assertEqual(
            recorder.calls,
            [("search_episodes", (), {"query": "ping", "limit": 3})],
        )

        created = bridge.make_brain()
        self.addCleanup(created.episodic_memory.close)
        calls = bridge.wrap_episodic(created.episodic_memory)
        with bridge.silence():
            think_result = created.think(bridge.RECALL, use_router=True)
        self.assertEqual(
            bridge.method_names(calls),
            ["start_episode", "search_episodes", "close_episode"],
        )
        self.assertEqual(calls[1][1], ())
        self.assertEqual(calls[1][2], {"query": bridge.RECALL, "limit": 3})
        self.assertNotIn("include_current_open", calls[1][2])
        self.assertNotIn("status", calls[1][2])
        self.assertNotIn("episodes", think_result)

    def test_n9_existing_tests_are_not_edited(self):
        proc = subprocess.run(
            ["git", "diff", "--name-only", BASE],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        changed = [
            line for line in proc.stdout.splitlines()
            if line.startswith("test_") and line != THIS_FILE
        ]
        self.assertEqual(changed, [])
        self.assertNotIn("core/constitutional_brain.py", proc.stdout.splitlines())

    def test_n10_like_literal_baseline_file_passes(self):
        catalog = (ROOT / "tools/cognitivo_mutation.py").read_text(encoding="utf-8")
        self.assertIn(LIKE_LITERAL, catalog)
        proc = subprocess.run(
            ["python3", "-m", "unittest", LIKE_LITERAL],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(ROOT),
                "NEXUS_AUDIT_DB": "/tmp/nexus_audit_b2b_n10.sqlite3",
                "LC_ALL": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn("Baseline failed before mutation", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
