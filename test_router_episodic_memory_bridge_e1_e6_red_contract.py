"""RED da errata de C5 e de E1–E6.

Pré-registro aprovado em docs/cognitivo/PONTE_EPISODIC_MEMORY_SECAO_B_ERRATA_C5_E1_E6_2026-11-12.md.
Na main e21d25c a chave episodes ainda não existe: E1, E2 e E6 falham por isso.
E3, E4 e E5 exigem a ausência da chave e podem passar. Este arquivo não implementa o GREEN.
"""
import unittest

import test_router_episodic_memory_bridge_contract as bridge
from core.constitutional_brain import RealEpisodicMemory

RECALL = bridge.RECALL
FIVE = ("episode_id", "title", "trigger_query", "importance", "status")


def _projection(episode_id, title, trigger_query, importance, status="closed"):
    return {
        "episode_id": episode_id,
        "title": title,
        "trigger_query": trigger_query,
        "importance": importance,
        "status": status,
    }


def _count_search(memory):
    calls = []
    original = memory.search_episodes

    def wrapped(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        return original(*args, **kwargs)

    memory.search_episodes = wrapped
    return calls


class ErrataC5AndE2EmptySuccess(unittest.TestCase):
    def test_successful_empty_search_projects_an_empty_list_and_caches_it(self):
        router, recorder = bridge.harness_h()
        with bridge.silence():
            first = router.route(RECALL, {})
            second = router.route(RECALL, {})
        self.assertEqual(
            recorder.calls,
            [("search_episodes", (), {"query": RECALL, "limit": 3})],
        )
        self.assertIs(first["success"], True)
        self.assertIs(first["degraded"], False)
        self.assertEqual(first["request_type"], "recall")
        self.assertEqual(first["answer"], "No answer generated")
        self.assertEqual(
            first["modules_executed"],
            ["jailbreak", "budget", "memory", "episodic", "log"],
        )
        self.assertNotIn("episodic", first)
        self.assertNotIn("episode_id", first)
        self.assertIs(second["cached"], True)
        self.assertEqual(bridge.method_names(recorder.calls).count("search_episodes"), 1)
        self.assertIn("episodes", first)
        self.assertEqual(first["episodes"], [])
        self.assertEqual(second["episodes"], [])


class E1ClosedEpisodeProjection(unittest.TestCase):
    def test_closed_episode_is_projected_and_the_second_route_repeats_it(self):
        memory = RealEpisodicMemory(":memory:")
        self.addCleanup(memory.close)
        episode_id = memory.start_episode(RECALL)
        memory.close_episode(episode_id)
        calls = _count_search(memory)
        router, _recorder = bridge.harness_h(memory)
        with bridge.silence():
            first = router.route(RECALL, {})
            second = router.route(RECALL, {})
        self.assertEqual(calls, [((), {"query": RECALL, "limit": 3})])
        self.assertNotIn("episode_id", first)
        self.assertNotIn("episodic", first)
        self.assertIs(second["cached"], True)
        self.assertEqual(len(calls), 1)
        expected = [_projection(episode_id, RECALL, None, 0.6)]
        self.assertIn("episodes", first)
        self.assertEqual(list(first["episodes"][0]), list(FIVE))
        self.assertEqual(first["episodes"], expected)
        self.assertEqual(second["episodes"], expected)


class E3PromptsWithoutEpisodes(unittest.TestCase):
    PROMPTS = (
        "ping",
        "",
        "learn this fact",
        "Tell me about item 9",
        "should I plan my week",
        "what did I say about the right choice",
    )

    def test_prompts_that_do_not_search_have_no_episodes_key(self):
        for prompt in self.PROMPTS:
            with self.subTest(prompt=prompt):
                router, _recorder = bridge.harness_h()
                with bridge.silence():
                    result = router.route(prompt, {})
                self.assertNotIn("episodes", result)


class E4NoEpisodesWhenTheSearchDoesNotSucceed(unittest.TestCase):
    def test_missing_module_degrades_without_episodes(self):
        router, recorder = bridge.harness_h(include_episodic=False)
        with bridge.silence():
            result = router.route(RECALL, {})
        self.assertIs(result["success"], True)
        self.assertIs(result["degraded"], True)
        self.assertEqual(result["failed_modules"], ["episodic"])
        self.assertEqual(
            result["failed_modules_detail"],
            [{"module": "episodic", "error": "Module not found in brain"}],
        )
        self.assertEqual(result["request_type"], "recall")
        self.assertEqual(recorder.calls, [])
        self.assertNotIn(result.get("error"), bridge.BLOCKING_ERRORS)
        self.assertNotIn("episodes", result)

    def test_search_exception_degrades_and_is_not_cached(self):
        recorder = bridge.EpisodicRecorder(search_raises=ValueError("query must be non-empty"))
        router, recorder = bridge.harness_h(recorder)
        with bridge.silence():
            first = router.route(RECALL, {})
            second = router.route(RECALL, {})
        self.assertIs(first["success"], True)
        self.assertIs(first["degraded"], True)
        self.assertEqual(first["failed_modules"], ["episodic"])
        self.assertEqual(
            first["failed_modules_detail"],
            [{"module": "episodic", "error": "query must be non-empty"}],
        )
        self.assertIs(first["cached"], False)
        self.assertIs(second["cached"], False)
        self.assertEqual(bridge.method_names(recorder.calls).count("search_episodes"), 2)
        self.assertNotIn(first.get("error"), bridge.BLOCKING_ERRORS)
        self.assertNotIn(second.get("error"), bridge.BLOCKING_ERRORS)
        self.assertNotIn("episodes", first)
        self.assertNotIn("episodes", second)

    def test_detected_jailbreak_has_no_episodes(self):
        router, recorder = bridge.harness_h(jailbreak_detected=True)
        with bridge.silence():
            result = router.route(RECALL, {})
        self.assertIs(result["success"], False)
        self.assertEqual(result["error"], "JAILBREAK_DETECTED")
        self.assertEqual(result["failed_modules"], ["jailbreak"])
        self.assertEqual(recorder.calls, [])
        self.assertNotIn("episodes", result)

    def test_empty_map_has_no_episodes(self):
        router = bridge.empty_router()
        for prompt in ("ping", RECALL):
            with self.subTest(prompt=prompt):
                with bridge.silence():
                    result = router.route(prompt, {})
                self.assertIs(result["success"], False)
                self.assertEqual(result["error"], "JAILBREAK_CHECK_FAILED")
                self.assertEqual(result["failed_modules"], ["jailbreak"])
                self.assertNotIn("episodes", result)


class E5ThinkDoesNotGainEpisodes(unittest.TestCase):
    def test_recall_think_keeps_c12_and_has_no_episodes(self):
        created = bridge.make_brain()
        self.addCleanup(created.episodic_memory.close)
        calls = bridge.wrap_episodic(created.episodic_memory)
        with bridge.silence():
            result = created.think(RECALL, use_router=True)
        self.assertNotIn("error", result)
        self.assertEqual(
            bridge.method_names(calls),
            ["start_episode", "search_episodes", "close_episode"],
        )
        self.assertEqual(calls[0][2], {
            "title": "Q: remember the last time",
            "trigger_query": RECALL,
            "importance": 0.7,
        })
        self.assertEqual(calls[1][1], ())
        self.assertEqual(calls[1][2], {"query": RECALL, "limit": 3})
        self.assertEqual(result["episode_id"], calls[0][3])
        self.assertEqual(calls[2][1], (result["episode_id"],))
        self.assertEqual(calls[2][2], {})
        self.assertNotIn("get_episode_memories", bridge.method_names(calls))
        self.assertNotIn("episodes", result)


class E6ForcedLearningProjection(unittest.TestCase):
    def test_forced_learning_projects_the_closed_ping_episode(self):
        memory = RealEpisodicMemory(":memory:")
        self.addCleanup(memory.close)
        episode_id = memory.start_episode("ping")
        memory.close_episode(episode_id)
        router, _recorder = bridge.harness_h(memory)
        router.request_type_proposers = (lambda state: ("learning",),)
        with bridge.silence():
            result = router.route("ping", {})
        self.assertEqual(result["request_type"], "learning")
        self.assertNotIn("episode_id", result)
        self.assertNotIn("episodic", result)
        expected = [_projection(episode_id, "ping", None, 0.6)]
        self.assertIn("episodes", result)
        self.assertEqual(list(result["episodes"][0]), list(FIVE))
        self.assertEqual(result["episodes"], expected)

    def test_learn_this_fact_without_proposer_stays_in_e3(self):
        router, _recorder = bridge.harness_h()
        with bridge.silence():
            result = router.route("learn this fact", {})
        self.assertNotIn("episodes", result)


if __name__ == "__main__":
    unittest.main()
