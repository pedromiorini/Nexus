"""C4–C7 da B2c: a rota vê o episódio aberto corrente.

Critérios aceitos em docs/PONTE_EPISODIC_B2C_CRITERIOS_2026-11-19.md.
Este arquivo só observa. Não altera produção.
"""
import unittest

import test_router_episodic_memory_bridge_contract as bridge
from core.cognitivo.episodic_memory import Episode

RECALL = "remember the last time"
C1_TITLE = "Q: remember the last time"
EPISODE_KEYS = {"episode_id", "title", "trigger_query", "importance", "status"}


def _trace(memory):
    calls = []
    for name in bridge.EPISODIC_METHODS:
        original = getattr(memory, name)

        def wrapped(*args, original=original, name=name, **kwargs):
            result = original(*args, **kwargs)
            current = memory.current_episode_id if name == "search_episodes" else None
            calls.append((name, args, dict(kwargs), result, current))
            return result

        setattr(memory, name, wrapped)
    return calls


def _fields(episodes):
    return [
        (episode.episode_id, episode.title, episode.trigger_query, episode.importance, episode.status)
        for episode in episodes
    ]


def _episode_dict(episode_id, title, trigger_query, importance, status):
    return {
        "episode_id": episode_id,
        "title": title,
        "trigger_query": trigger_query,
        "importance": importance,
        "status": status,
    }


class EpisodicB2cRouteOpenCurrentContractTests(unittest.TestCase):
    def _brain(self):
        created = bridge.make_brain()
        self.addCleanup(created.episodic_memory.close)
        return created

    def _open_c1(self, memory):
        return memory.start_episode(C1_TITLE, trigger_query=RECALL, importance=0.7)

    def _closed_plain(self, memory):
        episode_id = memory.start_episode(RECALL)
        memory.close_episode(episode_id)
        return episode_id

    def test_c4_think_recall_search_returns_the_episode_it_just_opened(self):
        created = self._brain()
        calls = _trace(created.episodic_memory)
        with bridge.silence():
            result = created.think(RECALL, use_router=True)
        self.assertEqual(
            bridge.method_names(calls),
            ["start_episode", "search_episodes", "close_episode"],
        )
        self.assertEqual(calls[1][1], ())
        self.assertEqual(calls[1][2]["query"], RECALL)
        self.assertEqual(calls[1][2]["limit"], 3)
        self.assertEqual(calls[1][4], result["episode_id"])
        self.assertNotIn("episodes", result)
        self.assertEqual(
            _fields(calls[1][3]),
            [(result["episode_id"], C1_TITLE, RECALL, 0.7, "open")],
        )
        self.assertIsInstance(calls[1][3][0], Episode)

    def test_c5_think_long_recall_returns_truncated_title_and_full_trigger(self):
        question = "remember " + ("x" * 80)
        created = self._brain()
        calls = _trace(created.episodic_memory)
        with bridge.silence():
            result = created.think(question, use_router=True)
        self.assertEqual(result["request_type"], "recall")
        self.assertEqual(
            bridge.method_names(calls),
            ["start_episode", "search_episodes", "close_episode"],
        )
        self.assertEqual(calls[1][1], ())
        self.assertEqual(calls[1][2]["query"], question)
        self.assertEqual(calls[1][2]["limit"], 3)
        self.assertEqual(calls[1][4], result["episode_id"])
        self.assertNotIn("episodes", result)
        self.assertEqual(
            _fields(calls[1][3]),
            [(
                result["episode_id"],
                "Q: " + question[:50],
                question,
                0.7,
                "open",
            )],
        )

    def test_c6_think_appends_its_open_episode_and_keeps_the_prior_open_out(self):
        created = self._brain()
        memory = created.episodic_memory
        closed_id = self._closed_plain(memory)
        prior_id = memory.start_episode("remember the last time prior")
        calls = _trace(memory)
        with bridge.silence():
            result = created.think(RECALL, use_router=True)
        self.assertEqual(
            bridge.method_names(calls),
            ["start_episode", "search_episodes", "close_episode"],
        )
        self.assertEqual(calls[1][2]["query"], RECALL)
        self.assertEqual(calls[1][2]["limit"], 3)
        self.assertEqual(calls[1][4], result["episode_id"])
        self.assertNotEqual(calls[1][4], prior_id)
        self.assertEqual(
            _fields(calls[1][3]),
            [
                (closed_id, RECALL, None, 0.6, "closed"),
                (result["episode_id"], C1_TITLE, RECALL, 0.7, "open"),
            ],
        )
        self.assertNotIn(prior_id, [episode.episode_id for episode in calls[1][3]])

    def test_c7_route_projects_the_appended_open_episode(self):
        created = self._brain()
        memory = created.episodic_memory
        closed_id = self._closed_plain(memory)
        open_id = self._open_c1(memory)
        with bridge.silence():
            result = created.router.route(RECALL, {})
        self.assertIs(result["success"], True)
        self.assertIs(result["degraded"], False)
        self.assertNotIn("episode_id", result)
        self.assertNotIn("episodic", result)
        self.assertEqual(memory.current_episode_id, open_id)
        self.assertEqual(
            result["episodes"],
            [
                _episode_dict(closed_id, RECALL, None, 0.6, "closed"),
                _episode_dict(open_id, C1_TITLE, RECALL, 0.7, "open"),
            ],
        )
        for episode in result["episodes"]:
            self.assertEqual(set(episode), EPISODE_KEYS)


if __name__ == "__main__":
    unittest.main()
