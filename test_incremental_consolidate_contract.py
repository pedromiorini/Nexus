"""Contrato de idempotência do incremental learner (C1–C9).

Critérios pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md antes
desta correção. C10 é limite declarado e não tem teste: o estado novo vive
só na instância. O fixture JSON é o fingerprint IL6 em 26e38e5, antes da
mudança de comportamento.
"""

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.cognitivo.incremental_learner import RealIncrementalLearner

ROOT = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "test_incremental_fingerprint_26e38e5.json"
STATS_KEYS = (
    "total_promotions",
    "total_reinforcements",
    "total_patterns",
    "total_consolidations",
    "total_learning_events",
    "pattern_registry_size",
)
DECLARED_PREFIXES = (
    "promotion.again",
    "promotion.again_stores",
    "promotion.again_stats",
    "patterns.own_synthetic_second",
    "patterns.own_stores",
)


class _Mem:
    def __init__(self, mode="empty"):
        self.mode = mode
        self.stores = []

    def store(self, content, tier, importance):
        self.stores.append((content, tier, importance))
        return len(self.stores)

    def retrieve(self, query, limit=1):
        if self.mode == "echo":
            return [{"content": item[0]} for item in self.stores]
        if self.mode == "tag":
            return [{"content": query}]
        if self.mode == "other":
            return [{"content": "unrelated"}]
        return []


class _Log:
    def __init__(self):
        self.events = []

    def log_event(self, kind, payload):
        self.events.append((kind, payload))


class _Epi:
    def __init__(self, memories):
        self.memories = list(memories)

    def get_episode_memories(self, episode_id):
        return list(self.memories)


class _Book:
    def __init__(self, episodes):
        self.episodes = episodes
        self.searches = []

    def search_episodes(self, status="closed", limit=10):
        self.searches.append({"status": status, "limit": limit})
        return list(self.episodes)[:limit]

    def get_episode_memories(self, episode_id):
        for episode in self.episodes:
            if episode.episode_id == episode_id:
                return list(episode.memories)
        return []


def _mem_item(mem_id, content, relevance, importance):
    return {
        "id": mem_id,
        "content": content,
        "relevance_to_episode": relevance,
        "importance": importance,
    }


def _same(left, right):
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return left == right and type(left) is type(right)


def _diff_paths(live, frozen, prefix=""):
    if isinstance(live, dict) and isinstance(frozen, dict):
        found = set()
        for key in set(live) | set(frozen):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in live or key not in frozen:
                found.add(path)
            else:
                found |= _diff_paths(live[key], frozen[key], path)
        return found
    if isinstance(live, list) and isinstance(frozen, list):
        if len(live) != len(frozen):
            return {prefix}
        found = set()
        for index, (item, other) in enumerate(zip(live, frozen)):
            found |= _diff_paths(item, other, f"{prefix}[{index}]")
        return found
    if _same(live, frozen):
        return set()
    return {prefix}


def _allowed(path):
    for prefix in DECLARED_PREFIXES:
        if path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "["):
            return True
    return False


class IncrementalConsolidateContractTests(unittest.TestCase):
    def test_consolidate_same_episodes_three_times_is_idempotent(self):
        episodes = [
            SimpleNamespace(episode_id="ep-a", memories=[_mem_item(1, "solo", 0.9, 0.8)]),
            SimpleNamespace(episode_id="ep-b", memories=[_mem_item(2, "solo", 0.1, 0.1)]),
            SimpleNamespace(episode_id="ep-c", memories=[_mem_item(3, "only-once", 0.1, 0.1)]),
        ]
        memory = _Mem()
        learner = RealIncrementalLearner(_Book(episodes), memory, _Log())
        self.assertEqual(tuple(learner.get_statistics()), STATS_KEYS)
        first, second, third = (learner.consolidate_learning(last_n_episodes=10) for _ in range(3))
        self.assertEqual(first.memories_promoted, 1)
        self.assertEqual(first.patterns_detected, 1)
        self.assertGreaterEqual(first.episodes_processed, 1)
        for report in (second, third):
            self.assertEqual(report.episodes_processed, first.episodes_processed)
            self.assertEqual(report.memories_promoted, 0)
            self.assertEqual(report.memories_reinforced, 0)
            self.assertEqual(report.patterns_detected, 0)
            self.assertEqual(report.total_learning_events, 0)
        self.assertEqual(learner.total_promotions, 1)
        self.assertEqual(learner.total_reinforcements, 0)
        self.assertEqual(learner.total_patterns, 1)
        self.assertEqual(learner.total_consolidations, 3)
        self.assertEqual(dict(learner.pattern_registry), {"solo": 2, "only-once": 1})
        self.assertEqual(
            [row[0] for row in memory.stores],
            ["solo", "[PATTERN freq=2] solo"],
        )
        self.assertEqual(tuple(learner.get_statistics()), STATS_KEYS)

    def test_pattern_counts_distinct_episodes_not_occurrences(self):
        with self.subTest("duas ocorrências no mesmo episódio"):
            epi = _Epi([
                _mem_item(1, "Repeat Me", 0.1, 0.1),
                _mem_item(2, "  REPEAT ME  ", 0.1, 0.1),
            ])
            memory = _Mem()
            learner = RealIncrementalLearner(epi, memory, None)
            learner.learn_from_episode("one")
            self.assertEqual(learner.pattern_registry.get("repeat me"), 1)
            self.assertEqual(learner.detect_and_store_patterns(), 0)
            self.assertEqual(memory.stores, [])
            epi.memories = [_mem_item(3, "repeat me", 0.1, 0.1)]
            learner.learn_from_episode("two")
            self.assertEqual(learner.pattern_registry.get("repeat me"), 2)
            self.assertEqual(learner.detect_and_store_patterns(), 1)

        with self.subTest("o mesmo episódio não conta de novo"):
            epi = _Epi([_mem_item(1, "Repeat Me", 0.1, 0.1)])
            learner = RealIncrementalLearner(epi, _Mem(), None)
            learner.learn_from_episode("one")
            learner.learn_from_episode("one")
            self.assertEqual(learner.pattern_registry.get("repeat me"), 1)

        with self.subTest("nan e relevância negativa contam um episódio"):
            for relevance in (float("nan"), -1):
                epi = _Epi([_mem_item(1, "quiet", relevance, 0.9)])
                learner = RealIncrementalLearner(epi, _Mem(), None)
                learner.learn_from_episode("one")
                learner.learn_from_episode("one")
                self.assertEqual(learner.pattern_registry.get("quiet"), 1)
                self.assertEqual(learner.learning_history, [])

        with self.subTest("janelas sobrepostas não recontam episódio"):
            episodes = [
                SimpleNamespace(episode_id=f"e{i}", memories=[_mem_item(i + 1, "same text", 0.2, 0.2)])
                for i in range(4)
            ]
            learner = RealIncrementalLearner(_Book(episodes), _Mem(), None)
            learner.consolidate_learning(last_n_episodes=2, run_pattern_detection=False, run_decay=False)
            learner.consolidate_learning(last_n_episodes=4, run_pattern_detection=False, run_decay=False)
            self.assertEqual(learner.pattern_registry.get("same text"), 4)

    def test_pattern_emitted_at_most_once_regardless_of_retrieve(self):
        def twice(mode):
            epi = _Epi([_mem_item(1, "Repeat Me", 0.1, 0.1)])
            log = _Log()
            memory = _Mem(mode)
            learner = RealIncrementalLearner(epi, memory, log)
            learner.learn_from_episode("a")
            epi.memories = [_mem_item(2, "repeat me", 0.1, 0.1)]
            learner.learn_from_episode("b")
            first = learner.detect_and_store_patterns()
            second = learner.detect_and_store_patterns()
            return first, second, memory, log, learner

        with self.subTest("retrieve ecoa o sintético"):
            first, second, memory, _log, learner = twice("echo")
            self.assertEqual(first, 1)
            self.assertEqual(second, 0)
            self.assertEqual(memory.stores, [("[PATTERN freq=2] repeat me", "long_term", 0.92)])
            self.assertEqual(learner.total_patterns, 1)

        with self.subTest("retrieve vazio"):
            first, second, memory, log, learner = twice("empty")
            self.assertEqual((first, second), (1, 0))
            self.assertEqual(len(memory.stores), 1)
            self.assertEqual(log.events, [("PATTERN_DETECTED", {"content": "repeat me", "frequency": 2})])
            self.assertEqual(learner.total_patterns, 1)

        with self.subTest("retrieve devolve texto que não é a tag"):
            _first, second, memory, _log, _learner = twice("other")
            self.assertEqual(second, 0)
            self.assertEqual(len(memory.stores), 1)

        with self.subTest("retrieve que devolve a tag continua sem gravar"):
            first, second, memory, _log, learner = twice("tag")
            self.assertEqual((first, second), (0, 0))
            self.assertEqual(memory.stores, [])
            self.assertEqual(learner.total_patterns, 0)

    def test_reinforcement_stores_once_per_episode_memory(self):
        with self.subTest("duas chamadas"):
            memory = _Mem()
            learner = RealIncrementalLearner(_Epi([_mem_item(2, "keep", 0.6, 0.8)]), memory, None)
            first = learner.learn_from_episode("ep")
            second = learner.learn_from_episode("ep")
            third = learner.learn_from_episode("ep")
            self.assertEqual(len(first), 1)
            self.assertFalse(first[0].promotion)
            self.assertEqual(first[0].new_importance, min(0.99, 0.8 + 0.08))
            self.assertEqual(second, [])
            self.assertEqual(third, [])
            self.assertEqual(memory.stores, [("keep", "long_term", 0.88)])
            self.assertEqual(learner.total_reinforcements, 1)

        with self.subTest("o mesmo id duas vezes na mesma lista"):
            memory = _Mem()
            learner = RealIncrementalLearner(_Epi([
                _mem_item(3, "keep", 0.6, 0.8),
                _mem_item(3, "keep", 0.6, 0.8),
            ]), memory, None)
            events = learner.learn_from_episode("ep")
            self.assertEqual(len(events), 1)
            self.assertEqual(len(memory.stores), 1)

        with self.subTest("bool não é rejeitado e o par entra uma vez"):
            memory = _Mem()
            learner = RealIncrementalLearner(_Epi([_mem_item(4, "keep", 0.6, True)]), memory, None)
            first = learner.learn_from_episode("ep")
            second = learner.learn_from_episode("ep")
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0].new_importance, 0.99)
            self.assertIs(type(first[0].old_importance), bool)
            self.assertEqual(second, [])
            self.assertEqual(len(memory.stores), 1)

        with self.subTest("str levanta e não consome o par"):
            item = _mem_item(5, "keep", 0.6, "0.8")
            memory = _Mem()
            learner = RealIncrementalLearner(_Epi([item]), memory, None)
            with self.assertRaises(TypeError):
                learner.learn_from_episode("ep")
            self.assertEqual(memory.stores, [])
            item["importance"] = 0.8
            self.assertEqual(len(learner.learn_from_episode("ep")), 1)
            self.assertEqual(learner.learn_from_episode("ep"), [])
            self.assertEqual(len(memory.stores), 1)

    def test_promoted_memory_not_reinforced_on_same_episode(self):
        with self.subTest("o mesmo episódio"):
            memory = _Mem()
            learner = RealIncrementalLearner(_Epi([_mem_item(7, "alpha fact", 0.9, 0.8)]), memory, None)
            first = learner.learn_from_episode("ep")
            second = learner.learn_from_episode("ep")
            self.assertEqual(len(first), 1)
            self.assertTrue(first[0].promotion)
            self.assertEqual(second, [])
            self.assertEqual(learner.total_promotions, 1)
            self.assertEqual(learner.total_reinforcements, 0)
            self.assertEqual(len(memory.stores), 1)
            self.assertEqual(learner.pattern_registry.get("alpha fact"), 1)

        with self.subTest("relevância True e inf"):
            for relevance in (True, float("inf")):
                memory = _Mem()
                learner = RealIncrementalLearner(_Epi([_mem_item(8, "edge", relevance, 0.5)]), memory, None)
                first = learner.learn_from_episode("ep")
                second = learner.learn_from_episode("ep")
                self.assertEqual(len(first), 1)
                self.assertTrue(first[0].promotion)
                self.assertEqual(second, [])
                self.assertEqual(len(memory.stores), 1)

        with self.subTest("outro episódio com o mesmo id ainda reforça"):
            epi = _Epi([_mem_item(9, "alpha fact", 0.9, 0.8)])
            memory = _Mem()
            learner = RealIncrementalLearner(epi, memory, None)
            learner.learn_from_episode("ep-a")
            epi.memories = [_mem_item(9, "alpha fact", 0.6, 0.8)]
            second = learner.learn_from_episode("ep-b")
            self.assertEqual(len(second), 1)
            self.assertFalse(second[0].promotion)
            self.assertEqual(learner.total_promotions, 1)
            self.assertEqual(learner.total_reinforcements, 1)
            self.assertEqual(len(memory.stores), 2)

    def test_report_zero_events_on_reconsolidation(self):
        episodes = [
            SimpleNamespace(episode_id="ep-a", memories=[_mem_item(1, "solo", 0.9, 0.8)]),
            SimpleNamespace(episode_id="ep-b", memories=[_mem_item(2, "solo", 0.6, 0.8)]),
        ]
        log = _Log()
        learner = RealIncrementalLearner(_Book(episodes), _Mem(), log)
        first = learner.consolidate_learning(last_n_episodes=10)
        second = learner.consolidate_learning(last_n_episodes=10)
        third = learner.consolidate_learning(last_n_episodes=10)
        self.assertEqual(first.total_learning_events, 2)
        for report in (second, third):
            self.assertEqual(report.episodes_processed, 2)
            self.assertEqual(report.memories_promoted, 0)
            self.assertEqual(report.memories_reinforced, 0)
            self.assertEqual(report.patterns_detected, 0)
            self.assertEqual(report.total_learning_events, 0)
        kinds = [kind for kind, _payload in log.events]
        self.assertEqual(kinds.count("PATTERN_DETECTED"), 1)
        self.assertEqual(kinds.count("DECAY_APPLIED"), 1)
        self.assertEqual(kinds.count("LEARNING_CONSOLIDATION"), 3)
        consolidations = [payload for kind, payload in log.events if kind == "LEARNING_CONSOLIDATION"]
        self.assertEqual(consolidations[1]["events_generated"], 0)
        self.assertEqual(consolidations[1]["patterns_found"], 0)
        self.assertEqual(consolidations[2]["events_generated"], 0)
        self.assertEqual(consolidations[2]["patterns_found"], 0)

    def test_late_linked_memory_processed_once(self):
        episode = SimpleNamespace(episode_id="late", memories=[_mem_item(1, "first", 0.9, 0.8)])
        memory = _Mem()
        learner = RealIncrementalLearner(_Book([episode]), memory, _Log())
        learner.consolidate_learning(last_n_episodes=10)
        episode.memories.append(_mem_item(2, "second", 0.9, 0.5))
        second = learner.consolidate_learning(last_n_episodes=10)
        third = learner.consolidate_learning(last_n_episodes=10)
        self.assertEqual(second.memories_promoted, 1)
        self.assertEqual(second.memories_reinforced, 0)
        self.assertEqual(third.memories_promoted, 0)
        self.assertEqual(third.memories_reinforced, 0)
        self.assertEqual(third.total_learning_events, 0)
        self.assertEqual([row[0] for row in memory.stores if row[0] == "first"], ["first"])
        self.assertEqual([row[0] for row in memory.stores if row[0] == "second"], ["second"])

    def test_new_independent_episode_still_accumulates(self):
        epi = _Epi([_mem_item(1, "solo", 0.8, 0.5)])
        memory = _Mem()
        learner = RealIncrementalLearner(epi, memory, None)
        first = learner.learn_from_episode("ep-a")
        epi.memories = [_mem_item(2, "solo", 0.8, 0.5)]
        second = learner.learn_from_episode("ep-b")
        self.assertEqual(len(first), 1)
        self.assertTrue(first[0].promotion)
        self.assertEqual(len(second), 1)
        self.assertTrue(second[0].promotion)
        self.assertEqual(learner.total_promotions, 2)
        self.assertEqual(learner.pattern_registry.get("solo"), 2)
        detected = learner.detect_and_store_patterns()
        self.assertEqual(detected, 1)

        reinforced = _Epi([_mem_item(3, "keep", 0.6, 0.8)])
        other = RealIncrementalLearner(reinforced, _Mem(), None)
        other.learn_from_episode("r-a")
        reinforced.memories = [_mem_item(4, "keep", 0.6, 0.8)]
        again = other.learn_from_episode("r-b")
        self.assertEqual(len(again), 1)
        self.assertFalse(again[0].promotion)
        self.assertEqual(other.total_reinforcements, 2)

        episodes = [
            SimpleNamespace(episode_id="c1", memories=[_mem_item(5, "batch", 0.8, 0.4)]),
            SimpleNamespace(episode_id="c2", memories=[_mem_item(6, "batch", 0.8, 0.4)]),
        ]
        batch = RealIncrementalLearner(_Book(episodes), _Mem(), None)
        report = batch.consolidate_learning(last_n_episodes=10, run_pattern_detection=False, run_decay=False)
        self.assertEqual(report.memories_promoted, 2)
        self.assertEqual(report.memories_reinforced, 0)
        self.assertEqual(batch.pattern_registry.get("batch"), 2)

    def test_fingerprint_changes_only_declared_keys(self):
        import core.cognitivo.incremental_learner as incremental
        from tools.cognitivo_eval import incremental_fingerprint

        learner = RealIncrementalLearner(_Epi([]), _Mem(), None)
        self.assertEqual(tuple(learner.get_statistics()), STATS_KEYS)
        frozen = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        live = incremental_fingerprint(incremental)
        self.assertEqual(live["promotion"]["again"], [])
        self.assertEqual(live["promotion"]["again_stores"], live["promotion"]["stores"])
        self.assertEqual(live["promotion"]["again_promoted"], [7])
        self.assertEqual(live["promotion"]["again_stats"]["total_promotions"], 1)
        self.assertEqual(live["promotion"]["again_stats"]["total_reinforcements"], 0)
        self.assertEqual(live["promotion"]["again_stats"]["total_learning_events"], 1)
        self.assertEqual(live["promotion"]["again_stats"]["pattern_registry_size"], 1)
        self.assertEqual(live["patterns"]["own_synthetic_first"], 1)
        self.assertEqual(live["patterns"]["own_synthetic_second"], 0)
        self.assertEqual(
            live["patterns"]["own_stores"],
            [["[PATTERN freq=2] repeat me", "long_term", 0.92]],
        )
        diff = _diff_paths(live, frozen)
        unexpected = sorted(path for path in diff if not _allowed(path))
        self.assertEqual(unexpected, [])
        for prefix in DECLARED_PREFIXES:
            self.assertTrue(
                any(path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[") for path in diff),
                prefix,
            )
