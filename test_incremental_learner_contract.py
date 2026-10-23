"""Contrato do incremental learner: tipos inválidos e extração strangler.

Critérios IL1–IL6 pré-registrados em docs/cognitivo/AUDITORIA_2026-10-22.md
(commit f11610d) antes do move e antes do eval. O harness é
tools/cognitivo_eval.py --capability incremental.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.constitutional_brain import RealIncrementalLearner

ROOT = Path(__file__).resolve().parent
_NAMES = ("LearningEvent", "ConsolidationReport", "RealIncrementalLearner")
_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.incremental_learner as mod
class _Mem:
    def store(self, content, tier, importance):
        self.content = content
        return 7
    def retrieve(self, query, limit=1):
        return []
class _Epi:
    def get_episode_memories(self, episode_id):
        return [{"id": 1, "content": "a", "relevance_to_episode": 0.8, "importance": 0.5}]
events = mod.RealIncrementalLearner(_Epi(), _Mem()).learn_from_episode("e")
assert len(events) == 1 and events[0].promotion is True
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    import core.cognitivo.incremental_learner as incremental
    return incremental


class _Mem:
    def __init__(self):
        self.stores = []

    def store(self, content, tier, importance):
        self.stores.append((content, tier, importance))
        return len(self.stores)

    def retrieve(self, query, limit=1):
        return []


class _Log:
    def log_event(self, kind, payload):
        return None


class _Epi:
    def __init__(self, memories):
        self.memories = memories

    def get_episode_memories(self, episode_id):
        return self.memories


class _Book:
    def __init__(self, episodes):
        self.episodes = episodes

    def search_episodes(self, status="closed", limit=10):
        return list(self.episodes)[:limit]

    def get_episode_memories(self, episode_id):
        for episode in self.episodes:
            if episode.episode_id == episode_id:
                return episode.memories
        return []


class IncrementalLearnerInvalidInputContractTests(unittest.TestCase):
    def test_string_relevance_and_importance_are_not_coerced(self):
        learner = RealIncrementalLearner(_Epi([{
            "id": 1, "content": "b", "relevance_to_episode": "0.9", "importance": 0.5,
        }]), _Mem())
        with self.assertRaises(TypeError):
            learner.learn_from_episode("e")
        promoted = RealIncrementalLearner(_Epi([{
            "id": 1, "content": "b", "relevance_to_episode": 0.9, "importance": "0.5",
        }]), _Mem())
        with self.assertRaises(TypeError):
            promoted.learn_from_episode("e")
        reinforced = RealIncrementalLearner(_Epi([{
            "id": 1, "content": "b", "relevance_to_episode": 0.6, "importance": "0.8",
        }]), _Mem())
        with self.assertRaises(TypeError):
            reinforced.learn_from_episode("e")

    def test_plan_rejects_text_reward_missing_sequence_and_null_goal(self):
        learner = RealIncrementalLearner(_Epi([]), _Mem())
        self.assertIsNone(learner.learn_from_plan(None))
        self.assertIsNone(learner.learn_from_plan(SimpleNamespace(
            expected_reward=0.699, goal="g", best_action_sequence=["a"],
        )))
        with self.assertRaises(TypeError):
            learner.learn_from_plan(SimpleNamespace(
                expected_reward="0.9", goal="g", best_action_sequence=["a"],
            ))
        with self.assertRaises(AttributeError):
            learner.learn_from_plan(SimpleNamespace(expected_reward=0.9, goal="g"))
        with self.assertRaises(TypeError):
            learner.learn_from_plan(SimpleNamespace(
                expected_reward=0.9, goal=None, best_action_sequence=["a"],
            ))

    def test_non_dict_memory_item_raises(self):
        learner = RealIncrementalLearner(_Epi(["nope"]), _Mem())
        with self.assertRaises(AttributeError):
            learner.learn_from_episode("e")

    def test_consolidation_counters_follow_the_promotion_flag(self):
        promoted = RealIncrementalLearner(_Book([SimpleNamespace(episode_id="p", memories=[{
            "id": 1, "content": "a", "relevance_to_episode": 0.8, "importance": 0.2,
        }])]), _Mem(), _Log())
        only_promotion = promoted.consolidate_learning(
            last_n_episodes=5, run_pattern_detection=False, run_decay=False,
        )
        self.assertEqual(only_promotion.memories_promoted, 1)
        self.assertEqual(only_promotion.memories_reinforced, 0)
        reinforced = RealIncrementalLearner(_Book([SimpleNamespace(episode_id="r", memories=[{
            "id": 2, "content": "b", "relevance_to_episode": 0.6, "importance": 0.8,
        }])]), _Mem(), _Log())
        only_reinforcement = reinforced.consolidate_learning(
            last_n_episodes=5, run_pattern_detection=False, run_decay=False,
        )
        self.assertEqual(only_reinforcement.memories_promoted, 0)
        self.assertEqual(only_reinforcement.memories_reinforced, 1)


class IncrementalLearnerExtractionContractTests(unittest.TestCase):
    def test_monolith_and_package_reexport_the_same_objects(self):
        incremental = _module()
        import core.cognitivo as package
        import core.constitutional_brain as brain
        for name in _NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(incremental, name))
                self.assertIs(getattr(package, name), getattr(incremental, name))
                self.assertEqual(getattr(incremental, name).__module__, "core.cognitivo.incremental_learner")

    def test_module_imports_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run(
            [sys.executable, "-c", _BLOCKER],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_pre_registered_checks_match_the_frozen_fingerprint(self):
        _module()
        from tools.cognitivo_eval import IL_BEHAVIOR_SHA, IL_SOURCE_SHA, eval_incremental
        checks = {item["id"]: item for item in eval_incremental()}
        self.assertEqual(set(checks), {"IL1", "IL2", "IL3", "IL4", "IL5", "IL6"})
        for cid, item in checks.items():
            with self.subTest(cid=cid):
                self.assertTrue(item["passed"], item)
        self.assertEqual(checks["IL6"]["value"]["sha_module"], IL_BEHAVIOR_SHA)
        self.assertEqual(checks["IL6"]["value"]["sha_brain"], IL_BEHAVIOR_SHA)
        self.assertEqual(checks["IL6"]["value"]["source_sha"], IL_SOURCE_SHA)
        self.assertEqual(checks["IL6"]["value"]["clock"], "wall")
        self.assertTrue(checks["IL6"]["value"]["identity"])


if __name__ == "__main__":
    unittest.main()
