"""Contrato de extração (strangler) de RealEpisodicMemory para core.cognitivo.episodic_memory.

Teste dos três casos (critérios escritos antes da extração):
(a) caso de origem: a capacidade passa a existir num módulo próprio, importável
    sem numpy, vita ou o monólito, e o monólito reexporta a MESMA classe;
(b) não regressão: o comportamento observável numa fixture fixa (estatísticas,
    busca, ligações) é idêntico ao valor medido na main 0e11093;
(c) caso incompatível: entradas inválidas (limit=-1, limit=True, "3", 1.5)
    continuam rejeitadas antes de qualquer consulta; limit=0 devolve vazio.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.episodic_memory as mod
m = mod.RealEpisodicMemory(":memory:")
assert m.get_statistics()["episodes_created"] == 0
m.close()
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    from core.cognitivo import episodic_memory
    return episodic_memory


class EpisodicMemoryExtractionContractTests(unittest.TestCase):
    def test_monolith_reexports_the_module_classes(self):
        mod = _module()
        import core.constitutional_brain as brain
        self.assertIs(brain.RealEpisodicMemory, mod.RealEpisodicMemory)
        self.assertIs(brain.Episode, mod.Episode)
        self.assertEqual(mod.RealEpisodicMemory.__module__, "core.cognitivo.episodic_memory")

    def test_module_imports_standalone_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", _BLOCKER], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_invalid_limits_are_rejected_and_zero_returns_empty(self):
        mem = _module().RealEpisodicMemory(":memory:")
        self.addCleanup(mem.close)
        with self.assertRaises(ValueError):
            mem.search_episodes(limit=-1)
        for bad in (True, False, "3", 1.5, None):
            with self.subTest(limit=bad), self.assertRaises(TypeError):
                mem.search_episodes(limit=bad)
        self.assertEqual(mem.search_episodes(limit=0), [])

    def test_empty_database_is_not_an_error(self):
        mem = _module().RealEpisodicMemory(":memory:")
        self.addCleanup(mem.close)
        self.assertEqual(mem.get_statistics(),
                         {"episodes_created": 0, "episodes_closed": 0,
                          "total_links": 0, "avg_importance": 0.0})
        self.assertEqual(mem.search_episodes(), [])
        self.assertEqual(mem.search_episodes("x", limit=5), [])
        self.assertEqual(mem.get_episode_memories("missing"), [])

    def test_fixed_fixture_matches_main_baseline(self):
        mem = _module().RealEpisodicMemory(":memory:")
        self.addCleanup(mem.close)
        a = mem.start_episode("alpha", trigger_query="q1", importance=0.9)
        b = mem.start_episode("beta", importance=0.3)
        mem.link_memory_to_episode(a, 1, 0.5)
        mem.link_memory_to_episode(a, 2)
        mem.link_memory_to_episode(b, 3, 0.2)
        mem.close_episode(a)
        # Valores medidos com core.constitutional_brain na main 0e11093.
        self.assertEqual(mem.get_statistics(),
                         {"episodes_created": 2, "episodes_closed": 1,
                          "total_links": 3, "avg_importance": 0.6})
        found = mem.search_episodes(limit=10)
        self.assertEqual([(e.episode_id, e.title, e.trigger_query, e.importance, e.status) for e in found],
                         [(a, "alpha", "q1", 0.9, "closed")])
        self.assertEqual([e.episode_id for e in mem.search_episodes("alp", limit=5)], [a])
        self.assertEqual(mem.get_episode_memories(a), [])
        with self.assertRaises(ValueError):
            mem.link_memory_to_episode("missing", 1)


if __name__ == "__main__":
    unittest.main()
