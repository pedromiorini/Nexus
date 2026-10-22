"""Contrato de extração (strangler) de RealMCTSPlanner para core.cognitivo.mcts_planner.

Teste dos três casos (critérios escritos antes da extração):
(a) caso de origem: o planejador passa a existir num módulo próprio, importável sem numpy,
    vita ou o monólito, e o monólito reexporta as MESMAS classes;
(b) não regressão: com random.seed(20261016), o PlanResult (sem planning_time) tem o mesmo
    SHA-256 medido na main 4d54183;
(c) caso incompatível: entrada ausente (goal=None) levanta erro; casos-limite atuais
    (max_depth<=0, exploration_constant=NaN, max_iterations<=0, goal vazio) ficam
    CARACTERIZADOS como estão hoje -- este PR não muda comportamento; endurecê-los é decisão separada.
"""

import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GOLDEN_SHA256 = "47c6fda003aeeaa15c649a416eea7920d8d4bb6f6b23237cf64bc4524054ea12"
SEED = 20261016
GOAL = "secure the system"
EMPTY_PLAN = {"best_action_sequence": [], "expected_reward": 0.0, "tree_nodes_created": 1,
              "success_probability": 0.0, "alternative_plans": []}

_BLOCKER = r"""
import random, sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.mcts_planner as mod
random.seed(1)
r = mod.RealMCTSPlanner(max_depth=1).plan("secure the system", max_iterations=3)
assert isinstance(r, mod.PlanResult) and r.iterations_run == 3
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    from core.cognitivo import mcts_planner
    return mcts_planner


def _plan(goal=GOAL, iterations=50, **kwargs):
    random.seed(SEED)
    result = _module().RealMCTSPlanner(**kwargs).plan(goal, max_iterations=iterations)
    data = dataclasses.asdict(result)
    data.pop("planning_time")
    return data


class MCTSPlannerExtractionContractTests(unittest.TestCase):
    def test_monolith_reexports_the_module_classes(self):
        mod = _module()
        import core.constitutional_brain as brain
        for name in ("RealMCTSPlanner", "MCTSNode", "PlanResult"):
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(mod, name))
        self.assertEqual(mod.RealMCTSPlanner.__module__, "core.cognitivo.mcts_planner")

    def test_module_imports_standalone_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", _BLOCKER], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_fixed_seed_plan_matches_main(self):
        data = _plan()
        self.assertEqual(data["best_action_sequence"], ["ESCALATE", "IGNORE_ALERT"])
        text = json.dumps(data, sort_keys=True)
        self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(), GOLDEN_SHA256)

    def test_missing_goal_is_rejected(self):
        with self.assertRaises(AttributeError):
            _plan(goal=None)

    def test_characterize_non_positive_depth_returns_empty_plan(self):
        for depth in (0, -1):
            with self.subTest(max_depth=depth):
                data = _plan(max_depth=depth)
                self.assertEqual({k: data[k] for k in EMPTY_PLAN}, EMPTY_PLAN)
                self.assertEqual(data["iterations_run"], 50)

    def test_characterize_nan_exploration_is_accepted(self):
        data = _plan(exploration_constant=float("nan"))
        self.assertEqual(data["best_action_sequence"], ["ESCALATE"])
        self.assertEqual(data["tree_nodes_created"], 44)

    def test_characterize_non_positive_iterations_and_empty_goal(self):
        self.assertEqual(_plan(iterations=0)["iterations_run"], 0)
        self.assertEqual(_plan(iterations=-1)["iterations_run"], -1)
        self.assertEqual(_plan(iterations=-1)["best_action_sequence"], [])
        self.assertEqual(_plan(goal="")["best_action_sequence"], ["ESCALATE", "IGNORE_ALERT"])


if __name__ == "__main__":
    unittest.main()
