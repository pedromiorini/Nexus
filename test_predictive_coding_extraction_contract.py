"""Contrato de extração (strangler) de RealPredictiveCodingSystem para core.cognitivo.predictive_coding.

Teste dos três casos (critérios escritos antes da extração):
(a) caso de origem: a capacidade passa a existir num módulo próprio, importável
    sem numpy, vita ou o monólito, e o monólito reexporta as MESMAS classes;
(b) não regressão: o JSON do benchmark POMDP v2 (seed 20261016), com os campos
    de ambiente normalizados, tem o mesmo SHA-256 medido na main 425f1f0, e os
    4 controles inválidos continuam rejeitados;
(c) caso incompatível: entrada ausente ou vazia em update_belief_state levanta
    erro explícito, nunca devolve sucesso.
"""

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
GOLDEN_SHA256 = "6ba403bbc8070eb3d67d6091f74af73de85fb97f26c2ad3585f17a310238fb41"

_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.predictive_coding as mod
pc = mod.RealPredictiveCodingSystem()
post = pc.update_belief_state(["a", "b"], [0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
assert post == {"a": 1.0, "b": 0.0}, post
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    from core.cognitivo import predictive_coding
    return predictive_coding


class PredictiveCodingExtractionContractTests(unittest.TestCase):
    def test_monolith_reexports_the_module_classes(self):
        mod = _module()
        import core.constitutional_brain as brain
        for name in ("RealPredictiveCodingSystem", "Prediction", "PredictionError"):
            with self.subTest(name=name):
                self.assertIs(getattr(brain, name), getattr(mod, name))
        self.assertEqual(mod.RealPredictiveCodingSystem.__module__, "core.cognitivo.predictive_coding")

    def test_module_imports_standalone_without_numpy_vita_or_monolith(self):
        _module()
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", _BLOCKER], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LEAKED=\n", proc.stdout)

    def test_pomdp_v2_output_matches_main_and_controls_rejected(self):
        _module()
        from benchmarks.nexus_pomdp_benchmark_v2 import run_benchmark
        env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_SHA", "CI_COMMIT_SHA")}
        with mock.patch.dict(os.environ, env, clear=True):
            report = run_benchmark(20261016)
        controls = report["tasks"]["belief_state_update"]["rejection_controls"]
        self.assertEqual(controls, {"cases": 4, "rejected": 4, "accepted": 0})
        report["metadata"]["python_version"] = "X"
        report["metadata"]["source_revision"] = "X"
        text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(), GOLDEN_SHA256)

    def test_missing_or_empty_input_is_rejected(self):
        pc = _module().RealPredictiveCodingSystem()
        identity = [[1.0, 0.0], [0.0, 1.0]]
        cases = {
            "empty_states": ([], [], [], []),
            "none_states": (None, [1.0], identity, [1.0]),
            "none_prior": (["a", "b"], None, identity, [1.0, 0.0]),
            "empty_transition": (["a", "b"], [0.5, 0.5], [], [1.0, 0.0]),
            "empty_likelihoods": (["a", "b"], [0.5, 0.5], identity, []),
        }
        for name, args in cases.items():
            with self.subTest(case=name), self.assertRaises((ValueError, TypeError)):
                pc.update_belief_state(*args)
        with self.assertRaisesRegex(ValueError, "states must be a non-empty"):
            pc.update_belief_state([], [], [], [])


if __name__ == "__main__":
    unittest.main()
