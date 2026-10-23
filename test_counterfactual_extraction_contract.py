"""Contrato de extração (strangler) de RealCounterfactualReasoningEngine
para core.cognitivo.counterfactual. Escrito antes da extração.

(a) identidade: monólito reexporta os MESMOS objetos do módulo novo;
(b) import standalone sem numpy, vita ou o monólito;
(c) golden SHA-256 medido na main (uuid.uuid4 sequencial controlado);
(d) casos ausentes/inválidos: diferença "" vs "" = 0; parse sem "what if" → default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
GOLDEN_SHA256 = "e71725490a4240690606d3191d202188b2b6e5c8b0a0b08db4c68ade4f36d882"
NAMES = (
    "CounterfactualScenario",
    "CausalIntervention",
    "OutcomeComparison",
    "RealCounterfactualReasoningEngine",
)

_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.counterfactual as mod
eng = mod.RealCounterfactualReasoningEngine()
assert eng._predict_outcome("s", None) == "outcome_without_intervention"
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    from core.cognitivo import counterfactual
    return counterfactual


class _SeqUUID:
    def __init__(self):
        self.i = 0

    def __call__(self):
        self.i += 1
        return uuid.UUID(int=self.i)


def golden_payload(mod):
    Eng = mod.RealCounterfactualReasoningEngine
    eng = Eng()
    sc = eng.generate_counterfactual_scenario("system vulnerable", "apply patch")
    inter = eng.model_causal_intervention("alert_level", "high", "low")
    cmp_ = eng.compare_outcomes(sc)
    what = eng.explore_what_if("What if we isolate the host?")

    from core.cognitivo.causal_world_model import RealCausalReasoning

    class WM:
        def predict_outcome(self, situation, intervention):
            return f"wm:{situation}:{intervention}"

    eng2 = Eng(causal_reasoning=RealCausalReasoning(), world_model=WM())
    sc2 = eng2.generate_counterfactual_scenario("alarm", "patch")
    return {
        "sc": dataclasses.asdict(sc),
        "inter": dataclasses.asdict(inter),
        "cmp": dataclasses.asdict(cmp_),
        "what_plaus": what["plausibility"],
        "what_insights": what["insights"],
        "stats": eng.get_statistics(),
        "sc2": dataclasses.asdict(sc2),
        "chain": eng2._build_causal_chain("alarm", "patch"),
        "pred": eng2._predict_outcome("s", "i"),
        "stats2": eng2.get_statistics(),
        "empty_diff": eng._calculate_outcome_difference("", ""),
        "same_diff": eng._calculate_outcome_difference("x", "x"),
        "parse_default": list(eng._parse_what_if_question("no marker here")),
    }


def golden_sha(mod):
    with patch.object(mod.uuid, "uuid4", _SeqUUID()):
        return hashlib.sha256(
            json.dumps(golden_payload(mod), sort_keys=True, default=str).encode()
        ).hexdigest()


class CounterfactualExtractionContractTests(unittest.TestCase):
    def test_monolith_reexports_the_module_classes(self):
        import core.constitutional_brain as brain

        mod = _module()
        for name in NAMES:
            self.assertIs(getattr(brain, name), getattr(mod, name), name)

    def test_module_imports_standalone(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        out = subprocess.run(
            [sys.executable, "-c", _BLOCKER],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "LEAKED=")

    def test_golden_sha_matches_main_measurement(self):
        self.assertEqual(golden_sha(_module()), GOLDEN_SHA256)

    def test_empty_and_missing_cases(self):
        eng = _module().RealCounterfactualReasoningEngine()
        self.assertEqual(eng._calculate_outcome_difference("", ""), 0.0)
        self.assertEqual(eng._parse_what_if_question("no marker here"), ("unknown_situation", "no marker here"))


if __name__ == "__main__":
    unittest.main()
