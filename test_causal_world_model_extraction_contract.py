"""Contrato de extração (strangler) de RealCausalReasoning e RealWorldModelSimulator
para core.cognitivo.causal_world_model. Escrito antes da extração.

(a) caso de origem: as classes (e dataclasses CausalRelation, WorldState, SimulationResult)
    existem num módulo próprio importável sem numpy, vita ou o monólito, e o monólito
    reexporta os MESMOS objetos;
(b) não regressão: saídas de uma sequência fixa têm SHA-256 fixado (golden medido na main);
(c) casos ausentes/inválidos registram o comportamento atual (não endurecido aqui):
    texto vazio -> [], enrich_context([]) -> zeros, compare_futures([]) -> ValueError.
"""

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GOLDEN_SHA256 = "feb0815a29f141a2b8bb8137c4a75f48192e939eaa12c510e64700b000fbbab5"
NAMES = ("CausalRelation", "RealCausalReasoning", "WorldState", "SimulationResult", "RealWorldModelSimulator")

_BLOCKER = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"numpy", "vita"} or name == "core.constitutional_brain":
            raise ImportError("blocked dependency: " + name)
        return None
sys.meta_path.insert(0, _Block())
import core.cognitivo.causal_world_model as mod
assert mod.RealCausalReasoning().extract_causal_relations("rain causes floods")[0].effect == "floods"
assert mod.RealWorldModelSimulator().simulate(["APPLY_PATCH"]).success is True
leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"numpy", "vita"} or n == "core.constitutional_brain")
print("LEAKED=" + ",".join(leaked))
"""


def _module():
    from core.cognitivo import causal_world_model
    return causal_world_model


def golden_payload(mod):
    cr = mod.RealCausalReasoning()
    texts = ["Smoking causes cancer", "rain CAUSE wet roads", "no relation here", "",
             "Heat causes expansion which causes stress"]
    rels = [[dataclasses.asdict(r) for r in cr.extract_causal_relations(t)] for t in texts]
    enriched = cr.enrich_context([{"content": "Bugs cause outages"}, {"content": "nothing"}, {}])
    wm = mod.RealWorldModelSimulator(causal_reasoning=cr)
    sims = []
    for seq in ([], ["APPLY_PATCH"], ["IGNORE_ALERT"], ["APPLY_PATCH", "IGNORE_ALERT"], ["MONITOR"]):
        r = wm.simulate(seq)
        sims.append({"final": r.final_state.state, "action": r.final_state.action_taken,
                     "success": r.success, "warnings": r.warnings})
    cmp_ = wm.compare_futures([["IGNORE_ALERT"], ["APPLY_PATCH"], ["MONITOR"]])
    return {"relations": rels, "enriched": enriched, "stats": cr.get_statistics(), "sims": sims,
            "best": cmp_["best_future_index"], "initial": wm.current_state.state}


def golden_sha(mod):
    return hashlib.sha256(json.dumps(golden_payload(mod), sort_keys=True).encode("utf-8")).hexdigest()


class CausalWorldModelExtractionContractTests(unittest.TestCase):
    def test_monolith_reexports_the_module_classes(self):
        import core.constitutional_brain as brain
        mod = _module()
        for name in NAMES:
            self.assertIs(getattr(brain, name), getattr(mod, name), name)

    def test_module_imports_standalone(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        out = subprocess.run([sys.executable, "-c", _BLOCKER], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "LEAKED=")

    def test_fixed_outputs_are_byte_identical(self):
        self.assertEqual(golden_sha(_module()), GOLDEN_SHA256)

    def test_missing_or_invalid_inputs_keep_current_behaviour(self):
        mod = _module()
        cr = mod.RealCausalReasoning()
        self.assertEqual(cr.extract_causal_relations(""), [])
        self.assertEqual(cr.enrich_context([]), {"causal_relations_found": 0, "relations": []})
        self.assertEqual(cr.get_statistics(), {"total_relations_extracted": 0, "graph_nodes": 0})
        with self.assertRaises(ValueError):
            mod.RealWorldModelSimulator().compare_futures([])

    def test_simulation_does_not_mutate_current_state(self):
        wm = _module().RealWorldModelSimulator()
        before = dict(wm.current_state.state)
        wm.simulate(["IGNORE_ALERT"])
        self.assertEqual(wm.current_state.state, before)


if __name__ == "__main__":
    unittest.main()
