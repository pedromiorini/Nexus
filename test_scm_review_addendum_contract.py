"""Contrato SC10–SC13 (adendo da revisão do Pearl; pré-registro em
docs/cognitivo/AUDITORIA_2026-10-22.md, seção «adendo da revisão do Pearl»). Escrito antes da correção.

Seeds fixas: random.Random(20261031 + k), k = 0..19. Referência escrita aqui, sem chamar o módulo
para calcular o esperado.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import string
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator

with contextlib.redirect_stdout(io.StringIO()):
    from core.cognitivo.counterfactual import RealCounterfactualReasoningEngine

ROOT = Path(__file__).resolve().parent
SEEDS = [random.Random(20261031 + k) for k in range(20)]
TABLE_ACTIONS = {"APPLY_PATCH", "IGNORE_ALERT", "ROLLBACK", "ESCALATE", "MONITOR", "ISOLATE_SYSTEM",
                 "RESTORE_BACKUP", "DEPLOY_UPDATE", "INCREASE_LOGGING", "SHUTDOWN_SERVICE"}


def _name(rng):
    while True:
        name = "FIX" + "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.randint(4, 8)))
        if name not in TABLE_ACTIONS:
            return name


def _engine(causal):
    with contextlib.redirect_stdout(io.StringIO()):
        return RealCounterfactualReasoningEngine(causal_reasoning=causal, world_model=RealWorldModelSimulator(causal))


class SC10NoLeakInCounterfactualChain(unittest.TestCase):
    def test_non_schema_pairs_never_reach_the_chain(self):
        for k, rng in enumerate(SEEDS):
            with self.subTest(k=k):
                action = _name(rng)
                junk = rng.choice(["foo_bar galaxy", "owner root", "system_status owned", "alert_level very high"])
                causal = RealCausalReasoning()
                causal.extract_causal_relations(f"{action} causes {junk}")
                causal.extract_causal_relations(f"{action} causes alert_level low")
                scenario = _engine(causal).generate_counterfactual_scenario("system vulnerable", action)
                joined = "\n".join(scenario.causal_chain)
                self.assertNotIn("foo_bar", joined)
                self.assertNotIn("owner", joined)
                self.assertNotIn("owned", joined)
                self.assertNotIn("very high", joined)
                self.assertIn(f"do({action}): alert_level=low", scenario.causal_chain)

    def test_table_style_pair_with_schema_value_is_kept(self):
        causal = RealCausalReasoning()
        causal.extract_causal_relations("DEPLOY_FIX causes system_status secure")
        scenario = _engine(causal).generate_counterfactual_scenario("system vulnerable", "DEPLOY_FIX")
        self.assertEqual(scenario.causal_chain[0], "do(DEPLOY_FIX): system_status=secure")


class SC11ClaimsAreMarkedUnverified(unittest.TestCase):
    def test_claim_in_outcome_is_recorded_per_scenario(self):
        causal = RealCausalReasoning()
        causal.extract_causal_relations("DEPLOY_FIX causes system_status secure")
        engine = _engine(causal)
        scenario = engine.generate_counterfactual_scenario("system vulnerable", "DEPLOY_FIX")
        self.assertEqual(
            scenario.counterfactual_outcome,
            "state:alert_level=high,security_patch_applied=False,system_status=secure",
        )
        self.assertEqual(engine.unverified_claims[scenario.scenario_id], ["DEPLOY_FIX: system_status=secure"])
        other = engine.generate_counterfactual_scenario("system vulnerable", "APPLY_PATCH")
        self.assertEqual(engine.unverified_claims.get(other.scenario_id, []), [])
        self.assertNotEqual(other.scenario_id, scenario.scenario_id)

    def test_table_and_declared_actions_carry_no_claim(self):
        causal = RealCausalReasoning()
        causal.declare_intervention("FIXDECL", {"system_status": "secure"})
        engine = _engine(causal)
        for action in ("APPLY_PATCH", "FIXDECL"):
            scenario = engine.generate_counterfactual_scenario("system vulnerable", action)
            self.assertEqual(engine.unverified_claims.get(scenario.scenario_id, []), [], action)


CHILD = r"""
import json, sys, io, contextlib
from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator
spec = json.loads(sys.argv[1])
cr = RealCausalReasoning()
for text in spec["texts"]:
    cr.extract_causal_relations(text)
for key, value in spec.get("declared", {}).items():
    cr.declare_intervention(spec["action"], {key: value})
sim = RealWorldModelSimulator(cr)
result = sim.simulate([spec["action"]])
print(json.dumps({"state": result.final_state.state, "claims": result.claims, "warnings": result.warnings,
                  "success": result.success, "do": sim._do(spec["action"]),
                  "verified_state": getattr(result, "verified_state", None),
                  "claim_items": getattr(result, "claim_items", None)}, sort_keys=False))
"""


def _run_child(spec, seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", CHILD, json.dumps(spec)], capture_output=True, text=True,
                         env=env, cwd=ROOT, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


class SC12ConflictingClaimsAreDeterministic(unittest.TestCase):
    def _spec(self, rng):
        action = _name(rng)
        statuses = rng.sample(["secure", "stable", "isolated", "offline", "compromised"], rng.randint(2, 4))
        texts = [f"{action} causes system_status {s}" for s in statuses]
        texts.append(f"{action} causes alert_level low")
        texts.append(f"{action} causes logging_level verbose")
        return action, statuses, texts

    def test_five_hash_seeds_give_identical_results_and_conflict_is_excluded(self):
        for k in range(4):
            action, statuses, texts = self._spec(SEEDS[k])
            with self.subTest(k=k):
                runs = [_run_child({"action": action, "texts": texts}, seed) for seed in range(5)]
                self.assertTrue(all(run == runs[0] for run in runs), runs)
                run = runs[0]
                self.assertEqual(run["state"]["system_status"], "vulnerable")
                self.assertNotIn("system_status", run["do"])
                self.assertEqual(run["do"], {"alert_level": "low", "logging_level": "verbose"})
                self.assertEqual(run["state"]["alert_level"], "low")
                self.assertFalse(run["success"])
                self.assertFalse([c for c in run["claims"] if "system_status" in c])
                expected = f"conflicting causal claims: {action!r} -> system_status in {', '.join(sorted(statuses))}"
                self.assertIn(expected, run["warnings"])

    def test_same_value_with_different_case_is_not_a_conflict(self):
        action = "FIXCASE"
        run = _run_child({"action": action, "texts": [f"{action} causes system_status Secure",
                                                      f"{action} causes system_status secure"]}, 0)
        self.assertEqual(run["state"]["system_status"], "secure")
        self.assertFalse([w for w in run["warnings"] if "conflicting" in w])

    def test_declared_key_beats_conflicting_text_without_conflict_warning(self):
        action = "FIXDEC"
        run = _run_child({"action": action, "declared": {"system_status": "stable"},
                          "texts": [f"{action} causes system_status secure",
                                    f"{action} causes system_status compromised"]}, 0)
        self.assertEqual(run["state"]["system_status"], "stable")
        self.assertTrue(run["success"])
        self.assertFalse([w for w in run["warnings"] if "conflicting" in w])

    def test_conflicting_key_is_not_a_step_in_the_counterfactual_chain(self):
        causal = RealCausalReasoning()
        for text in ("FIXCONF causes system_status secure", "FIXCONF causes system_status compromised",
                     "FIXCONF causes alert_level low"):
            causal.extract_causal_relations(text)
        scenario = _engine(causal).generate_counterfactual_scenario("system vulnerable", "FIXCONF")
        joined = "\n".join(scenario.causal_chain)
        self.assertNotIn("system_status=", joined)
        self.assertIn("do(FIXCONF): alert_level=low", scenario.causal_chain)


class SC13VerifiedStateIsSeparate(unittest.TestCase):
    def test_forged_claim_stays_out_of_verified_state_and_is_structured(self):
        for k, rng in enumerate(SEEDS):
            with self.subTest(k=k):
                action = _name(rng)
                value = rng.choice(["secure", "stable"])
                causal = RealCausalReasoning()
                causal.extract_causal_relations(f"{action} causes system_status {value}")
                result = RealWorldModelSimulator(causal).simulate([action])
                self.assertEqual(result.final_state.state["system_status"], value)
                self.assertEqual(result.verified_state["system_status"], "vulnerable")
                self.assertEqual(result.claim_items, [{"action": action, "key": "system_status", "value": value,
                                                       "source": "free_text", "confidence": None}])

    def test_table_only_sequences_have_equal_verified_and_final_state(self):
        names = sorted(TABLE_ACTIONS)
        for k, rng in enumerate(SEEDS):
            with self.subTest(k=k):
                seq = [rng.choice(names) for _ in range(rng.randint(1, 4))]
                result = RealWorldModelSimulator().simulate(seq)
                self.assertEqual(result.verified_state, result.final_state.state)
                self.assertEqual(result.claim_items, [])

    def test_benchmark_composition_ignores_forged_claims(self):
        import benchmarks.nexus_capability_benchmark as bench

        def forging_simulator():
            causal = RealCausalReasoning()
            causal.extract_causal_relations("FIXFORGE causes system_status secure")
            return RealWorldModelSimulator(causal)

        case = {"split": "seen_length_1_2", "actions": ["FIXFORGE"], "target_state": {"system_status": "secure"}}
        with mock.patch.object(bench, "RealWorldModelSimulator", forging_simulator):
            result = bench._score_composition([case])
        self.assertEqual(result["seen_length_1_2"]["exact_match_accuracy"], 0.0)

    def test_counterfactual_outcome_is_unchanged_for_the_cf9_edge(self):
        causal = RealCausalReasoning()
        causal.extract_causal_relations("DEPLOY_FIX causes system_status secure")
        scenario = _engine(causal).generate_counterfactual_scenario("system vulnerable", "DEPLOY_FIX")
        self.assertEqual(scenario.counterfactual_outcome,
                         "state:alert_level=high,security_patch_applied=False,system_status=secure")


if __name__ == "__main__":
    unittest.main()
