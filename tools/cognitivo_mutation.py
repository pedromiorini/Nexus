"""Mutation testing direcionado para capacidades extraídas em core/cognitivo/.

Catálogo separado de tools/targeted_mutation.py (arquivo no escopo do Axis e
alterado por PRs abertas). Reutiliza a mesma maquinaria (cópia temporária,
árvore de trabalho intocada) e grava cognitivo_mutation_report.json.
Exit 0 só se todo mutante declarado for aplicável e morto.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.targeted_mutation import COPY_IGNORE, Mutation, run_tests  # noqa: E402

REPORT = ROOT / "cognitivo_mutation_report.json"
EPISODIC = "core/cognitivo/episodic_memory.py"
EXTRACTION = ("test_episodic_memory_extraction_contract.py",)
LIMITS = ("test_episodic_memory_search_limits_contract.py", "test_episodic_memory_extraction_contract.py")
STATS = ("test_episodic_memory_statistics_contract.py", "test_episodic_memory_extraction_contract.py")
PREDICTIVE = "core/cognitivo/predictive_coding.py"
POMDP = ("test_predictive_coding_extraction_contract.py", "test_pomdp_belief_filter_contract.py")
PC_EXTRACTION = ("test_predictive_coding_extraction_contract.py",)
MCTS = "core/cognitivo/mcts_planner.py"
MCTS_INPUT = ("test_mcts_planner_input_contract.py", "test_mcts_planner_extraction_contract.py")
MCTS_TESTS = ("test_mcts_planner_extraction_contract.py", "test_mcts_action_contract.py")
_GUARD = '"""Buscar episódios"""\n        if isinstance(limit, bool) or not isinstance(limit, int):\n            raise TypeError("limit must be an integer")'

CAUSAL = "core/cognitivo/causal_world_model.py"
COUNTERFACTUAL = "core/cognitivo/counterfactual.py"
EFFECTS = "core/cognitivo/action_effects.py"
R1_TESTS = ("test_action_effects_contract.py",)
R2_TESTS = ("test_futures_ranking_contract.py",)
R3_TESTS = ("test_causal_extraction_contract.py",)
R4_TESTS = ("test_scm_intervention_contract.py",)
CF_TESTS = ("test_counterfactual_extraction_contract.py", "test_counterfactual_contract.py")
CAUSAL_TESTS = ("test_causal_world_model_extraction_contract.py",)

MCTS_QUALITY = ("test_mcts_planner_quality_contract.py",)

PC_LEARNING = ("test_predictive_learning_contract.py",)

RETRIEVAL = ("test_episodic_retrieval_contract.py",)

MUTATIONS = [
    Mutation("episodic_fts_ranks_by_bm25", EPISODIC, "ORDER BY bm25(episodic_fts), m.start_timestamp DESC",
             "ORDER BY m.start_timestamp DESC", tests=RETRIEVAL),
    Mutation("episodic_fts_backfills_legacy_rows", EPISODIC, "WHERE episode_id NOT IN (SELECT episode_id FROM episodic_fts)",
             "WHERE 0", tests=RETRIEVAL),
    Mutation("episodic_fts_indexes_new_episode", EPISODIC, "            if self._fts_enabled:\n                cursor.execute(",
             "            if False:\n                cursor.execute(", tests=RETRIEVAL),
    Mutation("episodic_like_fallback_keeps_substrings", EPISODIC, "        if len(rows) < limit:", "        if False:",
             tests=RETRIEVAL),
    Mutation("predictive_level0_uses_learned_argmax", PREDICTIVE,
             "max(self._sensory_distribution.items(), key=lambda item: item[1])",
             "min(self._sensory_distribution.items(), key=lambda item: item[1])", tests=PC_LEARNING),
    Mutation("predictive_cycle_learns_observation", PREDICTIVE,
             '        self._learn_sensory(observation.get("sensory", "input"))\n', "", tests=PC_LEARNING),
    Mutation("predictive_warmup_sample_mean", PREDICTIVE,
             "alpha = max(self.learning_rates[0], 1.0 / self._sensory_observations)",
             "alpha = self.learning_rates[0]", tests=PC_LEARNING),
    Mutation("predictive_decays_previous_mass", PREDICTIVE,
             "self._sensory_distribution[symbol] *= 1.0 - alpha", "pass", tests=PC_LEARNING),
    Mutation("mcts_copies_caller_action_list", MCTS,
             "untried_actions=list(allowed) if allowed is not None else (",
             "untried_actions=allowed if allowed is not None else (", tests=MCTS_QUALITY),
    Mutation("mcts_available_actions_every_depth", MCTS,
             "all_actions = (list(self._plan_actions) if self._plan_actions is not None",
             "all_actions = (list(self._plan_actions) if False", tests=MCTS_QUALITY),
    Mutation("mcts_prefix_competes_for_best_plan", MCTS,
             "        if depth > 0:  # prefixo", "        if False:  # prefixo", tests=MCTS_QUALITY),
    Mutation("mcts_returns_best_found_sequence", MCTS,
             "best_sequence = (self._best_found[1][prefix:] if self._best_found is not None",
             "best_sequence = (self._best_found[1][prefix:] if False", tests=MCTS_QUALITY),
    Mutation("mcts_keeps_strictly_better_found", MCTS,
             "reward > 0 and (self._best_found is None or reward > self._best_found[0])",
             "reward > 0", tests=MCTS_QUALITY),
    Mutation("episodic_search_rejects_boolean_and_non_integer_limit", EPISODIC, _GUARD,
             '"""Buscar episódios"""\n        if False:\n            raise TypeError("limit must be an integer")', tests=LIMITS),
    Mutation("episodic_search_rejects_negative_limit", EPISODIC,
             'if limit < 0:\n            raise ValueError("limit must be non-negative")',
             'if False:\n            raise ValueError("limit must be non-negative")', tests=LIMITS),
    Mutation("episodic_search_zero_limit_skips_database", EPISODIC,
             "if limit == 0:\n            return []", "if False:\n            return []", tests=LIMITS),
    Mutation("episodic_close_counter", EPISODIC,
             "self.episodes_closed += 1", "self.episodes_closed += 2", tests=STATS),
    Mutation("episodic_close_status", EPISODIC,
             "SET end_timestamp = ?, status = 'closed'", "SET end_timestamp = ?, status = 'open'", tests=STATS),
    Mutation("episodic_avg_importance", EPISODIC,
             'cursor.execute("SELECT AVG(importance) FROM episodic_memories")',
             'cursor.execute("SELECT MAX(importance) FROM episodic_memories")', tests=STATS),
    Mutation("episodic_module_pulls_monolith", EPISODIC,
             "import sqlite3\n", "import sqlite3\nimport core.constitutional_brain  # noqa: F401\n", tests=EXTRACTION),
    Mutation("monolith_stops_reexporting", "core/constitutional_brain.py",
             "from core.cognitivo.episodic_memory import Episode, RealEpisodicMemory",
             "from core.cognitivo.episodic_memory import Episode as _E, RealEpisodicMemory as _R\n"
             "class RealEpisodicMemory(_R):\n    pass\nEpisode = _E",
             tests=EXTRACTION),
    Mutation("predictive_empty_states_accepted", PREDICTIVE,
             "if not is_sequence(states) or len(states) == 0:",
             "if not is_sequence(states):", tests=POMDP),
    Mutation("predictive_prior_normalization", PREDICTIVE,
             'if abs(math.fsum(prior_values) - 1.0) > tolerance:\n            raise ValueError("prior must be normalized to 1")',
             'if False:\n            raise ValueError("prior must be normalized to 1")', tests=POMDP),
    Mutation("predictive_zero_evidence_accepted", PREDICTIVE,
             "if not math.isfinite(log_evidence):", "if False:", tests=POMDP),
    Mutation("predictive_posterior_normalization", PREDICTIVE,
             "math.exp(log_weight - log_evidence)", "math.exp(log_weight)", tests=POMDP),
    Mutation("predictive_likelihood_length", PREDICTIVE,
             "if not is_sequence(observation_likelihoods) or len(observation_likelihoods) != state_count:",
             "if not is_sequence(observation_likelihoods):", tests=POMDP),
    Mutation("predictive_module_pulls_monolith", PREDICTIVE,
             "import math\n", "import math\nimport core.constitutional_brain  # noqa: F401\n", tests=PC_EXTRACTION),
    Mutation("monolith_stops_reexporting_predictive", "core/constitutional_brain.py",
             "from core.cognitivo.predictive_coding import Prediction, PredictionError, RealPredictiveCodingSystem",
             "from core.cognitivo.predictive_coding import Prediction, PredictionError, RealPredictiveCodingSystem as _R\n"
             "class RealPredictiveCodingSystem(_R):\n    pass",
             tests=PC_EXTRACTION),
    Mutation("mcts_exploration_term_dropped", MCTS,
             "exploration = exploration_constant * math.sqrt(", "exploration = 0.0 * math.sqrt(", tests=MCTS_TESTS),
    Mutation("mcts_backprop_skips_reward", MCTS,
             "current.total_reward += reward", "current.total_reward += 0.0", tests=MCTS_TESTS),
    Mutation("mcts_depth_limit_off_by_one", MCTS,
             "if depth >= self.max_depth:", "if depth > self.max_depth:",
             tests=MCTS_TESTS + MCTS_QUALITY),
    Mutation("mcts_best_child_by_min_reward", MCTS,
             "best_child = max(root.children, key=lambda c: c.avg_reward)",
             "best_child = min(root.children, key=lambda c: c.avg_reward)", tests=MCTS_TESTS),
    Mutation("mcts_module_pulls_monolith", MCTS,
             "import copy\n", "import copy\nimport core.constitutional_brain  # noqa: F401\n",
             tests=("test_mcts_planner_extraction_contract.py",)),
    Mutation("mcts_guard_exploration_type", MCTS,
             'if isinstance(exploration_constant, bool) or not isinstance(exploration_constant, Real):',
             'if False:', tests=MCTS_INPUT),
    Mutation("mcts_guard_exploration_bool", MCTS,
             'if isinstance(exploration_constant, bool) or not',
             'if not', tests=MCTS_INPUT),
    Mutation("mcts_guard_exploration_finite", MCTS,
             'if not math.isfinite(exploration_constant):',
             'if False:', tests=MCTS_INPUT),
    Mutation("mcts_guard_depth_type", MCTS,
             'if isinstance(max_depth, bool) or not isinstance(max_depth, int):',
             'if False:', tests=MCTS_INPUT),
    Mutation("mcts_guard_depth_bool", MCTS,
             'if isinstance(max_depth, bool) or not isinstance(max_depth, int):',
             'if not isinstance(max_depth, int):', tests=MCTS_INPUT),
    Mutation("mcts_guard_depth_positive", MCTS,
             'if max_depth <= 0:',
             'if max_depth < 0:', tests=MCTS_INPUT),
    Mutation("mcts_guard_goal_none", MCTS,
             'if goal is None:\n            raise ValueError',
             'if False:\n            raise ValueError', tests=MCTS_INPUT),
    Mutation("mcts_guard_goal_type", MCTS,
             'if not isinstance(goal, str):',
             'if False:', tests=MCTS_INPUT),
    Mutation("mcts_guard_iterations_type", MCTS,
             'if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):',
             'if False:', tests=MCTS_INPUT),
    Mutation("mcts_guard_iterations_bool", MCTS,
             'if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):',
             'if not isinstance(max_iterations, int):', tests=MCTS_INPUT),
    Mutation("mcts_guard_iterations_negative", MCTS,
             'if max_iterations < 0:',
             'if max_iterations < -1:', tests=MCTS_INPUT),
    Mutation("mcts_guard_iterations_zero_allowed", MCTS,
             'if max_iterations < 0:',
             'if max_iterations <= 0:', tests=MCTS_INPUT),
    Mutation("mcts_guard_exploration_negative", MCTS,
             'if exploration_constant < 0:', 'if exploration_constant < -1:', tests=MCTS_INPUT),
    Mutation("mcts_guard_exploration_zero_allowed", MCTS,
             'if exploration_constant < 0:', 'if exploration_constant <= 0:', tests=MCTS_INPUT),
    Mutation("causal_lazy_cause_group", CAUSAL, r"r'(.+?)\s+causes?\s+(.+)'", r"r'(.+)\s+causes?\s+(.+)'",
             tests=CAUSAL_TESTS),
    Mutation("causal_case_sensitive", CAUSAL, "_F = re.IGNORECASE", "_F = 0", tests=CAUSAL_TESTS),
    Mutation("causal_confidence", CAUSAL, '"direct_cause", 0.9', '"direct_cause", 0.8', tests=CAUSAL_TESTS),
    Mutation("causal_counter_not_incremented", CAUSAL, "self.relations_extracted += 1", "pass",
             tests=CAUSAL_TESTS),
    Mutation("causal_graph_not_updated", CAUSAL, 'self.causal_graph["adjacency"][cause].add(effect)', "pass",
             tests=CAUSAL_TESTS),
    Mutation("world_patch_not_secure", EFFECTS, '"low", "system_status": "secure"}},\n    "IGNORE',
             '"low", "system_status": "stable_"}},\n    "IGNORE', tests=CAUSAL_TESTS),
    Mutation("world_ignore_alert_noop", EFFECTS, '"world": {"alert_level": "critical", "system_status": "compromised"}',
             '"world": {}', tests=CAUSAL_TESTS),
    Mutation("r1_unknown_action_silent", CAUSAL, 'warnings.append(f"unknown action: {action!r}")', "pass",
             tests=R1_TESTS),
    Mutation("r1_unknown_action_applies_nothing_but_breaks", CAUSAL,
             "            if not applied:\n                warnings.append(f\"unknown action: {action!r}\")\n",
             "            if not applied:\n                warnings.append(f\"unknown action: {action!r}\")\n                break\n",
             tests=R1_TESTS),
    Mutation("r1_planner_shares_mutable_table", EFFECTS, "dict(entry[\"metrics\"])", "entry[\"metrics\"]",
             tests=R1_TESTS),
    Mutation("r1_rollback_noop", EFFECTS,
             '"world": {"security_patch_applied": False, "alert_level": "medium", "system_status": "stable"}',
             '"world": {}', tests=R1_TESTS),
    Mutation("r1_delta_changed", EFFECTS, '"SHUTDOWN_SERVICE": {"metrics": {"security": +0.8',
             '"SHUTDOWN_SERVICE": {"metrics": {"security": +0.7', tests=R1_TESTS),
    Mutation("world_shallow_copy_mutates_state", CAUSAL, "copy.deepcopy(self.current_state)",
             "self.current_state", tests=CAUSAL_TESTS),
    Mutation("world_compare_picks_min", CAUSAL, "best_idx = max(", "best_idx = min(", tests=CAUSAL_TESTS),
    Mutation("r2_secure_equals_stable", CAUSAL, '{"secure": 3, "stable": 2,', '{"secure": 2, "stable": 2,',
             tests=R2_TESTS),
    Mutation("r2_vulnerable_equals_compromised", CAUSAL, '"vulnerable": 1,\n', '"vulnerable": 0,\n', tests=R2_TESTS),
    Mutation("r2_unknown_status_ranks_high", CAUSAL, '"system_status"), -1)', '"system_status"), 9)',
             tests=R2_TESTS),
    Mutation("r3_negation_ignored", CAUSAL, "            if not first or not second or _negated(text[:match.start(2)], first, kind):",
             "            if not first or not second:", tests=R3_TESTS),
    Mutation("r3_negation_window_empty", CAUSAL, "window = words[-6:-1] if", "window = words[-1:-1] if",
             tests=R3_TESTS),
    Mutation("r3_nt_contraction_missed", CAUSAL, ' or w.endswith("n\'t")', "", tests=R3_TESTS),
    Mutation("r3_noun_cause_allowed", CAUSAL, 'return bool(last) and last[-1] in _DETERMINERS', "return False",
             tests=R3_TESTS),
    Mutation("r3_because_forward", CAUSAL, 'because\\s+(.+)", _F), "backward_cause", 0.8, True)',
             'because\\s+(.+)", _F), "backward_cause", 0.8, False)', tests=R3_TESTS),
    Mutation("r3_because_of_after_because", CAUSAL,
             '    (re.compile(r"(.+?)\\s+because\\s+of\\s+(.+)", _F), "backward_cause", 0.8, True),\n',
             "", tests=R3_TESTS),
    Mutation("r3_multiple_relations", CAUSAL, "            self.causal_graph[\"adjacency\"][cause].add(effect)\n            break\n",
             "            self.causal_graph[\"adjacency\"][cause].add(effect)\n", tests=R3_TESTS + CAUSAL_TESTS),
    Mutation("r3_negation_continues_search", CAUSAL, "_negated(text[:match.start(2)], first, kind):\n                break\n",
             "_negated(text[:match.start(2)], first, kind):\n                continue\n", tests=R3_TESTS),
    Mutation("c4_do_ignored", CAUSAL, "            intervened = self._do(action)\n            if intervened:\n                state.state.update(intervened)\n                applied = True\n",
             "", tests=R4_TESTS),
    Mutation("c4_do_skips_split", CAUSAL, "            if len(parts) == 2:\n                effects[parts[0]] = parts[1]",
             "            if False:\n                effects[parts[0]] = parts[1]", tests=R4_TESTS),
    Mutation("c4_table_skipped_when_do", CAUSAL, "            if entry is not None:\n                state.state.update(entry[\"world\"])\n                applied = True\n",
             "            if entry is not None and not self._do(action):\n                state.state.update(entry[\"world\"])\n                applied = True\n",
             tests=R4_TESTS + R1_TESTS),
    Mutation("c4_warns_even_when_do", CAUSAL, "            if not applied:\n                warnings.append(f\"unknown action: {action!r}\")",
             "            if entry is None:\n                warnings.append(f\"unknown action: {action!r}\")",
             tests=R4_TESTS),
    Mutation("c4_adjacency_wrong_key", CAUSAL, "        for effect in adjacency.get(action, ()):",
             "        for effect in adjacency.get(action.lower(), ()):", tests=R4_TESTS),
    Mutation("cf_module_pulls_monolith", COUNTERFACTUAL,
             "from typing import Any, Dict, List, Optional",
             "from typing import Any, Dict, List, Optional\nfrom core.constitutional_brain import RealCausalReasoning  # pull",
             tests=CF_TESTS),
    Mutation("cf_monolith_stops_reexporting", "core/constitutional_brain.py",
             "from core.cognitivo.counterfactual import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    CausalIntervention, CounterfactualScenario, OutcomeComparison, RealCounterfactualReasoningEngine)\n",
             "CausalIntervention = None  # broken reexport\nCounterfactualScenario = None\n"
             "OutcomeComparison = None\nRealCounterfactualReasoningEngine = None\n",
             tests=CF_TESTS),
    Mutation("cf_predict_skips_world_model", COUNTERFACTUAL,
             "        predictor = getattr(self.world_model, \"predict_outcome\", None)\n        if callable(predictor):\n            return str(predictor(situation, intervention))",
             "        predictor = None\n        if False:\n            return str(predictor(situation, intervention))",
             tests=CF_TESTS),
    Mutation("cf_causal_chain_skips_extractor", COUNTERFACTUAL,
             "        extractor = getattr(self.causal, \"extract_causal_relations\", None)\n        if callable(extractor):",
             "        extractor = None\n        if False:",
             tests=CF_TESTS),
    Mutation("cf_plausibility_ignores_chain_len", COUNTERFACTUAL,
             "        chain_penalty = len(causal_chain) * 0.05",
             "        chain_penalty = 0.0",
             tests=("test_counterfactual_extraction_contract.py",)),
    Mutation("cf_compare_skips_insight_extend", COUNTERFACTUAL,
             "        self.insights.extend(insights)",
             "        pass",
             tests=("test_counterfactual_extraction_contract.py",)),
    Mutation("cf_what_if_skips_compare", COUNTERFACTUAL,
             "        comparison = self.compare_outcomes(scenario)",
             "        comparison = scenario  # wrong type/shape",
             tests=("test_counterfactual_extraction_contract.py",)),
    Mutation("cf_parse_what_if_always_default", COUNTERFACTUAL,
             "        if \"what if\" in question_lower:",
             "        if False:",
             tests=("test_counterfactual_extraction_contract.py",)),
    Mutation("cf_stats_scenarios_not_counted", COUNTERFACTUAL,
             "        self.total_scenarios_generated += 1",
             "        pass",
             tests=("test_counterfactual_extraction_contract.py",)),
]


def main() -> int:
    results = []
    with tempfile.TemporaryDirectory(prefix="nexus-cognitivo-mutation-") as scratch:
        workdir = Path(scratch) / "repo"
        shutil.copytree(ROOT, workdir, ignore=COPY_IGNORE)
        for tests in dict.fromkeys(m.tests for m in MUTATIONS):
            code, output = run_tests(workdir, tests)
            if code != 0:
                raise SystemExit(f"Baseline failed before mutation ({', '.join(tests)}):\n{output}")
        for m in MUTATIONS:
            target = workdir / m.target
            original = target.read_text(encoding="utf-8")
            matches = original.count(m.needle)
            if matches != 1:
                results.append({"name": m.name, "target": m.target, "status": "not_applied", "matches": matches})
                continue
            target.write_text(original.replace(m.needle, m.replacement, 1), encoding="utf-8")
            started = time.time()
            try:
                code, output = run_tests(workdir, m.tests)
            finally:
                target.write_text(original, encoding="utf-8")
            results.append({"name": m.name, "target": m.target, "tests": list(m.tests),
                            "status": "killed" if code != 0 else "survived", "exit_code": code,
                            "duration_seconds": round(time.time() - started, 3), "output_tail": output})
    killed = sum(r["status"] == "killed" for r in results)
    summary = {"declared": len(MUTATIONS), "killed": killed,
               "survived": sum(r["status"] == "survived" for r in results),
               "not_applied": sum(r["status"] == "not_applied" for r in results)}
    REPORT.write_text(json.dumps({"tool": "nexus-cognitivo-mutation", "mutations": results, "summary": summary},
                                 indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    for r in results:
        if r["status"] != "killed":
            print(f"  {r['status']}: {r['name']} ({r['target']})", file=sys.stderr)
    return 0 if killed == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
