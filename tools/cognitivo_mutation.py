"""Mutation testing direcionado para capacidades extraídas em core/cognitivo/.

Catálogo separado de tools/targeted_mutation.py (arquivo no escopo do Axis e
alterado por PRs abertas). Reutiliza a mesma maquinaria (cópia temporária,
árvore de trabalho intocada) e grava cognitivo_mutation_report.json.
Exit 0 só se todo mutante declarado for aplicável e morto.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import queue
import re
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
LIKE_LITERAL = ("test_episodic_like_literal_contract.py",)
STATS = ("test_episodic_memory_statistics_contract.py", "test_episodic_memory_extraction_contract.py")
PREDICTIVE = "core/cognitivo/predictive_coding.py"
POMDP = ("test_predictive_coding_extraction_contract.py", "test_pomdp_belief_filter_contract.py")
PC_EXTRACTION = ("test_predictive_coding_extraction_contract.py",)
MCTS = "core/cognitivo/mcts_planner.py"
MCTS_INPUT = ("test_mcts_planner_input_contract.py", "test_mcts_planner_extraction_contract.py")
MCTS_TESTS = ("test_mcts_planner_extraction_contract.py", "test_mcts_action_contract.py")
_GUARD = '"""Buscar episódios"""\n        if isinstance(limit, bool) or not isinstance(limit, int):\n            raise TypeError("limit must be an integer")'

CAUSAL = "core/cognitivo/causal_world_model.py"
BRAIN = "core/constitutional_brain.py"
RT_TESTS = ("test_router_fail_closed_contract.py", "test_central_router_contract.py")
BS_TESTS = ("test_router_budget_swarm_contract.py", "test_router_fail_closed_contract.py", "test_router_review_addendum_contract.py", "test_central_router_contract.py")
RV_TESTS = ("test_router_review_addendum_contract.py", "test_router_fail_closed_contract.py", "test_central_router_contract.py")
COUNTERFACTUAL = "core/cognitivo/counterfactual.py"
EFFECTS = "core/cognitivo/action_effects.py"
R1_TESTS = ("test_action_effects_contract.py",)
R2_TESTS = ("test_futures_ranking_contract.py",)
R3_TESTS = ("test_causal_extraction_contract.py",)
R4_TESTS = ("test_scm_intervention_contract.py", "test_scm_schema_contract.py")
SR_TESTS = ("test_scm_review_addendum_contract.py",)
SC_TESTS = ("test_scm_schema_contract.py", "test_scm_intervention_contract.py")
SC3R_TESTS = ("test_scm_outside_table_contract.py", "test_scm_schema_contract.py")
CF_TESTS = ("test_counterfactual_extraction_contract.py", "test_counterfactual_contract.py",
            "test_counterfactual_gap3_contract.py",
            "test_counterfactual_gap4_contract.py",
            "test_counterfactual_planner_contract.py",
            "test_counterfactual_dox_contract.py")
CAUSAL_TESTS = ("test_causal_world_model_extraction_contract.py",)

MCTS_QUALITY = ("test_mcts_planner_quality_contract.py",)
MW_TESTS = ("test_mcts_world_effects_contract.py",)
IJ_TESTS = ("test_mcts_prompt_injection_contract.py",)
ER_TESTS = ("test_mcts_expected_reward_contract.py",)

PC_LEARNING = ("test_predictive_learning_contract.py",)

RETRIEVAL = ("test_episodic_retrieval_contract.py",)
WM = "core/cognitivo/working_memory.py"
WM_TESTS = ("test_working_memory_contract.py",)
ATT = "core/cognitivo/attention.py"
AT_TESTS = ("test_attention_contract.py",)
IL = "core/cognitivo/incremental_learner.py"
IL_TESTS = ("test_incremental_learner_contract.py",)
EF = "core/cognitivo/episodic_future_thinking.py"
EF_TESTS = ("test_episodic_future_thinking_contract.py",)
MS = "core/cognitivo/mental_simulation.py"
MS_TESTS = ("test_mental_simulation_contract.py", "test_mental_simulation_goal_credit.py")
AR = "core/cognitivo/analogical_reasoning.py"
AR_TESTS = ("test_analogical_reasoning_contract.py",)
CM = "core/cognitivo/contextual_memory.py"
CM_TESTS = ("test_contextual_memory_contract.py",)
EW = "core/cognitivo/emotional_memory_weighting.py"
EW_TESTS = ("test_emotional_memory_weighting_contract.py",)
SN = "core/cognitivo/semantic_networks.py"
SN_TESTS = ("test_semantic_networks_contract.py",)

MUTATIONS = [
    Mutation("episodic_fts_ranks_by_bm25", EPISODIC, "ORDER BY bm25(episodic_fts), m.start_timestamp DESC",
             "ORDER BY m.start_timestamp DESC", tests=RETRIEVAL),
    Mutation("episodic_fts_backfills_legacy_rows", EPISODIC, "WHERE episode_id NOT IN (SELECT episode_id FROM episodic_fts)",
             "WHERE 0", tests=RETRIEVAL),
    Mutation("episodic_fts_indexes_new_episode", EPISODIC, "            if self._fts_enabled:\n                cursor.execute(",
             "            if False:\n                cursor.execute(", tests=RETRIEVAL),
    Mutation("episodic_like_fallback_keeps_substrings", EPISODIC, "        if len(rows) < limit:", "        if False:",
             tests=RETRIEVAL),
    Mutation("episodic_search_rejects_non_string_query", EPISODIC,
             'if query is not None and not isinstance(query, str):\n            raise TypeError("query must be a string")',
             'if False:\n            raise TypeError("query must be a string")', tests=LIKE_LITERAL),
    Mutation("episodic_search_rejects_empty_query", EPISODIC,
             'if query == "":\n            raise ValueError("query must be non-empty")',
             'if False:\n            raise ValueError("query must be non-empty")', tests=LIKE_LITERAL),
    Mutation("episodic_search_validates_query_before_zero_limit", EPISODIC,
             'if query == "":\n            raise ValueError("query must be non-empty")',
             'if query == "" and limit != 0:\n            raise ValueError("query must be non-empty")',
             tests=LIKE_LITERAL),
    Mutation("episodic_search_backslash_query_skips_fts", EPISODIC,
             'terms = [] if "\\\\" in query else re.findall(r"\\w+", query)',
             'terms = re.findall(r"\\w+", query)', tests=LIKE_LITERAL),
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
             "if self._best_found is None or reward > self._best_found[0]:",
             "if True:", tests=MCTS_QUALITY),
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
    Mutation("mw_world_evaluation_off", MCTS, "return self._evaluate_world(state)",
             "pass", tests=MW_TESTS),
    Mutation("mw_scale_wrong", MCTS, "utility / 3.0", "utility / 4.0", tests=MW_TESTS),
    Mutation("mw_unknown_status_scores_high", MCTS,
             'STATUS_UTILITY.get(simulated.verified_status, 0)',
             'STATUS_UTILITY.get(simulated.verified_status, 3)', tests=MW_TESTS),
    Mutation("mw_reads_wrong_world_key", MCTS, "simulated.verified_status", 'simulated.verified_state.get("alert_level")',
             tests=MW_TESTS),
    Mutation("mw_cache_never_reset", MCTS,
             "            self._best_found = None\n            self._world_rewards = {}",
             "            self._best_found = None\n            pass", tests=MW_TESTS),
    Mutation("mw_cache_key_by_length", MCTS, 'steps = tuple(state.get("steps_taken", []))',
             'steps = (len(state.get("steps_taken", [])),)', tests=MW_TESTS),
    Mutation("mw_tie_prefers_longer", MCTS,
             'if len(state.get("steps_taken", [])) < len(self._best_found[1]):',
             'if len(state.get("steps_taken", [])) > len(self._best_found[1]):', tests=MW_TESTS),
    Mutation("mw_tie_applies_without_model", MCTS,
             "reward == self._best_found[0] and self._uses_world_effects():",
             "reward == self._best_found[0]:", tests=MW_TESTS + MCTS_TESTS + MCTS_QUALITY + MCTS_INPUT),
    Mutation("mw_unmodeled_ignored", MCTS, "if any(self._is_unmodeled(a) for a in best_sequence):",
             "if False:", tests=MW_TESTS),
    Mutation("mw_unmodeled_requires_all", MCTS, "if any(self._is_unmodeled(a) for a in best_sequence):",
             "if all(self._is_unmodeled(a) for a in best_sequence):", tests=MW_TESTS),
    Mutation("mw_unmodeled_ignores_warnings", MCTS,
             'return bool(getattr(self.world_model.simulate([action]), "warnings", []))',
             "return False", tests=MW_TESTS),
    Mutation("mw_unmodeled_ignores_table", MCTS, "return action not in self._action_effects",
             "return False", tests=MW_TESTS),
    Mutation("mw_world_forced_without_simulate", MCTS,
             'callable(getattr(self.world_model, "simulate", None))', "True", tests=MW_TESTS),
    Mutation("er_reward_from_best_child", MCTS,
             "        if self._uses_world_effects():\n            # Modo mundo: a recompensa da sequência",
             "        if False:\n            # Modo mundo: a recompensa da sequência", tests=ER_TESTS),
    Mutation("er_empty_sequence_scored", MCTS, "                if best_sequence else 0.0",
             "                if True else 0.0", tests=ER_TESTS),
    Mutation("er_scores_only_first_action", MCTS, '"steps_taken": list(initial_state.get("steps_taken", [])) + best_sequence})',
             '"steps_taken": list(initial_state.get("steps_taken", [])) + best_sequence[:1]})', tests=ER_TESTS),
    Mutation("er_ignores_initial_steps", MCTS, 'self._evaluate_world({"steps_taken": list(initial_state.get("steps_taken", [])) + best_sequence})',
             'self._evaluate_world({"steps_taken": best_sequence})', tests=ER_TESTS),
    Mutation("er_world_reward_applies_without_model", MCTS,
             "        if self._uses_world_effects():\n            # Modo mundo: a recompensa da sequência",
             "        if True:\n            # Modo mundo: a recompensa da sequência",
             tests=ER_TESTS + MCTS_TESTS + MCTS_QUALITY + MCTS_INPUT),
    Mutation("er_success_from_best_child", MCTS, "success_probability = round(min(1.0, expected_reward * 1.1), 4)",
             "success_probability = round(min(1.0, (best_child.avg_reward if best_child else 0.0) * 1.1), 4)",
             tests=ER_TESTS),
    Mutation("er_success_without_cap", MCTS, "success_probability = round(min(1.0, expected_reward * 1.1), 4)",
             "success_probability = round(expected_reward * 1.1, 4)", tests=ER_TESTS),
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
    Mutation("r2_unknown_status_ranks_high", CAUSAL, 'verified_status, -1)', 'verified_status, 9)',
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
    Mutation("c4_do_ignored", CAUSAL, "            if declared:\n                state.state.update(declared)\n                verified.update(declared)\n                applied = True\n",
             "", tests=R4_TESTS),
    Mutation("c4_do_skips_split", CAUSAL, "        if len(parts) != 2:\n            continue",
             "        if True:\n            continue", tests=R4_TESTS),
    Mutation("c4_table_skipped_when_do", CAUSAL, "            if entry is not None:\n                state.state.update(entry[\"world\"])\n                verified.update(entry[\"world\"])\n                applied = True\n",
             "            if entry is not None and not self._do(action):\n                state.state.update(entry[\"world\"])\n                verified.update(entry[\"world\"])\n                applied = True\n",
             tests=R4_TESTS + R1_TESTS),
    Mutation("c4_warns_even_when_do", CAUSAL, "            if not applied:\n                warnings.append(f\"unknown action: {action!r}\")",
             "            if entry is None:\n                warnings.append(f\"unknown action: {action!r}\")",
             tests=R4_TESTS),
    Mutation("c4_adjacency_wrong_key", CAUSAL, "adjacency.get(action, ())",
             "adjacency.get(action.lower(), ())", tests=R4_TESTS),
    # SC1–SC8: texto livre não forja sucesso (WORLD_SCHEMA, claims não verificadas)
    Mutation("sc_claim_promotes_success", CAUSAL, 'success = verified.get("system_status") in',
             'success = state.state.get("system_status") in', tests=SC_TESTS),
    Mutation("sc_verified_status_from_state", CAUSAL, 'verified_status=verified.get("system_status")',
             'verified_status=state.state.get("system_status")', tests=SC_TESTS),
    Mutation("sc_claim_updates_verified", CAUSAL, "                state.state.update(claims)\n",
             "                state.state.update(claims)\n                verified.update(claims)\n", tests=SC_TESTS),
    Mutation("sc_claim_not_applied_to_state", CAUSAL, "                state.state.update(claims)\n",
             "", tests=SC_TESTS),
    Mutation("sc_unknown_key_accepted", CAUSAL, "    if allowed is None:\n        return None\n",
             "    if allowed is None:\n        return (canon_key, value)\n", tests=SC_TESTS),
    Mutation("sc_unknown_value_accepted", CAUSAL, "canon_value in [v for v in allowed if isinstance(v, str)]",
             "True", tests=SC_TESTS),
    Mutation("sc_value_case_sensitive", CAUSAL, "canon_value = value.strip().lower()",
             "canon_value = value.strip()", tests=SC_TESTS),
    Mutation("sc_key_case_sensitive", CAUSAL, "canon_key = key.strip().lower()",
             "canon_key = key.strip()", tests=SC_TESTS),
    Mutation("sc_bool_unchecked", CAUSAL, "if any(isinstance(v, bool) for v in allowed) else None",
             "if True else None", tests=SC_TESTS),
    Mutation("sc_non_str_key_accepted", CAUSAL, "    if not isinstance(key, str):\n        return None\n",
             "", tests=SC_TESTS),
    Mutation("sc_text_overrides_declared", CAUSAL, "        elif pair[0] not in skip:\n",
             "        else:\n", tests=SC_TESTS),
    # SC10–SC13 (adendo da revisão do Pearl): sem vazamento, conflito determinístico, estado verificado separado
    Mutation("sr_conflict_kept_as_claim", CAUSAL, "if len(values) == 1}", "if len(values) >= 1}", tests=SR_TESTS),
    Mutation("sr_conflict_warning_dropped", CAUSAL,
             "                warnings.append(f\"conflicting causal claims: {action!r} -> {key} in {', '.join(values)}\")\n",
             "                pass\n", tests=SR_TESTS),
    Mutation("sr_conflict_warning_wrong_text", CAUSAL, "conflicting causal claims: ", "conflicting claims: ", tests=SR_TESTS),
    Mutation("sr_effects_unsorted", CAUSAL, "for effect in sorted(str(item).strip() for item in (effects or ())):",
             "for effect in list(str(item).strip() for item in (effects or ())):", tests=SR_TESTS),
    Mutation("sr_conflict_values_unsorted", CAUSAL, "conflicts = {key: sorted(values) for", "conflicts = {key: list(values) for", tests=SR_TESTS),
    Mutation("sr_verified_state_has_claims", CAUSAL, "            verified_state=verified,\n",
             "            verified_state=dict(state.state),\n", tests=SR_TESTS),
    Mutation("sr_claim_items_dropped", CAUSAL, "            claim_items=claim_items,\n", "            claim_items=[],\n", tests=SR_TESTS),
    Mutation("sr_claim_source_wrong", CAUSAL, '"source": "free_text"', '"source": "table"', tests=SR_TESTS),
    Mutation("sr_claim_confidence_invented", CAUSAL, '"confidence": None', '"confidence": 1.0', tests=SR_TESTS),
    Mutation("sr_do_steps_unfiltered", COUNTERFACTUAL, "if pair is not None and pair[0] in claims:", "if True:", tests=SR_TESTS),
    Mutation("sr_do_steps_keeps_conflict", COUNTERFACTUAL, "if pair is not None and pair[0] in claims:",
             "if pair is not None:", tests=SR_TESTS),
    Mutation("sr_claims_not_recorded", COUNTERFACTUAL, "self.unverified_claims[scenario.scenario_id] = claims_seen", "pass", tests=SR_TESTS),
    Mutation("sr_claims_not_collected", COUNTERFACTUAL, "if isinstance(items, list):", "if False:", tests=SR_TESTS),
    Mutation("sr_claims_stale_across_scenarios", COUNTERFACTUAL,
             "        claims_seen = list(self._pending_claims)\n        self._pending_claims = []\n", "        claims_seen = list(self._pending_claims)\n", tests=SR_TESTS),
    Mutation("sr_benchmark_reads_mixed_state", "benchmarks/nexus_capability_benchmark.py",
             ".simulate(case[\"actions\"]).verified_state", ".simulate(case[\"actions\"]).final_state.state", tests=SR_TESTS),
    Mutation("sc3r_warning_dropped", CAUSAL,
             "                    warnings.append(f\"declared intervention for action outside ACTION_EFFECTS: {action!r}\")\n",
             "                    pass\n", tests=SC3R_TESTS),
    Mutation("sc3r_warns_for_table_actions", CAUSAL, "                if entry is None:\n                    # SC3 (errata E2)",
             "                if True:\n                    # SC3 (errata E2)", tests=SC3R_TESTS),
    Mutation("sc3r_warning_text_changed", CAUSAL, "declared intervention for action outside ACTION_EFFECTS: {action!r}",
             "declared intervention for action outside ACTION_EFFECTS", tests=SC3R_TESTS),
    Mutation("sc3r_outside_table_not_verified", CAUSAL,
             "                verified.update(declared)\n                applied = True\n                if entry is None:",
             "                verified.update(declared if entry is not None else {})\n                applied = True\n                if entry is None:",
             tests=SC3R_TESTS),
    Mutation("sc_claim_warning_dropped", CAUSAL, "                    warnings.append(f\"unverified causal claim (free text): {action!r} -> {key}={value}\")\n",
             "                    pass\n", tests=SC_TESTS),
    Mutation("sc_ignored_warning_dropped", CAUSAL, "                warnings.append(f\"ignored causal claim outside WORLD_SCHEMA: {action!r} -> {pair}\")\n",
             "                pass\n", tests=SC_TESTS),
    Mutation("sc_claim_log_dropped", CAUSAL, "                    claim_log.append(f\"{action}: {key}={value}\")\n",
             "                    pass\n", tests=SC_TESTS),
    Mutation("sc_compare_uses_state", CAUSAL, "STATUS_UTILITY.get(results[i].verified_status, -1)",
             'STATUS_UTILITY.get(results[i].final_state.state.get("system_status"), -1)', tests=SC_TESTS + R2_TESTS),
    Mutation("sc_declare_accepts_non_dict", CAUSAL, '        if not isinstance(effects, dict):\n            raise TypeError("effects deve ser dict")\n',
             "", tests=SC_TESTS),
    Mutation("sc_declare_blank_action", CAUSAL, "if not isinstance(action, str) or not action.strip():",
             "if not isinstance(action, str):", tests=SC_TESTS),
    Mutation("sc_declare_not_stored", CAUSAL, "            self.declared_interventions.setdefault(action, {}).update(accepted)\n",
             "            pass\n", tests=SC_TESTS),
    Mutation("sc_declare_rejected_hidden", CAUSAL, "        return rejected\n",
             "        return []\n", tests=SC_TESTS),
    Mutation("sc_declared_revalidation_dropped", CAUSAL, "                pair = _schema_pair(key, value)\n                if pair is not None:\n                    declared[pair[0]] = pair[1]\n",
             "                declared[key] = value\n", tests=SC_TESTS),
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
             tests=("test_counterfactual_plausibility_contract.py",
                    "test_counterfactual_extraction_contract.py")),
    Mutation("cf_plausibility_ignores_graph_evidence", COUNTERFACTUAL,
             "        if attested:\n"
             "            fraction = attested / len(causal_chain)\n"
             "            return min(1.0, length_score + 0.2 * fraction)\n"
             "        if _graph_has_edge(adjacency):\n"
             "            return max(0.1, length_score - 0.2)\n"
             "        return length_score",
             "        return length_score",
             tests=("test_counterfactual_plausibility_contract.py",)),
    Mutation("cf_plausibility_empty_uses_length", COUNTERFACTUAL,
             "        if intervention is False or (isinstance(intervention, str) and intervention.strip() == \"\"):\n"
             "            return 0.1\n",
             "        if False:\n"
             "            return 0.1\n",
             tests=("test_counterfactual_plausibility_contract.py",)),
    Mutation("cf_plausibility_do_text_counts", COUNTERFACTUAL,
             "    if do_match and _do_step_in_graph(do_match, adjacency):\n",
             "    if do_match:\n",
             tests=("test_counterfactual_plausibility_contract.py",)),
    Mutation("cf_plausibility_causes_text_counts", COUNTERFACTUAL,
             "    if causes_match and _causes_step_in_graph(causes_match, adjacency):\n",
             "    if causes_match:\n",
             tests=("test_counterfactual_plausibility_contract.py",)),
    Mutation("cf_plausibility_bonus_not_fraction", COUNTERFACTUAL,
             "return min(1.0, length_score + 0.2 * fraction)",
             "return min(1.0, length_score + 0.2)",
             tests=("test_counterfactual_plausibility_contract.py",)),
    Mutation("cf_plausibility_no_unsupported_penalty", COUNTERFACTUAL,
             "return max(0.1, length_score - 0.2)",
             "return length_score",
             tests=("test_counterfactual_plausibility_contract.py",)),
    Mutation("cf_plausibility_ignores_attestation", COUNTERFACTUAL,
             "attested = sum(1 for step in causal_chain if _step_attested(step, adjacency))",
             "attested = 0",
             tests=("test_counterfactual_plausibility_contract.py",)),
    Mutation("cf_plausibility_iterates_string_edge", COUNTERFACTUAL,
             "    if isinstance(raw, (str, bytes)) or not isinstance(raw, (set, list, tuple, frozenset)):\n"
             "        return []\n",
             "    if False:\n"
             "        return []\n",
             tests=("test_counterfactual_plausibility_contract.py",)),
    Mutation("cf_compare_skips_insight_extend", COUNTERFACTUAL,
             "        self.insights.extend(insights)",
             "        pass",
             tests=("test_counterfactual_extraction_contract.py",)),
    Mutation("cf_what_if_skips_compare", COUNTERFACTUAL,
             "        comparison = self.compare_outcomes(scenario)",
             "        comparison = scenario  # wrong type/shape",
             tests=("test_counterfactual_extraction_contract.py",)),
    Mutation("cf_diff_reverts_to_zip", COUNTERFACTUAL,
             "tokens_a = set(re.findall(r\"[a-z0-9]+\", actual.lower()))",
             "tokens_a = set(actual)  # mutant: char-set not tokens",
             tests=CF_TESTS),
    Mutation("cf_diff_always_one", COUNTERFACTUAL,
             "return 1.0 - (intersection / union)",
             "return 1.0  # mutant: ignore overlap",
             tests=CF_TESTS),
    Mutation("cf_explicit_kwargs_ignored", COUNTERFACTUAL,
             "situation is not None and intervention is not None",
             "False  # mutant: ignore explicit API",
             tests=CF_TESTS),
    Mutation("cf_explicit_swaps_sit_inter", COUNTERFACTUAL,
             "sit, inter = situation, intervention",
             "sit, inter = intervention, situation  # mutant: swap",
             tests=CF_TESTS),
    Mutation("cf_parse_what_if_always_default", COUNTERFACTUAL,
             "        if \"what if\" in question_lower:",
             "        if False:",
             tests=("test_counterfactual_extraction_contract.py",)),
    Mutation("cf_stats_scenarios_not_counted", COUNTERFACTUAL,
             "        self.total_scenarios_generated += 1",
             "        pass",
             tests=("test_counterfactual_extraction_contract.py",)),
    Mutation("cf_planner_never_consulted", COUNTERFACTUAL,
             "        planner = self.planner\n"
             "        plan = getattr(planner, \"plan\", None) if planner is not None else None\n"
             "        if not callable(plan):\n"
             "            return []",
             "        planner = None  # mutant: planner injetado ignorado\n"
             "        plan = None\n"
             "        if True:\n"
             "            return []",
             tests=("test_counterfactual_planner_contract.py",)),
    Mutation("cf_planner_sequence_discarded", COUNTERFACTUAL,
             "        return [f\"planner: {action}\" for action in sequence if isinstance(action, str)]",
             "        return []  # mutant: sequência do planner descartada",
             tests=("test_counterfactual_planner_contract.py",)),
    Mutation("cf_do_never_consulted", COUNTERFACTUAL,
             "        simulate = getattr(world, \"simulate\", None)\n"
             "        if callable(simulate):\n"
             "            result = simulate([action])\n",
             "        simulate = None  # mutant: do(X) nunca consultado\n"
             "        if False:\n"
             "            result = simulate([action])\n",
             tests=("test_counterfactual_dox_contract.py",)),
    Mutation("cf_scm_edges_ignored", COUNTERFACTUAL,
             "        do_steps = self._scm_do_steps(intervention)\n",
             "        do_steps = []  # mutant: arestas SCM ignoradas\n",
             tests=("test_counterfactual_dox_contract.py",)),
    Mutation("cf_do_swallows_simulate_error", COUNTERFACTUAL,
             "            result = simulate([action])\n",
             "            try:\n"
             "                result = simulate([action])\n"
             "            except Exception:\n"
             "                return None\n",
             tests=("test_counterfactual_dox_contract.py",)),
    Mutation("cf_token_set_match_dropped", COUNTERFACTUAL,
             "        if len(matches) == 1:\n"
             "            return matches[0]\n",
             "        if False and len(matches) == 1:\n"
             "            return matches[0]\n",
             tests=("test_counterfactual_dox_contract.py",)),
    Mutation("cf_do_hook_prefers_private", COUNTERFACTUAL,
             "        for hook_name in (\"do\", \"_do\"):\n",
             "        for hook_name in (\"_do\", \"do\"):\n",
             tests=("test_counterfactual_dox_contract.py",)),
    Mutation("wm_phono_evicts_newest", WM,
             "oldest = min(self.buffer, key=lambda x: x.last_rehearsed)",
             "oldest = max(self.buffer, key=lambda x: x.last_rehearsed)", tests=WM_TESTS),
    Mutation("wm_phono_ignores_capacity", WM,
             "        # Se buffer cheio, remover item mais antigo\n"
             "        if len(self.buffer) >= self.capacity:",
             "        # Se buffer cheio, remover item mais antigo\n"
             "        if len(self.buffer) > self.capacity:", tests=WM_TESTS),
    Mutation("wm_visual_evicts_highest", WM,
             "            # Remover menos ativado\n"
             "            least_active = min(self.buffer, key=lambda x: x.activation)",
             "            # Remover menos ativado\n"
             "            least_active = max(self.buffer, key=lambda x: x.activation)", tests=WM_TESTS),
    Mutation("wm_spatial_evicts_by_entered_at", WM,
             "        \"\"\"Armazenar informação espacial\"\"\"\n"
             "        if len(self.buffer) >= self.capacity:\n"
             "            least_active = min(self.buffer, key=lambda x: x.activation)",
             "        \"\"\"Armazenar informação espacial\"\"\"\n"
             "        if len(self.buffer) >= self.capacity:\n"
             "            least_active = min(self.buffer, key=lambda x: x.entered_at)", tests=WM_TESTS),
    Mutation("wm_decay_keeps_low_activation", WM,
             "if i.activation > 0.1", "if i.activation > 0.0", tests=WM_TESTS),
    Mutation("wm_decay_uses_entered_at", WM,
             "time_since_rehearsal = current_time - item.last_rehearsed",
             "time_since_rehearsal = current_time - item.entered_at", tests=WM_TESTS),
    Mutation("wm_rehearse_no_boost", WM,
             "item.activation = min(1.0, item.activation + 0.3)",
             "item.activation = min(1.0, item.activation + 0.0)", tests=WM_TESTS),
    Mutation("wm_rehearse_no_cap", WM,
             "item.activation = min(1.0, item.activation + 0.3)",
             "item.activation = item.activation + 0.3", tests=WM_TESTS),
    Mutation("wm_rehearse_skips_timestamp", WM,
             "        item.activation = min(1.0, item.activation + 0.3)\n"
             "        item.last_rehearsed = time.time()",
             "        item.activation = min(1.0, item.activation + 0.3)", tests=WM_TESTS),
    Mutation("wm_retrieve_case_sensitive", WM,
             "if query_lower in str(item.content).lower():",
             "if query_lower in str(item.content):", tests=WM_TESTS),
    Mutation("wm_allocate_no_clamp", WM,
             "normalized_strength = max(0.0, min(1.0, float(strength)))",
             "normalized_strength = float(strength)", tests=WM_TESTS),
    Mutation("wm_allocate_ignores_attention", WM,
             '"delegated": self.attention_system is not None,',
             '"delegated": False,', tests=WM_TESTS),
    Mutation("wm_refresh_no_boost", WM,
             "        item.activation = min(1.0, item.activation + 0.2)\n"
             "        item.last_rehearsed = time.time()\n"
             "\n"
             "        return True",
             "        item.activation = min(1.0, item.activation + 0.0)\n"
             "        item.last_rehearsed = time.time()\n"
             "\n"
             "        return True", tests=WM_TESTS),
    Mutation("wm_switch_same_task_costs", WM,
             "            return 0.0  # Sem custo se mesma tarefa",
             "            return self.task_switching_cost  # Sem custo se mesma tarefa", tests=WM_TESTS),
    Mutation("wm_integrate_evicts_by_activation", WM,
             "oldest = min(self.episodic_buffer, key=lambda x: x.entered_at)",
             "oldest = min(self.episodic_buffer, key=lambda x: x.activation)", tests=WM_TESTS),
    Mutation("wm_maintain_rehearses_none", WM,
             "for item in self.phonological_loop.buffer[:3]:  # Top 3",
             "for item in self.phonological_loop.buffer[:0]:  # Top 3", tests=WM_TESTS),
    Mutation("wm_transfer_inverted", WM,
             "        if not self.ltm:\n            return",
             "        if self.ltm:\n            return", tests=WM_TESTS),
    Mutation("wm_consciousness_skipped", WM,
             "        if self.consciousness:",
             "        if False and self.consciousness:", tests=WM_TESTS),
    Mutation("wm_retrieve_wm_no_boost", WM,
             "            item.activation = min(1.0, item.activation + 0.2)\n"
             "            return item",
             "            item.activation = min(1.0, item.activation + 0.0)\n"
             "            return item", tests=WM_TESTS),
    Mutation("wm_active_sort_ascending", WM,
             "all_items.sort(key=lambda x: x.activation, reverse=True)",
             "all_items.sort(key=lambda x: x.activation, reverse=False)", tests=WM_TESTS),
    Mutation("wm_spatial_modality_visual", WM,
             '            modality="spatial",',
             '            modality="visual",', tests=WM_TESTS),
    Mutation("wm_rehearse_verbal_ignores_hit", WM,
             "        success = self.phonological_loop.rehearse(item_id)\n"
             "\n"
             "        if success:",
             "        success = self.phonological_loop.rehearse(item_id)\n"
             "\n"
             "        if False:", tests=WM_TESTS),
    Mutation("wm_module_pulls_monolith", WM,
             "import time\n",
             "import time\nimport core.constitutional_brain  # noqa: F401\n", tests=WM_TESTS),
    Mutation("wm_monolith_stops_reexporting", "core/constitutional_brain.py",
             "from core.cognitivo.working_memory import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    CentralExecutive,\n"
             "    PhonologicalLoop,\n"
             "    RealWorkingMemorySystem,\n"
             "    RehearsalEvent,\n"
             "    VisuospatialSketchpad,\n"
             "    WorkingMemoryItem,\n"
             ")\n",
             "from core.cognitivo.working_memory import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    CentralExecutive as _CentralExecutive,\n"
             "    PhonologicalLoop,\n"
             "    RealWorkingMemorySystem,\n"
             "    RehearsalEvent,\n"
             "    VisuospatialSketchpad,\n"
             "    WorkingMemoryItem,\n"
             ")\n"
             "CentralExecutive = type(\"CentralExecutive\", (_CentralExecutive,), {})\n",
             tests=WM_TESTS),
    Mutation("at_novelty_always_familiar", ATT,
             "            return 0.9  # Alta novidade",
             "            return 0.2  # Alta novidade", tests=AT_TESTS),
    Mutation("at_novelty_always_new", ATT,
             "        return 0.2  # Baixa novidade (familiar)",
             "        return 0.9  # Baixa novidade (familiar)", tests=AT_TESTS),
    Mutation("at_intensity_visual_is_internal", ATT,
             "            return 0.7",
             "            return 0.4", tests=AT_TESTS),
    Mutation("at_weights_drop_novelty", ATT,
             '            "novelty": 0.3,      # Novidade',
             '            "novelty": 0.0,      # Novidade', tests=AT_TESTS),
    Mutation("at_contrast_fallback_zero", ATT,
             "        # Fallback heurístico explícito para estímulos sem observação.\n"
             "        return 0.5",
             "        # Fallback heurístico explícito para estímulos sem observação.\n"
             "        return 0.0", tests=AT_TESTS),
    Mutation("at_motion_fallback_zero", ATT,
             "        # Fallback heurístico explícito para estímulos sem observação.\n"
             "        return 0.6",
             "        # Fallback heurístico explícito para estímulos sem observação.\n"
             "        return 0.0", tests=AT_TESTS),
    Mutation("at_contrast_skips_clamp", ATT,
             "            return min(1.0, max(0.0, stimulus.contrast))",
             "            return stimulus.contrast", tests=AT_TESTS),
    Mutation("at_contrast_ignores_context", ATT,
             '        context_contrast = context.get("contrast")\n',
             "        context_contrast = None\n", tests=AT_TESTS),
    Mutation("at_topdown_no_goals_zero", ATT,
             "            return 0.3  # Sem objetivos = relevância baixa",
             "            return 0.0  # Sem objetivos = relevância baixa", tests=AT_TESTS),
    Mutation("at_goal_match_is_equality", ATT,
             "        matches = sum(1 for word in goal_words if word in stimulus_str)",
             "        matches = sum(1 for word in goal_words if word == stimulus_str)", tests=AT_TESTS),
    Mutation("at_goal_relevance_sums", ATT,
             "        return max(relevances) if relevances else 0.3",
             "        return sum(relevances) if relevances else 0.3", tests=AT_TESTS),
    Mutation("at_priority_no_clamp", ATT,
             "        self.goal_relevance[goal] = min(1.0, max(0.0, priority))",
             "        self.goal_relevance[goal] = priority", tests=AT_TESTS),
    Mutation("at_set_goals_resets_priority", ATT,
             "            if goal not in self.goal_relevance:\n"
             "                self.goal_relevance[goal] = 0.5",
             "            if True or goal not in self.goal_relevance:\n"
             "                self.goal_relevance[goal] = 0.5", tests=AT_TESTS),
    Mutation("at_spotlight_evicts_highest", ATT,
             "            least_salient = min(self.spotlight, key=lambda x: x.salience)",
             "            least_salient = max(self.spotlight, key=lambda x: x.salience)", tests=AT_TESTS),
    Mutation("at_spotlight_ignores_capacity", ATT,
             "        if len(self.spotlight) >= self.capacity:",
             "        if len(self.spotlight) > self.capacity:", tests=AT_TESTS),
    Mutation("at_refocus_appends_again", ATT,
             "        if stimulus in self.spotlight:\n"
             "            # Já focado\n"
             "            return\n",
             "        if False and stimulus in self.spotlight:\n"
             "            # Já focado\n"
             "            return\n", tests=AT_TESTS),
    Mutation("at_attend_sorts_ascending", ATT,
             "            reverse=True",
             "            reverse=False", tests=AT_TESTS),
    Mutation("at_attend_score_is_salience", ATT,
             "            stimulus.attention_score = (stimulus.salience + stimulus.priority) / 2.0",
             "            stimulus.attention_score = stimulus.salience", tests=AT_TESTS),
    Mutation("at_shift_missing_returns_true", ATT,
             "            return False",
             "            return True", tests=AT_TESTS),
    Mutation("at_shift_drops_cross_modal_cost", ATT,
             "                base_cost += 0.3  # Cross-modal shift cost",
             "                base_cost += 0.0  # Cross-modal shift cost", tests=AT_TESTS),
    Mutation("at_shift_same_modal_costs_extra", ATT,
             "            if current_type != target.stimulus_type:",
             "            if current_type == target.stimulus_type:", tests=AT_TESTS),
    Mutation("at_clear_keeps_an_hour", ATT,
             "            if current_time - s.timestamp < 60.0  # Keep last 60s",
             "            if current_time - s.timestamp < 3600.0  # Keep last 60s", tests=AT_TESTS),
    Mutation("at_attend_empty_counts_episode", ATT,
             "        if not self.available_stimuli:\n"
             "            return []\n",
             "        if False and not self.available_stimuli:\n"
             "            return []\n", tests=AT_TESTS),
    Mutation("at_consciousness_skipped", ATT,
             "        if self.consciousness:",
             "        if False and self.consciousness:", tests=AT_TESTS),
    Mutation("at_release_drops_filter", ATT,
             "        self.spotlight = [s for s in self.spotlight if s.stimulus_id != stimulus_id]",
             "        self.spotlight = [s for s in self.spotlight if True]", tests=AT_TESTS),
    Mutation("at_module_pulls_monolith", ATT,
             "import time\n",
             "import time\nimport core.constitutional_brain  # noqa: F401\n", tests=AT_TESTS),
    Mutation("at_monolith_stops_reexporting", "core/constitutional_brain.py",
             "from core.cognitivo.attention import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    AttentionShift,\n"
             "    AttentionalFocus,\n"
             "    AttentionalSpotlight,\n"
             "    BottomUpAttention,\n"
             "    RealAttentionSystem,\n"
             "    StimulusItem,\n"
             "    TopDownAttention,\n"
             ")\n",
             "from core.cognitivo.attention import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    AttentionShift,\n"
             "    AttentionalFocus,\n"
             "    AttentionalSpotlight,\n"
             "    BottomUpAttention as _BottomUpAttention,\n"
             "    RealAttentionSystem,\n"
             "    StimulusItem,\n"
             "    TopDownAttention,\n"
             ")\n"
             "BottomUpAttention = type(\"BottomUpAttention\", (_BottomUpAttention,), {})\n",
             tests=AT_TESTS),
    Mutation("il_promotion_threshold_raised", IL,
             "PROMOTION_RELEVANCE_THRESHOLD: float = 0.75   # relevância mínima para promoção",
             "PROMOTION_RELEVANCE_THRESHOLD: float = 0.90   # relevância mínima para promoção",
             tests=IL_TESTS),
    Mutation("il_promotion_drops_multiplier", IL,
             "old_imp * 1.15", "old_imp * 1.0", tests=IL_TESTS),
    Mutation("il_promotion_drops_cap", IL,
             "new_imp = min(self.IMPORTANCE_CAP, old_imp * 1.15)",
             "new_imp = old_imp * 1.15", tests=IL_TESTS),
    Mutation("il_promoted_ids_ignored", IL,
             "and mem_id not in self.promoted_ids", "and True", tests=IL_TESTS),
    Mutation("il_none_id_not_skipped", IL,
             "if mem_id is None:", "if False and mem_id is None:", tests=IL_TESTS),
    Mutation("il_promotion_stores_short_term", IL,
             "new_imp = min(self.IMPORTANCE_CAP, old_imp * 1.15)\n"
             "                self.memory.store(content, \"long_term\", new_imp)",
             "new_imp = min(self.IMPORTANCE_CAP, old_imp * 1.15)\n"
             "                self.memory.store(content, \"short_term\", new_imp)",
             tests=IL_TESTS),
    Mutation("il_reinforcement_boost_zero", IL,
             "REINFORCEMENT_IMPORTANCE_BOOST: float = 0.08  # boost por acesso extra",
             "REINFORCEMENT_IMPORTANCE_BOOST: float = 0.0  # boost por acesso extra",
             tests=IL_TESTS),
    Mutation("il_reinforcement_ignores_relevance", IL,
             "elif relevance >= 0.5 and old_imp >= 0.7:",
             "elif old_imp >= 0.7:", tests=IL_TESTS),
    Mutation("il_reinforcement_ignores_importance", IL,
             "elif relevance >= 0.5 and old_imp >= 0.7:",
             "elif relevance >= 0.5:", tests=IL_TESTS),
    Mutation("il_promotion_also_reinforces", IL,
             "elif relevance >= 0.5 and old_imp >= 0.7:",
             "if relevance >= 0.5 and old_imp >= 0.7:", tests=IL_TESTS),
    Mutation("il_pattern_freq_three", IL,
             "PATTERN_MIN_FREQ: int = 2                      # freq. mínima para virar padrão",
             "PATTERN_MIN_FREQ: int = 3                      # freq. mínima para virar padrão",
             tests=IL_TESTS),
    Mutation("il_pattern_importance_half", IL,
             "PATTERN_IMPORTANCE: float = 0.92               # importância de memórias sintéticas",
             "PATTERN_IMPORTANCE: float = 0.50               # importância de memórias sintéticas",
             tests=IL_TESTS),
    Mutation("il_pattern_key_keeps_case", IL,
             "key = content[:60].lower().strip()", "key = content[:60]", tests=IL_TESTS),
    Mutation("il_pattern_guard_removed", IL,
             "if not already_stored:", "if True:", tests=IL_TESTS),
    Mutation("il_pattern_synthetic_is_raw_key", IL,
             'synthetic = f"[PATTERN freq={freq}] {content_key}"',
             "synthetic = content_key", tests=IL_TESTS),
    Mutation("il_decay_ignores_log", IL,
             "if self.log and memory_ids_accessed:",
             "if memory_ids_accessed:", tests=IL_TESTS),
    Mutation("il_decay_runs_on_empty", IL,
             "if self.log and memory_ids_accessed:",
             "if self.log:", tests=IL_TESTS),
    Mutation("il_decay_count_is_zero", IL,
             "decayed = max(0, 5 - len(memory_ids_accessed))  # heurística",
             "decayed = 0  # heurística", tests=IL_TESTS),
    Mutation("il_consolidate_ignores_pattern_flag", IL,
             "new_patterns = self.detect_and_store_patterns() if run_pattern_detection else 0",
             "new_patterns = self.detect_and_store_patterns()", tests=IL_TESTS),
    Mutation("il_consolidate_ignores_decay_flag", IL,
             "if run_decay:\n            self.apply_decay(accessed_ids)",
             "if True:\n            self.apply_decay(accessed_ids)", tests=IL_TESTS),
    Mutation("il_consolidate_promoted_counts_all", IL,
             "memories_promoted=sum(1 for e in all_events if e.promotion),",
             "memories_promoted=sum(1 for e in all_events if True),", tests=IL_TESTS),
    Mutation("il_consolidate_reinforced_counts_promotions", IL,
             "memories_reinforced=sum(1 for e in all_events if not e.promotion),",
             "memories_reinforced=sum(1 for e in all_events if e.promotion),", tests=IL_TESTS),
    Mutation("il_plan_threshold_dropped", IL,
             "if reward < 0.7:", "if reward < 0.0:", tests=IL_TESTS),
    Mutation("il_plan_missing_reward_defaults_high", IL,
             'reward = getattr(plan_result, "expected_reward", 0.0)',
             'reward = getattr(plan_result, "expected_reward", 0.9)', tests=IL_TESTS),
    Mutation("il_plan_importance_ignores_reward", IL,
             "importance = min(self.IMPORTANCE_CAP, 0.6 + reward * 0.35)",
             "importance = min(self.IMPORTANCE_CAP, 0.6)", tests=IL_TESTS),
    Mutation("il_module_pulls_monolith", IL,
             "import time\n",
             "import time\nimport core.constitutional_brain  # noqa: F401\n", tests=IL_TESTS),
    Mutation("il_monolith_stops_reexporting", "core/constitutional_brain.py",
             "from core.cognitivo.incremental_learner import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    LearningEvent,\n"
             "    ConsolidationReport,\n"
             "    RealIncrementalLearner,\n"
             ")\n",
             "from core.cognitivo.incremental_learner import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    LearningEvent as _LearningEvent,\n"
             "    ConsolidationReport,\n"
             "    RealIncrementalLearner,\n"
             ")\n"
             "LearningEvent = type(\"LearningEvent\", (_LearningEvent,), {})\n",
             tests=IL_TESTS),
    Mutation("ef_memory_gate_closed", EF, "if self.memory:", "if False:", tests=EF_TESTS),
    Mutation("ef_emotion_gate_closed", EF, "if self.emotion:", "if False:", tests=EF_TESTS),
    Mutation("ef_past_emotions_ignored", EF, "if past_emotions:", "if False:", tests=EF_TESTS),
    Mutation("ef_challenge_keyword_dropped", EF, 'if "challenge" in description.lower():', "if False:", tests=EF_TESTS),
    Mutation("ef_vividness_ignores_past_size", EF, "0.3 + detail_count * 0.15", "0.3", tests=EF_TESTS),
    Mutation("ef_plausibility_ignores_past", EF, "if past_elements:", "if False:", tests=EF_TESTS),
    Mutation("ef_common_events_ignored", EF,
             "if any(event in description.lower() for event in common_events):",
             "if False:", tests=EF_TESTS),
    Mutation("ef_false_details_not_coerced", EF, "details = details or {}",
             "details = {} if details is None else details", tests=EF_TESTS),
    Mutation("ef_significance_treats_zero_as_missing", EF, 'details.get("significance", 0.5)',
             'details.get("significance") or 0.5', tests=EF_TESTS),
    Mutation("ef_blank_trigger_stays_blank", EF, "trigger if trigger else when", "trigger", tests=EF_TESTS),
    Mutation("ef_skips_open_intentions", EF, "if intention.fulfilled:", "if not intention.fulfilled:", tests=EF_TESTS),
    Mutation("ef_time_trigger_is_equality", EF, 'intention.when in str(context.get("time"))',
             'intention.when == str(context.get("time"))', tests=EF_TESTS),
    Mutation("ef_where_trigger_is_equality", EF, 'intention.where in str(context.get("location"))',
             'intention.where == str(context.get("location"))', tests=EF_TESTS),
    Mutation("ef_fulfill_is_idempotent", EF, "if intention:\n",
             "if intention and not intention.fulfilled:\n", tests=EF_TESTS),
    Mutation("ef_preexperience_does_not_count", EF, "self.total_preexperiences += 1",
             "self.total_preexperiences += 0", tests=EF_TESTS),
    Mutation("ef_active_counts_fulfilled", EF, "if not p.fulfilled", "if p.fulfilled", tests=EF_TESTS),
    Mutation("ef_empty_average_divides_by_zero", EF, "max(1, len(self.imagined_events))",
             "len(self.imagined_events)", tests=EF_TESTS),
    Mutation("ef_year_requires_plural", EF, 'if "year" in horizon.lower():',
             'if "years" in horizon.lower():', tests=EF_TESTS),
    Mutation("ef_growth_wisdom_changes", EF, '"wisdom": 0.6', '"wisdom": 0.5', tests=EF_TESTS),
    Mutation("ef_vivid_tie_takes_first", EF, "event1.vividness > event2.vividness",
             "event1.vividness >= event2.vividness", tests=EF_TESTS),
    Mutation("ef_preferred_tie_takes_first", EF, "event1.vividness * event1.personal_significance >",
             "event1.vividness * event1.personal_significance >=", tests=EF_TESTS),
    Mutation("ef_module_pulls_monolith", EF, "import time\n",
             "import time\nimport core.constitutional_brain  # noqa: F401\n", tests=EF_TESTS),
    Mutation("ef_monolith_stops_reexporting", "core/constitutional_brain.py",
             "from core.cognitivo.episodic_future_thinking import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    FutureEvent,\n"
             "    ProspectiveMemory,\n"
             "    RealEpisodicFutureThinkingSystem,\n"
             ")\n",
             "from core.cognitivo.episodic_future_thinking import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    FutureEvent as _FutureEvent,\n"
             "    ProspectiveMemory,\n"
             "    RealEpisodicFutureThinkingSystem,\n"
             ")\n"
             "FutureEvent = type(\"FutureEvent\", (_FutureEvent,), {})\n",
             tests=EF_TESTS),
    Mutation("ef_package_stops_reexporting", "core/cognitivo/__init__.py",
             "from core.cognitivo.episodic_future_thinking import (\n"
             "    FutureEvent,\n"
             "    ProspectiveMemory,\n"
             "    RealEpisodicFutureThinkingSystem,\n"
             ")\n",
             "from core.cognitivo.episodic_future_thinking import (\n"
             "    FutureEvent,\n"
             "    ProspectiveMemory,\n"
             "    RealEpisodicFutureThinkingSystem as _RealEpisodicFutureThinkingSystem,\n"
             ")\n"
             "RealEpisodicFutureThinkingSystem = type(\"RealEpisodicFutureThinkingSystem\", (_RealEpisodicFutureThinkingSystem,), {})\n",
             tests=EF_TESTS),
    Mutation("il_package_stops_reexporting", "core/cognitivo/__init__.py",
             "from core.cognitivo.incremental_learner import (\n"
             "    ConsolidationReport,\n"
             "    LearningEvent,\n"
             "    RealIncrementalLearner,\n"
             ")\n",
             "from core.cognitivo.incremental_learner import (\n"
             "    ConsolidationReport,\n"
             "    LearningEvent,\n"
             "    RealIncrementalLearner as _RealIncrementalLearner,\n"
             ")\n"
             "RealIncrementalLearner = type(\"RealIncrementalLearner\", (_RealIncrementalLearner,), {})\n",
             tests=IL_TESTS),
    Mutation('ms_move_substring_dropped', MS, 'if "move" in action.lower():\n            next_state["position"]', 'if False:\n            next_state["position"]', tests=MS_TESTS),
    Mutation('ms_move_adds_two', MS, 'next_state.get("position", 0) + 1', 'next_state.get("position", 0) + 2', tests=MS_TESTS),
    Mutation('ms_acquire_substring_dropped', MS, 'if "acquire" in action.lower():\n            next_state["resources"]', 'if False:\n            next_state["resources"]', tests=MS_TESTS),
    Mutation('ms_build_substring_dropped', MS, 'if "build" in action.lower():', 'if False:', tests=MS_TESTS),
    Mutation('ms_state_not_copied', MS, 'next_state = state.copy()', 'next_state = state', tests=MS_TESTS),
    Mutation('ms_initial_state_not_copied', MS, 'current_state = scenario.initial_state.copy()', 'current_state = scenario.initial_state', tests=MS_TESTS),
    Mutation('ms_safety_threshold_moved', MS, 'state.get("safety", 1.0) < 0.5', 'state.get("safety", 1.0) <= 0.5', tests=MS_TESTS),
    Mutation('ms_safety_default_zero', MS, 'state.get("safety", 1.0) < 0.5', 'state.get("safety", 0.0) < 0.5', tests=MS_TESTS),
    Mutation('ms_resource_overflow_threshold', MS, 'state.get("resources", 0) > 10', 'state.get("resources", 0) >= 10', tests=MS_TESTS),
    Mutation('ms_invalid_position_dropped', MS, 'if next_state.get("position", 0) < 0:', 'if False:', tests=MS_TESTS),
    Mutation('ms_resource_gain_dropped', MS, 'if next_state.get("resources", 0) > state.get("resources", 0):', 'if False:', tests=MS_TESTS),
    Mutation('ms_structure_growth_dropped', MS, 'if next_state.get("structures", 0) > state.get("structures", 0):', 'if False:', tests=MS_TESTS),
    Mutation('ms_resource_goal_threshold', MS, 'state.get("resources", 0) >= 5', 'state.get("resources", 0) >= 4', tests=MS_TESTS),
    Mutation('ms_structure_goal_threshold', MS, 'state.get("structures", 0) >= 3', 'state.get("structures", 0) >= 2', tests=MS_TESTS),
    Mutation('ms_position_goal_threshold', MS, 'state.get("position", 0) >= 10', 'state.get("position", 0) >= 11', tests=MS_TESTS),
    Mutation('ms_goal_case_sensitive', MS, 'goal_lower = goal.lower()\n\n        if "resource" in goal_lower and state', 'goal_lower = goal\n\n        if "resource" in goal_lower and state', tests=MS_TESTS),
    Mutation('ms_success_does_not_stop', MS, '                success = True\n                break', '                success = True', tests=MS_TESTS),
    Mutation('ms_success_quality_changed', MS, '            return 0.9', '            return 0.8', tests=MS_TESTS),
    Mutation('ms_base_quality_changed', MS, 'quality = 0.3', 'quality = 0.4', tests=MS_TESTS),
    Mutation('ms_resource_credit_dropped', MS, 'quality += 0.2\n        if final_state.get("structures"', 'quality += 0.0\n        if final_state.get("structures"', tests=MS_TESTS),
    Mutation('ms_structure_credit_dropped', MS, 'if final_state.get("structures", 0) > 0:\n            quality += 0.2', 'if final_state.get("structures", 0) > 0:\n            quality += 0.0', tests=MS_TESTS),
    Mutation('ms_goal_credit_position_case_sensitive', MS, 'goal_lower = goal.lower()\n        return "position"', 'goal_lower = goal\n        return "position"', tests=MS_TESTS),
    Mutation('ms_goal_credit_ignores_resource_exclusion', MS, ' and "resource" not in goal_lower', '', tests=MS_TESTS),
    Mutation('ms_goal_credit_ignores_structure_exclusion', MS, ' and "structure" not in goal_lower', '', tests=MS_TESTS),
    Mutation('ms_goal_credit_requires_no_position_word', MS, 'return "position" in goal_lower and', 'return "position" not in goal_lower and', tests=MS_TESTS),
    Mutation('ms_goal_credit_non_string_guard_dropped', MS, 'if not isinstance(goal, str):\n            return False\n        goal_lower', 'if False:\n            return False\n        goal_lower', tests=MS_TESTS),
    Mutation('ms_goal_credit_branch_disabled', MS, 'if self._is_position_only_goal(goal):', 'if False:', tests=MS_TESTS),
    Mutation('ms_goal_credit_base_changed', MS, 'return min(1.0, 0.3 + 0.4 *', 'return min(1.0, 0.4 + 0.4 *', tests=MS_TESTS),
    Mutation('ms_goal_credit_weight_changed', MS, '0.3 + 0.4 * self._position_progress', '0.3 + 0.5 * self._position_progress', tests=MS_TESTS),
    Mutation('ms_goal_credit_cap_dropped', MS, 'return min(1.0, 0.3 + 0.4 * self._position_progress(final_state))', 'return 0.3 + 0.4 * self._position_progress(final_state)', tests=MS_TESTS),
    Mutation('ms_goal_credit_divisor_changed', MS, 'max(0.0, position) / 10.0', 'max(0.0, position) / 9.0', tests=MS_TESTS),
    Mutation('ms_goal_credit_negative_not_floored', MS, 'max(0.0, position) / 10.0', 'position / 10.0', tests=MS_TESTS),
    Mutation('ms_goal_credit_progress_cap_dropped', MS, 'return min(1.0, max(0.0, position) / 10.0)', 'return max(0.0, position) / 10.0', tests=MS_TESTS),
    Mutation('ms_goal_credit_nan_accepted', MS, 'isinstance(position, (int, float)) and position == position', 'isinstance(position, (int, float))', tests=MS_TESTS),
    Mutation('ms_goal_credit_str_position_accepted', MS, 'isinstance(position, (int, float)) and position == position', 'position == position', tests=MS_TESTS),
    Mutation('ms_goal_credit_ignores_final_state', MS, 'self._position_progress(final_state)', 'self._position_progress({})', tests=MS_TESTS),
    Mutation('ms_failure_utility_factor', MS, '(1.0 if success else 0.5)', '(1.0 if success else 0.6)', tests=MS_TESTS),
    Mutation('ms_cache_write_dropped', MS, 'self.simulation_cache[scenario.scenario_id] = result\n', 'pass\n', tests=MS_TESTS),
    Mutation('ms_counter_dropped', MS, '        self.total_simulations += 1\n', '        self.total_simulations += 0\n', tests=MS_TESTS),
    Mutation('ms_world_model_hook_ignored', MS, "if self.world_model and hasattr(self.world_model, 'predict_next_state'):", 'if False:', tests=MS_TESTS),
    Mutation('ms_world_model_truth_gate_dropped', MS, "if self.world_model and hasattr(self.world_model, 'predict_next_state'):", "if hasattr(self.world_model, 'predict_next_state'):", tests=MS_TESTS),
    Mutation('ms_rollout_drops_first_action', MS, 'actions=[action] + random.choices(possible_actions, k=3)', 'actions=random.choices(possible_actions, k=3)', tests=MS_TESTS),
    Mutation('ms_rollout_horizon_changed', MS, 'random.choices(possible_actions, k=3)', 'random.choices(possible_actions, k=2)', tests=MS_TESTS),
    Mutation('ms_rollout_picks_worst', MS, 'best_action = max(action_utilities.items()', 'best_action = min(action_utilities.items()', tests=MS_TESTS),
    Mutation('ms_rollout_sums_instead_of_mean', MS, 'total_utility / num_rollouts', 'total_utility', tests=MS_TESTS),
    Mutation('ms_rollout_reuses_scenario_id', MS, 'scenario_id=str(uuid.uuid4()),', 'scenario_id="rollout",', tests=MS_TESTS),
    Mutation('ms_counterfactual_keeps_first_action', MS, '[alternative_action] + actual_scenario.actions[1:]', '[alternative_action] + actual_scenario.actions', tests=MS_TESTS),
    Mutation('ms_counterfactual_id_unprefixed', MS, 'f"counterfactual_{actual_scenario.scenario_id}"', 'actual_scenario.scenario_id', tests=MS_TESTS),
    Mutation('ms_compare_tie_prefers_first', MS, 'result1.predicted_utility > result2.predicted_utility', 'result1.predicted_utility >= result2.predicted_utility', tests=MS_TESTS),
    Mutation('ms_compare_difference_sign', MS, 'result1.predicted_utility - result2.predicted_utility', 'result2.predicted_utility - result1.predicted_utility', tests=MS_TESTS),
    Mutation('ms_stats_cache_count_dropped', MS, '"cached_simulations": len(self.simulation_cache)', '"cached_simulations": 0', tests=MS_TESTS),
    Mutation("ms_module_pulls_monolith", MS, "import random\n",
             "import random\nimport core.constitutional_brain  # noqa: F401\n", tests=MS_TESTS),
    Mutation("ms_monolith_stops_reexporting", "core/constitutional_brain.py",
             "from core.cognitivo.mental_simulation import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    MentalScenario,\n"
             "    MentalSimulationResult,\n"
             "    RealMentalSimulationSystem,\n"
             ")\n",
             "from core.cognitivo.mental_simulation import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    MentalScenario,\n"
             "    MentalSimulationResult,\n"
             "    RealMentalSimulationSystem as _RealMentalSimulationSystem,\n"
             ")\n"
             "RealMentalSimulationSystem = type(\"RealMentalSimulationSystem\", (_RealMentalSimulationSystem,), {})\n",
             tests=MS_TESTS),
    Mutation("ms_package_stops_reexporting", "core/cognitivo/__init__.py",
             "from core.cognitivo.mental_simulation import (\n"
             "    MentalScenario,\n"
             "    MentalSimulationResult,\n"
             "    RealMentalSimulationSystem,\n"
             ")\n",
             "from core.cognitivo.mental_simulation import (\n"
             "    MentalScenario,\n"
             "    MentalSimulationResult,\n"
             "    RealMentalSimulationSystem as _RealMentalSimulationSystem,\n"
             ")\n"
             "RealMentalSimulationSystem = type(\"RealMentalSimulationSystem\", (_RealMentalSimulationSystem,), {})\n",
             tests=MS_TESTS),
    Mutation('ar_same_domain_not_skipped', AR, 'if source.domain == target_domain:\n                continue', 'if False:\n                continue', tests=AR_TESTS),
    Mutation('ar_domain_case_insensitive', AR, 'if source.domain == target_domain:', 'if source.domain.lower() == target_domain.lower():', tests=AR_TESTS),
    Mutation('ar_tie_prefers_later', AR, 'mapping.quality_score > best_score', 'mapping.quality_score >= best_score', tests=AR_TESTS),
    Mutation('ar_quality_denominator_min', AR, 'max(len(source.relations), len(target.relations))', 'min(len(source.relations), len(target.relations))', tests=AR_TESTS),
    Mutation('ar_mapping_no_break', AR, 'matched_relations += 1\n                    break', 'matched_relations += 1', tests=AR_TESTS),
    Mutation('ar_matched_counter_dropped', AR, 'matched_relations += 1', 'matched_relations += 0', tests=AR_TESTS),
    Mutation('ar_unmatched_returns_mapping', AR, 'if matched_relations == 0:\n            return None', 'if False:\n            return None', tests=AR_TESTS),
    Mutation('ar_target_id_changed', AR, 'situation_id="target",', 'situation_id="t",', tests=AR_TESTS),
    Mutation('ar_target_entities_copied', AR, 'entities=target_entities,', 'entities=list(target_entities),', tests=AR_TESTS),
    Mutation('ar_known_relations_copied', AR, 'domain=domain,\n            entities=entities,\n            relations=relations,', 'domain=domain,\n            entities=entities,\n            relations=list(relations),', tests=AR_TESTS),
    Mutation('ar_known_situation_not_stored', AR, 'self.known_situations.append(situation)', 'pass', tests=AR_TESTS),
    Mutation('ar_analogy_counter_dropped', AR, 'self.total_analogies_found += 1', 'self.total_analogies_found += 0', tests=AR_TESTS),
    Mutation('ar_transfer_guard_dropped', AR, 'if not mapping.source_situation.solution:\n            return None', 'if False:\n            return None', tests=AR_TESTS),
    Mutation('ar_transfer_counter_dropped', AR, 'self.successful_transfers += 1', 'self.successful_transfers += 0', tests=AR_TESTS),
    Mutation('ar_mapping_reuses_source_relations', AR, 'if index in used_relations or t_rel != s_rel:', 'if t_rel != s_rel:', tests=AR_TESTS),
    Mutation('ar_mapping_forgets_used_relation', AR, 'used_relations.add(index)', 'pass', tests=AR_TESTS),
    Mutation('ar_mapping_not_injective', AR, 'elif any(image == t_entity for image in tentative.values()):', 'elif False:', tests=AR_TESTS),
    Mutation('ar_mapping_not_forward_consistent', AR, 'consistent = tentative[s_entity] == t_entity', 'consistent = True', tests=AR_TESTS),
    Mutation('ar_mapping_inconsistency_not_final', AR, 'if not consistent:\n                        break', 'pass', tests=AR_TESTS),
    Mutation('ar_mapping_leaks_partial_pairs', AR, 'tentative = dict(entity_mappings)', 'tentative = entity_mappings', tests=AR_TESTS),
    Mutation('ar_mapping_never_committed', AR, 'entity_mappings = tentative\n', 'pass\n', tests=AR_TESTS),
    Mutation('ar_transfer_non_str_not_rejected', AR, 'if mapping.entity_mappings and not isinstance(source_solution, str):', 'if False:', tests=AR_TESTS),
    Mutation('ar_transfer_non_str_checked_without_mapping', AR, 'if mapping.entity_mappings and not isinstance(source_solution, str):', 'if not isinstance(source_solution, str):', tests=AR_TESTS),
    Mutation('ar_transfer_empty_keys_kept', AR, 'keys = [key for key in mapping.entity_mappings if key]', 'keys = list(mapping.entity_mappings)', tests=AR_TESTS),
    Mutation('ar_transfer_shortest_key_first', AR, 'keys.sort(key=len, reverse=True)', 'keys.sort(key=len)', tests=AR_TESTS),
    Mutation('ar_transfer_keys_unordered', AR, 'keys.sort(key=len, reverse=True)', 'pass', tests=AR_TESTS),
    Mutation('ar_transfer_keys_unescaped', AR, 're.escape(key)', 'key', tests=AR_TESTS),
    Mutation('ar_transfer_left_boundary_dropped', AR, '(?<!\\w)', '', tests=AR_TESTS),
    Mutation('ar_transfer_right_boundary_dropped', AR, '(?!\\w)', '', tests=AR_TESTS),
    Mutation('ar_transfer_replacement_looked_up_wrongly', AR, 'mapping.entity_mappings[found.group(0)]', 'found.group(0)', tests=AR_TESTS),
    Mutation('ar_transfer_skipped_without_keys', AR, 'if keys:\n            keys.sort', 'if not keys:\n            keys.sort', tests=AR_TESTS),
    Mutation('ar_reason_ignores_entities', AR, 'if len(entities) >= 2:', 'if False:', tests=AR_TESTS),
    Mutation('ar_reason_needs_three_entities', AR, 'if len(entities) >= 2:', 'if len(entities) > 2:', tests=AR_TESTS),
    Mutation('ar_reason_one_entity_enough', AR, 'if len(entities) >= 2:', 'if len(entities) >= 1:', tests=AR_TESTS),
    Mutation('ar_reason_swaps_entities', AR, '"entity1": entities[0], "entity2": entities[1]', '"entity1": entities[1], "entity2": entities[0]', tests=AR_TESTS),
    Mutation('ar_reason_overwrites_real_names', AR, 'names.get(e1, e1)', 'entities[0]', tests=AR_TESTS),
    Mutation('ar_reason_skips_transfer', AR, 'solution = self.transfer_solution(mapping)', 'solution = None', tests=AR_TESTS),
    Mutation('ar_reason_returns_none_early', AR, 'if not mapping:\n            return None\n\n        # Transferir', 'if True:\n            return None\n\n        # Transferir', tests=AR_TESTS),
    Mutation('ar_entities_limit_changed', AR, 'return entities[:5]', 'return entities[:4]', tests=AR_TESTS),
    Mutation('ar_entities_strip_dropped', AR, "w.strip('.,!?')", 'w', tests=AR_TESTS),
    Mutation('ar_entities_len_guard_dropped', AR, 'and len(w) > 1', 'and len(w) > 0', tests=AR_TESTS),
    Mutation('ar_relation_similar_dropped', AR, 'if "similar to" in text.lower():', 'if False:', tests=AR_TESTS),
    Mutation('ar_relation_causes_dropped', AR, 'if "causes" in text.lower():', 'if False:', tests=AR_TESTS),
    Mutation('ar_relation_part_of_dropped', AR, 'if "part of" in text.lower():', 'if False:', tests=AR_TESTS),
    Mutation('ar_relations_case_sensitive', AR, 'if "causes" in text.lower():', 'if "causes" in text:', tests=AR_TESTS),
    Mutation('ar_stats_rate_divisor_unguarded', AR, 'max(1, self.total_analogies_found)', 'self.total_analogies_found', tests=AR_TESTS),
    Mutation('ar_stats_known_dropped', AR, '"known_situations": len(self.known_situations)', '"known_situations": 0', tests=AR_TESTS),
    Mutation('ar_collaborator_not_stored', AR, 'self.memory = memory', 'self.memory = None', tests=AR_TESTS),
    Mutation("ar_module_pulls_monolith", AR, "import uuid\n",
             "import uuid\nimport core.constitutional_brain  # noqa: F401\n", tests=AR_TESTS),
    Mutation("ar_monolith_stops_reexporting", "core/constitutional_brain.py",
             "from core.cognitivo.analogical_reasoning import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    AnalogyMapping,\n"
             "    AnalogySituation,\n"
             "    RealAnalogicalReasoningSystem,\n"
             ")\n",
             "from core.cognitivo.analogical_reasoning import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    AnalogyMapping,\n"
             "    AnalogySituation,\n"
             "    RealAnalogicalReasoningSystem as _RealAnalogicalReasoningSystem,\n"
             ")\n"
             "RealAnalogicalReasoningSystem = type(\"RealAnalogicalReasoningSystem\", (_RealAnalogicalReasoningSystem,), {})\n",
             tests=AR_TESTS),
    Mutation("ar_package_stops_reexporting", "core/cognitivo/__init__.py",
             "from core.cognitivo.analogical_reasoning import (\n"
             "    AnalogyMapping,\n"
             "    AnalogySituation,\n"
             "    RealAnalogicalReasoningSystem,\n"
             ")\n",
             "from core.cognitivo.analogical_reasoning import (\n"
             "    AnalogyMapping,\n"
             "    AnalogySituation,\n"
             "    RealAnalogicalReasoningSystem as _RealAnalogicalReasoningSystem,\n"
             ")\n"
             "RealAnalogicalReasoningSystem = type(\"RealAnalogicalReasoningSystem\", (_RealAnalogicalReasoningSystem,), {})\n",
             tests=AR_TESTS),
    Mutation('cm_dimension_dropped', CM, '            "social"       # Contexto social\n', '            # Contexto social\n', tests=CM_TESTS),
    Mutation('cm_dimension_order_swapped', CM, '"location",    # Onde\n            "time",        # Quando', '"time",        # Quando\n            "location",    # Onde', tests=CM_TESTS),
    Mutation('cm_missing_value_not_unknown', CM, 'context.get(dim, "unknown")', 'context.get(dim, "")', tests=CM_TESTS),
    Mutation('cm_hash_modulus_changed', CM, '% 100 / 100.0', '% 101 / 100.0', tests=CM_TESTS),
    Mutation('cm_encoding_hash_builtin', CM, 'int(digest[:8], 16) % 100 / 100.0', 'hash(digest) % 100 / 100.0', tests=CM_TESTS),
    Mutation('cm_encoding_of_repr', CM, 'hashlib.sha256(str(value).encode', 'hashlib.sha256(repr(value).encode', tests=CM_TESTS),
    Mutation('cm_encoding_md5', CM, 'hashlib.sha256(', 'hashlib.md5(', tests=CM_TESTS),
    Mutation('cm_encoding_surrogates_strict', CM, '"utf-8", "surrogatepass"', '"utf-8", "replace"', tests=CM_TESTS),
    Mutation('cm_encoding_digest_slice', CM, 'digest[:8]', 'digest[:6]', tests=CM_TESTS),
    Mutation('cm_match_empty_guard_dropped', CM, 'if not vec1 or not vec2:\n            return 0.0\n\n        indexes', 'if False:\n            return 0.0\n\n        indexes', tests=CM_TESTS),
    Mutation('cm_match_one_empty_only', CM, 'if not vec1 or not vec2:\n            return 0.0\n\n        indexes', 'if not vec1 and not vec2:\n            return 0.0\n\n        indexes', tests=CM_TESTS),
    Mutation('cm_match_default_indexes_max', CM, 'range(min(len(vec1), len(vec2)))', 'range(max(len(vec1), len(vec2)))', tests=CM_TESTS),
    Mutation('cm_match_default_indexes_ignored', CM, 'if cue_indexes is None else cue_indexes', 'if True else cue_indexes', tests=CM_TESTS),
    Mutation('cm_match_no_cue_scores_zero', CM, 'if not indexes:\n            return 1.0', 'if not indexes:\n            return 0.0', tests=CM_TESTS),
    Mutation('cm_match_no_cue_guard_dropped', CM, 'if not indexes:\n            return 1.0', 'if False:\n            return 1.0', tests=CM_TESTS),
    Mutation('cm_match_counts_differences', CM, 'if vec1[i] == vec2[i]', 'if vec1[i] != vec2[i]', tests=CM_TESTS),
    Mutation('cm_match_denominator_fixed', CM, '/ len(indexes)', '/ 5', tests=CM_TESTS),
    Mutation('cm_retrieval_cues_all_dimensions', CM, 'if dim in context]', 'if True]', tests=CM_TESTS),
    Mutation('cm_retrieval_cues_dropped', CM, 'retrieval_vector,\n                cued\n', 'retrieval_vector\n', tests=CM_TESTS),
    Mutation('cm_retrieval_uses_cosine', CM, 'self.context_encoder.compute_match(', 'self.context_encoder.compute_similarity(', tests=CM_TESTS),
    Mutation('cm_retrieval_score_ignores_context', CM, 'match_score = similarity\n', 'match_score = 1.0\n', tests=CM_TESTS),
    Mutation('cm_retrieval_query_score_multiplied', CM, 'match_score = (similarity + content_match) / 2.0', 'match_score = similarity * content_match', tests=CM_TESTS),
    Mutation('cm_retrieval_query_miss_value', CM, '1.0 if query_lower in content_lower else 0.3', '1.0 if query_lower in content_lower else 0.0', tests=CM_TESTS),
    Mutation('cm_similarity_empty_guard_dropped', CM, 'if not vec1 or not vec2:\n            return 0.0\n\n        # Similaridade', 'if False:\n            return 0.0\n\n        # Similaridade', tests=CM_TESTS),
    Mutation('cm_similarity_one_empty_only', CM, 'if not vec1 or not vec2:\n            return 0.0\n\n        # Similaridade', 'if not vec1 and not vec2:\n            return 0.0\n\n        # Similaridade', tests=CM_TESTS),
    Mutation('cm_similarity_not_normalised', CM, 'return dot_product / (mag1 * mag2)', 'return dot_product', tests=CM_TESTS),
    Mutation('cm_similarity_zero_guard_dropped', CM, 'if mag1 == 0 or mag2 == 0:', 'if False:', tests=CM_TESTS),
    Mutation('cm_similarity_zero_guard_and', CM, 'if mag1 == 0 or mag2 == 0:', 'if mag1 == 0 and mag2 == 0:', tests=CM_TESTS),
    Mutation('cm_similarity_dot_wrong', CM, 'a * b for a, b in zip(vec1, vec2)', 'a + b for a, b in zip(vec1, vec2)', tests=CM_TESTS),
    Mutation('cm_set_context_vector_dropped', CM, 'self.current_context_vector = self.context_encoder.encode_context(context)', 'pass', tests=CM_TESTS),
    Mutation('cm_set_context_copied', CM, 'self.current_context = context\n', 'self.current_context = dict(context)\n', tests=CM_TESTS),
    Mutation('cm_store_ignores_current_context', CM, 'if context is None:\n            context = self.current_context\n\n        # Codificar contexto\n', 'if context is None:\n            context = {}\n\n        # Codificar contexto\n', tests=CM_TESTS),
    Mutation('cm_store_context_copied', CM, '            context=context,\n            context_vector', '            context=dict(context),\n            context_vector', tests=CM_TESTS),
    Mutation('cm_store_not_appended', CM, 'self.contextual_memories.append(memory)', 'pass', tests=CM_TESTS),
    Mutation('cm_encoding_counter_dropped', CM, 'self.total_encodings += 1', 'self.total_encodings += 0', tests=CM_TESTS),
    Mutation('cm_ltm_always_called', CM, 'if self.ltm:', 'if True:', tests=CM_TESTS),
    Mutation('cm_ltm_content_not_str', CM, 'str(content),\n                "contextual"', 'content,\n                "contextual"', tests=CM_TESTS),
    Mutation('cm_ltm_importance_changed', CM, 'importance=0.7', 'importance=0.5', tests=CM_TESTS),
    Mutation('cm_ltm_kind_changed', CM, '"contextual",\n                importance', '"context",\n                importance', tests=CM_TESTS),
    Mutation('cm_retrieval_ranked_ascending', CM, 'matches.sort(key=lambda m: m.match_score, reverse=True)', 'matches.sort(key=lambda m: m.match_score)', tests=CM_TESTS),
    Mutation('cm_retrieval_unsorted', CM, 'matches.sort(key=lambda m: m.match_score, reverse=True)', 'pass', tests=CM_TESTS),
    Mutation('cm_threshold_strict', CM, 'if match_score >= threshold:', 'if match_score > threshold:', tests=CM_TESTS),
    Mutation('cm_default_threshold_changed', CM, 'threshold: float = 0.5)', 'threshold: float = 0.6)', tests=CM_TESTS),
    Mutation('cm_score_not_averaged', CM, '(similarity + content_match) / 2.0', 'similarity + content_match', tests=CM_TESTS),
    Mutation('cm_score_ignores_content', CM, '(similarity + content_match) / 2.0', 'similarity', tests=CM_TESTS),
    Mutation('cm_query_mismatch_weight_changed', CM, 'else 0.3', 'else 0.2', tests=CM_TESTS),
    Mutation('cm_query_case_sensitive', CM, 'query_lower = str(query).lower()', 'query_lower = str(query)', tests=CM_TESTS),
    Mutation('cm_content_case_sensitive', CM, 'content_lower = str(memory.content).lower()', 'content_lower = str(memory.content)', tests=CM_TESTS),
    Mutation('cm_query_ignored', CM, '            if query:\n                query_lower', '            if False:\n                query_lower', tests=CM_TESTS),
    Mutation('cm_retrieval_count_dropped', CM, 'memory.retrieval_count += 1', 'pass', tests=CM_TESTS),
    Mutation('cm_retrieval_count_always', CM, '                # Incrementar retrieval count\n                memory.retrieval_count += 1', '                pass\n            memory.retrieval_count += 1', tests=CM_TESTS),
    Mutation('cm_retrieval_counter_dropped', CM, 'self.total_retrievals += 1\n', 'self.total_retrievals += 0\n', tests=CM_TESTS),
    Mutation('cm_matched_counter_always', CM, 'if matches:\n            self.context_matched_retrievals += 1', 'self.context_matched_retrievals += 1', tests=CM_TESTS),
    Mutation('cm_retrieval_ignores_current_context', CM, '            context = self.current_context\n\n        # Codificar contexto de retrieval', '            context = {}\n\n        # Codificar contexto de retrieval', tests=CM_TESTS),
    Mutation('cm_features_equality_dropped', CM, 'if encoding_context[key] == retrieval_context[key]:', 'if True:', tests=CM_TESTS),
    Mutation('cm_features_membership_dropped', CM, 'if key in retrieval_context:\n                if', 'if True:\n                if', tests=CM_TESTS),
    Mutation('cm_features_iterates_retrieval', CM, 'for key in encoding_context:', 'for key in retrieval_context:', tests=CM_TESTS),
    Mutation('cm_reinstate_missing_raises', CM, 'if not memory:\n            return None', 'if not memory:\n            raise KeyError(memory_id)', tests=CM_TESTS),
    Mutation('cm_reinstate_does_not_set_context', CM, 'self.set_current_context(memory.context)', 'pass', tests=CM_TESTS),
    Mutation('cm_reinstate_returns_none', CM, 'return memory.context\n', 'return None\n', tests=CM_TESTS),
    Mutation('cm_cues_format_changed', CM, 'f"{key}={value}"', 'f"{key}:{value}"', tests=CM_TESTS),
    Mutation('cm_cues_values_only', CM, 'f"{key}={value}"', 'f"{value}"', tests=CM_TESTS),
    Mutation('cm_specificity_threshold_raised', CM, 'threshold=0.0)', 'threshold=0.5)', tests=CM_TESTS),
    Mutation('cm_specificity_missing_returns_one', CM, '        return 0.0\n\n    def get_statistics', '        return 1.0\n\n    def get_statistics', tests=CM_TESTS),
    Mutation('cm_specificity_wrong_context', CM, 'self.retrieve_by_context(content, retrieval_context', 'self.retrieve_by_context(content, encoding_context', tests=CM_TESTS),
    Mutation('cm_stats_rate_unguarded', CM, 'max(1, self.total_retrievals)', 'self.total_retrievals', tests=CM_TESTS),
    Mutation('cm_stats_rate_always_computed', CM, 'if self.total_retrievals > 0\n            else 0.0', 'if True\n            else 0.0', tests=CM_TESTS),
    Mutation('cm_stats_dimensions_dropped', CM, '"current_context_dimensions": len(self.current_context)', '"current_context_dimensions": 0', tests=CM_TESTS),
    Mutation('cm_stats_stored_dropped', CM, '"stored_memories": len(self.contextual_memories)', '"stored_memories": self.total_encodings', tests=CM_TESTS),
    Mutation('cm_collaborator_not_stored', CM, 'self.wm = working_memory', 'self.wm = None', tests=CM_TESTS),
    Mutation('cm_banner_dropped', CM, 'print("🧠🔍 POST 14', 'pass  # print("🧠🔍 POST 14', tests=CM_TESTS),
    Mutation("cm_module_pulls_monolith", CM, "import uuid\n",
             "import uuid\nimport core.constitutional_brain  # noqa: F401\n", tests=CM_TESTS),
    Mutation("cm_monolith_stops_reexporting", "core/constitutional_brain.py",
             "from core.cognitivo.contextual_memory import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    ContextEncoder,\n"
             "    ContextMatch,\n"
             "    ContextualMemory,\n"
             "    RealContextualMemorySystem,\n"
             ")\n",
             "from core.cognitivo.contextual_memory import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    ContextEncoder,\n"
             "    ContextMatch,\n"
             "    ContextualMemory,\n"
             "    RealContextualMemorySystem as _RealContextualMemorySystem,\n"
             ")\n"
             "RealContextualMemorySystem = type(\"RealContextualMemorySystem\", (_RealContextualMemorySystem,), {})\n",
             tests=CM_TESTS),
    Mutation("cm_package_stops_reexporting", "core/cognitivo/__init__.py",
             "from core.cognitivo.contextual_memory import (\n"
             "    ContextEncoder,\n"
             "    ContextMatch,\n"
             "    ContextualMemory,\n"
             "    RealContextualMemorySystem,\n"
             ")\n",
             "from core.cognitivo.contextual_memory import (\n"
             "    ContextEncoder,\n"
             "    ContextMatch,\n"
             "    ContextualMemory,\n"
             "    RealContextualMemorySystem as _RealContextualMemorySystem,\n"
             ")\n"
             "RealContextualMemorySystem = type(\"RealContextualMemorySystem\", (_RealContextualMemorySystem,), {})\n",
             tests=CM_TESTS),
    Mutation('ew_valence_default_changed', EW, "emotional_context.get('valence', 0.0)", "emotional_context.get('valence', 0.1)", tests=EW_TESTS),
    Mutation('ew_arousal_default_changed', EW, "emotional_context.get('arousal', 0.0)", "emotional_context.get('arousal', 0.1)", tests=EW_TESTS),
    Mutation('ew_emotion_guard_dropped', EW, "if self.emotion and 'agent_id' in emotional_context:", "if 'agent_id' in emotional_context:", tests=EW_TESTS),
    Mutation('ew_agent_guard_dropped', EW, "if self.emotion and 'agent_id' in emotional_context:", 'if self.emotion:', tests=EW_TESTS),
    Mutation('ew_agent_membership_dropped', EW, 'if agent_id in self.emotion.emotional_states:', 'if True:', tests=EW_TESTS),
    Mutation('ew_state_valence_not_used', EW, 'valence = state.pleasure', 'pass', tests=EW_TESTS),
    Mutation('ew_state_arousal_not_used', EW, 'arousal = state.arousal', 'pass', tests=EW_TESTS),
    Mutation('ew_state_valence_from_arousal', EW, 'valence = state.pleasure', 'valence = state.arousal', tests=EW_TESTS),
    Mutation('ew_intensity_not_normalised', EW, 'math.sqrt(valence**2 + arousal**2) / math.sqrt(2)', 'math.sqrt(valence**2 + arousal**2)', tests=EW_TESTS),
    Mutation('ew_intensity_sum_not_euclidean', EW, 'math.sqrt(valence**2 + arousal**2)', 'abs(valence) + abs(arousal)', tests=EW_TESTS),
    Mutation('ew_intensity_ignores_valence', EW, 'math.sqrt(valence**2 + arousal**2)', 'math.sqrt(arousal**2)', tests=EW_TESTS),
    Mutation('ew_intensity_ignores_arousal', EW, 'math.sqrt(valence**2 + arousal**2)', 'math.sqrt(valence**2)', tests=EW_TESTS),
    Mutation('ew_boost_base_changed', EW, 'consolidation_boost = 1.0 + (arousal', 'consolidation_boost = 0.0 + (arousal', tests=EW_TESTS),
    Mutation('ew_boost_ignores_factor', EW, '(arousal * self.arousal_consolidation_factor)', 'arousal', tests=EW_TESTS),
    Mutation('ew_boost_factor_changed', EW, 'self.arousal_consolidation_factor = 2.0', 'self.arousal_consolidation_factor = 1.0', tests=EW_TESTS),
    Mutation('ew_priority_factor_changed', EW, 'emotional_intensity * 1.5', 'emotional_intensity * 1.0', tests=EW_TESTS),
    Mutation('ew_weight_not_stored', EW, 'self.emotional_weights[memory_id] = weight', 'pass', tests=EW_TESTS),
    Mutation('ew_counter_dropped', EW, 'self.total_emotional_memories += 1', 'self.total_emotional_memories += 0', tests=EW_TESTS),
    Mutation('ew_stats_update_dropped', EW, '        # Update statistics\n        self._update_statistics()\n', '        # Update statistics\n', tests=EW_TESTS),
    Mutation('ew_high_arousal_strict_changed', EW, 'if arousal > 0.7:', 'if arousal >= 0.7:', tests=EW_TESTS),
    Mutation('ew_high_arousal_threshold_changed', EW, 'if arousal > 0.7:', 'if arousal > 0.8:', tests=EW_TESTS),
    Mutation('ew_high_arousal_counter_dropped', EW, 'self.high_arousal_count += 1', 'self.high_arousal_count += 0', tests=EW_TESTS),
    Mutation('ew_average_not_divided', EW, 'sum(intensities) / len(intensities)', 'sum(intensities)', tests=EW_TESTS),
    Mutation('ew_average_guard_dropped', EW, 'if self.emotional_weights:\n            intensities', 'if True:\n            intensities', tests=EW_TESTS),
    Mutation('ew_unknown_consolidation_changed', EW, 'return 1.0  # Sem boost', 'return 0.0  # Sem boost', tests=EW_TESTS),
    Mutation('ew_consolidation_returns_priority', EW, 'return weight.consolidation_boost', 'return weight.recall_priority', tests=EW_TESTS),
    Mutation('ew_base_score_changed', EW, 'base_score = 1.0', 'base_score = 0.5', tests=EW_TESTS),
    Mutation('ew_untagged_dropped', EW, 'scored_memories.append((mem_id, 0.0))', 'pass', tests=EW_TESTS),
    Mutation('ew_untagged_score_restored', EW, 'scored_memories.append((mem_id, 0.0))', 'scored_memories.append((mem_id, base_score))', tests=EW_TESTS),
    Mutation('ew_untagged_score_half', EW, 'scored_memories.append((mem_id, 0.0))', 'scored_memories.append((mem_id, 0.5))', tests=EW_TESTS),
    Mutation('ew_untagged_score_negative', EW, 'scored_memories.append((mem_id, 0.0))', 'scored_memories.append((mem_id, -1.0))', tests=EW_TESTS),
    Mutation('ew_clamp_valence_dropped', EW, 'valence = self._clamp_to_domain(valence, -1.0, 1.0)', 'pass', tests=EW_TESTS),
    Mutation('ew_clamp_arousal_dropped', EW, 'arousal = self._clamp_to_domain(arousal, 0.0, 1.0)', 'pass', tests=EW_TESTS),
    Mutation('ew_clamp_valence_low_bound_changed', EW, '_clamp_to_domain(valence, -1.0, 1.0)', '_clamp_to_domain(valence, -0.9, 1.0)', tests=EW_TESTS),
    Mutation('ew_clamp_valence_high_bound_changed', EW, '_clamp_to_domain(valence, -1.0, 1.0)', '_clamp_to_domain(valence, -1.0, 0.9)', tests=EW_TESTS),
    Mutation('ew_clamp_arousal_low_bound_changed', EW, '_clamp_to_domain(arousal, 0.0, 1.0)', '_clamp_to_domain(arousal, 0.1, 1.0)', tests=EW_TESTS),
    Mutation('ew_clamp_arousal_high_bound_changed', EW, '_clamp_to_domain(arousal, 0.0, 1.0)', '_clamp_to_domain(arousal, 0.0, 0.9)', tests=EW_TESTS),
    Mutation('ew_clamp_arousal_low_bound_to_valence_domain', EW, '_clamp_to_domain(arousal, 0.0, 1.0)', '_clamp_to_domain(arousal, -1.0, 1.0)', tests=EW_TESTS),
    Mutation('ew_clamp_valence_reads_arousal', EW, 'valence = self._clamp_to_domain(valence, -1.0, 1.0)', 'valence = self._clamp_to_domain(arousal, -1.0, 1.0)', tests=EW_TESTS),
    Mutation('ew_clamp_arousal_reads_valence', EW, 'arousal = self._clamp_to_domain(arousal, 0.0, 1.0)', 'arousal = self._clamp_to_domain(valence, 0.0, 1.0)', tests=EW_TESTS),
    Mutation('ew_clamp_low_inclusive', EW, 'if value < low:', 'if value <= low:', tests=EW_TESTS),
    Mutation('ew_clamp_high_inclusive', EW, 'if value > high:', 'if value >= high:', tests=EW_TESTS),
    Mutation('ew_clamp_low_condition_inverted', EW, 'if value < low:', 'if value > low:', tests=EW_TESTS),
    Mutation('ew_clamp_high_condition_inverted', EW, 'if value > high:', 'if value < high:', tests=EW_TESTS),
    Mutation('ew_clamp_low_returns_high', EW, '            return low\n', '            return high\n', tests=EW_TESTS),
    Mutation('ew_clamp_high_returns_low', EW, '            return high\n', '            return low\n', tests=EW_TESTS),
    Mutation('ew_clamp_in_domain_converted', EW, '        return value\n', '        return float(value)\n', tests=EW_TESTS),
    Mutation('ew_clamp_low_branch_dropped', EW, 'if value < low:\n            return low\n        if value > high:', 'if False:\n            return low\n        if value > high:', tests=EW_TESTS),
    Mutation('ew_clamp_block_removed', EW, '        # Limitar ao domínio declarado (valência -1 a +1, arousal 0 a 1)\n        valence = self._clamp_to_domain(valence, -1.0, 1.0)\n        arousal = self._clamp_to_domain(arousal, 0.0, 1.0)\n', '', tests=EW_TESTS),
    Mutation('ew_clamp_skips_agent_valence', EW, 'valence = self._clamp_to_domain(valence, -1.0, 1.0)', "valence = self._clamp_to_domain(valence, -1.0, 1.0) if 'agent_id' not in emotional_context else valence", tests=EW_TESTS),
    Mutation('ew_clamp_skips_agent_arousal', EW, 'arousal = self._clamp_to_domain(arousal, 0.0, 1.0)', "arousal = self._clamp_to_domain(arousal, 0.0, 1.0) if 'agent_id' not in emotional_context else arousal", tests=EW_TESTS),
    Mutation('ew_clamp_ignores_subclass_hook', EW, 'valence = self._clamp_to_domain(valence, -1.0, 1.0)', 'valence = RealEmotionalMemoryWeighting._clamp_to_domain(valence, -1.0, 1.0)', tests=EW_TESTS),
    Mutation('ew_priority_not_multiplied', EW, 'score = base_score * weight.recall_priority', 'score = weight.recall_priority + base_score', tests=EW_TESTS),
    Mutation('ew_query_guard_dropped', EW, "if emotional_query and 'valence' in emotional_query:", 'if emotional_query:', tests=EW_TESTS),
    Mutation('ew_query_ignored', EW, "if emotional_query and 'valence' in emotional_query:", 'if False:', tests=EW_TESTS),
    Mutation('ew_similarity_not_inverted', EW, 'valence_similarity = 1.0 - abs(query_valence - weight.valence)', 'valence_similarity = abs(query_valence - weight.valence)', tests=EW_TESTS),
    Mutation('ew_similarity_signed', EW, '1.0 - abs(query_valence - weight.valence)', '1.0 - (query_valence - weight.valence)', tests=EW_TESTS),
    Mutation('ew_valence_sensitivity_ignored', EW, '(1.0 + valence_similarity * self.valence_sensitivity)', '(1.0 + valence_similarity)', tests=EW_TESTS),
    Mutation('ew_valence_sensitivity_changed', EW, 'self.valence_sensitivity = 0.8', 'self.valence_sensitivity = 0.5', tests=EW_TESTS),
    Mutation('ew_score_added_not_multiplied', EW, 'score *= (1.0 + valence_similarity', 'score += (1.0 + valence_similarity', tests=EW_TESTS),
    Mutation('ew_recall_sorted_ascending', EW, 'scored_memories.sort(key=lambda x: x[1], reverse=True)', 'scored_memories.sort(key=lambda x: x[1])', tests=EW_TESTS),
    Mutation('ew_recall_unsorted', EW, 'scored_memories.sort(key=lambda x: x[1], reverse=True)', 'pass', tests=EW_TESTS),
    Mutation('ew_valence_threshold_strict', EW, 'if valence_diff <= threshold:', 'if valence_diff < threshold:', tests=EW_TESTS),
    Mutation('ew_valence_default_threshold_changed', EW, 'threshold: float = 0.3,', 'threshold: float = 0.4,', tests=EW_TESTS),
    Mutation('ew_valence_default_limit_changed', EW, 'limit: int = 10)', 'limit: int = 5)', tests=EW_TESTS),
    Mutation('ew_valence_similarity_changed', EW, 'similarity = 1.0 - valence_diff', 'similarity = valence_diff', tests=EW_TESTS),
    Mutation('ew_valence_unsorted', EW, 'candidates.sort(key=lambda x: x[1], reverse=True)', 'pass', tests=EW_TESTS),
    Mutation('ew_valence_sorted_ascending', EW, 'candidates.sort(key=lambda x: x[1], reverse=True)', 'candidates.sort(key=lambda x: x[1])', tests=EW_TESTS),
    Mutation('ew_valence_limit_ignored', EW, 'candidates[:limit]', 'candidates', tests=EW_TESTS),
    Mutation('ew_valence_diff_signed', EW, 'valence_diff = abs(target_valence - weight.valence)', 'valence_diff = target_valence - weight.valence', tests=EW_TESTS),
    Mutation('ew_profile_missing_flag_changed', EW, 'return {"has_emotional_weight": False}', 'return {"has_emotional_weight": True}', tests=EW_TESTS),
    Mutation('ew_profile_category_dropped', EW, '"emotional_category": self._categorize_emotion(weight.valence, weight.arousal)', '"emotional_category": "neutral"', tests=EW_TESTS),
    Mutation('ew_profile_intensity_key_changed', EW, '"intensity": weight.emotional_intensity', '"intensity": weight.recall_priority', tests=EW_TESTS),
    Mutation('ew_category_arousal_high_changed', EW, 'if arousal > 0.6:', 'if arousal >= 0.6:', tests=EW_TESTS),
    Mutation('ew_category_arousal_low_changed', EW, 'elif arousal < 0.3:', 'elif arousal <= 0.3:', tests=EW_TESTS),
    Mutation('ew_category_positive_strict_changed', EW, 'if valence > 0.3:\n                return "excited_positive"', 'if valence >= 0.3:\n                return "excited_positive"', tests=EW_TESTS),
    Mutation('ew_category_negative_changed', EW, 'elif valence < -0.3:\n                return "distressed_negative"', 'elif valence < -0.2:\n                return "distressed_negative"', tests=EW_TESTS),
    Mutation('ew_category_high_neutral_renamed', EW, 'return "high_arousal_neutral"', 'return "neutral"', tests=EW_TESTS),
    Mutation('ew_category_calm_swapped', EW, 'return "calm_positive"', 'return "sad_negative"', tests=EW_TESTS),
    Mutation('ew_category_sad_swapped', EW, 'return "sad_negative"', 'return "calm_positive"', tests=EW_TESTS),
    Mutation('ew_category_low_neutral_renamed', EW, 'return "low_arousal_neutral"', 'return "neutral"', tests=EW_TESTS),
    Mutation('ew_category_pleasant_swapped', EW, 'return "pleasant"', 'return "unpleasant"', tests=EW_TESTS),
    Mutation('ew_category_unpleasant_swapped', EW, 'return "unpleasant"', 'return "pleasant"', tests=EW_TESTS),
    Mutation('ew_stats_weighted_dropped', EW, '"weighted_memories": len(self.emotional_weights)', '"weighted_memories": self.total_emotional_memories', tests=EW_TESTS),
    Mutation('ew_stats_high_arousal_dropped', EW, '"high_arousal_count": self.high_arousal_count', '"high_arousal_count": 0', tests=EW_TESTS),
    Mutation('ew_collaborator_not_stored', EW, 'self.memory = memory_system', 'self.memory = None', tests=EW_TESTS),
    Mutation('ew_emotion_not_stored', EW, 'self.emotion = emotion_system', 'self.emotion = None', tests=EW_TESTS),
    Mutation('ew_timestamp_constant', EW, 'timestamp=time.time()', 'timestamp=0.0', tests=EW_TESTS),
    Mutation('ew_banner_dropped', EW, 'print("💗📝 POST 56', 'pass  # print("💗📝 POST 56', tests=EW_TESTS),
    Mutation("ew_module_pulls_monolith", EW, "import math\n",
             "import math\nimport core.constitutional_brain  # noqa: F401\n", tests=EW_TESTS),
    Mutation("ew_monolith_stops_reexporting", "core/constitutional_brain.py",
             "from core.cognitivo.emotional_memory_weighting import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    EmotionalMemoryWeight,\n"
             "    RealEmotionalMemoryWeighting,\n"
             ")\n",
             "from core.cognitivo.emotional_memory_weighting import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    EmotionalMemoryWeight,\n"
             "    RealEmotionalMemoryWeighting as _RealEmotionalMemoryWeighting,\n"
             ")\n"
             "RealEmotionalMemoryWeighting = type(\"RealEmotionalMemoryWeighting\", (_RealEmotionalMemoryWeighting,), {})\n",
             tests=EW_TESTS),
    Mutation("ew_package_stops_reexporting", "core/cognitivo/__init__.py",
             "from core.cognitivo.emotional_memory_weighting import (\n"
             "    EmotionalMemoryWeight,\n"
             "    RealEmotionalMemoryWeighting,\n"
             ")\n",
             "from core.cognitivo.emotional_memory_weighting import (\n"
             "    EmotionalMemoryWeight,\n"
             "    RealEmotionalMemoryWeighting as _RealEmotionalMemoryWeighting,\n"
             ")\n"
             "RealEmotionalMemoryWeighting = type(\"RealEmotionalMemoryWeighting\", (_RealEmotionalMemoryWeighting,), {})\n",
             tests=EW_TESTS),
    # RT1–RT7: CentralRouter fail-closed (segurança e módulos obrigatórios)
    Mutation("rt_block_skipped", BRAIN, '            if blocked is not None:\n', '            if False:\n', tests=RT_TESTS),
    Mutation("rt_detected_ignored", BRAIN, '            if verdict:\n', '            if False:\n', tests=RT_TESTS),
    Mutation("rt_jb_failure_ignored", BRAIN, '            if not result.success:\n                return {"error": "JAILBREAK_CHECK_FAILED"', '            if False:\n                return {"error": "JAILBREAK_CHECK_FAILED"', tests=RT_TESTS),
    Mutation("rt_no_verdict_ignored", BRAIN, '            if verdict is self._NO_VERDICT:\n', '            if False:\n', tests=RT_TESTS),
    Mutation("rt_dict_verdict_ignored", BRAIN, 'verdict = data.get("detected", self._NO_VERDICT)', 'verdict = self._NO_VERDICT', tests=RT_TESTS),
    Mutation("rt_degraded_always_false", BRAIN, 'final_result["degraded"] = bool(failed_modules)', 'final_result["degraded"] = False', tests=RT_TESTS),
    Mutation("rt_failed_modules_empty", BRAIN, 'final_result["failed_modules"] = failed_modules', 'final_result["failed_modules"] = []', tests=RT_TESTS),
    Mutation("rt_degraded_cached", BRAIN, '        if cacheable and not failed_modules:\n', '        if cacheable:\n', tests=RT_TESTS),
    Mutation("rt_failed_names_inverted", BRAIN, 'if not result.success and result.module.value not in names:', 'if result.success and result.module.value not in names:', tests=RT_TESTS),
    Mutation("rt_blocked_success_true", BRAIN, '"success": False,\n            "error": blocked["error"],', '"success": True,\n            "error": blocked["error"],', tests=RT_TESTS),
    Mutation("rt_blocked_not_degraded", BRAIN, '"degraded": True,\n            "failed_modules": [blocked', '"degraded": False,\n            "failed_modules": [blocked', tests=RT_TESTS),
    Mutation("rt_blocked_modules_empty", BRAIN, '"failed_modules": [blocked["module"].value],', '"failed_modules": [],', tests=RT_TESTS),
    Mutation("rt_blocked_prompt_untruncated", BRAIN, '"prompt": prompt[:100],\n            "degraded": True,', '"prompt": prompt,\n            "degraded": True,', tests=RT_TESTS),
    Mutation("rt_blocked_wrong_code", BRAIN, '"error": blocked["error"],\n            "reason": blocked["reason"],', '"error": "ETHICS_VIOLATION",\n            "reason": blocked["reason"],', tests=RT_TESTS),
    # BS1–BS10: orçamento fail-closed e Swarm obrigatório além de ETHICAL_DECISION
    Mutation("rt_budget_denied_ignored", BRAIN, '            if allowed is False:\n', '            if False:\n', tests=BS_TESTS),
    Mutation("rt_swarm_rule_off", BRAIN, '        if swarm_required:\n            swarm_ran = False', '        if False:\n            swarm_ran = False', tests=BS_TESTS),
    Mutation("rt_swarm_rule_all_types", BRAIN, '        if swarm_required:\n            swarm_ran = False', '        if True:\n            swarm_ran = False', tests=BS_TESTS),
    Mutation("rt_swarm_success_inverted", BRAIN, '                swarm_ran = True\n                if not result.success:', '                swarm_ran = True\n                if result.success:', tests=BS_TESTS),
    Mutation("bs_budget_failure_ignored", BRAIN, '            if not result.success:\n                return {"error": "BUDGET_CHECK_FAILED"', '            if False:\n                return {"error": "BUDGET_CHECK_FAILED"', tests=BS_TESTS),
    Mutation("bs_budget_no_verdict_ignored", BRAIN, '            if allowed is not True:\n', '            if False:\n', tests=BS_TESTS),
    Mutation("bs_budget_truthy_allowed", BRAIN, '            if allowed is not True:\n', '            if not allowed:\n', tests=BS_TESTS),
    Mutation("bs_budget_default_allowed", BRAIN, 'result.data.get("allowed", self._NO_VERDICT)', 'result.data.get("allowed", True)', tests=BS_TESTS),
    Mutation("bs_budget_non_dict_allowed", BRAIN, 'if isinstance(result.data, dict) else self._NO_VERDICT\n', 'if isinstance(result.data, dict) else True\n', tests=BS_TESTS),
    Mutation("bs_budget_none_as_denied", BRAIN, '            if allowed is False:\n', '            if allowed is False or allowed is None:\n', tests=BS_TESTS),
    Mutation("bs_budget_reason_text_dropped", BRAIN, 'f"budget check failed: {result.error}"', '"budget check failed"', tests=BS_TESTS),
    Mutation("bs_budget_ran_not_tracked", BRAIN, '            budget_ran = True\n', '            budget_ran = False\n', tests=BS_TESTS),
    Mutation("bs_budget_completeness_off", BRAIN, '        if complete and not budget_ran:\n', '        if False and not budget_ran:\n', tests=BS_TESTS),
    Mutation("bs_budget_completeness_early", BRAIN, '        if complete and not budget_ran:\n', '        if not budget_ran:\n', tests=BS_TESTS),
    Mutation("bs_budget_missing_wrong_module", BRAIN, '"module": ModuleType.BUDGET,\n                    "reason": "budget check was not executed"', '"module": ModuleType.LOG,\n                    "reason": "budget check was not executed"', tests=BS_TESTS),
    Mutation("bs_swarm_ran_not_tracked", BRAIN, '                swarm_ran = True\n', '                swarm_ran = False\n', tests=BS_TESTS),
    Mutation("bs_swarm_completeness_off", BRAIN, '            if complete and not swarm_ran:\n', '            if False and not swarm_ran:\n', tests=BS_TESTS),
    Mutation("bs_swarm_completeness_early", BRAIN, '            if complete and not swarm_ran:\n', '            if not swarm_ran:\n', tests=BS_TESTS),
    Mutation("bs_swarm_missing_wrong_module", BRAIN, '"module": ModuleType.SWARM,\n                        "reason": "swarm is mandatory for this request and was not executed"', '"module": ModuleType.LOG,\n                        "reason": "swarm is mandatory for this request and was not executed"', tests=BS_TESTS),
    Mutation("bs_final_check_dropped", BRAIN, 'swarm_required, complete=True)\n        if blocked is not None:', 'swarm_required, complete=True)\n        if False:', tests=BS_TESTS),
    Mutation("bs_final_check_not_complete", BRAIN, 'swarm_required, complete=True)', 'swarm_required, complete=False)', tests=BS_TESTS),
    Mutation("bs_swarm_flag_not_passed", BRAIN, 'self._blocking_decision(all_results, swarm_required)\n', 'self._blocking_decision(all_results, False)\n', tests=BS_TESTS),
    Mutation("bs_swarm_flag_not_passed_final", BRAIN, 'self._blocking_decision(all_results, swarm_required, complete=True)', 'self._blocking_decision(all_results, False, complete=True)', tests=BS_TESTS),
    Mutation("bs_ethical_floor_dropped", BRAIN, '        if request_type == RequestType.ETHICAL_DECISION:\n            return True\n        hook', '        if False:\n            return True\n        hook', tests=BS_TESTS),
    Mutation("bs_hook_missing_mandatory", BRAIN, '        if not callable(hook):\n            return False', '        if not callable(hook):\n            return True', tests=BS_TESTS),
    Mutation("bs_hook_missing_ignored_check", BRAIN, '        if not callable(hook):\n            return False', '        if False:\n            return False', tests=BS_TESTS),
    Mutation("bs_hook_error_not_mandatory", BRAIN, '        except Exception:\n            return True\n', '        except Exception:\n            return False\n', tests=BS_TESTS),
    Mutation("bs_hook_result_ignored", BRAIN, 'return bool(hook(prompt))', 'return False', tests=BS_TESTS),
    Mutation("bs_hook_result_inverted", BRAIN, 'return bool(hook(prompt))', 'return not hook(prompt)', tests=BS_TESTS),
    Mutation("bs_requires_swarm_case_sensitive", BRAIN, '        lowered = prompt.lower()\n        return any(flag in lowered', '        lowered = prompt\n        return any(flag in lowered', tests=BS_TESTS),
    Mutation("bs_requires_swarm_always", BRAIN, 'return any(flag in lowered for flag in self.ethical_red_flags)', 'return True', tests=BS_TESTS),
    Mutation("bs_requires_swarm_never", BRAIN, 'return any(flag in lowered for flag in self.ethical_red_flags)', 'return False', tests=BS_TESTS),
    Mutation("bs_requires_swarm_needs_all", BRAIN, 'return any(flag in lowered for flag in self.ethical_red_flags)', 'return all(flag in lowered for flag in self.ethical_red_flags)', tests=BS_TESTS),
    Mutation("bs_p4_check_off", BRAIN, 'has_ethical_concerns = self.requires_swarm(prompt)', 'has_ethical_concerns = False', tests=BS_TESTS),
    Mutation("bs_swarm_reason_text_dropped", BRAIN, 'f"swarm is mandatory for this request and failed: {result.error}"', '"swarm is mandatory for this request and failed"', tests=BS_TESTS),
    Mutation('rv_think_drops_error', BRAIN, '"success", "error", "reason", "degraded"', '"success", "reason", "degraded"', tests=RV_TESTS),
    Mutation('rv_think_drops_degraded', BRAIN, '"reason", "degraded", "failed_modules", "failed_modules_detail"]', '"reason", "failed_modules", "failed_modules_detail"]', tests=RV_TESTS),
    Mutation('rv_think_drops_failed_modules', BRAIN, '"degraded", "failed_modules", "failed_modules_detail"]', '"degraded", "failed_modules_detail"]', tests=RV_TESTS),
    Mutation('rv_think_drops_detail', BRAIN, '"degraded", "failed_modules", "failed_modules_detail"]', '"degraded", "failed_modules"]', tests=RV_TESTS),
    Mutation('rv_think_drops_success', BRAIN, 'for field in ["success", "error"', 'for field in ["error"', tests=RV_TESTS),
    Mutation('rv_block_not_logged', BRAIN, '        if log_module is not None:\n            try:\n                log_module.log_event("ROUTER_BLOCKED"', '        if False:\n            try:\n                log_module.log_event("ROUTER_BLOCKED"', tests=RV_TESTS),
    Mutation('rv_block_logged_twice', BRAIN, 'log_module.log_event("ROUTER_BLOCKED", {\n', 'log_module.log_event("ROUTER_BLOCKED", {}); log_module.log_event("ROUTER_BLOCKED", {\n', tests=RV_TESTS),
    Mutation('rv_block_log_failure_propagates', BRAIN, '            except Exception as exc:  # o log não pode', '            except ZeroDivisionError as exc:  # o log não pode', tests=RV_TESTS),
    Mutation('rv_block_log_prompt_untruncated', BRAIN, '"prompt": prompt[:100],\n                })', '"prompt": prompt,\n                })', tests=RV_TESTS),
    Mutation('rv_block_log_wrong_module', BRAIN, '"module": blocked["module"].value,\n                    "request_type"', '"module": blocked["error"],\n                    "request_type"', tests=RV_TESTS),
    Mutation('rv_block_log_wrong_request_type', BRAIN, '"request_type": request_type.value,\n                    "prompt"', '"request_type": "x",\n                    "prompt"', tests=RV_TESTS),
    Mutation('rv_block_stats_not_updated', BRAIN, '        self._update_statistics((time.time() - start_time) * 1000, plan, results)\n        return response', '        return response', tests=RV_TESTS),
    Mutation('rv_block_not_counted', BRAIN, 'self.stats["blocked_requests"] = self.stats.get("blocked_requests", 0) + 1', 'pass', tests=RV_TESTS),
    Mutation('rv_block_cause_not_counted', BRAIN, 'by_error[blocked["error"]] += 1', 'pass', tests=RV_TESTS),
    Mutation('rv_block_stats_not_exposed', BRAIN, '"blocked_requests": self.stats.get("blocked_requests", 0),', '"blocked_requests": 0,', tests=RV_TESTS),
    Mutation('rv_block_cause_not_exposed', BRAIN, '"blocked_by_error": dict(self.stats.get("blocked_by_error", {})),', '"blocked_by_error": {},', tests=RV_TESTS),
    Mutation('rv_detail_blocked_wrong_cause', BRAIN, '"failed_modules_detail": [{"module": blocked["module"].value, "error": blocked["reason"]}],', '"failed_modules_detail": [{"module": blocked["module"].value, "error": blocked["error"]}],', tests=RV_TESTS),
    Mutation('rv_detail_cause_empty', BRAIN, 'details.append({"module": result.module.value, "error": str(result.error)})', 'details.append({"module": result.module.value, "error": ""})', tests=RV_TESTS),
    Mutation('rv_detail_dropped', BRAIN, 'final_result["failed_modules_detail"] = self._failed_module_details(all_results)', 'final_result["failed_modules_detail"] = []', tests=RV_TESTS),
    Mutation('rv_detail_duplicates', BRAIN, 'if not result.success and result.module.value not in seen:\n                seen.append', 'if not result.success:\n                seen.append', tests=RV_TESTS),
    Mutation('rv_detail_includes_success', BRAIN, 'if not result.success and result.module.value not in seen:\n                seen.append', 'if result.module.value not in seen:\n                seen.append', tests=RV_TESTS),
    Mutation('sn_banner_dropped', SN, 'print("🧠🕸️ POST 26', 'pass  # print("🧠🕸️ POST 26', tests=SN_TESTS),
    Mutation('sn_kg_not_stored', SN, 'self.kg = knowledge_graph', 'self.kg = None', tests=SN_TESTS),
    Mutation('sn_language_not_stored', SN, 'self.language = language', 'self.language = None', tests=SN_TESTS),
    Mutation('sn_decay_changed', SN, 'self.activation_decay = 0.1', 'self.activation_decay = 0.2', tests=SN_TESTS),
    Mutation('sn_spread_factor_changed', SN, 'self.spread_factor = 0.7', 'self.spread_factor = 0.8', tests=SN_TESTS),
    Mutation('sn_concept_counter_dropped', SN, 'self.total_concepts += 1', 'self.total_concepts += 0', tests=SN_TESTS),
    Mutation('sn_relation_counter_dropped', SN, 'self.total_relations += 1', 'self.total_relations += 0', tests=SN_TESTS),
    Mutation('sn_activation_counter_dropped', SN, 'self.activations_performed += 1', 'self.activations_performed += 0', tests=SN_TESTS),
    Mutation('sn_properties_default_dropped', SN, 'properties = properties or {}', 'properties = properties', tests=SN_TESTS),
    Mutation('sn_properties_copied_on_add', SN, 'properties = properties or {}', 'properties = dict(properties or {})', tests=SN_TESTS),
    Mutation('sn_concept_id_truncated', SN, 'concept_id=str(uuid.uuid4())', 'concept_id=str(uuid.uuid4())[:8]', tests=SN_TESTS),
    Mutation('sn_relation_id_truncated', SN, 'relation_id=str(uuid.uuid4())', 'relation_id=str(uuid.uuid4())[:8]', tests=SN_TESTS),
    Mutation('sn_created_at_monotonic', SN, 'created_at=time.time()', 'created_at=time.monotonic()', tests=SN_TESTS),
    Mutation('sn_created_at_zero', SN, 'created_at=time.time()', 'created_at=0.0', tests=SN_TESTS),
    Mutation('sn_initial_activation_changed', SN, 'activation=0.0,', 'activation=0.1,', tests=SN_TESTS),
    Mutation('sn_concept_not_stored', SN, 'self.concepts[concept.concept_id] = concept', 'pass', tests=SN_TESTS),
    Mutation('sn_relation_not_stored', SN, 'self.relations.append(relation)', 'pass', tests=SN_TESTS),
    Mutation('sn_relation_strength_ignored', SN, 'strength=strength', 'strength=1.0', tests=SN_TESTS),
    Mutation('sn_relation_type_ignored', SN, 'relation_type=relation_type,', 'relation_type="IS-A",', tests=SN_TESTS),
    Mutation('sn_relation_ends_swapped', SN, 'source_id=source.concept_id,\n            target_id=target.concept_id,', 'source_id=target.concept_id,\n            target_id=source.concept_id,', tests=SN_TESTS),
    Mutation('sn_strength_default_changed', SN, 'strength: float = 1.0', 'strength: float = 0.5', tests=SN_TESTS),
    Mutation('sn_find_case_sensitive', SN, 'if concept.label.lower() == label.lower():\n                return concept', 'if concept.label == label:\n                return concept', tests=SN_TESTS),
    Mutation('sn_find_never_reuses', SN, 'if concept.label.lower() == label.lower():\n                return concept', 'if False:\n                return concept', tests=SN_TESTS),
    Mutation('sn_activation_not_clamped', SN, 'concept.activation = min(1.0, activation)', 'concept.activation = activation', tests=SN_TESTS),
    Mutation('sn_activation_clamp_changed', SN, 'concept.activation = min(1.0, activation)', 'concept.activation = min(0.9, activation)', tests=SN_TESTS),
    Mutation('sn_spread_threshold_inclusive', SN, 'if start[relation.source_id] > 0.1:', 'if start[relation.source_id] >= 0.1:', tests=SN_TESTS),
    Mutation('sn_spread_threshold_raised', SN, 'if start[relation.source_id] > 0.1:', 'if start[relation.source_id] > 0.2:', tests=SN_TESTS),
    Mutation('sn_spread_factor_unused', SN, 'start[relation.source_id] * self.spread_factor * relation.strength', 'start[relation.source_id] * relation.strength', tests=SN_TESTS),
    Mutation('sn_spread_strength_unused', SN, 'start[relation.source_id] * self.spread_factor * relation.strength', 'start[relation.source_id] * self.spread_factor', tests=SN_TESTS),
    Mutation('sn_spread_target_unclamped', SN, 'min(1.0, start[target_id] + sum(sorted(amounts)))', 'start[target_id] + sum(sorted(amounts))', tests=SN_TESTS),
    Mutation('sn_spread_overwrites_target', SN, 'min(1.0, start[target_id] + sum(sorted(amounts)))', 'min(1.0, sum(sorted(amounts)))', tests=SN_TESTS),
    Mutation('sn_spread_decay_unused', SN, 'concept.activation = max(0.0, concept.activation - self.activation_decay)', 'pass', tests=SN_TESTS),
    Mutation('sn_spread_decay_floor_dropped', SN, 'concept.activation = max(0.0, concept.activation - self.activation_decay)', 'concept.activation = concept.activation - self.activation_decay', tests=SN_TESTS),
    Mutation('sn_spread_iterations_ignored', SN, 'for _ in range(iterations):', 'for _ in range(1):', tests=SN_TESTS),
    Mutation('sn_spread_result_inclusive', SN, 'if c.activation > 0.1\n        }', 'if c.activation >= 0.1\n        }', tests=SN_TESTS),
    Mutation('sn_spread_result_keyed_by_id', SN, 'c.label: c.activation', 'c.concept_id: c.activation', tests=SN_TESTS),
    Mutation('sn_related_case_sensitive', SN, 'if c.label.lower() == label.lower():\n                source = c', 'if c.label == label:\n                source = c', tests=SN_TESTS),
    Mutation('sn_related_type_ignored', SN, 'if relation_type is None or relation.relation_type == relation_type:', 'if True:', tests=SN_TESTS),
    Mutation('sn_related_none_matches_nothing', SN, 'if relation_type is None or relation.relation_type == relation_type:', 'if relation_type is not None and relation.relation_type == relation_type:', tests=SN_TESTS),
    Mutation('sn_related_reversed_direction', SN, 'if relation.source_id == source.concept_id:', 'if relation.target_id == source.concept_id:', tests=SN_TESTS),
    Mutation('sn_related_returns_sources', SN, 'related_ids.append(relation.target_id)', 'related_ids.append(relation.source_id)', tests=SN_TESTS),
    Mutation('sn_related_creates_missing', SN, 'if not source:\n            return []', 'if not source:\n            self.add_concept(label)\n            return []', tests=SN_TESTS),
    Mutation('sn_inherit_mutates_own_properties', SN, 'inherited = concept.properties.copy()', 'inherited = concept.properties', tests=SN_TESTS),
    Mutation('sn_inherit_deep_copies', SN, 'inherited = concept.properties.copy()', "inherited = __import__('copy').deepcopy(concept.properties)", tests=SN_TESTS),
    Mutation('sn_inherit_parent_overrides', SN, 'if key not in inherited:', 'if True:', tests=SN_TESTS),
    Mutation('sn_inherit_ignores_parents', SN, 'parent_concepts = self.get_related_concepts(concept_label, "IS-A")', 'parent_concepts = []', tests=SN_TESTS),
    Mutation('sn_inherit_any_relation', SN, 'parent_concepts = self.get_related_concepts(concept_label, "IS-A")', 'parent_concepts = self.get_related_concepts(concept_label)', tests=SN_TESTS),
    Mutation('sn_inherit_not_recursive', SN, 'parent_props = self._inherit(parent.label, path)', 'parent_props = parent.properties', tests=SN_TESTS),
    Mutation('sn_verify_case_sensitive', SN, 'if label.lower() == target:', 'if label == target:', tests=SN_TESTS),
    Mutation('sn_verify_identity_false', SN, 'return True\n\n            # Ciclo IS-A', 'return False\n\n            # Ciclo IS-A', tests=SN_TESTS),
    Mutation('sn_verify_any_relation', SN, 'self.get_related_concepts(label, "IS-A"))', 'self.get_related_concepts(label))', tests=SN_TESTS),
    Mutation('sn_verify_not_recursive', SN, 'pending.extend(parent.label for parent in self.get_related_concepts(label, "IS-A"))', 'pending.extend(parent.label for parent in self.get_related_concepts(label, "IS-A") if label == child_label)', tests=SN_TESTS),
    Mutation('sn_verify_default_true', SN, 'return False\n\n    def query_by_properties', 'return True\n\n    def query_by_properties', tests=SN_TESTS),
    Mutation('sn_query_uses_own_properties', SN, 'all_props = self.inherit_properties(concept.label)', 'all_props = concept.properties', tests=SN_TESTS),
    Mutation('sn_query_value_not_compared', SN, 'if key not in all_props or all_props[key] != value:', 'if key not in all_props:', tests=SN_TESTS),
    Mutation('sn_query_value_equality_inverted', SN, 'all_props[key] != value', 'all_props[key] == value', tests=SN_TESTS),
    Mutation('sn_query_mismatch_ignored', SN, 'match = False\n                    break', 'match = True\n                    break', tests=SN_TESTS),
    Mutation('sn_stats_activated_inclusive', SN, 'if c.activation > 0.1)', 'if c.activation >= 0.1)', tests=SN_TESTS),
    Mutation('sn_stats_activated_threshold_raised', SN, 'if c.activation > 0.1)', 'if c.activation > 0.2)', tests=SN_TESTS),
    Mutation('sn_stats_activated_zero', SN, '"activated_concepts": activated_concepts,', '"activated_concepts": 0,', tests=SN_TESTS),
    Mutation('sn_stats_activation_count_zero', SN, '"activations_performed": self.activations_performed,', '"activations_performed": 0,', tests=SN_TESTS),
    Mutation('sn_stats_avg_floor_two', SN, 'max(1, len(self.concepts))', 'max(2, len(self.concepts))', tests=SN_TESTS),
    Mutation('sn_stats_avg_unguarded', SN, 'max(1, len(self.concepts))', 'len(self.concepts)', tests=SN_TESTS),
    Mutation('sn_stats_avg_only_active', SN, 'sum(c.activation for c in self.concepts.values())', 'sum(c.activation for c in self.concepts.values() if c.activation > 0.1)', tests=SN_TESTS),
    Mutation('sn_spread_snapshot_once', SN, '        for _ in range(iterations):\n            # Síncrono: toda relação lê a ativação do início da iteração\n            start = {cid: c.activation for cid, c in self.concepts.items()}\n', '        start = {cid: c.activation for cid, c in self.concepts.items()}\n        for _ in range(iterations):\n            # Síncrono: toda relação lê a ativação do início da iteração\n', tests=SN_TESTS),
    Mutation('sn_spread_sum_unsorted', SN, 'sum(sorted(amounts))', 'sum(amounts)', tests=SN_TESTS),
    Mutation('sn_spread_sum_keeps_last', SN, 'sum(sorted(amounts))', 'amounts[-1]', tests=SN_TESTS),
    Mutation('sn_spread_sum_keeps_max', SN, 'sum(sorted(amounts))', 'max(amounts)', tests=SN_TESTS),
    Mutation('sn_spread_incoming_overwritten', SN, 'incoming.setdefault(relation.target_id, []).append(spread_amount)', 'incoming[relation.target_id] = [spread_amount]', tests=SN_TESTS),
    Mutation('sn_spread_cap_per_contribution', SN, 'min(1.0, start[target_id] + sum(sorted(amounts)))', 'sum(min(1.0, start[target_id] + a) for a in sorted(amounts)) - (len(amounts) - 1) * start[target_id]', tests=SN_TESTS),
    Mutation('sn_spread_applies_in_place', SN, 'incoming.setdefault(relation.target_id, []).append(spread_amount)', 'target.activation = min(1.0, target.activation + spread_amount)', tests=SN_TESTS),
    Mutation('sn_spread_decay_before_apply', SN, '            # Soma em ordem canônica e limite 1.0 uma vez por alvo\n', '            for concept in self.concepts.values():\n                concept.activation = max(0.0, concept.activation - self.activation_decay)\n            # Soma em ordem canônica e limite 1.0 uma vez por alvo\n', tests=SN_TESTS),
    Mutation('sn_spread_target_from_zero', SN, 'min(1.0, start[target_id] + sum(sorted(amounts)))', 'min(1.0, sum(sorted(amounts)) + 0.0)', tests=SN_TESTS),
    Mutation('sn_inherit_path_not_updated', SN, 'path = path | {concept.concept_id}', 'path = path', tests=SN_TESTS),
    Mutation('sn_inherit_cycle_check_dropped', SN, 'if self._find_or_create_concept(parent.label).concept_id in path:\n                continue', 'if False:\n                continue', tests=SN_TESTS),
    Mutation('sn_inherit_cycle_stops_all_parents', SN, 'concept_id in path:\n                continue', 'concept_id in path:\n                return inherited', tests=SN_TESTS),
    Mutation('sn_inherit_path_reset', SN, 'parent_props = self._inherit(parent.label, path)', 'parent_props = self._inherit(parent.label, frozenset())', tests=SN_TESTS),
    Mutation('sn_inherit_cycle_skip_by_label', SN, 'if self._find_or_create_concept(parent.label).concept_id in path:', 'if parent.label == concept_label:', tests=SN_TESTS),
    Mutation('sn_verify_cycle_returns_true', SN, 'if label.lower() in visited:\n                continue', 'if label.lower() in visited:\n                return True', tests=SN_TESTS),
    Mutation('sn_verify_cycle_returns_false', SN, 'if label.lower() in visited:\n                continue', 'if label.lower() in visited:\n                return False', tests=SN_TESTS),
    Mutation('sn_verify_first_parent_only', SN, 'pending.extend(parent.label for parent in self.get_related_concepts(label, "IS-A"))', 'pending.extend([parent.label for parent in self.get_related_concepts(label, "IS-A")][:1])', tests=SN_TESTS),
    Mutation('sn_verify_target_case_sensitive', SN, 'target = ancestor_label.lower()', 'target = ancestor_label', tests=SN_TESTS),
    Mutation("sn_module_pulls_monolith", SN, "import uuid\n",
             "import uuid\nimport core.constitutional_brain  # noqa: F401\n", tests=SN_TESTS),
    Mutation("sn_monolith_stops_reexporting", "core/constitutional_brain.py",
             "from core.cognitivo.semantic_networks import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    ConceptNode,\n"
             "    RealSemanticNetworksSystem,\n"
             "    SemanticRelation,\n"
             ")\n",
             "from core.cognitivo.semantic_networks import (  # noqa: E402,F401  (strangler: extraído, reexportado aqui)\n"
             "    ConceptNode,\n"
             "    RealSemanticNetworksSystem as _RealSemanticNetworksSystem,\n"
             "    SemanticRelation,\n"
             ")\n"
             "RealSemanticNetworksSystem = type(\"RealSemanticNetworksSystem\", (_RealSemanticNetworksSystem,), {})\n",
             tests=SN_TESTS),
    Mutation("sn_package_stops_reexporting", "core/cognitivo/__init__.py",
             "from core.cognitivo.semantic_networks import (\n"
             "    ConceptNode,\n"
             "    RealSemanticNetworksSystem,\n"
             "    SemanticRelation,\n"
             ")\n",
             "from core.cognitivo.semantic_networks import (\n"
             "    ConceptNode,\n"
             "    RealSemanticNetworksSystem as _RealSemanticNetworksSystem,\n"
             "    SemanticRelation,\n"
             ")\n"
             "RealSemanticNetworksSystem = type(\"RealSemanticNetworksSystem\", (_RealSemanticNetworksSystem,), {})\n",
             tests=SN_TESTS),
    Mutation("ij_reward_from_final_state", MCTS, "STATUS_UTILITY.get(simulated.verified_status, 0)",
             'STATUS_UTILITY.get(simulated.final_state.state.get("system_status"), 0)', tests=IJ_TESTS),
    Mutation("ij_reward_max_of_final_and_verified", MCTS, "utility = STATUS_UTILITY.get(simulated.verified_status, 0)",
             "utility = max(STATUS_UTILITY.get(simulated.verified_status, 0),\n"
             '                          STATUS_UTILITY.get(simulated.final_state.state.get("system_status"), 0))',
             tests=IJ_TESTS),
    Mutation("ij_verified_status_from_final_state", CAUSAL, 'verified_status=verified.get("system_status"),',
             'verified_status=state.state.get("system_status"),', tests=IJ_TESTS),
    Mutation("ij_claims_leak_into_verified", CAUSAL, "            if claims:\n                state.state.update(claims)",
             "            if claims:\n                state.state.update(claims)\n                verified.update(claims)",
             tests=IJ_TESTS),
    Mutation("om_unmodeled_flag_never_set", CAUSAL, "unmodeled = True  # P1a", "pass  # P1a", tests=IJ_TESTS),
    Mutation("om_success_ignores_unmodeled", CAUSAL, " and not unmodeled", "", tests=IJ_TESTS),
    Mutation("om_text_claim_counts_as_modeled", CAUSAL, "if entry is None and not declared:",
             "if entry is None and not (declared or claims):", tests=IJ_TESTS),
    Mutation("om_declared_counts_as_unmodeled", CAUSAL, "if entry is None and not declared:",
             "if entry is None:", tests=IJ_TESTS),
    Mutation("om_only_last_action_checked", CAUSAL, "unmodeled = True  # P1a",
             "unmodeled = action is actions[-1]  # P1a", tests=IJ_TESTS),
    Mutation("om_flag_reset_each_action", CAUSAL, "            entry = ACTION_EFFECTS.get(action)\n",
             "            unmodeled = False\n            entry = ACTION_EFFECTS.get(action)\n", tests=IJ_TESTS),
    Mutation("om_isolated_counts_as_success", CAUSAL, 'in ["secure", "stable"] and not unmodeled',
             'in ["secure", "stable", "isolated"] and not unmodeled', tests=IJ_TESTS),
]


MAX_JOBS = 32
DEFAULT_MAX_JOBS = 8
USAGE = "uso: python tools/cognitivo_mutation.py [--jobs N]  (N: inteiro de 1 a 32; padrão: núcleos, no máximo 8)"


def parse_jobs(argv: list[str]) -> int:
    """Lê `--jobs N`/`--jobs=N`; qualquer outra entrada sai com 2 sem executar nada."""

    def invalid(reason: str) -> SystemExit:
        print(f"entrada inválida: {reason}\n{USAGE}", file=sys.stderr)
        return SystemExit(2)

    jobs = None
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--jobs":
            if jobs is not None:
                raise invalid("--jobs repetido")
            if not args:
                raise invalid("--jobs sem valor")
            value = args.pop(0)
        elif arg.startswith("--jobs="):
            if jobs is not None:
                raise invalid("--jobs repetido")
            value = arg[len("--jobs="):]
        else:
            raise invalid(f"argumento não reconhecido: {arg!r}")
        if re.fullmatch(r"[1-9][0-9]{0,1}", value) is None or not 1 <= int(value) <= MAX_JOBS:
            raise invalid(f"--jobs deve ser inteiro de 1 a {MAX_JOBS}, recebi {value!r}")
        jobs = int(value)
    if jobs is None:
        jobs = max(1, min(os.cpu_count() or 1, DEFAULT_MAX_JOBS))
    return jobs


def evaluate_mutation(workdir: Path, m: Mutation) -> dict:
    """Aplica o mutante na cópia `workdir`, roda seus testes e restaura o arquivo."""
    target = workdir / m.target
    original = target.read_text(encoding="utf-8")
    matches = original.count(m.needle)
    if matches != 1:
        return {"name": m.name, "target": m.target, "status": "not_applied", "matches": matches}
    target.write_text(original.replace(m.needle, m.replacement, 1), encoding="utf-8")
    started = time.time()
    try:
        code, output = run_tests(workdir, m.tests)
    finally:
        target.write_text(original, encoding="utf-8")
    return {
        "name": m.name,
        "target": m.target,
        "tests": list(m.tests),
        "status": "killed" if code != 0 else "survived",
        "exit_code": code,
        "duration_seconds": round(time.time() - started, 3),
        "output_tail": output,
    }


def run_catalog(mutations: list[Mutation], root: Path, jobs: int, skip_baseline: bool = False) -> list[dict]:
    """Roda a linha de base e os mutantes com `jobs` trabalhadores, cada um com a sua cópia de `root`.

    Os resultados voltam na ordem de `mutations`, qualquer que seja a ordem de término.
    """
    workers = max(1, min(jobs, max(len(mutations), 1)))
    with tempfile.TemporaryDirectory(prefix="nexus-cognitivo-mutation-") as scratch:
        free: queue.Queue[Path] = queue.Queue()

        def make_copy(index: int) -> Path:
            workdir = Path(scratch) / f"repo{index}"
            shutil.copytree(root, workdir, ignore=COPY_IGNORE)
            return workdir

        def with_copy(fn, *args):
            workdir = free.get()
            try:
                return fn(workdir, *args)
            finally:
                free.put(workdir)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for workdir in pool.map(make_copy, range(workers)):
                free.put(workdir)
            if not skip_baseline:
                suites = list(dict.fromkeys(m.tests for m in mutations))
                outcomes = list(pool.map(lambda tests: with_copy(run_tests, tests), suites))
                for tests, (code, output) in zip(suites, outcomes):
                    if code != 0:
                        raise SystemExit(f"Baseline failed before mutation ({', '.join(tests)}):\n{output}")
            return list(pool.map(lambda m: with_copy(evaluate_mutation, m), mutations))


def main(argv: list[str] | None = None, *, mutations: list[Mutation] | None = None,
         root: Path = ROOT, report: Path = REPORT) -> int:
    try:
        jobs = parse_jobs(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return int(exc.code)
    catalog = MUTATIONS if mutations is None else mutations
    results = run_catalog(catalog, root, jobs)
    killed = sum(r["status"] == "killed" for r in results)
    summary = {
        "declared": len(catalog),
        "killed": killed,
        "survived": sum(r["status"] == "survived" for r in results),
        "not_applied": sum(r["status"] == "not_applied" for r in results),
    }
    report.write_text(
        json.dumps({"tool": "nexus-cognitivo-mutation", "mutations": results, "summary": summary},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    for r in results:
        if r["status"] != "killed":
            print(f"  {r['status']}: {r['name']} ({r['target']})", file=sys.stderr)
    return 0 if killed == len(catalog) else 1


if __name__ == "__main__":
    raise SystemExit(main())
