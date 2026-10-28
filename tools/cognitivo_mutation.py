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
CF_TESTS = ("test_counterfactual_extraction_contract.py", "test_counterfactual_contract.py",
            "test_counterfactual_gap3_contract.py",
            "test_counterfactual_gap4_contract.py",
            "test_counterfactual_planner_contract.py",
            "test_counterfactual_dox_contract.py")
CAUSAL_TESTS = ("test_causal_world_model_extraction_contract.py",)

MCTS_QUALITY = ("test_mcts_planner_quality_contract.py",)

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
