# Bandit Security Triage

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 303
bandit_low: 131
-->

> This is a conservative inventory, not a clearance report. Findings remain open until reviewed and fixed or explicitly justified.

- Total findings: **131**
- Severity counts: LOW=131

## Dispositions

| Disposition | Count | Meaning |
|---|---:|---|
| `controlled_subprocess_review` | 4 | Mutation harness subprocess; keep inputs fixed and review execution boundaries. |
| `embedded_demo_assert_review` | 111 | Assertions inside the legacy __main__ demonstration block; keep them out of production contracts and migrate incrementally. |
| `simulation_only_random_review` | 16 | Randomness reviewed as simulation/heuristic behavior; keep it out of secrets and security decisions. |

## Findings

Agrupados por escopo (sem números de linha, que mudam a cada edição).

| ID | Severity | File | Scope | Count | Disposition |
|---|---|---|---|---:|---|
| `B101` | LOW | `core/constitutional_brain.py` | `<__main__ block>` | 111 | `embedded_demo_assert_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `AudioProcessor.detect_emotion_from_voice` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `CognitiveFlexibility.adapt_to_context` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `ConceptualBlender.generate_multiple_blends` | 2 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `DivergentThinker.generate_alternatives` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `HyperparameterOptimizer.suggest_next_value` | 2 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `ImitationLearner.imitate` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `InhibitoryControl.inhibit_response` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `QuestionGenerator.generate_followup` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `QuestionGenerator.generate_question` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealCreativitySystem.generate_creative_idea` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealMCTSPlanner._expand` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealMCTSPlanner._simulate` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealMentalSimulationSystem.rollout_planning` | 1 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | `RealRSI.optimize_step` | 1 | `simulation_only_random_review` |
| `B404` | LOW | `tools/state_manifest.py` | `<module>` | 1 | `controlled_subprocess_review` |
| `B404` | LOW | `tools/targeted_mutation.py` | `<module>` | 1 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/state_manifest.py` | `_git` | 1 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/targeted_mutation.py` | `run_tests` | 1 | `controlled_subprocess_review` |
