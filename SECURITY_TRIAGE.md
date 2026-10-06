# Bandit Security Triage

<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 116
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

| ID | Severity | File | Line | Disposition |
|---|---|---|---:|---|
| `B311` | LOW | `core/constitutional_brain.py` | 1401 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 2211 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 2244 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 7131 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 8196 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 8315 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10439 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10461 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10897 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11240 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11361 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11362 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11496 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11860 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11862 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 12071 | `simulation_only_random_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25030 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25105 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25131 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25132 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25140 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25146 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25159 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25168 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25198 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25214 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25215 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25219 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25248 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25249 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25280 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25281 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25361 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25362 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25363 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25438 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25439 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25440 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25441 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25442 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25443 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25553 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25554 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25555 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25556 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25557 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25643 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25644 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25645 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25646 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25647 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25728 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25729 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25730 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25731 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25732 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25733 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25734 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25811 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25812 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25813 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25814 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25815 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25816 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25817 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25893 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25894 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25895 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25896 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25897 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25898 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25961 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25962 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25963 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25964 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25965 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25966 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26036 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26037 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26038 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26039 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26040 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26041 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26042 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26089 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26099 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26100 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26101 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26102 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26122 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26154 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26155 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26156 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26202 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26203 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26204 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26205 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26268 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26269 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26270 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26271 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26322 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26323 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26324 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26325 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26383 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26384 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26385 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26386 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26434 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26435 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26436 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26479 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26480 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26532 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26533 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26534 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26581 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26582 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26583 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26631 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26632 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26633 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26680 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26681 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26682 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26683 | `embedded_demo_assert_review` |
| `B404` | LOW | `tools/state_manifest.py` | 14 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/state_manifest.py` | 27 | `controlled_subprocess_review` |
| `B404` | LOW | `tools/targeted_mutation.py` | 11 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/targeted_mutation.py` | 49 | `controlled_subprocess_review` |
