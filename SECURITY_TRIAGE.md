# Bandit Security Triage
<!-- NEXUS-CURRENT-STATE
commit: HEAD
ci_run: CURRENT_RUN
tests: 173
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
| `B311` | LOW | `core/constitutional_brain.py` | 1578 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 2390 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 2423 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 7310 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 8375 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 8494 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10618 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10640 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11076 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11419 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11540 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11541 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11675 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 12039 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 12041 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 12250 | `simulation_only_random_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25215 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25290 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25316 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25317 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25325 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25331 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25344 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25353 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25383 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25399 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25400 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25404 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25433 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25434 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25465 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25466 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25546 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25547 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25548 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25623 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25624 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25625 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25626 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25627 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25628 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25738 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25739 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25740 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25741 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25742 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25828 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25829 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25830 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25831 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25832 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25913 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25914 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25915 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25916 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25917 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25918 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25919 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25996 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25997 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25998 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25999 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26000 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26001 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26002 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26078 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26079 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26080 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26081 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26082 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26083 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26146 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26147 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26148 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26149 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26150 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26151 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26221 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26222 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26223 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26224 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26225 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26226 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26227 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26274 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26284 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26285 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26286 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26287 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26307 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26339 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26340 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26341 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26387 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26388 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26389 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26390 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26453 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26454 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26455 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26456 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26507 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26508 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26509 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26510 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26568 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26569 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26570 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26571 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26619 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26620 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26621 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26664 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26665 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26717 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26718 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26719 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26766 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26767 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26768 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26816 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26817 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26818 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26865 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26866 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26867 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26868 | `embedded_demo_assert_review` |
| `B404` | LOW | `tools/state_manifest.py` | 15 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/state_manifest.py` | 29 | `controlled_subprocess_review` |
| `B404` | LOW | `tools/targeted_mutation.py` | 11 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/targeted_mutation.py` | 49 | `controlled_subprocess_review` |
