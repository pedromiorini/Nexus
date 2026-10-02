# Bandit Security Triage

> This is a conservative inventory, not a clearance report. Findings remain open until reviewed and fixed or explicitly justified.

- Total findings: **129**
- Severity counts: LOW=129

## Dispositions

| Disposition | Count | Meaning |
|---|---:|---|
| `controlled_subprocess_review` | 2 | Mutation harness subprocess; keep inputs fixed and review execution boundaries. |
| `embedded_demo_assert_review` | 111 | Assertions inside the legacy __main__ demonstration block; keep them out of production contracts and migrate incrementally. |
| `simulation_only_random_review` | 16 | Randomness reviewed as simulation/heuristic behavior; keep it out of secrets and security decisions. |

## Findings

| ID | Severity | File | Line | Disposition |
|---|---|---|---:|---|
| `B311` | LOW | `core/constitutional_brain.py` | 1378 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 2171 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 2204 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 7044 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 8109 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 8228 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10352 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10374 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10810 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11153 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11274 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11275 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11409 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11773 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11775 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11984 | `simulation_only_random_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24843 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24918 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24944 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24945 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24953 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24959 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24972 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24981 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25011 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25027 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25028 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25032 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25061 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25062 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25093 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25094 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25174 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25175 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25176 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25251 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25252 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25253 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25254 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25255 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25256 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25366 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25367 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25368 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25369 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25370 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25456 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25457 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25458 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25459 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25460 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25541 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25542 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25543 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25544 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25545 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25546 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25547 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25624 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25625 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25626 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25627 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25628 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25629 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25630 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25706 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25707 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25708 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25709 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25710 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25711 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25774 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25775 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25776 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25777 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25778 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25779 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25849 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25850 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25851 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25852 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25853 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25854 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25855 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25902 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25912 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25913 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25914 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25915 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25935 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25967 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25968 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25969 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26015 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26016 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26017 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26018 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26081 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26082 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26083 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26084 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26135 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26136 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26137 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26138 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26196 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26197 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26198 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26199 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26247 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26248 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26249 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26292 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26293 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26345 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26346 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26347 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26394 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26395 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26396 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26444 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26445 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26446 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26493 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26494 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26495 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26496 | `embedded_demo_assert_review` |
| `B404` | LOW | `tools/targeted_mutation.py` | 11 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/targeted_mutation.py` | 49 | `controlled_subprocess_review` |
