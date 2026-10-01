# Bandit Security Triage

> This is a conservative inventory, not a clearance report. Findings remain open until reviewed and fixed or explicitly justified.

- Total findings: **132**
- Severity counts: LOW=132

## Dispositions

| Disposition | Count | Meaning |
|---|---:|---|
| `controlled_subprocess_review` | 2 | Mutation harness subprocess; keep inputs fixed and review execution boundaries. |
| `legacy_demo_assert_review` | 112 | Assertions in legacy/demo paths; review whether they are appropriate and isolated. |
| `non_cryptographic_random_review` | 16 | Non-cryptographic randomness; confirm it is never used for secrets or security decisions. |
| `silent_exception_review` | 2 | Silent exception handling; review whether fallback behavior hides failures. |

## Findings

| ID | Severity | File | Line | Disposition |
|---|---|---|---:|---|
| `B311` | LOW | `core/constitutional_brain.py` | 1378 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 2171 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 2204 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 7044 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 8109 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 8228 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10352 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10374 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10810 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11153 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11274 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11275 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11409 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11773 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11775 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11984 | `non_cryptographic_random_review` |
| `B110` | LOW | `core/constitutional_brain.py` | 20105 | `silent_exception_review` |
| `B110` | LOW | `core/constitutional_brain.py` | 20114 | `silent_exception_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24843 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24888 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24917 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24943 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24944 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24952 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24958 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24971 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24980 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25010 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25026 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25027 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25031 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25060 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25061 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25092 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25093 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25173 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25174 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25175 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25250 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25251 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25252 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25253 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25254 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25255 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25365 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25366 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25367 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25368 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25369 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25455 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25456 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25457 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25458 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25459 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25540 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25541 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25542 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25543 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25544 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25545 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25546 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25623 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25624 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25625 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25626 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25627 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25628 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25629 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25705 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25706 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25707 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25708 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25709 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25710 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25773 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25774 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25775 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25776 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25777 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25778 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25848 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25849 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25850 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25851 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25852 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25853 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25854 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25901 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25911 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25912 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25913 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25914 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25934 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25966 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25967 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25968 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26014 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26015 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26016 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26017 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26080 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26081 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26082 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26083 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26134 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26135 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26136 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26137 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26195 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26196 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26197 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26198 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26246 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26247 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26248 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26291 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26292 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26344 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26345 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26346 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26393 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26394 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26395 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26443 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26444 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26445 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26492 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26493 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26494 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26495 | `legacy_demo_assert_review` |
| `B404` | LOW | `tools/targeted_mutation.py` | 11 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/targeted_mutation.py` | 49 | `controlled_subprocess_review` |
