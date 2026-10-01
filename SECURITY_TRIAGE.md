# Bandit Security Triage

> This is a conservative inventory, not a clearance report. Findings remain open until reviewed and fixed or explicitly justified.

- Total findings: **133**
- Severity counts: LOW=132, MEDIUM=1

## Dispositions

| Disposition | Count | Meaning |
|---|---:|---|
| `controlled_subprocess_review` | 2 | Mutation harness subprocess; keep inputs fixed and review execution boundaries. |
| `legacy_demo_assert_review` | 112 | Assertions in legacy/demo paths; review whether they are appropriate and isolated. |
| `non_cryptographic_random_review` | 16 | Non-cryptographic randomness; confirm it is never used for secrets or security decisions. |
| `silent_exception_review` | 2 | Silent exception handling; review whether fallback behavior hides failures. |
| `sql_construction_high_priority_review` | 1 | SQL construction in legacy core; highest-priority manual review for parameterization and trust boundaries. |

## Findings

| ID | Severity | File | Line | Disposition |
|---|---|---|---:|---|
| `B608` | MEDIUM | `core/constitutional_brain.py` | 390 | `sql_construction_high_priority_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 1369 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 2162 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 2195 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 7035 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 8100 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 8219 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10343 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10365 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10801 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11144 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11265 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11266 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11400 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11764 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11766 | `non_cryptographic_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11975 | `non_cryptographic_random_review` |
| `B110` | LOW | `core/constitutional_brain.py` | 20096 | `silent_exception_review` |
| `B110` | LOW | `core/constitutional_brain.py` | 20105 | `silent_exception_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24834 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24879 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24908 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24934 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24935 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24943 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24949 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24962 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 24971 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25001 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25017 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25018 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25022 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25051 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25052 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25083 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25084 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25164 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25165 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25166 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25241 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25242 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25243 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25244 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25245 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25246 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25356 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25357 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25358 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25359 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25360 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25446 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25447 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25448 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25449 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25450 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25531 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25532 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25533 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25534 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25535 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25536 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25537 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25614 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25615 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25616 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25617 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25618 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25619 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25620 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25696 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25697 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25698 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25699 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25700 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25701 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25764 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25765 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25766 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25767 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25768 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25769 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25839 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25840 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25841 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25842 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25843 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25844 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25845 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25892 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25902 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25903 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25904 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25905 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25925 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25957 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25958 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25959 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26005 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26006 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26007 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26008 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26071 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26072 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26073 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26074 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26125 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26126 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26127 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26128 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26186 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26187 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26188 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26189 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26237 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26238 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26239 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26282 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26283 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26335 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26336 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26337 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26384 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26385 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26386 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26434 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26435 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26436 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26483 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26484 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26485 | `legacy_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26486 | `legacy_demo_assert_review` |
| `B404` | LOW | `tools/targeted_mutation.py` | 11 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/targeted_mutation.py` | 49 | `controlled_subprocess_review` |
