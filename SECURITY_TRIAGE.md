# Bandit Security Triage

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
| `B311` | LOW | `core/constitutional_brain.py` | 2199 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 2232 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 7119 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 8184 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 8303 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10427 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10449 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 10885 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11228 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11349 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11350 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11484 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11848 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 11850 | `simulation_only_random_review` |
| `B311` | LOW | `core/constitutional_brain.py` | 12059 | `simulation_only_random_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25018 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25093 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25119 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25120 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25128 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25134 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25147 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25156 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25186 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25202 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25203 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25207 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25236 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25237 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25268 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25269 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25349 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25350 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25351 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25426 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25427 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25428 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25429 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25430 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25431 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25541 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25542 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25543 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25544 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25545 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25631 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25632 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25633 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25634 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25635 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25716 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25717 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25718 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25719 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25720 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25721 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25722 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25799 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25800 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25801 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25802 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25803 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25804 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25805 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25881 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25882 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25883 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25884 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25885 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25886 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25949 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25950 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25951 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25952 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25953 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 25954 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26024 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26025 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26026 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26027 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26028 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26029 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26030 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26077 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26087 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26088 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26089 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26090 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26110 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26142 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26143 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26144 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26190 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26191 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26192 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26193 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26256 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26257 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26258 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26259 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26310 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26311 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26312 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26313 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26371 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26372 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26373 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26374 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26422 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26423 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26424 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26467 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26468 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26520 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26521 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26522 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26569 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26570 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26571 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26619 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26620 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26621 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26668 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26669 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26670 | `embedded_demo_assert_review` |
| `B101` | LOW | `core/constitutional_brain.py` | 26671 | `embedded_demo_assert_review` |
| `B404` | LOW | `tools/state_manifest.py` | 14 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/state_manifest.py` | 27 | `controlled_subprocess_review` |
| `B404` | LOW | `tools/targeted_mutation.py` | 11 | `controlled_subprocess_review` |
| `B603` | LOW | `tools/targeted_mutation.py` | 49 | `controlled_subprocess_review` |
