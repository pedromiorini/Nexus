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
_GUARD = '"""Buscar episódios"""\n        if isinstance(limit, bool) or not isinstance(limit, int):\n            raise TypeError("limit must be an integer")'

MUTATIONS = [
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
