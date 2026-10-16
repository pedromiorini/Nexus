"""Benchmark determinístico do filtro de crença POMDP de um passo (protocolo v2).

Mede somente a atualização de um belief state com modelo explícito fornecido;
não mede aprendizagem de parâmetros, descoberta de estados ou planejamento.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

with contextlib.redirect_stdout(io.StringIO()):
    from core.constitutional_brain import RealPredictiveCodingSystem


BENCHMARK_VERSION = "2.0.0"
SCHEMA_VERSION = "nexus.pomdp-belief-benchmark.v2"
TASKSET_SCHEMA_VERSION = "nexus.pomdp-belief-taskset.v2"
DEFAULT_SEED = 20261016
DEFAULT_CASE_COUNT = 128
MIN_STATE_COUNT = 2
MAX_STATE_COUNT = 8
POSTERIOR_TOLERANCE = 1e-12


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _simplex(rng: random.Random, count: int) -> list[float]:
    weights = [rng.random() + 0.001 for _ in range(count)]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def _reference_update(
    prior: list[float],
    transition_model: list[list[float]],
    observation_likelihoods: list[float],
) -> list[float]:
    """Independent direct reference implementation for generated valid cases."""
    state_count = len(prior)
    predicted = [
        math.fsum(prior[i] * transition_model[i][j] for i in range(state_count))
        for j in range(state_count)
    ]
    weights = [predicted[j] * observation_likelihoods[j] for j in range(state_count)]
    evidence = math.fsum(weights)
    if evidence <= 0.0:
        raise ValueError("task generator produced a zero-evidence case")
    return [weight / evidence for weight in weights]


def _rejection_controls() -> list[dict[str, Any]]:
    identity = [[1.0, 0.0], [0.0, 1.0]]
    return [
        {
            "case_id": "zero_evidence",
            "states": ["a", "b"],
            "prior": [1.0, 0.0],
            "transition_model": identity,
            "observation_likelihoods": [0.0, 1.0],
        },
        {
            "case_id": "unnormalized_prior",
            "states": ["a", "b"],
            "prior": [0.4, 0.4],
            "transition_model": identity,
            "observation_likelihoods": [0.5, 0.5],
        },
        {
            "case_id": "non_stochastic_transition_row",
            "states": ["a", "b"],
            "prior": [0.5, 0.5],
            "transition_model": [[0.6, 0.6], [0.0, 1.0]],
            "observation_likelihoods": [0.5, 0.5],
        },
        {
            "case_id": "likelihood_out_of_range",
            "states": ["a", "b"],
            "prior": [0.5, 0.5],
            "transition_model": identity,
            "observation_likelihoods": [1.1, 0.5],
        },
    ]


def build_taskset(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = random.Random(seed)
    cases = []
    for case_index in range(DEFAULT_CASE_COUNT):
        state_count = MIN_STATE_COUNT + (case_index % (MAX_STATE_COUNT - MIN_STATE_COUNT + 1))
        states = [f"state_{index}" for index in range(state_count)]
        prior = _simplex(rng, state_count)
        transition_model = [_simplex(rng, state_count) for _ in range(state_count)]
        likelihoods = [0.02 + 0.98 * rng.random() for _ in range(state_count)]
        for index in range(state_count):
            if rng.random() < 0.2:
                likelihoods[index] = 0.0
        if not any(likelihoods):
            likelihoods[case_index % state_count] = 0.02
        target = _reference_update(prior, transition_model, likelihoods)
        cases.append(
            {
                "case_id": f"belief_{case_index:03d}",
                "states": states,
                "prior": prior,
                "transition_model": transition_model,
                "observation_likelihoods": likelihoods,
                "target_posterior": target,
            }
        )
    return {
        "schema_version": TASKSET_SCHEMA_VERSION,
        "seed": seed,
        "cases": cases,
        "rejection_controls": _rejection_controls(),
    }


def _score_taskset(taskset: dict[str, Any]) -> dict[str, Any]:
    absolute_errors: list[float] = []
    max_case_errors: list[float] = []
    state_counts: dict[str, int] = {}
    for case in taskset["cases"]:
        result = RealPredictiveCodingSystem.update_belief_state(
            case["states"],
            case["prior"],
            case["transition_model"],
            case["observation_likelihoods"],
        )
        targets = case["target_posterior"]
        errors = [abs(result[state] - targets[index]) for index, state in enumerate(case["states"])]
        if set(result) != set(case["states"]):
            raise AssertionError(f"unexpected posterior keys for {case['case_id']}")
        absolute_errors.extend(errors)
        max_case_errors.append(max(errors))
        state_counts[str(len(case["states"]))] = state_counts.get(str(len(case["states"])), 0) + 1

    controls_rejected = 0
    controls_accepted = 0
    for control in taskset["rejection_controls"]:
        try:
            RealPredictiveCodingSystem.update_belief_state(
                control["states"],
                control["prior"],
                control["transition_model"],
                control["observation_likelihoods"],
            )
        except ValueError:
            controls_rejected += 1
        else:
            controls_accepted += 1

    case_count = len(taskset["cases"])
    control_count = len(taskset["rejection_controls"])
    maximum_error = max(max_case_errors, default=0.0)
    within_tolerance_cases = sum(
        error <= POSTERIOR_TOLERANCE for error in max_case_errors
    )
    posterior_gate_passed = case_count > 0 and within_tolerance_cases == case_count
    controls_gate_passed = (
        control_count > 0
        and controls_rejected == control_count
        and controls_accepted == 0
    )
    failed_conditions = []
    if not posterior_gate_passed:
        failed_conditions.append("posterior_cases_outside_tolerance")
    if not controls_gate_passed:
        failed_conditions.append("rejection_controls_not_all_rejected")
    return {
        "status": "measured",
        "behavior": "one-step Bayesian belief update under an explicit finite POMDP model",
        "cases": case_count,
        "state_probability_estimates": len(absolute_errors),
        "state_cardinality_counts": state_counts,
        "mean_absolute_error": math.fsum(absolute_errors) / max(1, len(absolute_errors)),
        "maximum_absolute_error": maximum_error,
        "within_tolerance_cases": within_tolerance_cases,
        "within_tolerance_case_rate": within_tolerance_cases / max(1, case_count),
        "posterior_tolerance": POSTERIOR_TOLERANCE,
        "rejection_controls": {
            "cases": control_count,
            "rejected": controls_rejected,
            "accepted": controls_accepted,
        },
        "quality_gate": {
            "passed": not failed_conditions,
            "posterior_cases_passed": posterior_gate_passed,
            "rejection_controls_passed": controls_gate_passed,
            "failed_conditions": failed_conditions,
        },
        "training_examples_used": 0,
        "model_parameters_provided": True,
        "interpretation_limit": (
            "Mede somente uma atualização de crença com estados, prior, transições e likelihoods fornecidos. "
            "Não mede aprendizagem do modelo, descoberta de estados, planejamento de política ou desempenho no mundo real."
        ),
    }


def run_benchmark(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    taskset = build_taskset(seed)
    taskset_digest = hashlib.sha256(_canonical_json(taskset).encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "metadata": {
            "seed": seed,
            "python_version": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
            "source_revision": os.environ.get("GITHUB_SHA")
            or os.environ.get("CI_COMMIT_SHA")
            or "unrecorded-local",
            "taskset_sha256": taskset_digest,
            "external_network_used": False,
        },
        "tasks": {"belief_state_update": _score_taskset(taskset)},
        "case_counts": {
            "belief_state_update": len(taskset["cases"]),
            "rejection_controls": len(taskset["rejection_controls"]),
        },
        "global_interpretation_limit": (
            "Benchmark sintético e versionado de um filtro bayesiano com modelo explícito; "
            "não valida aprendizado, planejamento ou alegações cognitivas gerais."
        ),
    }


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump-cases", type=Path)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed deve ser um inteiro não negativo")
    report = run_benchmark(args.seed)
    if args.output:
        _write_json(args.output, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    if args.dump_cases:
        _write_json(args.dump_cases, build_taskset(args.seed))
    quality_gate = report["tasks"]["belief_state_update"]["quality_gate"]
    if not quality_gate["passed"]:
        print(
            "POMDP v2 quality gate failed: " + ", ".join(quality_gate["failed_conditions"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
