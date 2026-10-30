"""Benchmark sintético e determinístico de comportamentos observáveis do Nexus.

Este módulo mede interfaces experimentais existentes; não é uma avaliação de
inteligência geral, cognição humana ou causalidade científica no mundo real.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import random
from itertools import product
from pathlib import Path
from typing import Any, Iterable

# O núcleo emite avisos de dependências opcionais no import. Eles não devem
# misturar texto com o JSON que este CLI promete em stdout.
with contextlib.redirect_stdout(io.StringIO()):
    from core.constitutional_brain import (
        RealCausalReasoning,
        RealCounterfactualReasoningEngine,
        RealPredictiveCodingSystem,
        RealUncertaintyQuantificationEngine,
        RealWorldModelSimulator,
    )


BENCHMARK_VERSION = "1.0.0"
SCHEMA_VERSION = "nexus.capability-benchmark.v1"
BASE_COMMIT = "5f53a9922b71d6ef3c565a31630f56091a454671"
DEFAULT_SEED = 20261016
COMPOSITION_ACTIONS = ("APPLY_PATCH", "IGNORE_ALERT", "WAIT")
PROBABILITY_LEVELS = tuple(index / 10 for index in range(1, 10))
UNCERTAINTY_REPETITIONS_PER_LEVEL = 128
POMDP_STATES = ("intact", "compromised")
POMDP_OBSERVATIONS = {
    "clear": {"intact": 0.9, "compromised": 0.2},
    "alarm": {"intact": 0.1, "compromised": 0.8},
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _binary_metrics(predictions: Iterable[bool], labels: Iterable[bool]) -> dict[str, Any]:
    pairs = list(zip(predictions, labels))
    if not pairs:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
        }
    true_positive = sum(prediction and label for prediction, label in pairs)
    false_positive = sum(prediction and not label for prediction, label in pairs)
    true_negative = sum(not prediction and not label for prediction, label in pairs)
    false_negative = sum(not prediction and label for prediction, label in pairs)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return {
        "accuracy": _ratio(true_positive + true_negative, len(pairs)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def _target_state_after(actions: Iterable[str]) -> dict[str, Any]:
    """Especificação de referência independente do simulador sob teste."""
    state = {
        "security_patch_applied": False,
        "alert_level": "high",
        "system_status": "vulnerable",
    }
    for action in actions:
        if action == "APPLY_PATCH":
            state.update(
                security_patch_applied=True,
                alert_level="low",
                system_status="secure",
            )
        elif action == "IGNORE_ALERT":
            state.update(alert_level="critical", system_status="compromised")
        # WAIT é o elemento neutro da composição.
    return state


def _posterior_compromised(observation: str) -> float:
    """Posterior exato no POMDP binário sintético, com prior uniforme."""
    likelihoods = POMDP_OBSERVATIONS[observation]
    joint_compromised = 0.5 * likelihoods["compromised"]
    joint_intact = 0.5 * likelihoods["intact"]
    return joint_compromised / (joint_compromised + joint_intact)


def build_taskset(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Gera tarefas e rótulos fixos; o seed controla apenas os resultados binários."""
    rng = random.Random(seed)

    composition_cases = []
    for length in (1, 2, 3):
        split = "seen_length_1_2" if length <= 2 else "held_out_length_3"
        for sequence in product(COMPOSITION_ACTIONS, repeat=length):
            actions = list(sequence)
            composition_cases.append(
                {
                    "actions": actions,
                    "split": split,
                    "target_state": _target_state_after(actions),
                }
            )

    causal_cases = []
    for index in range(8):
        cause = f"cause_{index:02d}"
        effect = f"effect_{index:02d}"
        causal_cases.append(
            {"text": f"{cause} causes {effect}", "gold_edges": [[cause, effect]], "kind": "positive"}
        )
        causal_cases.append(
            {
                "text": f"{cause} correlates with {effect}",
                "gold_edges": [],
                "kind": "association_control",
            }
        )
        causal_cases.append(
            {
                "text": f"{cause} does not cause {effect}",
                "gold_edges": [],
                "kind": "negation_control",
            }
        )

    intervention_cases = []
    for index in range(8):
        intervention_cases.append(
            {
                "variable": f"connected_{index:02d}",
                "original_value": 0,
                "counterfactual_value": 1,
                "has_descendant": True,
                "kind": "causal_change",
            }
        )
        intervention_cases.append(
            {
                "variable": f"isolated_{index:02d}",
                "original_value": 0,
                "counterfactual_value": 1,
                "has_descendant": False,
                "kind": "disconnected_change",
            }
        )
        intervention_cases.append(
            {
                "variable": f"connected_{index:02d}",
                "original_value": 0,
                "counterfactual_value": 0,
                "has_descendant": False,
                "kind": "no_op",
            }
        )

    probability_cases = []
    for probability in PROBABILITY_LEVELS:
        for repetition in range(UNCERTAINTY_REPETITIONS_PER_LEVEL):
            probability_cases.append(
                {
                    "positive_probability": probability,
                    "outcome": int(rng.random() < probability),
                    "repetition": repetition,
                }
            )

    partial_observation_cases = []
    for observation in POMDP_OBSERVATIONS:
        posterior = _posterior_compromised(observation)
        for repetition in range(4):
            partial_observation_cases.append(
                {
                    "observation": observation,
                    "repetition": repetition,
                    "target_belief": {
                        "intact": 1.0 - posterior,
                        "compromised": posterior,
                    },
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "composition": composition_cases,
        "causal_extraction": causal_cases,
        "causal_intervention": intervention_cases,
        "uncertainty": probability_cases,
        "partial_observation": partial_observation_cases,
    }


def _score_composition(cases: list[dict[str, Any]]) -> dict[str, Any]:
    simulator = RealWorldModelSimulator()
    outcomes: dict[str, list[bool]] = {"seen_length_1_2": [], "held_out_length_3": []}
    for case in cases:
        actual = simulator.simulate(case["actions"]).verified_state
        target = case["target_state"]
        exact_match = all(actual.get(key) == value for key, value in target.items())
        outcomes[case["split"]].append(exact_match)
    all_outcomes = [item for split_results in outcomes.values() for item in split_results]
    return {
        "status": "measured",
        "behavior": "replay exato de transições em sequências compostas",
        "cases": len(all_outcomes),
        "training_examples_used": 0,
        "seen_length_1_2": {
            "cases": len(outcomes["seen_length_1_2"]),
            "exact_match_accuracy": _ratio(sum(outcomes["seen_length_1_2"]), len(outcomes["seen_length_1_2"])),
        },
        "held_out_length_3": {
            "cases": len(outcomes["held_out_length_3"]),
            "exact_match_accuracy": _ratio(sum(outcomes["held_out_length_3"]), len(outcomes["held_out_length_3"])),
        },
        "exact_match_accuracy": _ratio(sum(all_outcomes), len(all_outcomes)),
        "interpretation_limit": (
            "A separação testa consistência de composição em sequências de comprimento 3; "
            "não há treinamento, aquisição de regras nem evidência de generalização aprendida."
        ),
    }


def _score_causal_extraction(cases: list[dict[str, Any]]) -> dict[str, Any]:
    causal = RealCausalReasoning()
    true_positive = false_positive = false_negative = 0
    negative_examples = 0
    negative_false_positives = 0
    for case in cases:
        relations = causal.extract_causal_relations(case["text"])
        predicted_edges = {(relation.cause, relation.effect) for relation in relations}
        gold_edges = {tuple(edge) for edge in case["gold_edges"]}
        true_positive += len(predicted_edges & gold_edges)
        false_positive += len(predicted_edges - gold_edges)
        false_negative += len(gold_edges - predicted_edges)
        if case["kind"] != "positive":
            negative_examples += 1
            negative_false_positives += bool(predicted_edges)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return {
        "status": "measured",
        "behavior": "extração lexical de relações expressas em texto",
        "cases": len(cases),
        "gold_positive_edges": true_positive + false_negative,
        "predicted_edges": true_positive + false_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "negative_control_false_positive_rate": _ratio(negative_false_positives, negative_examples),
        "interpretation_limit": "Não mede descoberta causal, identificação de efeitos ou validade de uma relação no mundo.",
    }


def _score_causal_interventions(cases: list[dict[str, Any]]) -> dict[str, Any]:
    causal = RealCausalReasoning()
    for index in range(8):
        causal.extract_causal_relations(f"connected_{index:02d} causes downstream_{index:02d}")
    engine = RealCounterfactualReasoningEngine(causal_reasoning=causal)
    predictions = []
    labels = []
    for case in cases:
        intervention = engine.model_causal_intervention(
            case["variable"], case["original_value"], case["counterfactual_value"]
        )
        predicts_downstream_effect = any(
            "downstream effects" in effect.lower() for effect in intervention.expected_effects
        )
        predictions.append(predicts_downstream_effect)
        labels.append(case["has_descendant"])
    metrics = _binary_metrics(predictions, labels)
    return {
        "status": "measured",
        "behavior": "alegação textual de efeitos downstream após intervenção",
        "cases": len(cases),
        "metrics": metrics,
        "interpretation_limit": (
            "O resultado mede somente se a API afirma haver efeitos downstream; "
            "não compara valores contrafactuais nem estima o efeito de do(X=x)."
        ),
    }


def _expected_calibration_error(confidences: list[float], correctness: list[bool], bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, correct in zip(confidences, correctness):
        index = min(bins - 1, int(confidence * bins))
        buckets[index].append((confidence, correct))
    total = len(confidences)
    ece = 0.0
    reliability = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        ece += len(bucket) / total * abs(mean_confidence - accuracy)
        reliability.append(
            {
                "bin_lower": index / bins,
                "bin_upper": (index + 1) / bins,
                "cases": len(bucket),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return ece, reliability


def _score_uncertainty(cases: list[dict[str, Any]]) -> dict[str, Any]:
    engine = RealUncertaintyQuantificationEngine()
    brier_sum = 0.0
    baseline_brier_sum = 0.0
    confidences: list[float] = []
    correctness: list[bool] = []
    instance_calibration_scores: list[float] = []
    for case in cases:
        positive_probability = case["positive_probability"]
        prediction = int(positive_probability >= 0.5)
        assessment = engine.assess_confidence(
            prediction=prediction,
            ground_truth=case["outcome"],
            prediction_probabilities=[1.0 - positive_probability, positive_probability],
        )
        confidence = float(assessment.confidence_score)
        # A saída expõe apenas confiança da classe vencedora; recupera-se p(y=1)
        # para pontuar o que foi efetivamente reportado junto com a classe escolhida.
        reported_positive_probability = confidence if prediction == 1 else 1.0 - confidence
        brier_sum += (reported_positive_probability - case["outcome"]) ** 2
        baseline_brier_sum += (0.5 - case["outcome"]) ** 2
        confidences.append(confidence)
        correctness.append(prediction == case["outcome"])
        instance_calibration_scores.append(float(assessment.calibration_score))
    ece, reliability = _expected_calibration_error(confidences, correctness)

    spread_engine = RealUncertaintyQuantificationEngine()
    spread_levels = (0.0, 0.1, 0.2, 0.4)
    spread_estimates = []
    for spread in spread_levels:
        estimate = spread_engine.estimate_uncertainty(
            value=1.0,
            predictions=[1.0 - spread, 1.0, 1.0 + spread],
            uncertainty_type="total",
        )
        spread_estimates.append(float(estimate.uncertainty))
    spread_monotonic = all(
        left <= right for left, right in zip(spread_estimates, spread_estimates[1:])
    )
    return {
        "status": "measured_input_probability_path",
        "behavior": "confiança extraída de probabilidades fornecidas pelo benchmark",
        "cases": len(cases),
        "brier_score": brier_sum / len(cases),
        "expected_calibration_error": ece,
        "baseline_uniform_0_5": {
            "brier_score": baseline_brier_sum / len(cases),
            "expected_calibration_error": abs(0.5 - sum(correctness) / len(correctness)),
        },
        "mean_engine_instance_calibration_score": sum(instance_calibration_scores) / len(instance_calibration_scores),
        "reliability_bins": reliability,
        "probability_vector_input": True,
        "probability_vector_output": False,
        "uncertainty_spread_probe": {
            "input_spread_levels": list(spread_levels),
            "reported_normalized_uncertainty": spread_estimates,
            "nondecreasing_with_spread": spread_monotonic,
        },
        "interpretation_limit": (
            "O benchmark fornece as probabilidades de entrada; avalia apenas a extração "
            "da confiança e o estimador de dispersão, não a geração/calibração de previsões do Nexus. "
            "O campo calibration_score por exemplo não é usado como proper scoring rule."
        ),
    }


def _is_distribution(value: Any, states: tuple[str, ...]) -> bool:
    if not isinstance(value, dict) or not all(state in value for state in states):
        return False
    probabilities = [value[state] for state in states]
    return all(isinstance(item, (int, float)) and 0.0 <= item <= 1.0 for item in probabilities) and abs(sum(probabilities) - 1.0) < 1e-9


def _score_partial_observation(cases: list[dict[str, Any]]) -> dict[str, Any]:
    engine = RealPredictiveCodingSystem()
    outputs = []
    belief_output_count = 0
    for case in cases:
        result = engine.hierarchical_prediction_cycle({"sensory": case["observation"]})
        level_zero_prediction = result["predictions"][-1]
        prediction_value = level_zero_prediction.predicted_value
        outputs.append((str(prediction_value), float(level_zero_prediction.confidence)))
        belief_output_count += _is_distribution(prediction_value, POMDP_STATES)

    comparable_pairs = []
    distinguished_pairs = 0
    for left_index, left in enumerate(cases):
        for right_index in range(left_index + 1, len(cases)):
            right = cases[right_index]
            left_target = left["target_belief"]["compromised"]
            right_target = right["target_belief"]["compromised"]
            if left_target != right_target:
                comparable_pairs.append((left_index, right_index))
                if outputs[left_index] != outputs[right_index]:
                    distinguished_pairs += 1
    return {
        "status": "unsupported_interface",
        "behavior": "tentativa de inferência de estado latente a partir de observações parciais",
        "cases": len(cases),
        "target_state_space": list(POMDP_STATES),
        "distinct_target_posterior_values": sorted(
            {round(case["target_belief"]["compromised"], 6) for case in cases}
        ),
        "belief_distribution_outputs": int(belief_output_count),
        "posterior_accuracy": None,
        "observation_conditioned_pair_rate": _ratio(distinguished_pairs, len(comparable_pairs)),
        "distinct_model_outputs": len(set(outputs)),
        "reason": (
            "A API atual retorna previsões hierárquicas de valor/confiança, mas não uma "
            "distribuição de crença sobre estados ocultos; por isso não há posterior comparável ao alvo."
        ),
    }


def run_benchmark(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    taskset = build_taskset(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        composition = _score_composition(taskset["composition"])
        causal_extraction = _score_causal_extraction(taskset["causal_extraction"])
        causal_interventions = _score_causal_interventions(taskset["causal_intervention"])
        uncertainty = _score_uncertainty(taskset["uncertainty"])
        partial_observation = _score_partial_observation(taskset["partial_observation"])

    digest = hashlib.sha256(_canonical_json(taskset).encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "metadata": {
            "seed": seed,
            "python_version": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
            "base_commit": BASE_COMMIT,
            "source_revision": os.environ.get("GITHUB_SHA") or os.environ.get("CI_COMMIT_SHA") or "unrecorded-local",
            "taskset_sha256": digest,
            "external_network_used": False,
        },
        "tasks": {
            "compositional_generalization": composition,
            "causal_relation_extraction": causal_extraction,
            "causal_intervention_effect_claims": causal_interventions,
            "uncertainty": uncertainty,
            "partial_observability": partial_observation,
        },
        "case_counts": {
            "compositional_generalization": len(taskset["composition"]),
            "causal_relation_extraction": len(taskset["causal_extraction"]),
            "causal_intervention_effect_claims": len(taskset["causal_intervention"]),
            "uncertainty": len(taskset["uncertainty"]),
            "partial_observability": len(taskset["partial_observation"]),
        },
        "global_interpretation_limit": (
            "Resultado de software sintético, específico a estas APIs, tarefas e versão do benchmark; "
            "não valida teorias cognitivas nem sustenta alegações de AGI, consciência ou autonomia geral."
        ),
    }


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed determinística dos resultados simulados.")
    parser.add_argument("--output", type=Path, help="Caminho para gravar o relatório JSON.")
    parser.add_argument("--dump-cases", type=Path, help="Caminho opcional para gravar tarefas e rótulos de referência.")
    parser.add_argument(
        "--require-full-support",
        action="store_true",
        help="Retorna código 2 se alguma trilha não possuir interface mensurável.",
    )
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed deve ser um inteiro não negativo")

    result = run_benchmark(args.seed)
    if args.output:
        _write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    if args.dump_cases:
        _write_json(args.dump_cases, build_taskset(args.seed))
    if args.require_full_support and any(
        task["status"] == "unsupported_interface" for task in result["tasks"].values()
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
