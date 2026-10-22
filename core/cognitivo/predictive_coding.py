"""Codificação preditiva (POST 33) extraída de core/constitutional_brain.py.

Extração strangler: o código abaixo foi movido sem alteração de comportamento;
core.constitutional_brain reexporta Prediction, PredictionError e
RealPredictiveCodingSystem no ponto original.
"""

import math
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, List

__all__ = ["Prediction", "PredictionError", "RealPredictiveCodingSystem"]


@dataclass
class Prediction:
    """Predição em um nível hierárquico"""
    prediction_id: str
    level: int  # Nível hierárquico
    predicted_value: Any
    confidence: float  # Precision (inverso da variância)
    timestamp: float

@dataclass
class PredictionError:
    """Erro de predição"""
    error_id: str
    level: int
    predicted: Any
    observed: Any
    error_magnitude: float
    weighted_error: float  # Error * precision

class RealPredictiveCodingSystem:
    """
    POST33: Predictive Coding System — REAL Bayesian Prediction Machine
    
    Codificação Preditiva = cérebro como máquina de predição Bayesiana.
    Constantemente gera predições top-down e minimiza erros bottom-up.
    
    Princípio Central:
    - O cérebro gera predições sobre inputs sensoriais
    - Compara predições com observações reais
    - Propaga erros de predição para cima
    - Atualiza predições para minimizar erro
    - Ponderação por precision (confiança)
    
    Níveis Hierárquicos:
    Level 0: Sensory input (observações)
    Level 1: Feature detection
    Level 2: Object recognition  
    Level 3: Scene understanding
    Level 4: Abstract concepts
    
    Componentes:
    - Generative Model: Modelo gerativo hierárquico
    - Prediction Generation: Gera predições top-down
    - Error Computation: Calcula prediction errors
    - Error Minimization: Minimiza via updates
    - Precision Weighting: Pondera por confiança
    
    Multiplica:
    • × POST13 (WorldModel): Modelo gerativo do mundo
    • × POST12 (Multimodal): Predições multimodais
    • × POST9 (Learning): Aprende de erros
    • × POST25 (Attention): Atenção = precision weighting
    
    Baseado em:
    - Friston's Free Energy Principle (2010)
    - Rao & Ballard's Predictive Coding (1999)
    - Clark's Predictive Processing (2013)
    - Hohwy's Prediction Error Minimization (2013)
    """

    MAX_BELIEF_STATES = 128

    def __init__(self, world_model=None, perception=None, learning=None, attention=None):
        self.world_model = world_model
        self.perception = perception
        self.learning = learning
        self.attention = attention

        # Hierarchical levels (0 = sensory, higher = abstract)
        self.num_levels = 5

        # Predictions per level
        self.predictions: Dict[int, List[Prediction]] = {i: [] for i in range(self.num_levels)}

        # Prediction errors per level
        self.errors: Dict[int, List[PredictionError]] = {i: [] for i in range(self.num_levels)}

        # Learning rate per level
        self.learning_rates = {
            0: 0.1,   # Fast learning at sensory level
            1: 0.08,
            2: 0.05,
            3: 0.03,
            4: 0.01   # Slow learning at abstract level
        }

        # Estatísticas
        self.total_predictions = 0
        self.total_errors_computed = 0
        self.total_updates = 0
        self.avg_error_magnitude = 0.0
        self.generative_model_updates: List[Dict[str, Any]] = []

        print("🧠📊 POST 33 - RealPredictiveCodingSystem initialized (Bayesian prediction machine active)")

    @classmethod
    def update_belief_state(
        cls,
        states: Sequence[str],
        prior: Sequence[float],
        transition_model: Sequence[Sequence[float]],
        observation_likelihoods: Sequence[float],
    ) -> Dict[str, float]:
        """Apply one numerically stable Bayesian update to a supplied POMDP.

        ``transition_model[i][j]`` is P(next=states[j] | current=states[i], action),
        and each observation likelihood is P(observation | next=state, action).
        This pure filter neither learns model parameters nor retains the inputs.
        """
        tolerance = 1e-9
        negative_infinity = float("-inf")

        def is_sequence(value: Any) -> bool:
            return isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            )

        def probability_value(value: Any, name: str) -> float:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must contain real probabilities")
            try:
                probability = float(value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain finite probabilities") from exc
            if not math.isfinite(probability) or probability < 0.0:
                raise ValueError(f"{name} must contain finite non-negative probabilities")
            if probability == 0.0 and value != 0:
                raise ValueError(f"{name} contains a value below float precision")
            if probability > 1.0:
                raise ValueError(f"{name} probabilities must be in [0, 1]")
            return probability

        def logsumexp(values: Sequence[float]) -> float:
            maximum = max(values, default=negative_infinity)
            if maximum == negative_infinity:
                return negative_infinity
            return maximum + math.log(
                math.fsum(
                    math.exp(value - maximum)
                    for value in values
                    if value != negative_infinity
                )
            )

        if not is_sequence(states) or len(states) == 0:
            raise ValueError("states must be a non-empty sequence of unique names")
        state_count = len(states)
        if state_count > cls.MAX_BELIEF_STATES:
            raise ValueError(
                f"state count exceeds the maximum of {cls.MAX_BELIEF_STATES}"
            )
        state_names = tuple(states)
        if any(not isinstance(state, str) or not state.strip() for state in state_names):
            raise ValueError("states must contain non-empty strings")
        if len(set(state_names)) != state_count:
            raise ValueError("states must be unique")

        if not is_sequence(prior) or len(prior) != state_count:
            raise ValueError("prior length must match the number of states")
        prior_values = [probability_value(value, "prior") for value in prior]
        if abs(math.fsum(prior_values) - 1.0) > tolerance:
            raise ValueError("prior must be normalized to 1")

        if not is_sequence(transition_model) or len(transition_model) != state_count:
            raise ValueError("transition_model must be a square matrix matching the states")
        transition_values = []
        for row in transition_model:
            if not is_sequence(row) or len(row) != state_count:
                raise ValueError("transition_model must be a square matrix matching the states")
            probabilities = [probability_value(value, "transition_model") for value in row]
            if abs(math.fsum(probabilities) - 1.0) > tolerance:
                raise ValueError("each transition_model row must be normalized to 1")
            transition_values.append(probabilities)

        if not is_sequence(observation_likelihoods) or len(observation_likelihoods) != state_count:
            raise ValueError("observation_likelihoods length must match the number of states")
        likelihood_values = [
            probability_value(value, "observation_likelihoods")
            for value in observation_likelihoods
        ]

        log_prior = [math.log(value) if value > 0.0 else negative_infinity for value in prior_values]
        log_transition = [
            [math.log(value) if value > 0.0 else negative_infinity for value in row]
            for row in transition_values
        ]
        log_likelihoods = [
            math.log(value) if value > 0.0 else negative_infinity
            for value in likelihood_values
        ]
        predicted_log_probabilities = [
            logsumexp(
                [
                    log_prior[i] + log_transition[i][j]
                    for i in range(state_count)
                    if prior_values[i] > 0.0 and transition_values[i][j] > 0.0
                ]
            )
            for j in range(state_count)
        ]
        unnormalized_log_weights = [
            predicted_log_probabilities[j] + log_likelihoods[j]
            if predicted_log_probabilities[j] != negative_infinity
            and log_likelihoods[j] != negative_infinity
            else negative_infinity
            for j in range(state_count)
        ]
        log_evidence = logsumexp(unnormalized_log_weights)
        if not math.isfinite(log_evidence):
            raise ValueError(
                "observation has zero probability under the supplied belief model"
            )

        posterior = [
            math.exp(log_weight - log_evidence)
            if log_weight != negative_infinity
            else 0.0
            for log_weight in unnormalized_log_weights
        ]
        return dict(zip(state_names, posterior, strict=True))

    def generate_prediction(self, level: int, context: Dict[str, Any]) -> Prediction:
        """
        Gerar predição top-down em um nível.
        
        Higher levels predict lower levels.
        
        Args:
            level: Nível hierárquico
            context: Contexto para predição
        
        Returns:
            Prediction
        """
        # Usar world model se disponível para predições
        if self.world_model and level >= 2:
            predicted_value = self._generate_abstract_prediction(context)
        else:
            predicted_value = self._generate_sensory_prediction(context, level)

        # Confidence baseado no nível (higher = more confident in abstractions)
        base_confidence = 0.5 + (level * 0.1)
        confidence = min(1.0, base_confidence)

        prediction = Prediction(
            prediction_id=str(uuid.uuid4()),
            level=level,
            predicted_value=predicted_value,
            confidence=confidence,
            timestamp=time.time()
        )

        self.predictions[level].append(prediction)
        self.total_predictions += 1

        return prediction

    def _generate_abstract_prediction(self, context: Dict) -> str:
        """Gerar predição abstrata (níveis altos)"""
        # Heurísticas para predições abstratas
        if "object" in str(context).lower():
            return "expected_object_category"
        if "scene" in str(context).lower():
            return "expected_scene_type"

        return "abstract_concept"

    def _generate_sensory_prediction(self, context: Dict, level: int) -> str:
        """Gerar predição sensorial (níveis baixos)"""
        if level == 0:
            return "sensory_features"
        elif level == 1:
            return "detected_features"

        return "intermediate_representation"

    def compute_prediction_error(self, prediction: Prediction, 
                                 observation: Any) -> PredictionError:
        """
        Computar erro de predição.
        
        Error = Observation - Prediction
        
        Args:
            prediction: Predição feita
            observation: Observação real
        
        Returns:
            PredictionError
        """
        # Calcular magnitude do erro (simplificado)
        if str(prediction.predicted_value) == str(observation):
            error_magnitude = 0.0
        else:
            # Erro proporcional à diferença
            error_magnitude = 1.0

        # Ponderar por precision (confidence)
        weighted_error = error_magnitude * prediction.confidence

        error = PredictionError(
            error_id=str(uuid.uuid4()),
            level=prediction.level,
            predicted=prediction.predicted_value,
            observed=observation,
            error_magnitude=error_magnitude,
            weighted_error=weighted_error
        )

        self.errors[prediction.level].append(error)
        self.total_errors_computed += 1

        # Update running average
        self.avg_error_magnitude = (
            (self.avg_error_magnitude * (self.total_errors_computed - 1) + error_magnitude) 
            / self.total_errors_computed
        )

        return error

    def minimize_prediction_error(self, error: PredictionError) -> bool:
        """
        Minimizar erro de predição via atualização.
        
        Two routes:
        1. Update prediction (perceptual inference)
        2. Update world/action (active inference)
        
        Args:
            error: Erro a minimizar
        
        Returns:
            Success
        """
        level = error.level
        learning_rate = self.learning_rates.get(level, 0.05)

        # Update predictions at this level
        if error.weighted_error > 0.1:  # Significant error
            # Propagate error up hierarchy
            if level < self.num_levels - 1:
                self._propagate_error_up(error, level + 1)

            # Update internal model
            self._update_generative_model(error, learning_rate)

            self.total_updates += 1
            return True

        return False

    def _propagate_error_up(self, error: PredictionError, target_level: int):
        """Propagar erro para níveis superiores"""
        # Níveis superiores ajustam predições baseado em erros de níveis inferiores
        if target_level not in self.errors:
            raise ValueError(f"target_level inválido: {target_level}")
        propagated = PredictionError(
            error_id=str(uuid.uuid4()),
            level=target_level,
            predicted=error.predicted,
            observed=error.observed,
            error_magnitude=error.error_magnitude,
            weighted_error=error.weighted_error,
        )
        self.errors[target_level].append(propagated)
        return propagated

    def _update_generative_model(self, error: PredictionError, learning_rate: float):
        """Atualizar modelo gerativo baseado em erro"""
        # O núcleo não conhece um contrato comum de ``learning``. Registra a
        # atualização proposta para manter a evidência observável sem afirmar
        # que um modelo externo foi realmente treinado.
        update = {
            "error_id": error.error_id,
            "level": error.level,
            "learning_rate": learning_rate,
            "weighted_error": error.weighted_error,
            "learning_available": self.learning is not None,
            "applied": False,
        }
        self.generative_model_updates.append(update)
        return update

    def precision_weighting(self, error: PredictionError, 
                           attention_weight: float = 1.0) -> float:
        """
        Ponderar erro por precision.
        
        Precision = 1 / variance = confidence
        Attention modulates precision.
        
        Args:
            error: Erro a ponderar
            attention_weight: Peso atencional (0-1)
        
        Returns:
            Weighted error
        """
        # Precision comes from prediction confidence
        # Attention amplifies precision
        effective_precision = attention_weight

        weighted = error.error_magnitude * effective_precision

        return weighted

    def hierarchical_prediction_cycle(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ciclo completo de predição hierárquica.
        
        1. Generate predictions top-down
        2. Compare with observations
        3. Compute errors
        4. Minimize errors via updates
        
        Args:
            observation: Observação sensorial
        
        Returns:
            Resultado do ciclo
        """
        results = {
            "predictions": [],
            "errors": [],
            "updates": []
        }

        # Generate predictions at each level (top-down)
        for level in reversed(range(self.num_levels)):
            pred = self.generate_prediction(level, observation)
            results["predictions"].append(pred)

        # Compute errors at sensory level (bottom-up)
        sensory_pred = results["predictions"][-1]  # Level 0
        error = self.compute_prediction_error(sensory_pred, observation.get("sensory", "input"))
        results["errors"].append(error)

        # Minimize errors
        if self.minimize_prediction_error(error):
            results["updates"].append(f"Updated level {error.level}")

        return results

    def get_statistics(self) -> Dict:
        """Estatísticas do Predictive Coding System"""
        return {
            "total_predictions": self.total_predictions,
            "total_errors_computed": self.total_errors_computed,
            "total_updates": self.total_updates,
            "avg_error_magnitude": self.avg_error_magnitude,
            "num_hierarchical_levels": self.num_levels
        }
