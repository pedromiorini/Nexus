"""Mental simulation (POST30) extraído de core/constitutional_brain.py.

Extração strangler: o código abaixo foi movido sem alteração de comportamento;
core.constitutional_brain reexporta as classes no ponto original.
Não há relógio injetável: rollout_planning usa random.choices no módulo global
e as simulações não leem tempo. world_model é a única colaboração lida
(predict_next_state); planner, reasoning e causal só são armazenados.
"""

import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

__all__ = [
    "MentalScenario",
    "MentalSimulationResult",
    "RealMentalSimulationSystem",
]


# ═══════════════════════════════════════════════════════════════════════════
# MÓDULO 30: MENTAL SIMULATION SYSTEM (POST 30)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MentalScenario:
    """Cenário para simulação mental"""
    scenario_id: str
    initial_state: Dict[str, Any]
    actions: List[str]
    constraints: List[str]
    goal: str

@dataclass
class MentalSimulationResult:
    """Resultado de simulação mental"""
    scenario_id: str
    final_state: Dict[str, Any]
    success: bool
    outcome_quality: float  # 0-1
    steps_taken: int
    predicted_utility: float
    risks: List[str]
    opportunities: List[str]

class RealMentalSimulationSystem:
    """
    POST30: Mental Simulation System — REAL Internal Scenario Testing
    
    Simulação Mental = rodar cenários mentalmente antes de executá-los.
    "Think before you act" - teste virtual de ações.
    
    Capacidades:
    - Simulate Scenarios: Simular cenários completos
    - Predict Outcomes: Prever resultados de ações
    - Rollout Planning: Planejamento via rollouts mentais
    - Risk Assessment: Avaliar riscos antes de agir
    - Counterfactual Testing: "E se eu tivesse feito X?"
    
    Multiplica:
    • × POST13 (WorldModel): Usa modelo do mundo para simular
    • × POST5 (Planning): Planning via simulação
    • × POST2 (Reasoning): Raciocínio sobre simulações
    • × POST11 (Causal): Causalidade em simulações
    
    Baseado em:
    - Craik's Internal Model Theory (1943)
    - Tolman's Cognitive Maps (1948)
    - Sutton & Barto's Model-Based RL (1998)
    - Doll et al.'s Mental Simulation (2015)
    """

    def __init__(self, world_model=None, planner=None, reasoning=None, causal=None):
        self.world_model = world_model
        self.planner = planner
        self.reasoning = reasoning
        self.causal = causal

        # Cache de simulações
        self.simulation_cache: Dict[str, MentalSimulationResult] = {}

        # Estatísticas
        self.total_simulations = 0
        self.successful_predictions = 0
        self.simulation_accuracy = 0.0

        print("🧠🎬 POST 30 - RealMentalSimulationSystem initialized (Internal scenario testing active)")

    def simulate_scenario(self, scenario: MentalScenario) -> MentalSimulationResult:
        """
        Simular cenário mentalmente.
        
        Args:
            scenario: Cenário a simular
        
        Returns:
            MentalSimulationResult
        """
        # Usar world model para simular se disponível
        current_state = scenario.initial_state.copy()
        steps = 0
        success = False
        risks = []
        opportunities = []

        # Simular cada ação
        for action in scenario.actions:
            steps += 1

            # Prever próximo estado
            next_state = self._predict_next_state(current_state, action)

            # Avaliar riscos/oportunidades
            action_risks = self._assess_risks(current_state, action, next_state)
            action_opps = self._identify_opportunities(current_state, action, next_state)

            risks.extend(action_risks)
            opportunities.extend(action_opps)

            # Atualizar estado
            current_state = next_state

            # Checar se atingiu goal
            if self._check_goal_achieved(current_state, scenario.goal):
                success = True
                break

        # Avaliar qualidade do outcome
        outcome_quality = self._evaluate_outcome(current_state, scenario.goal, success)

        # Utility previsto
        predicted_utility = outcome_quality * (1.0 if success else 0.5)

        result = MentalSimulationResult(
            scenario_id=scenario.scenario_id,
            final_state=current_state,
            success=success,
            outcome_quality=outcome_quality,
            steps_taken=steps,
            predicted_utility=predicted_utility,
            risks=risks,
            opportunities=opportunities
        )

        # Cache result
        self.simulation_cache[scenario.scenario_id] = result
        self.total_simulations += 1

        return result

    def _predict_next_state(self, state: Dict, action: str) -> Dict:
        """Prever próximo estado após ação"""
        # Usar world model se disponível
        if self.world_model and hasattr(self.world_model, 'predict_next_state'):
            return self.world_model.predict_next_state(state, action)

        # Simulação simplificada
        next_state = state.copy()

        # Heurísticas básicas
        if "move" in action.lower():
            next_state["position"] = next_state.get("position", 0) + 1
        if "acquire" in action.lower():
            next_state["resources"] = next_state.get("resources", 0) + 1
        if "build" in action.lower():
            next_state["structures"] = next_state.get("structures", 0) + 1

        return next_state

    def _assess_risks(self, state: Dict, action: str, next_state: Dict) -> List[str]:
        """Avaliar riscos de ação"""
        risks = []

        # Risk heuristics
        if "move" in action.lower() and state.get("safety", 1.0) < 0.5:
            risks.append("unsafe_movement")

        if "acquire" in action.lower() and state.get("resources", 0) > 10:
            risks.append("resource_overflow")

        if next_state.get("position", 0) < 0:
            risks.append("invalid_position")

        return risks

    def _identify_opportunities(self, state: Dict, action: str, next_state: Dict) -> List[str]:
        """Identificar oportunidades"""
        opps = []

        # Opportunity heuristics
        if next_state.get("resources", 0) > state.get("resources", 0):
            opps.append("resource_gain")

        if next_state.get("structures", 0) > state.get("structures", 0):
            opps.append("infrastructure_growth")

        return opps

    def _check_goal_achieved(self, state: Dict, goal: str) -> bool:
        """Checar se goal foi atingido"""
        # Simplificado: match keywords
        goal_lower = goal.lower()

        if "resource" in goal_lower and state.get("resources", 0) >= 5:
            return True
        if "structure" in goal_lower and state.get("structures", 0) >= 3:
            return True
        if "position" in goal_lower and state.get("position", 0) >= 10:
            return True

        return False

    def _evaluate_outcome(self, final_state: Dict, goal: str, success: bool) -> float:
        """Avaliar qualidade do outcome"""
        if success:
            return 0.9

        # Partial credit por progresso
        quality = 0.3

        if final_state.get("resources", 0) > 0:
            quality += 0.2
        if final_state.get("structures", 0) > 0:
            quality += 0.2

        return min(1.0, quality)

    def rollout_planning(self, initial_state: Dict, possible_actions: List[str], 
                        goal: str, num_rollouts: int = 5) -> str:
        """
        Planejamento por rollouts mentais.
        
        Simula múltiplas sequências de ações e escolhe a melhor.
        
        Args:
            initial_state: Estado inicial
            possible_actions: Ações possíveis
            goal: Objetivo
            num_rollouts: Número de rollouts
        
        Returns:
            Melhor ação
        """
        action_utilities = {action: 0.0 for action in possible_actions}

        # Rodar múltiplos rollouts por ação
        for action in possible_actions:
            total_utility = 0.0

            for _ in range(num_rollouts):
                # Criar cenário de teste
                scenario = MentalScenario(
                    scenario_id=str(uuid.uuid4()),
                    initial_state=initial_state,
                    actions=[action] + random.choices(possible_actions, k=3),
                    constraints=[],
                    goal=goal
                )

                # Simular
                result = self.simulate_scenario(scenario)
                total_utility += result.predicted_utility

            # Média de utility
            action_utilities[action] = total_utility / num_rollouts

        # Melhor ação
        best_action = max(action_utilities.items(), key=lambda x: x[1])[0]

        return best_action

    def counterfactual_simulation(self, actual_scenario: MentalScenario,
                                 alternative_action: str) -> MentalSimulationResult:
        """
        Simulação contrafactual: "E se eu tivesse feito X?"
        
        Args:
            actual_scenario: Cenário que realmente aconteceu
            alternative_action: Ação alternativa
        
        Returns:
            Resultado contrafactual
        """
        # Criar cenário contrafactual
        counterfactual = MentalScenario(
            scenario_id=f"counterfactual_{actual_scenario.scenario_id}",
            initial_state=actual_scenario.initial_state,
            actions=[alternative_action] + actual_scenario.actions[1:],
            constraints=actual_scenario.constraints,
            goal=actual_scenario.goal
        )

        # Simular
        result = self.simulate_scenario(counterfactual)

        return result

    def compare_scenarios(self, scenario1: MentalScenario, 
                         scenario2: MentalScenario) -> Dict[str, Any]:
        """Comparar dois cenários"""
        result1 = self.simulate_scenario(scenario1)
        result2 = self.simulate_scenario(scenario2)

        comparison = {
            "scenario1_success": result1.success,
            "scenario2_success": result2.success,
            "utility_difference": result1.predicted_utility - result2.predicted_utility,
            "preferred_scenario": 1 if result1.predicted_utility > result2.predicted_utility else 2,
            "risk_comparison": {
                "scenario1_risks": len(result1.risks),
                "scenario2_risks": len(result2.risks)
            }
        }

        return comparison

    def get_statistics(self) -> Dict:
        """Estatísticas do Mental Simulation System"""
        return {
            "total_simulations": self.total_simulations,
            "successful_predictions": self.successful_predictions,
            "simulation_accuracy": self.simulation_accuracy,
            "cached_simulations": len(self.simulation_cache)
        }
