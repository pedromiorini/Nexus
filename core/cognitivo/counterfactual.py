"""Counterfactual reasoning (POST37) — extraído do monólito (strangler, move-only)."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.cognitivo.action_effects import ACTION_EFFECTS
from core.cognitivo.causal_world_model import _schema_pair, classify_text_effects

# Sonda ao planner injetado. Não é o default 200 de RealMCTSPlanner.plan
# e não calibra plausibilidade.
PLANNER_PROBE_ITERATIONS = 8


def _normalize_action_key(text: str) -> str:
    """Mesmo normalizador do pré-registro CF9 (espaços/hífens, upper)."""
    collapsed = re.sub(r"[\s\-]+", "_", text.strip())
    return re.sub(r"_+", "_", collapsed).strip("_").upper()


def _serialize_state(state: Dict[str, object]) -> str:
    """Serialização estável do pré-registro: state:k=v com chaves ordenadas."""
    return "state:" + ",".join(f"{key}={state[key]}" for key in sorted(state))


# Passos atestáveis (pré-registro CF10). Texto sem aresta no grafo não conta.
_DO_STEP = re.compile(r"do\(([^)]+)\): ([^=]+)=(.+)\Z")
_CAUSES_STEP = re.compile(r"(.+) causes (.+)\Z")


def _effect_members(raw: object) -> List[object]:
    """Efeitos de uma aresta. str/bytes não são um conjunto (não fatiar em caracteres)."""
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (set, list, tuple, frozenset)):
        return []
    return list(raw)


def _graph_has_edge(adjacency: Dict[str, object]) -> bool:
    return any(_effect_members(value) for value in adjacency.values())


def _do_step_in_graph(match: re.Match[str], adjacency: Dict[str, object]) -> bool:
    action, key, value = match.group(1), match.group(2), match.group(3)
    for effect in _effect_members(adjacency.get(action)):
        parts = str(effect).strip().split(None, 1)
        if len(parts) == 2 and parts[0] == key and parts[1] == value:
            return True
    return False


def _causes_step_in_graph(match: re.Match[str], adjacency: Dict[str, object]) -> bool:
    cause, effect = match.group(1), match.group(2)
    return any(str(item) == effect for item in _effect_members(adjacency.get(cause)))


def _step_attested(step: str, adjacency: Dict[str, object]) -> bool:
    do_match = _DO_STEP.fullmatch(step)
    if do_match and _do_step_in_graph(do_match, adjacency):
        return True
    causes_match = _CAUSES_STEP.fullmatch(step)
    if causes_match and _causes_step_in_graph(causes_match, adjacency):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# MÓDULO 37: COUNTERFACTUAL REASONING ENGINE (POST 37)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CounterfactualScenario:
    """Cenário contrafactual ("What if?")"""
    scenario_id: str
    original_situation: str
    intervention: str  # O que mudou
    counterfactual_outcome: str
    actual_outcome: str
    causal_chain: List[str]
    plausibility_score: float  # 0-1
    
@dataclass
class CausalIntervention:
    """Intervenção causal"""
    intervention_id: str
    variable_changed: str
    original_value: Any
    counterfactual_value: Any
    expected_effects: List[str]
    
@dataclass
class OutcomeComparison:
    """Comparação de outcomes"""
    comparison_id: str
    actual_outcome: str
    counterfactual_outcome: str
    difference_score: float  # 0-1
    insights_learned: List[str]

class RealCounterfactualReasoningEngine:
    """
    POST37: Counterfactual Reasoning Engine — REAL "What If?" Reasoning
    
    Raciocínio Contrafactual = explorar cenários alternativos ("What if?").
    "Counterfactual thinking is essential for learning and planning" - Pearl (2009)
    
    Princípios Centrais:
    - Scenario generation: Gera cenários "what if?"
    - Causal intervention: Modela intervenções causais
    - Outcome prediction: Prediz outcomes alternativos
    - Comparison: Compara com outcomes reais
    - Learning: Aprende de contrafactuais
    
    Componentes:
    1. Scenario Generator: Gera cenários contrafactuais
    2. Intervention Modeler: Modela intervenções causais
    3. Outcome Predictor: Prediz outcomes de cenários
    4. Comparator: Compara outcomes reais vs. contrafactuais
    5. Insight Extractor: Extrai insights de comparações
    
    Multiplica:
    • × POST11 (Causal): Causal + Counterfactual = raciocínio completo
    • × POST5 (Planning): Planejamento via counterfactuals
    • × POST13 (WorldModel): Simula cenários alternativos
    • × POST30 (Simulation): Mental simulation de "what ifs"
    
    Baseado em:
    - Pearl's Causal Models (2009) - Structural Causal Models
    - Counterfactual Thinking (Roese & Olson, 1995)
    - Interventional Reasoning (Woodward, 2003)
    - Causal Inference (Spirtes et al., 2000)
    """
    
    def __init__(self, causal_reasoning=None, world_model=None, planner=None):
        self.causal = causal_reasoning
        self.world_model = world_model
        self.planner = planner
        
        # Counterfactual scenarios
        self.scenarios: List[CounterfactualScenario] = []
        self.interventions: List[CausalIntervention] = []
        
        # Comparisons
        self.outcome_comparisons: List[OutcomeComparison] = []
        
        # Insights learned
        self.insights: List[str] = []
        
        # Claims de texto livre (não verificadas) por cenário: scenario_id -> ["ação: chave=valor"]
        self.unverified_claims: Dict[str, List[str]] = {}
        self._pending_claims: List[str] = []

        # Estatísticas
        self.total_scenarios_generated = 0
        self.total_interventions_modeled = 0
        self.total_comparisons = 0
        self.avg_insight_value = 0.0
        
        print("🔮💭 POST 37 - RealCounterfactualReasoningEngine initialized (What-if reasoning active)")
    
    def generate_counterfactual_scenario(self, situation: str, intervention: str) -> CounterfactualScenario:
        """
        Gerar cenário contrafactual ("What if?").
        
        Args:
            situation: Situação original
            intervention: Intervenção contrafactual
        
        Returns:
            CounterfactualScenario gerado
        """
        # Consulta o planner antes das heurísticas: goal inválido propaga
        # sem normalizar e sem gravar cenário.
        planner_steps = self._planner_steps(situation, intervention)

        # Predizer outcome atual (sem intervenção)
        actual_outcome = self._predict_outcome(situation, intervention=None)
        
        # Predizer outcome contrafactual (com intervenção)
        counterfactual_outcome = self._predict_outcome(situation, intervention=intervention)
        claims_seen = list(self._pending_claims)
        self._pending_claims = []
        
        # Construir causal chain. Passos do planner, se houver, vêm depois.
        causal_chain = list(self._build_causal_chain(situation, intervention))
        causal_chain.extend(planner_steps)
        
        # Calcular plausibilidade
        plausibility = self._calculate_plausibility(intervention, causal_chain)
        
        # Criar cenário
        scenario = CounterfactualScenario(
            scenario_id=str(uuid.uuid4()),
            original_situation=situation,
            intervention=intervention,
            counterfactual_outcome=counterfactual_outcome,
            actual_outcome=actual_outcome,
            causal_chain=causal_chain,
            plausibility_score=plausibility
        )
        
        self.scenarios.append(scenario)
        if claims_seen:
            self.unverified_claims[scenario.scenario_id] = claims_seen
        self.total_scenarios_generated += 1
        
        return scenario
    
    def _planner_steps(self, situation: str, intervention: str) -> List[str]:
        """Pede uma sequência a ``planner.plan`` para o caminho what-if.

        Lista vazia quando não há planner ou ``plan`` não é chamável: a
        heurística de cadeia permanece. A intervenção pedida pelo chamador
        não é substituída. ``ValueError``/``TypeError`` de ``plan`` propagam.
        """
        planner = self.planner
        plan = getattr(planner, "plan", None) if planner is not None else None
        if not callable(plan):
            return []
        result = plan(
            situation,
            available_actions=[intervention],
            max_iterations=PLANNER_PROBE_ITERATIONS,
        )
        sequence = getattr(result, "best_action_sequence", None)
        if sequence is None and isinstance(result, dict):
            sequence = result.get("best_action_sequence")
        if isinstance(sequence, (str, bytes)) or not isinstance(sequence, (list, tuple)):
            return []  # str solta não vira lista de caracteres
        return [f"planner: {action}" for action in sequence if isinstance(action, str)]

    def _adjacency(self) -> Dict[str, object]:
        """Arestas já gravadas. O template lexical desta chamada não entra."""
        found: Dict[str, object] = {}
        graphs = []
        causal = self.causal
        if causal is not None:
            graphs.append(getattr(causal, "causal_graph", None))
        world = self.world_model
        nested = getattr(world, "causal_reasoning", None) if world is not None else None
        if nested is not None and nested is not causal:
            graphs.append(getattr(nested, "causal_graph", None))
        for graph in graphs:
            if not isinstance(graph, dict):
                continue
            adjacency = graph.get("adjacency")
            if not isinstance(adjacency, dict):
                continue
            for key, value in adjacency.items():
                found.setdefault(str(key), value)
        return found

    def _map_intervention(self, intervention: object) -> Optional[str]:
        """Uma ação do vocabulário ou da adjacência, ou None. Não coage não-str."""
        if not isinstance(intervention, str):
            return None
        raw = intervention.strip()
        if not raw:
            return None
        normalized = _normalize_action_key(intervention)
        adjacency = self._adjacency()
        if normalized in ACTION_EFFECTS:
            return normalized
        if raw in adjacency:
            return raw
        if normalized in adjacency:
            return normalized
        folded = [key for key in adjacency if _normalize_action_key(key) == normalized]
        if len(folded) == 1:
            return folded[0]
        if len(folded) > 1:
            return None
        tokens = set(re.findall(r"[a-z0-9]+", intervention.lower()))
        if not tokens:
            return None
        matches = [
            name
            for name in ACTION_EFFECTS
            if tokens == set(re.findall(r"[a-z0-9]+", name.lower()))
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _scm_do_steps(self, intervention: object) -> List[str]:
        """Passos do(ação) lidos da adjacência prévia, com o corte de C4."""
        action = self._map_intervention(intervention)
        if action is None:
            return []
        effects = self._adjacency().get(action)
        if not effects:
            return []
        members = _effect_members(effects)
        claims, _, _ = classify_text_effects(members)
        pairs = []
        for effect in members:
            parts = str(effect).strip().split(None, 1)
            if len(parts) != 2:
                continue
            pair = _schema_pair(parts[0], parts[1])
            if pair is not None and pair[0] in claims:
                pairs.append((parts[0], parts[1]))
        pairs.sort()
        return [f"do({action}): {key}={value}" for key, value in pairs]

    def _outcome_via_do(self, intervention: object) -> Optional[str]:
        """Outcome contrafactual via simulate, senão do, senão _do. None = heurística."""
        action = self._map_intervention(intervention)
        if action is None or self.world_model is None:
            return None
        world = self.world_model
        simulate = getattr(world, "simulate", None)
        if callable(simulate):
            result = simulate([action])
            state = getattr(getattr(result, "final_state", None), "state", None)
            if isinstance(state, dict):
                items = getattr(result, "claim_items", None)
                if isinstance(items, list):
                    self._pending_claims = [
                        f"{item['action']}: {item['key']}={item['value']}"
                        for item in items
                        if isinstance(item, dict) and {"action", "key", "value"} <= set(item)
                    ]
                return _serialize_state(state)
            return None
        for hook_name in ("do", "_do"):
            hook = getattr(world, hook_name, None)
            if not callable(hook):
                continue
            effects = hook(action)
            if not isinstance(effects, dict) or not effects:
                return None
            current = getattr(world, "current_state", None)
            base = getattr(current, "state", None)
            merged = dict(base) if isinstance(base, dict) else {}
            merged.update(effects)
            return _serialize_state(merged)
        return None

    def _predict_outcome(self, situation: str, intervention: Optional[str]) -> str:
        """Predizer outcome (com ou sem intervenção)"""
        predictor = getattr(self.world_model, "predict_outcome", None)
        if callable(predictor):
            return str(predictor(situation, intervention))
        modeled = self._outcome_via_do(intervention)
        if modeled is not None:
            return modeled

        # Heurística simples. Não-str truthy continua sem coerção (AttributeError).
        if intervention:
            return f"outcome_with_{intervention.replace(' ', '_')}"
        else:
            return "outcome_without_intervention"
    
    def _build_causal_chain(self, situation: str, intervention: str) -> List[str]:
        """Construir cadeia causal. Efeitos SCM prévios vêm antes do template."""
        do_steps = self._scm_do_steps(intervention)
        extractor = getattr(self.causal, "extract_causal_relations", None)
        if callable(extractor):
            relations = extractor(f"{situation} causes {intervention}")
            if relations:
                return do_steps + [
                    f"{relation.cause} causes {relation.effect}"
                    for relation in relations
                ]
        if do_steps:
            return do_steps
        
        # Heurística: steps básicos
        chain = [
            f"1. {situation}",
            f"2. Intervention: {intervention}",
            "3. Immediate effect",
            "4. Secondary effect",
            "5. Final outcome"
        ]
        
        return chain
    
    def _calculate_plausibility(self, intervention: object, causal_chain: object) -> float:
        """Heurística de plausibilidade (não é probabilidade).

        Comprimento da cadeia mais a fração de passos atestados por aresta
        já gravada no grafo injetado. Sem grafo, fica max(0.1, 0.7-0.05*n).
        Constantes e limites: pré-registro CF10. Não replica Roese e Olson.
        """
        if (
            not isinstance(causal_chain, list)
            or any(not isinstance(step, str) for step in causal_chain)
        ):
            raise TypeError("causal_chain must be a list of str")
        if intervention is False or (isinstance(intervention, str) and intervention.strip() == ""):
            return 0.1
        if not isinstance(intervention, str):
            raise TypeError("intervention must be str")
        chain_penalty = len(causal_chain) * 0.05
        length_score = max(0.1, 0.7 - chain_penalty)
        adjacency = self._adjacency()
        attested = sum(1 for step in causal_chain if _step_attested(step, adjacency))
        if attested:
            fraction = attested / len(causal_chain)
            return min(1.0, length_score + 0.2 * fraction)
        if _graph_has_edge(adjacency):
            return max(0.1, length_score - 0.2)
        return length_score
    
    def model_causal_intervention(self, variable: str, original_value: Any, 
                                  counterfactual_value: Any) -> CausalIntervention:
        """
        Modelar intervenção causal.
        
        Args:
            variable: Variável a intervir
            original_value: Valor original
            counterfactual_value: Valor contrafactual
        
        Returns:
            CausalIntervention
        """
        # Predizer efeitos da intervenção
        expected_effects = self._predict_intervention_effects(variable, 
                                                              original_value,
                                                              counterfactual_value)
        
        intervention = CausalIntervention(
            intervention_id=str(uuid.uuid4()),
            variable_changed=variable,
            original_value=original_value,
            counterfactual_value=counterfactual_value,
            expected_effects=expected_effects
        )
        
        self.interventions.append(intervention)
        self.total_interventions_modeled += 1
        
        return intervention
    
    def _predict_intervention_effects(self, variable: str, original: Any, 
                                     counterfactual: Any) -> List[str]:
        """Predizer efeitos de intervenção"""
        effects = []
        
        # Heurística: mudança causa efeito
        if original != counterfactual:
            effects.append(f"Change in {variable}")
            effects.append("Downstream effects on dependent variables")
            effects.append("Potential cascade effects")
        
        return effects
    
    def compare_outcomes(self, scenario: CounterfactualScenario) -> OutcomeComparison:
        """
        Comparar outcomes real vs. contrafactual.
        
        Args:
            scenario: Cenário contrafactual
        
        Returns:
            OutcomeComparison
        """
        # Calcular diferença
        difference = self._calculate_outcome_difference(
            scenario.actual_outcome,
            scenario.counterfactual_outcome
        )
        
        # Extrair insights
        insights = self._extract_insights(scenario, difference)
        
        comparison = OutcomeComparison(
            comparison_id=str(uuid.uuid4()),
            actual_outcome=scenario.actual_outcome,
            counterfactual_outcome=scenario.counterfactual_outcome,
            difference_score=difference,
            insights_learned=insights
        )
        
        self.outcome_comparisons.append(comparison)
        self.total_comparisons += 1
        
        # Adicionar insights ao conhecimento
        self.insights.extend(insights)
        
        # Update statistics
        self._update_insight_statistics(insights)
        
        return comparison
    
    def _calculate_outcome_difference(self, actual: str, counterfactual: str) -> float:
        """Calcular diferença entre outcomes via 1 − Jaccard de tokens alnum."""
        if actual == counterfactual:
            return 0.0

        tokens_a = set(re.findall(r"[a-z0-9]+", actual.lower()))
        tokens_b = set(re.findall(r"[a-z0-9]+", counterfactual.lower()))
        if not tokens_a and not tokens_b:
            return 0.0
        if not tokens_a or not tokens_b:
            return 1.0
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return 1.0 - (intersection / union)
    
    def _extract_insights(self, scenario: CounterfactualScenario, 
                         difference: float) -> List[str]:
        """Extrair insights de comparação"""
        insights = []
        
        if difference > 0.5:
            insights.append(f"Intervention '{scenario.intervention}' would cause significant change")
        else:
            insights.append(f"Intervention '{scenario.intervention}' would cause minor change")
        
        if scenario.plausibility_score > 0.7:
            insights.append("Scenario is highly plausible")
        
        insights.append(f"Key causal factor: {scenario.intervention}")
        
        return insights
    
    def explore_what_if(
        self,
        question: Optional[str] = None,
        *,
        situation: Optional[str] = None,
        intervention: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Explorar questão "What if?".

        Compatível com a API antiga (`explore_what_if(question)`).
        Preferir kwargs explícitos `situation=` / `intervention=` quando o
        chamador já conhece o contexto — assim a situação não é descartada
        pelo parse lexical (que hoje força `current_situation`).

        Args:
            question: Questão "what if?" (ex: "What if I had studied more?")
            situation: Situação/contexto explícito (opcional)
            intervention: Intervenção explícita (opcional; exige situation)

        Returns:
            Análise do cenário contrafactual
        """
        if situation is not None and intervention is not None:
            sit, inter = situation, intervention
            q_label = question if question is not None else f"What if {inter} ({sit})?"
        elif question is not None:
            sit, inter = self._parse_what_if_question(question)
            q_label = question
        else:
            raise TypeError(
                "explore_what_if requires question= or both situation= and intervention="
            )

        scenario = self.generate_counterfactual_scenario(sit, inter)
        comparison = self.compare_outcomes(scenario)
        return {
            "question": q_label,
            "scenario": scenario,
            "comparison": comparison,
            "plausibility": scenario.plausibility_score,
            "insights": comparison.insights_learned,
        }
    
    def _parse_what_if_question(self, question: str) -> tuple:
        """Parse questão 'what if?'"""
        # Heurística: extrair de "what if X?"
        question_lower = question.lower()
        
        if "what if" in question_lower:
            # Extrair parte depois de "what if"
            parts = question_lower.split("what if")
            if len(parts) > 1:
                intervention = parts[1].strip("? ")
                situation = "current_situation"
                return situation, intervention
        
        # Default
        return "unknown_situation", question
    
    def _update_insight_statistics(self, insights: List[str]):
        """Atualizar estatísticas de insights"""
        if insights:
            # Valor de insight baseado em quantidade
            insight_value = min(1.0, len(insights) / 5.0)
            
            if self.avg_insight_value == 0:
                self.avg_insight_value = insight_value
            else:
                self.avg_insight_value = (self.avg_insight_value + insight_value) / 2
    
    def get_statistics(self) -> Dict:
        """Estatísticas do Counterfactual Reasoning Engine"""
        return {
            "total_scenarios_generated": self.total_scenarios_generated,
            "total_interventions_modeled": self.total_interventions_modeled,
            "total_comparisons": self.total_comparisons,
            "avg_insight_value": self.avg_insight_value,
            "insights_learned": len(self.insights),
            "avg_plausibility": sum(s.plausibility_score for s in self.scenarios) / max(1, len(self.scenarios))
        }
