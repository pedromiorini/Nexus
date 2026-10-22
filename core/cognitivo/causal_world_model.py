"""Raciocínio causal (POST7) e modelo de mundo (POST13), extraídos do monólito
core/constitutional_brain.py (strangler, move-only: comportamento idêntico)."""

import copy
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CausalRelation:
    cause: str
    effect: str
    confidence: float
    pattern_type: str

class RealCausalReasoning:
    """POST7: Deep Causal Reasoning - CONSOLIDADO"""

    def __init__(self):
        self.relations_extracted = 0
        self.causal_graph = {"adjacency": defaultdict(set)}

    def extract_causal_relations(self, text: str) -> List[CausalRelation]:
        """Extrair relações causais"""
        relations = []

        # Pattern: "A causes B"
        if "cause" in text.lower():
            match = re.search(r'(.+?)\s+causes?\s+(.+)', text, re.IGNORECASE)
            if match:
                cause = match.group(1).strip()
                effect = match.group(2).strip()
                relations.append(CausalRelation(cause, effect, 0.9, "direct_cause"))
                self.relations_extracted += 1
                self.causal_graph["adjacency"][cause].add(effect)

        return relations

    def enrich_context(self, evidence: List[Dict]) -> Dict:
        """Enriquecer contexto com análise causal"""
        all_relations = []
        for ev in evidence:
            rels = self.extract_causal_relations(ev.get("content", ""))
            all_relations.extend(rels)

        return {
            "causal_relations_found": len(all_relations),
            "relations": [
                {"cause": r.cause, "effect": r.effect, "confidence": r.confidence}
                for r in all_relations
            ]
        }

    def get_statistics(self) -> Dict:
        return {
            "total_relations_extracted": self.relations_extracted,
            "graph_nodes": len(self.causal_graph["adjacency"])
        }


@dataclass
class WorldState:
    state: Dict[str, Any]
    timestamp: float
    action_taken: Optional[str] = None

@dataclass
class SimulationResult:
    initial_state: WorldState
    final_state: WorldState
    success: bool
    warnings: List[str]

class RealWorldModelSimulator:
    """POST13: World Model & Simulation - CONSOLIDADO"""

    def __init__(self, causal_reasoning=None):
        self.causal_reasoning = causal_reasoning
        self.current_state = WorldState(
            state={
                "security_patch_applied": False,
                "alert_level": "high",
                "system_status": "vulnerable"
            },
            timestamp=time.time()
        )

    def simulate(self, actions: List[str]) -> SimulationResult:
        """Simular sequência de ações"""
        state = copy.deepcopy(self.current_state)
        warnings = []

        for action in actions:
            if action == "APPLY_PATCH":
                state.state["security_patch_applied"] = True
                state.state["alert_level"] = "low"
                state.state["system_status"] = "secure"
            elif action == "IGNORE_ALERT":
                state.state["alert_level"] = "critical"
                state.state["system_status"] = "compromised"

        # Evaluate success
        success = state.state.get("system_status") in ["secure", "stable"]

        return SimulationResult(
            initial_state=self.current_state,
            final_state=state,
            success=success,
            warnings=warnings
        )

    def compare_futures(self, action_sequences: List[List[str]]) -> Dict:
        """Comparar múltiplos futuros"""
        results = [self.simulate(seq) for seq in action_sequences]
        best_idx = max(range(len(results)), key=lambda i: results[i].success)

        return {
            "results": results,
            "best_future_index": best_idx,
            "best_future": results[best_idx]
        }
