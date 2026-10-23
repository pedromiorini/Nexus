"""Raciocínio causal (POST7) e modelo de mundo (POST13), extraídos do monólito
core/constitutional_brain.py (strangler, move-only: comportamento idêntico)."""

import copy
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.cognitivo.action_effects import ACTION_EFFECTS

# R2: utilidade ordinal do estado final (status desconhecido fica abaixo de todos)
STATUS_UTILITY: Dict[str, int] = {"secure": 3, "stable": 2, "isolated": 1, "offline": 1, "vulnerable": 1,
                                  "compromised": 0}


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
        """Extrair relações causais (R3: banco de marcadores + escopo de negação).

        Banco de marcadores no estilo de Khoo et al. (2000): verbo "cause(s)" (palavra inteira),
        verbos causais, passiva "caused by", marcadores retroativos ("because [of]", "due to") e
        conectivos ("so", "therefore", "as a result of"). O primeiro padrão que casa vence
        (uma relação por frase, como antes). Negação estilo NegEx (Chapman et al., 2001): um gatilho
        nas 5 palavras antes do marcador bloqueia a relação; "cause" como substantivo (precedido de
        determinante) não é marcador.
        """
        relations = []
        for pattern, kind, confidence, backward in _CAUSAL_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            first, second = match.group(1).strip(), match.group(2).strip()
            if not first or not second or _negated(text[:match.start(2)], first, kind):
                break
            cause, effect = (second, first) if backward else (first, second)
            relations.append(CausalRelation(cause, effect, confidence, kind))
            self.relations_extracted += 1
            self.causal_graph["adjacency"][cause].add(effect)
            break

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

    def _do(self, action: str) -> Dict[str, object]:
        """Intervenção do(ação): efeitos simbólicos das arestas causa→efeito (Pearl)."""
        cr = self.causal_reasoning
        if cr is None:
            return {}
        adjacency = getattr(cr, "causal_graph", {}).get("adjacency") or {}
        effects: Dict[str, object] = {}
        for effect in adjacency.get(action, ()):
            parts = str(effect).strip().split(None, 1)
            if len(parts) == 2:
                effects[parts[0]] = parts[1]
        return effects

    def simulate(self, actions: List[str]) -> SimulationResult:
        """Simular sequência de ações; aplica ACTION_EFFECTS e do(ação)."""
        state = copy.deepcopy(self.current_state)
        warnings = []

        for action in actions:
            applied = False
            entry = ACTION_EFFECTS.get(action)
            if entry is not None:
                state.state.update(entry["world"])
                applied = True
            intervened = self._do(action)
            if intervened:
                state.state.update(intervened)
                applied = True
            if not applied:
                warnings.append(f"unknown action: {action!r}")

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
        best_idx = max(range(len(results)),
                       key=lambda i: (results[i].success,
                                      STATUS_UTILITY.get(results[i].final_state.state.get("system_status"), -1)))

        return {
            "results": results,
            "best_future_index": best_idx,
            "best_future": results[best_idx]
        }


# ─── R3: banco de marcadores causais e escopo de negação ─────────────────────
_F = re.IGNORECASE
_CAUSAL_PATTERNS = [
    (re.compile(r"(.+?)\s+(?:is|are|was|were)\s+(?:caused|triggered|produced|induced)\s+by\s+(.+)", _F),
     "passive_cause", 0.8, True),
    (re.compile(r'(.+?)\s+causes?\s+(.+)', _F), "direct_cause", 0.9, False),
    (re.compile(r"(.+?)\s+(?:leads?|led)\s+to\s+(.+)", _F), "causal_verb", 0.8, False),
    (re.compile(r"(.+?)\s+(?:results?|resulted)\s+in\s+(.+)", _F), "causal_verb", 0.8, False),
    (re.compile(r"(.+?)\s+(?:gives?|gave)\s+rise\s+to\s+(.+)", _F), "causal_verb", 0.8, False),
    (re.compile(r"(.+?)\s+(?:brings?|brought)\s+about\s+(.+)", _F), "causal_verb", 0.8, False),
    (re.compile(r"(.+?)\s+(?:contributes?|contributed)\s+to\s+(.+)", _F), "causal_verb", 0.8, False),
    (re.compile(r"(.+?)\s+(?:triggers?|triggered|produces?|produced|induces?|induced)\s+(.+)", _F),
     "causal_verb", 0.8, False),
    (re.compile(r"^as\s+a\s+result\s+of\s+(.+?),\s*(.+)", _F), "connective", 0.8, False),
    (re.compile(r"(.+?)\s+because\s+of\s+(.+)", _F), "backward_cause", 0.8, True),
    (re.compile(r"(.+?)\s+because\s+(.+)", _F), "backward_cause", 0.8, True),
    (re.compile(r"(.+?)\s+due\s+to\s+(.+)", _F), "backward_cause", 0.8, True),
    (re.compile(r"^(.+?),\s*(?:so|therefore|thus|hence)\s+(.+)", _F), "connective", 0.8, False),
]
_NEGATION_TRIGGERS = {"not", "no", "never", "without", "cannot", "false", "nothing", "neither", "nor", "unlikely"}
_DETERMINERS = {"the", "a", "an", "no", "any", "its", "their", "his", "her", "this", "that", "root", "main"}


def _negated(prefix: str, first: str, kind: str) -> bool:
    words = re.findall(r"[a-z']+", prefix.lower())
    window = words[-6:-1] if len(words) > 1 else []  # 5 palavras antes do marcador (NegEx)
    if any(w in _NEGATION_TRIGGERS or w.endswith("n't") for w in window):
        return True
    if kind == "direct_cause":
        last = re.findall(r"[a-z']+", first.lower())
        return bool(last) and last[-1] in _DETERMINERS
    return False
