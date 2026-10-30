"""Raciocínio causal (POST7) e modelo de mundo (POST13), extraídos do monólito
core/constitutional_brain.py (strangler, move-only: comportamento idêntico)."""

import copy
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.cognitivo.action_effects import ACTION_EFFECTS

# R2: utilidade ordinal do estado final (status desconhecido fica abaixo de todos)
STATUS_UTILITY: Dict[str, int] = {"secure": 3, "stable": 2, "isolated": 1, "offline": 1, "vulnerable": 1,
                                  "compromised": 0}


# SC: esquema declarado do mundo simbólico. Só pares (chave, valor) daqui podem virar efeito de
# do(X); texto livre fora do esquema não muda nada. Valores de texto comparam sem diferença de
# maiúsculas; booleanos são estritos (``True``/``False``, nunca "true" nem 1).
WORLD_SCHEMA: Dict[str, Tuple[Any, ...]] = {
    "security_patch_applied": (True, False),
    "alert_level": ("low", "medium", "high", "critical"),
    "system_status": ("secure", "stable", "isolated", "offline", "vulnerable", "compromised"),
    "logging_level": ("normal", "verbose"),
}


def _schema_pair(key: Any, value: Any) -> Optional[Tuple[str, Any]]:
    """Par (chave, valor) canônico se estiver no WORLD_SCHEMA; None caso contrário."""
    if not isinstance(key, str):
        return None
    canon_key = key.strip().lower()
    allowed = WORLD_SCHEMA.get(canon_key)
    if allowed is None:
        return None
    if isinstance(value, bool):
        return (canon_key, value) if any(isinstance(v, bool) for v in allowed) else None
    if isinstance(value, str):
        canon_value = value.strip().lower()
        return (canon_key, canon_value) if canon_value in [v for v in allowed if isinstance(v, str)] else None
    return None


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
        # Canal confiável (programático): ação -> efeitos validados contra o WORLD_SCHEMA.
        self.declared_interventions: Dict[str, Dict[str, Any]] = {}

    def declare_intervention(self, action: str, effects: Dict[str, Any]) -> List[str]:
        """Declara os efeitos verificados de uma ação. Guarda só pares do WORLD_SCHEMA e devolve os rejeitados."""
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action deve ser str não vazia")
        if not isinstance(effects, dict):
            raise TypeError("effects deve ser dict")
        accepted: Dict[str, Any] = {}
        rejected: List[str] = []
        for key, value in effects.items():
            pair = _schema_pair(key, value)
            if pair is None:
                rejected.append(f"{key!r}={value!r} fora do WORLD_SCHEMA")
            else:
                accepted[pair[0]] = pair[1]
        if accepted:
            self.declared_interventions.setdefault(action, {}).update(accepted)
        return rejected

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
    claims: List[str] = field(default_factory=list)  # afirmações de texto livre aplicadas, não verificadas
    verified_status: Optional[str] = None  # system_status só com tabela + intervenções declaradas

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

    def _split_do(self, action: str) -> Tuple[Dict[str, object], Dict[str, object], List[str]]:
        """Separa os efeitos de do(ação): (declarados e verificados, claims de texto válidas, ignorados)."""
        cr = self.causal_reasoning
        if cr is None:
            return {}, {}, []
        declared_raw = getattr(cr, "declared_interventions", None)
        declared: Dict[str, object] = {}
        if isinstance(declared_raw, dict):
            for key, value in (declared_raw.get(action) or {}).items():
                pair = _schema_pair(key, value)
                if pair is not None:
                    declared[pair[0]] = pair[1]
        adjacency = getattr(cr, "causal_graph", {}).get("adjacency") or {}
        claims: Dict[str, object] = {}
        ignored: List[str] = []
        for effect in adjacency.get(action, ()):
            parts = str(effect).strip().split(None, 1)
            if len(parts) != 2:
                continue
            pair = _schema_pair(parts[0], parts[1])
            if pair is None:
                ignored.append(f"{parts[0]}={parts[1]}")
            elif pair[0] not in declared:
                claims[pair[0]] = pair[1]
        return declared, claims, ignored

    def _do(self, action: str) -> Dict[str, object]:
        """Intervenção do(ação): só pares do WORLD_SCHEMA; declarado vence texto na mesma chave."""
        declared, claims, _ = self._split_do(action)
        effects = dict(claims)
        effects.update(declared)
        return effects

    def simulate(self, actions: List[str]) -> SimulationResult:
        """Simular sequência de ações; aplica ACTION_EFFECTS, intervenções declaradas e claims de texto.

        ``success`` e ``verified_status`` usam só o estado verificado (tabela + declaradas): uma
        afirmação extraída de texto livre aparece em ``final_state`` e em ``claims``, com aviso,
        mas não promove o sucesso.
        """
        state = copy.deepcopy(self.current_state)
        verified = dict(state.state)
        warnings = []
        claim_log: List[str] = []

        for action in actions:
            applied = False
            entry = ACTION_EFFECTS.get(action)
            if entry is not None:
                state.state.update(entry["world"])
                verified.update(entry["world"])
                applied = True
            declared, claims, ignored = self._split_do(action)
            if declared:
                state.state.update(declared)
                verified.update(declared)
                applied = True
            if claims:
                state.state.update(claims)
                applied = True
                for key, value in claims.items():
                    claim_log.append(f"{action}: {key}={value}")
                    warnings.append(f"unverified causal claim (free text): {action!r} -> {key}={value}")
            for pair in ignored:
                warnings.append(f"ignored causal claim outside WORLD_SCHEMA: {action!r} -> {pair}")
            if not applied:
                warnings.append(f"unknown action: {action!r}")

        # Evaluate success (só estado verificado)
        success = verified.get("system_status") in ["secure", "stable"]

        return SimulationResult(
            initial_state=self.current_state,
            final_state=state,
            success=success,
            warnings=warnings,
            claims=claim_log,
            verified_status=verified.get("system_status"),
        )

    def compare_futures(self, action_sequences: List[List[str]]) -> Dict:
        """Comparar múltiplos futuros"""
        results = [self.simulate(seq) for seq in action_sequences]
        best_idx = max(range(len(results)),
                       key=lambda i: (results[i].success,
                                      STATUS_UTILITY.get(results[i].verified_status, -1)))

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
