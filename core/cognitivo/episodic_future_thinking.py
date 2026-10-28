"""Episodic future thinking (POST31) extraído de core/constitutional_brain.py.

Extração strangler: o código abaixo foi movido sem alteração de comportamento;
core.constitutional_brain reexporta as classes no ponto original.
Não há relógio injetável: event_id e created_at usam uuid.uuid4() e time.time().
memory e emotion só entram pela verdade booleana; nenhum método deles é chamado.
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

__all__ = [
    "FutureEvent",
    "ProspectiveMemory",
    "RealEpisodicFutureThinkingSystem",
]


# ═══════════════════════════════════════════════════════════════════════════
# MÓDULO 31: EPISODIC FUTURE THINKING SYSTEM (POST 31)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FutureEvent:
    """Evento futuro imaginado"""
    event_id: str
    description: str
    time_point: str  # "tomorrow", "next week", etc.
    location: str
    participants: List[str]
    expected_emotions: List[str]
    vividness: float  # 0-1
    plausibility: float  # 0-1
    personal_significance: float  # 0-1
    created_at: float

@dataclass  
class ProspectiveMemory:
    """Intenção para o futuro"""
    intention_id: str
    what: str  # O que fazer
    when: str  # Quando fazer
    where: str  # Onde fazer
    trigger: str  # Gatilho que lembra
    importance: float
    fulfilled: bool

class RealEpisodicFutureThinkingSystem:
    """
    POST31: Episodic Future Thinking System — REAL Future Event Imagination
    
    Pensamento Futuro Episódico = capacidade de imaginar eventos futuros específicos.
    "Mental time travel to the future" - pré-viver experiências.
    
    Capacidades:
    - Construct Future Events: Construir eventos futuros detalhados
    - Pre-experience: "Pré-viver" eventos imaginando-os
    - Prospective Memory: Lembrar intenções futuras
    - Goal Simulation: Simular atingimento de objetivos
    - Emotional Pre-feeling: Antecipar emoções futuras
    - Future Self Projection: Projetar-se no futuro
    
    Multiplica:
    • × POST1 (Memory): Usa memórias passadas para construir futuro
    • × POST30 (MentalSimulation): Simula eventos futuros
    • × POST17 (Consciousness): Consciência do self futuro
    • × POST19 (Emotion): Emoções antecipadas
    
    Baseado em:
    - Tulving's Mental Time Travel (1985)
    - Schacter & Addis' Constructive Episodic Simulation (2007)
    - Atance & O'Neill's Episodic Foresight (2001)
    - Suddendorf & Corballis' Mental Time Travel (1997)
    """

    def __init__(self, memory=None, mental_simulation=None, consciousness=None, emotion=None):
        self.memory = memory
        self.mental_simulation = mental_simulation
        self.consciousness = consciousness
        self.emotion = emotion

        # Storage de eventos futuros imaginados
        self.imagined_events: List[FutureEvent] = []

        # Prospective memories (intenções futuras)
        self.prospective_memories: List[ProspectiveMemory] = []

        # Estatísticas
        self.total_events_imagined = 0
        self.total_preexperiences = 0
        self.fulfilled_intentions = 0

        print("🧠🔮 POST 31 - RealEpisodicFutureThinkingSystem initialized (Future imagination active)")

    def imagine_future_event(self, description: str, time_point: str,
                             details: Dict[str, Any] | None = None) -> FutureEvent:
        """
        Imaginar evento futuro específico.
        
        Uses past episodic memories to construct plausible future scenarios.
        
        Args:
            description: Descrição do evento
            time_point: Quando ("tomorrow", "next month", etc.)
            details: Detalhes adicionais
        
        Returns:
            FutureEvent imaginado
        """
        details = details or {}

        # Extrair componentes de memórias passadas similares
        past_elements = self._retrieve_similar_past_experiences(description)

        # Construir evento futuro combinando elementos
        location = details.get("location", past_elements.get("location", "unknown"))
        participants = details.get("participants", past_elements.get("participants", []))

        # Prever emoções usando emotion system
        expected_emotions = self._predict_future_emotions(description, past_elements)

        # Avaliar vividness e plausibility
        vividness = self._compute_vividness(past_elements)
        plausibility = self._assess_plausibility(description, past_elements)
        significance = details.get("significance", 0.5)

        event = FutureEvent(
            event_id=str(uuid.uuid4()),
            description=description,
            time_point=time_point,
            location=location,
            participants=participants,
            expected_emotions=expected_emotions,
            vividness=vividness,
            plausibility=plausibility,
            personal_significance=significance,
            created_at=time.time()
        )

        self.imagined_events.append(event)
        self.total_events_imagined += 1

        return event

    def _retrieve_similar_past_experiences(self, description: str) -> Dict[str, Any]:
        """Recuperar experiências passadas similares"""
        # Usar memory system se disponível
        if self.memory:
            # Simplificado: sem busca por keywords (retorno fixo abaixo)
            # Memory retrieval (simplified)
            return {
                "location": "familiar_place",
                "participants": ["friend", "colleague"],
                "emotions": ["happy", "excited"]
            }

        return {}

    def _predict_future_emotions(self, description: str, past_elements: Dict) -> List[str]:
        """Prever emoções futuras"""
        # Usar emotion system se disponível
        if self.emotion:
            # Baseado em emoções de experiências passadas similares
            past_emotions = past_elements.get("emotions", [])
            if past_emotions:
                return past_emotions

        # Heurística simples
        if "success" in description.lower() or "achievement" in description.lower():
            return ["pride", "joy"]
        if "challenge" in description.lower():
            return ["anxiety", "excitement"]
        if "meeting" in description.lower():
            return ["curiosity", "neutral"]

        return ["neutral"]

    def _compute_vividness(self, past_elements: Dict) -> float:
        """Computar vividez da imaginação"""
        # Mais detalhes de experiências passadas = mais vívido
        detail_count = len(past_elements)
        vividness = min(1.0, 0.3 + detail_count * 0.15)
        return vividness

    def _assess_plausibility(self, description: str, past_elements: Dict) -> float:
        """Avaliar plausibilidade do evento futuro"""
        # Mais similar a experiências passadas = mais plausível
        if past_elements:
            return 0.8

        # Eventos comuns são mais plausíveis
        common_events = ["meeting", "work", "lunch", "travel", "presentation"]
        if any(event in description.lower() for event in common_events):
            return 0.7

        return 0.5

    def preexperience_event(self, event: FutureEvent) -> Dict[str, Any]:
        """
        Pré-vivenciar evento = imaginar em detalhe sensorial.
        
        "Mental dress rehearsal" para o futuro.
        
        Args:
            event: Evento a pré-vivenciar
        
        Returns:
            Experiência simulada
        """
        self.total_preexperiences += 1

        # Pré-experiência com detalhes sensoriais
        return {
            "event_id": event.event_id,
            "sensory_details": {
                "visual": f"Imagining {event.location}",
                "emotional": event.expected_emotions,
                "social": event.participants
            },
            "vividness": event.vividness,
            "simulation_success": True
        }

    def set_prospective_memory(self, what: str, when: str, where: str = "",
                               trigger: str = "", importance: float = 0.5) -> ProspectiveMemory:
        """
        Definir memória prospectiva (intenção futura).
        
        "Remember to do X at Y" - lembrar de fazer algo no futuro.
        
        Args:
            what: O que fazer
            when: Quando fazer
            where: Onde fazer
            trigger: Gatilho que lembra
            importance: Importância (0-1)
        
        Returns:
            ProspectiveMemory criada
        """
        intention = ProspectiveMemory(
            intention_id=str(uuid.uuid4()),
            what=what,
            when=when,
            where=where,
            trigger=trigger if trigger else when,
            importance=importance,
            fulfilled=False
        )

        self.prospective_memories.append(intention)

        return intention

    def check_prospective_triggers(self, current_context: Dict[str, Any]) -> List[ProspectiveMemory]:
        """
        Checar se algum trigger de memória prospectiva foi ativado.
        
        Args:
            current_context: Contexto atual (time, location, etc.)
        
        Returns:
            Lista de intenções triggered
        """
        triggered = []

        for intention in self.prospective_memories:
            if intention.fulfilled:
                continue

            # Check triggers
            if self._trigger_matches(intention, current_context):
                triggered.append(intention)

        return triggered

    def _trigger_matches(self, intention: ProspectiveMemory, context: Dict) -> bool:
        """Verificar se trigger match"""
        # Time-based trigger
        if context.get("time") and intention.when in str(context.get("time")):
            return True

        # Location-based trigger
        if context.get("location") and intention.where in str(context.get("location")):
            return True

        # Event-based trigger
        if context.get("event") and intention.trigger in str(context.get("event")):
            return True

        return False

    def fulfill_intention(self, intention_id: str):
        """Marcar intenção como cumprida"""
        intention = next((i for i in self.prospective_memories if i.intention_id == intention_id), None)

        if intention:
            intention.fulfilled = True
            self.fulfilled_intentions += 1

    def project_future_self(self, time_horizon: str) -> Dict[str, Any]:
        """
        Projetar self no futuro.
        
        "Who will I be in X time?"
        
        Args:
            time_horizon: "1 year", "5 years", etc.
        
        Returns:
            Projeção do self futuro
        """
        # Projetar mudanças baseadas em horizon
        future_self = {
            "time_horizon": time_horizon,
            "projected_changes": self._project_changes(time_horizon),
            "anticipated_growth": self._anticipate_growth(time_horizon),
            "continuity_with_current_self": 0.7  # Mantém identidade
        }

        return future_self

    def _project_changes(self, horizon: str) -> List[str]:
        """Projetar mudanças no self"""
        if "year" in horizon.lower():
            return ["increased_experience", "new_skills", "evolved_goals"]
        if "month" in horizon.lower():
            return ["incremental_progress", "learning"]

        return ["minor_changes"]

    def _anticipate_growth(self, horizon: str) -> Dict[str, float]:
        """Antecipar crescimento"""
        return {
            "knowledge": 0.8,
            "skills": 0.7,
            "wisdom": 0.6
        }

    def compare_future_scenarios(self, event1: FutureEvent, event2: FutureEvent) -> Dict[str, Any]:
        """Comparar cenários futuros alternativos"""
        return {
            "more_vivid": event1.event_id if event1.vividness > event2.vividness else event2.event_id,
            "more_plausible": event1.event_id if event1.plausibility > event2.plausibility else event2.event_id,
            "more_significant": event1.event_id if event1.personal_significance > event2.personal_significance else event2.event_id,
            "preferred": event1.event_id if event1.vividness * event1.personal_significance > 
                                           event2.vividness * event2.personal_significance else event2.event_id
        }

    def get_statistics(self) -> Dict:
        """Estatísticas do Episodic Future Thinking System"""
        fulfillment_rate = (
            self.fulfilled_intentions / max(1, len(self.prospective_memories))
            if self.prospective_memories else 0.0
        )

        return {
            "total_events_imagined": self.total_events_imagined,
            "total_preexperiences": self.total_preexperiences,
            "prospective_memories_active": len([p for p in self.prospective_memories if not p.fulfilled]),
            "fulfilled_intentions": self.fulfilled_intentions,
            "intention_fulfillment_rate": fulfillment_rate,
            "avg_event_vividness": sum(e.vividness for e in self.imagined_events) / max(1, len(self.imagined_events))
        }
