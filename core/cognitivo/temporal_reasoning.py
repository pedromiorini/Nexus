"""Raciocínio temporal (POST41) extraído de core/constitutional_brain.py.

Extração strangler: o código foi movido do monólito, move-only (byte-idêntico,
SHA-256 da fonte em tools/cognitivo_eval.py::TR_SOURCE_SHA); core.constitutional_brain
reexporta as classes no ponto original.
Dependências: apenas a biblioteca padrão. O relógio é time.time() (timestamp de
cada evento) e os ids vêm de uuid.uuid4(). causal_reasoning, planner e world_model
são armazenados e nunca lidos. Lacunas caracterizadas, não corrigidas (TR4, lacunas
i–xii em docs/cognitivo/AUDITORIA_2026-10-22.md): get_events_between não devolve
eventos que atravessam a janela, a média de comprimento de sequência é
móvel (pesa o último comprimento pela metade), e entradas inválidas levantam
TypeError ou devolvem resultado vazio.
"""

import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = [
    "RealTemporalReasoningEngine",
    "TemporalEvent",
    "TemporalRelation",
    "TemporalSequence",
]


# ═══════════════════════════════════════════════════════════════════════════
# MÓDULO 41: TEMPORAL REASONING ENGINE (POST 41)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TemporalEvent:
    """Evento temporal"""
    event_id: str
    event_type: str
    timestamp: float
    duration: Optional[float]  # Duração do evento (se aplicável)
    description: str
    
@dataclass
class TemporalRelation:
    """Relação temporal entre eventos"""
    relation_id: str
    event1_id: str
    event2_id: str
    relation_type: str  # "before", "after", "during", "overlaps", "meets"
    confidence: float  # 0-1
    
@dataclass
class TemporalSequence:
    """Sequência temporal de eventos"""
    sequence_id: str
    events: List[TemporalEvent]
    total_duration: float
    causal_chain: List[str]  # IDs dos eventos em ordem causal

class RealTemporalReasoningEngine:
    """
    POST41: Temporal Reasoning Engine — REAL Temporal and Sequential Reasoning
    
    Raciocínio Temporal = raciocínio sobre tempo, sequências e relações temporais.
    "Time is the fabric of causality" - Allen's Interval Algebra (1983)
    
    Princípios Centrais:
    - Temporal relations: Relações entre eventos (antes, depois, durante)
    - Sequence reasoning: Raciocínio sobre sequências de eventos
    - Duration reasoning: Raciocínio sobre durações
    - Temporal planning: Planejamento com restrições temporais
    - Event ordering: Ordenação causal de eventos
    
    Componentes:
    1. Event Tracker: Rastreia eventos temporais
    2. Relation Detector: Detecta relações temporais
    3. Sequence Builder: Constrói sequências temporais
    4. Duration Calculator: Calcula durações
    5. Temporal Planner: Planeja com restrições temporais
    
    Multiplica:
    • × POST11 (Causal): Causal + Temporal = raciocínio espaço-temporal
    • × POST5 (Planning): Planning + Temporal = planejamento temporal
    • × POST13 (WorldModel): WorldModel simula evolução temporal
    • × POST31 (EpisodicFuture): Futuro + Temporal = previsão temporal
    
    Baseado em:
    - Allen's Interval Algebra (Allen, 1983)
    - Temporal Logic (Prior, 1957)
    - Event Calculus (Kowalski & Sergot, 1986)
    - Temporal Planning (Ghallab et al., 2004)
    """
    
    def __init__(self, causal_reasoning=None, planner=None, world_model=None):
        self.causal = causal_reasoning
        self.planner = planner
        self.world_model = world_model
        
        # Event tracking
        self.events: List[TemporalEvent] = []
        self.temporal_relations: List[TemporalRelation] = []
        
        # Sequences
        self.sequences: List[TemporalSequence] = []
        
        # Estatísticas
        self.total_events_tracked = 0
        self.total_relations_detected = 0
        self.total_sequences_built = 0
        self.avg_sequence_length = 0.0
        
        print("⏰🔗 POST 41 - RealTemporalReasoningEngine initialized (Temporal reasoning active)")
    
    def track_event(self, event_type: str, description: str, 
                   duration: Optional[float] = None) -> TemporalEvent:
        """
        Rastrear evento temporal.
        
        Args:
            event_type: Tipo do evento
            description: Descrição
            duration: Duração (opcional)
        
        Returns:
            TemporalEvent criado
        """
        event = TemporalEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            duration=duration,
            description=description
        )
        
        self.events.append(event)
        self.total_events_tracked += 1
        
        # Detectar relações com eventos anteriores
        self._detect_temporal_relations(event)
        
        return event
    
    def _detect_temporal_relations(self, new_event: TemporalEvent):
        """Detectar relações temporais com eventos anteriores"""
        for existing_event in self.events[:-1]:  # Todos exceto o novo
            relation = self._infer_temporal_relation(existing_event, new_event)
            if relation:
                self.temporal_relations.append(relation)
                self.total_relations_detected += 1
    
    def _infer_temporal_relation(self, event1: TemporalEvent, 
                                event2: TemporalEvent) -> Optional[TemporalRelation]:
        """Inferir relação temporal entre dois eventos"""
        # Comparar timestamps
        time_diff = event2.timestamp - event1.timestamp
        
        # Determinar tipo de relação
        if time_diff > 0.1:  # event1 antes de event2
            relation_type = "before"
            confidence = 0.9
        elif time_diff < -0.1:  # event1 depois de event2
            relation_type = "after"
            confidence = 0.9
        else:  # Simultâneos
            relation_type = "simultaneous"
            confidence = 0.7
        
        # Verificar overlaps se ambos têm duração
        if event1.duration and event2.duration:
            event1_end = event1.timestamp + event1.duration
            event2_end = event2.timestamp + event2.duration
            
            # Durante
            if event1.timestamp <= event2.timestamp and event1_end >= event2_end:
                relation_type = "during"
                confidence = 0.85
            # Overlaps
            elif (event1.timestamp < event2_end and event1_end > event2.timestamp):
                relation_type = "overlaps"
                confidence = 0.8
        
        relation = TemporalRelation(
            relation_id=str(uuid.uuid4()),
            event1_id=event1.event_id,
            event2_id=event2.event_id,
            relation_type=relation_type,
            confidence=confidence
        )
        
        return relation
    
    def build_temporal_sequence(self, event_ids: List[str]) -> TemporalSequence:
        """
        Construir sequência temporal a partir de eventos.
        
        Args:
            event_ids: IDs dos eventos
        
        Returns:
            TemporalSequence construída
        """
        # Buscar eventos
        sequence_events = [e for e in self.events if e.event_id in event_ids]
        
        # Ordenar por timestamp
        sequence_events.sort(key=lambda e: e.timestamp)
        
        # Calcular duração total
        if len(sequence_events) >= 2:
            start_time = sequence_events[0].timestamp
            end_time = sequence_events[-1].timestamp
            if sequence_events[-1].duration:
                end_time += sequence_events[-1].duration
            total_duration = end_time - start_time
        else:
            total_duration = 0.0
        
        # Construir causal chain (ordem dos eventos)
        causal_chain = [e.event_id for e in sequence_events]
        
        sequence = TemporalSequence(
            sequence_id=str(uuid.uuid4()),
            events=sequence_events,
            total_duration=total_duration,
            causal_chain=causal_chain
        )
        
        self.sequences.append(sequence)
        self.total_sequences_built += 1
        
        # Update statistics
        self._update_sequence_statistics(len(sequence_events))
        
        return sequence
    
    def _update_sequence_statistics(self, sequence_length: int):
        """Atualizar estatísticas de sequências"""
        if self.avg_sequence_length == 0:
            self.avg_sequence_length = float(sequence_length)
        else:
            self.avg_sequence_length = (self.avg_sequence_length + sequence_length) / 2
    
    def reason_about_duration(self, event_id: str) -> Optional[float]:
        """
        Raciocinar sobre duração de evento.
        
        Args:
            event_id: ID do evento
        
        Returns:
            Duração estimada (se disponível)
        """
        # Buscar evento
        event = next((e for e in self.events if e.event_id == event_id), None)
        
        if not event:
            return None
        
        # Se evento tem duração explícita, retornar
        if event.duration:
            return event.duration
        
        # Caso contrário, estimar baseado em eventos similares
        similar_events = [e for e in self.events 
                         if e.event_type == event.event_type and e.duration]
        
        if similar_events:
            avg_duration = sum(e.duration for e in similar_events) / len(similar_events)
            return avg_duration
        
        return None
    
    def get_events_between(self, start_time: float, end_time: float) -> List[TemporalEvent]:
        """
        Obter eventos em intervalo temporal.
        
        Args:
            start_time: Início do intervalo
            end_time: Fim do intervalo
        
        Returns:
            Lista de eventos no intervalo
        """
        events_in_range = []
        
        for event in self.events:
            # Evento começa no intervalo
            if start_time <= event.timestamp <= end_time:
                events_in_range.append(event)
            # Evento termina no intervalo (se tem duração)
            elif event.duration:
                event_end = event.timestamp + event.duration
                if start_time <= event_end <= end_time:
                    events_in_range.append(event)
        
        return events_in_range
    
    def predict_next_event(self, current_sequence_ids: List[str]) -> Optional[str]:
        """
        Predizer próximo evento em sequência.
        
        Args:
            current_sequence_ids: IDs dos eventos atuais
        
        Returns:
            Tipo do próximo evento (predição)
        """
        # Buscar sequências similares
        similar_sequences = []
        
        for seq in self.sequences:
            # Verificar se os primeiros N eventos são similares
            seq_event_ids = [e.event_id for e in seq.events[:len(current_sequence_ids)]]
            
            # Se começam similar, é candidata
            if len(set(seq_event_ids).intersection(set(current_sequence_ids))) >= len(current_sequence_ids) * 0.7:
                similar_sequences.append(seq)
        
        # Se encontrou sequências similares, predizer próximo evento
        if similar_sequences:
            # Pegar próximo evento da sequência mais similar
            next_events = []
            for seq in similar_sequences:
                if len(seq.events) > len(current_sequence_ids):
                    next_event = seq.events[len(current_sequence_ids)]
                    next_events.append(next_event.event_type)
            
            if next_events:
                # Retornar mais comum
                from collections import Counter
                most_common = Counter(next_events).most_common(1)[0][0]
                return most_common
        
        return None
    
    def get_statistics(self) -> Dict:
        """Estatísticas do Temporal Reasoning Engine"""
        return {
            "total_events_tracked": self.total_events_tracked,
            "total_relations_detected": self.total_relations_detected,
            "total_sequences_built": self.total_sequences_built,
            "avg_sequence_length": self.avg_sequence_length,
            "active_events": len(self.events),
            "temporal_relations": len(self.temporal_relations)
        }
