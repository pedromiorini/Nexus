"""Ponderação emocional de memória (POST56) extraída de core/constitutional_brain.py.

Extração strangler: o código foi movido do monólito, move-only (byte-idêntico,
SHA-256 da fonte em tools/cognitivo_eval.py::EW_SOURCE_SHA); core.constitutional_brain
reexporta as classes no ponto original.
Dependências: apenas a biblioteca padrão. O relógio é time.time() (timestamp de
cada peso). memory_system é armazenado e nunca lido; emotion_system só é lido
por emotional_states (opcional). emotional_decay_rate é declarado e nunca usado.
Lacunas registradas, fora desta extração (pré-registro EW1–EW6 em
docs/cognitivo/AUDITORIA_2026-10-22.md): EW3 (nenhum limite sobre valência e
ativação: intensidade acima de 1 e reforço negativo) e EW4 (memória sem rótulo
recebe prioridade 1.0, acima da emocional moderada).
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "EmotionalMemoryWeight",
    "RealEmotionalMemoryWeighting",
]


# ═══════════════════════════════════════════════════════════════════════════
# MÓDULO 56: EMOTIONAL MEMORY WEIGHTING (POST 56)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EmotionalMemoryWeight:
    """Peso emocional de uma memória"""
    memory_id: int
    valence: float  # -1 (negativo) a +1 (positivo)
    arousal: float  # 0 (calmo) a 1 (excitado)
    emotional_intensity: float  # 0-1
    consolidation_boost: float  # Multiplicador de consolidação
    recall_priority: float  # Prioridade de recall
    timestamp: float

class RealEmotionalMemoryWeighting:
    """
    POST56: Emotional Memory Weighting — REAL Emotion-Enhanced Memory
    
    Peso Emocional de Memórias = memórias com carga emocional são mais fortes.
    "Emotion strengthens memory" - McGaugh (2000)
    
    Princípios Centrais:
    - Memórias emocionais são consolidadas mais fortemente
    - Arousal aumenta a força de codificação (Amygdala enhancement)
    - Valência emocional influencia recuperação
    - Emoções guiam priorização de recall
    - Trade-off: memórias emocionais podem ser menos precisas mas mais vívidas
    
    Componentes:
    - Emotional Tagging: Marca memórias com peso emocional
    - Consolidation Boost: Fortalece consolidação de memórias emocionais
    - Recall Prioritization: Prioriza recall por emoção
    - Valence-Based Retrieval: Recupera por valência
    - Arousal Modulation: Modula força por arousal
    
    Multiplica:
    • × POST1 (Memory): Adiciona dimensão emocional à memória
    • × POST19 (Emotion): Usa estados emocionais para ponderar
    • × POST14 (ContextualMemory): Contexto emocional
    • × POST31 (EpisodicFutureThinking): Emoções antecipadas
    
    Baseado em:
    - McGaugh's Emotional Modulation of Memory (2000)
    - Cahill & McGaugh's Amygdala Enhancement (1995)
    - Kensinger & Corkin's Emotion-Memory Trade-off (2003)
    - LaBar & Cabeza's Emotional Memory Systems (2006)
    """
    
    def __init__(self, memory_system=None, emotion_system=None):
        self.memory = memory_system
        self.emotion = emotion_system
        
        # Pesos emocionais das memórias
        self.emotional_weights: Dict[int, EmotionalMemoryWeight] = {}
        
        # Parâmetros de modulação
        self.arousal_consolidation_factor = 2.0  # Arousal boosting
        self.valence_sensitivity = 0.8
        self.emotional_decay_rate = 0.95  # Emoção decai mais devagar
        
        # Estatísticas
        self.total_emotional_memories = 0
        self.avg_emotional_intensity = 0.0
        self.high_arousal_count = 0
        
        print("💗📝 POST 56 - RealEmotionalMemoryWeighting initialized (Emotion-enhanced memory active)")
    
    def tag_memory_emotionally(self, memory_id: int, emotional_context: Dict[str, Any]) -> EmotionalMemoryWeight:
        """
        Marca memória com peso emocional.
        
        Args:
            memory_id: ID da memória
            emotional_context: Contexto emocional (valence, arousal, emotions)
        
        Returns:
            EmotionalMemoryWeight criado
        """
        # Extrair valência e arousal do contexto
        valence = emotional_context.get('valence', 0.0)  # -1 a +1
        arousal = emotional_context.get('arousal', 0.0)   # 0 a 1
        
        # Se temos sistema de emoção, usar estado atual
        if self.emotion and 'agent_id' in emotional_context:
            agent_id = emotional_context['agent_id']
            if agent_id in self.emotion.emotional_states:
                state = self.emotion.emotional_states[agent_id]
                valence = state.pleasure
                arousal = state.arousal
        
        # Calcular intensidade emocional (distância da neutralidade)
        emotional_intensity = math.sqrt(valence**2 + arousal**2) / math.sqrt(2)
        
        # Arousal alto = consolidação mais forte (McGaugh 2000)
        consolidation_boost = 1.0 + (arousal * self.arousal_consolidation_factor)
        
        # Recall priority baseado em intensidade
        recall_priority = emotional_intensity * 1.5
        
        weight = EmotionalMemoryWeight(
            memory_id=memory_id,
            valence=valence,
            arousal=arousal,
            emotional_intensity=emotional_intensity,
            consolidation_boost=consolidation_boost,
            recall_priority=recall_priority,
            timestamp=time.time()
        )
        
        self.emotional_weights[memory_id] = weight
        self.total_emotional_memories += 1
        
        # Update statistics
        self._update_statistics()
        
        if arousal > 0.7:
            self.high_arousal_count += 1
        
        return weight
    
    def _update_statistics(self):
        """Atualizar estatísticas médias"""
        if self.emotional_weights:
            intensities = [w.emotional_intensity for w in self.emotional_weights.values()]
            self.avg_emotional_intensity = sum(intensities) / len(intensities)
    
    def consolidate_emotional_memory(self, memory_id: int) -> float:
        """
        Consolidar memória com boost emocional.
        
        Returns:
            Fator de consolidação (>= 1.0)
        """
        if memory_id not in self.emotional_weights:
            return 1.0  # Sem boost
        
        weight = self.emotional_weights[memory_id]
        
        # Consolidação fortalecida por arousal
        return weight.consolidation_boost
    
    def prioritize_recall(self, memory_ids: List[int], 
                         emotional_query: Optional[Dict] = None) -> List[Tuple[int, float]]:
        """
        Priorizar recall de memórias por emoção.
        
        Args:
            memory_ids: IDs de memórias candidatas
            emotional_query: Contexto emocional da query (opcional)
        
        Returns:
            Lista de (memory_id, priority_score) ordenada
        """
        scored_memories = []
        
        for mem_id in memory_ids:
            base_score = 1.0
            
            if mem_id in self.emotional_weights:
                weight = self.emotional_weights[mem_id]
                
                # Score base: recall priority
                score = base_score * weight.recall_priority
                
                # Se temos query emocional, match por valência
                if emotional_query and 'valence' in emotional_query:
                    query_valence = emotional_query['valence']
                    valence_similarity = 1.0 - abs(query_valence - weight.valence)
                    score *= (1.0 + valence_similarity * self.valence_sensitivity)
                
                scored_memories.append((mem_id, score))
            else:
                # Memória sem peso emocional - score base
                scored_memories.append((mem_id, base_score))
        
        # Ordenar por score descendente
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return scored_memories
    
    def retrieve_by_valence(self, target_valence: float, 
                           threshold: float = 0.3, 
                           limit: int = 10) -> List[int]:
        """
        Recuperar memórias por valência emocional.
        
        Args:
            target_valence: Valência alvo (-1 a +1)
            threshold: Threshold de similaridade
            limit: Máximo de resultados
        
        Returns:
            Lista de memory_ids
        """
        candidates = []
        
        for mem_id, weight in self.emotional_weights.items():
            valence_diff = abs(target_valence - weight.valence)
            
            if valence_diff <= threshold:
                # Score: quanto menor a diferença, melhor
                similarity = 1.0 - valence_diff
                candidates.append((mem_id, similarity))
        
        # Ordenar por similaridade
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [mem_id for mem_id, _ in candidates[:limit]]
    
    def get_emotional_profile(self, memory_id: int) -> Dict[str, Any]:
        """Obter perfil emocional de memória"""
        if memory_id not in self.emotional_weights:
            return {"has_emotional_weight": False}
        
        weight = self.emotional_weights[memory_id]
        
        return {
            "has_emotional_weight": True,
            "valence": weight.valence,
            "arousal": weight.arousal,
            "intensity": weight.emotional_intensity,
            "consolidation_boost": weight.consolidation_boost,
            "recall_priority": weight.recall_priority,
            "emotional_category": self._categorize_emotion(weight.valence, weight.arousal)
        }
    
    def _categorize_emotion(self, valence: float, arousal: float) -> str:
        """Categorizar emoção por valência e arousal"""
        if arousal > 0.6:
            if valence > 0.3:
                return "excited_positive"
            elif valence < -0.3:
                return "distressed_negative"
            else:
                return "high_arousal_neutral"
        elif arousal < 0.3:
            if valence > 0.3:
                return "calm_positive"
            elif valence < -0.3:
                return "sad_negative"
            else:
                return "low_arousal_neutral"
        else:  # medium arousal
            if valence > 0.3:
                return "pleasant"
            elif valence < -0.3:
                return "unpleasant"
            else:
                return "neutral"
    
    def get_statistics(self) -> Dict:
        """Estatísticas do Emotional Memory Weighting"""
        return {
            "total_emotional_memories": self.total_emotional_memories,
            "avg_emotional_intensity": self.avg_emotional_intensity,
            "high_arousal_count": self.high_arousal_count,
            "weighted_memories": len(self.emotional_weights)
        }
