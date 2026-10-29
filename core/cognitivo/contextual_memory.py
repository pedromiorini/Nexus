"""Memória contextual (POST14) extraída de core/constitutional_brain.py.

Extração strangler: o código foi movido do monólito, move-only (byte-idêntico,
SHA-256 da fonte em tools/cognitivo_eval.py::CM_SOURCE_SHA); core.constitutional_brain
reexporta as classes no ponto original.
Dependências: apenas a biblioteca padrão. O relógio é time.time() (timestamp de
cada memória) e os ids vêm de uuid.uuid4(). long_term_memory é só escrita
(store); working_memory e attention são armazenados e nunca lidos.
Lacunas registradas, não corrigidas aqui: limiar padrão não seletivo (CM3) e
vetor de contexto dependente de hash() por processo (CM4).
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = [
    "ContextEncoder",
    "ContextMatch",
    "ContextualMemory",
    "RealContextualMemorySystem",
]


# ═══════════════════════════════════════════════════════════════════════════
# MÓDULO 14: CONTEXTUAL MEMORY SYSTEM (POST 14)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ContextualMemory:
    """Memória com contexto associado"""
    memory_id: str
    content: Any
    context: Dict[str, Any]  # Contexto de codificação
    context_vector: List[float]  # Vetor de contexto
    timestamp: float
    retrieval_count: int


@dataclass
class ContextMatch:
    """Match entre contexto de codificação e recuperação"""
    memory: ContextualMemory
    match_score: float
    matched_features: List[str]


class ContextEncoder:
    """
    Codificador de Contexto.
    
    Transforma contexto situacional em representação.
    """

    def __init__(self):
        # Dimensões de contexto
        self.context_dimensions = [
            "location",    # Onde
            "time",        # Quando
            "mood",        # Estado emocional
            "task",        # Tarefa atual
            "social"       # Contexto social
        ]

    def encode_context(self, context: Dict[str, Any]) -> List[float]:
        """
        Codificar contexto em vetor.
        
        Args:
            context: Dicionário de contexto
        
        Returns:
            Vetor de contexto
        """
        vector = []

        for dim in self.context_dimensions:
            value = context.get(dim, "unknown")
            # Hash simples para valor
            encoded = hash(str(value)) % 100 / 100.0
            vector.append(encoded)

        return vector

    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Computar similaridade entre vetores de contexto"""
        if not vec1 or not vec2:
            return 0.0

        # Similaridade do cosseno simplificada
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)


class RealContextualMemorySystem:
    """
    POST14: Contextual Memory System — REAL Context-Dependent Recall
    
    Memória Contextual = recuperação dependente do contexto de codificação.
    
    Princípios:
    - Encoding Specificity: Melhor recall quando contexto de retrieval
      matches contexto de encoding
    - State-Dependent Memory: Recall melhor em mesmo estado interno
    - Cue-Dependent Memory: Pistas contextuais facilitam recall
    
    Componentes:
    - ContextEncoder: Codifica contexto situacional
    - Context Matching: Compara contextos
    - Context Reinstatement: Reinstancia contexto original
    
    Multiplica:
    • × POST1 (Memory): Memória contextualizada
    • × POST6 (WorkingMemory): Contexto em WM
    • × POST25 (Attention): Atenção a cues contextuais
    • × POST17 (Consciousness): Consciência de contexto
    
    Baseado em:
    - Tulving & Thomson's Encoding Specificity (1973)
    - Godden & Baddeley's Diving Study (1975)
    - Smith & Vela's Context-Dependent Memory (2001)
    - Eich's State-Dependent Learning (1980)
    """

    def __init__(self, long_term_memory=None, working_memory=None, attention=None):
        self.ltm = long_term_memory
        self.wm = working_memory
        self.attention = attention

        # Componentes
        self.context_encoder = ContextEncoder()

        # Storage
        self.contextual_memories: List[ContextualMemory] = []

        # Contexto atual
        self.current_context: Dict[str, Any] = {}
        self.current_context_vector: List[float] = []

        # Estatísticas
        self.total_encodings = 0
        self.total_retrievals = 0
        self.context_matched_retrievals = 0

        print("🧠🔍 POST 14 - RealContextualMemorySystem initialized (Context-dependent recall active)")

    def set_current_context(self, context: Dict[str, Any]):
        """
        Definir contexto atual.
        
        Args:
            context: Dicionário de contexto
        """
        self.current_context = context
        self.current_context_vector = self.context_encoder.encode_context(context)

    def store_with_context(self, content: Any, context: Dict[str, Any] | None = None) -> ContextualMemory:
        """
        Armazenar memória com contexto.
        
        Args:
            content: Conteúdo a armazenar
            context: Contexto de codificação (usa current se None)
        
        Returns:
            ContextualMemory criada
        """
        # Usar contexto atual se não fornecido
        if context is None:
            context = self.current_context

        # Codificar contexto
        context_vector = self.context_encoder.encode_context(context)

        # Criar memória contextual
        memory = ContextualMemory(
            memory_id=str(uuid.uuid4()),
            content=content,
            context=context,
            context_vector=context_vector,
            timestamp=time.time(),
            retrieval_count=0
        )

        self.contextual_memories.append(memory)
        self.total_encodings += 1

        # Também armazenar em LTM se disponível
        if self.ltm:
            self.ltm.store(
                str(content),
                "contextual",
                importance=0.7
            )

        return memory

    def retrieve_by_context(self, 
                           query: str | None = None,
                           context: Dict[str, Any] | None = None,
                           threshold: float = 0.5) -> List[ContextMatch]:
        """
        Recuperar memórias por contexto.
        
        Args:
            query: Query opcional
            context: Contexto de recuperação (usa current se None)
            threshold: Threshold de match
        
        Returns:
            Lista de ContextMatch
        """
        # Usar contexto atual se não fornecido
        if context is None:
            context = self.current_context

        # Codificar contexto de retrieval
        retrieval_vector = self.context_encoder.encode_context(context)

        # Buscar memórias com contexto similar
        matches = []

        for memory in self.contextual_memories:
            # Computar similaridade de contexto
            similarity = self.context_encoder.compute_similarity(
                memory.context_vector,
                retrieval_vector
            )

            # Se query fornecida, considerar também match de conteúdo
            content_match = 1.0
            if query:
                query_lower = str(query).lower()
                content_lower = str(memory.content).lower()
                content_match = 1.0 if query_lower in content_lower else 0.3

            # Score combinado
            match_score = (similarity + content_match) / 2.0

            if match_score >= threshold:
                # Identificar features matched
                matched_features = self._identify_matched_features(
                    memory.context,
                    context
                )

                match = ContextMatch(
                    memory=memory,
                    match_score=match_score,
                    matched_features=matched_features
                )

                matches.append(match)

                # Incrementar retrieval count
                memory.retrieval_count += 1

        # Ordenar por match_score
        matches.sort(key=lambda m: m.match_score, reverse=True)

        self.total_retrievals += 1
        if matches:
            self.context_matched_retrievals += 1

        return matches

    def _identify_matched_features(self, 
                                   encoding_context: Dict,
                                   retrieval_context: Dict) -> List[str]:
        """Identificar features de contexto que matcham"""
        matched = []

        for key in encoding_context:
            if key in retrieval_context:
                if encoding_context[key] == retrieval_context[key]:
                    matched.append(key)

        return matched

    def reinstate_context(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Reinstanciar contexto de memória.
        
        Context Reinstatement: Trazer de volta contexto original
        para facilitar recall adicional.
        
        Args:
            memory_id: ID da memória
        
        Returns:
            Contexto reinstanciado ou None
        """
        memory = next((m for m in self.contextual_memories if m.memory_id == memory_id), None)

        if not memory:
            return None

        # Definir como contexto atual
        self.set_current_context(memory.context)

        return memory.context

    def get_context_dependent_cues(self, memory: ContextualMemory) -> List[str]:
        """
        Obter cues contextuais para memória.
        
        Args:
            memory: Memória
        
        Returns:
            Lista de cues
        """
        cues = []

        for key, value in memory.context.items():
            cue = f"{key}={value}"
            cues.append(cue)

        return cues

    def test_encoding_specificity(self, 
                                  content: str,
                                  encoding_context: Dict,
                                  retrieval_context: Dict) -> float:
        """
        Testar princípio de encoding specificity.
        
        Returns:
            Match score (0-1)
        """
        # Encode
        memory = self.store_with_context(content, encoding_context)

        # Retrieve
        matches = self.retrieve_by_context(content, retrieval_context, threshold=0.0)

        # Find our memory
        our_match = next((m for m in matches if m.memory.memory_id == memory.memory_id), None)

        if our_match:
            return our_match.match_score

        return 0.0

    def get_statistics(self) -> Dict:
        """Estatísticas do Contextual Memory System"""
        context_match_rate = (
            self.context_matched_retrievals / max(1, self.total_retrievals)
            if self.total_retrievals > 0
            else 0.0
        )

        return {
            "total_encodings": self.total_encodings,
            "total_retrievals": self.total_retrievals,
            "context_matched_retrievals": self.context_matched_retrievals,
            "context_match_rate": context_match_rate,
            "stored_memories": len(self.contextual_memories),
            "current_context_dimensions": len(self.current_context)
        }