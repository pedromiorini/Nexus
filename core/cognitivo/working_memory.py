"""Working memory (POST6), extraída de core/constitutional_brain.py.

As classes continuam reexportadas por core.constitutional_brain; a construção e
toda reatribuição de capacity usam o protocolo de índice e exigem valor positivo.
Subclasses que sombreiam o descriptor `capacity` ou usam descritores de slot
customizados não são suportadas pelo contrato de validação de capacidade.
Não há relógio injetável: decay e rehearsal usam time.time().
"""

import time
import uuid
from dataclasses import dataclass
from operator import index
from types import MemberDescriptorType
from typing import Any, Dict, List, Optional, SupportsIndex

__all__ = [
    "CentralExecutive",
    "PhonologicalLoop",
    "RealWorkingMemorySystem",
    "RehearsalEvent",
    "VisuospatialSketchpad",
    "WorkingMemoryItem",
]


@dataclass
class WorkingMemoryItem:
    """Item na working memory"""
    item_id: str
    content: Any
    modality: str  # "verbal", "visual", "spatial", "multimodal"
    activation: float  # Nível de ativação (0-1)
    entered_at: float
    last_rehearsed: float


@dataclass
class RehearsalEvent:
    """Evento de rehearsal (repetição)"""
    item_id: str
    rehearsal_time: float
    activation_boost: float


def _positive_capacity(capacity: SupportsIndex) -> int:
    """Normalize an integer-like capacity and reject bool/non-positive values."""
    if isinstance(capacity, bool):
        raise TypeError("capacity must be a positive integer, not bool")
    normalized = int(index(capacity))
    if normalized <= 0:
        raise ValueError("capacity must be a positive integer")
    return normalized


def _restore_working_memory_state(instance: Any, state: Any) -> None:
    """Restore default dict/slot state with a validated capacity value."""
    if type(state) is dict:
        dict_state = state
        slot_state = None
    elif type(state) is tuple and len(state) == 2:
        dict_state, slot_state = state
        if dict_state is not None and type(dict_state) is not dict:
            raise TypeError("working memory state must be a dict or a (dict, dict) tuple")
        if slot_state is not None and type(slot_state) is not dict:
            raise TypeError("working memory state must be a dict or a (dict, dict) tuple")
    else:
        raise TypeError("working memory state must be a dict or a (dict, dict) tuple")

    restored_state = {} if dict_state is None else dict_state.copy()
    if any(type(name) is not str for name in restored_state):
        raise TypeError("working memory dict state keys must be strings")

    restored_slots = {} if slot_state is None else slot_state.copy()
    if any(type(name) is not str for name in restored_slots):
        raise TypeError("working memory slot state keys must be strings")
    valid_slots = {
        name
        for cls in type(instance).__mro__
        for name, descriptor in vars(cls).items()
        if isinstance(descriptor, MemberDescriptorType)
    }
    if any(name not in valid_slots for name in restored_slots):
        raise TypeError("working memory slot state contains an unsupported slot")

    capacity_values = []
    if "capacity" in restored_state:
        capacity_values.append(_positive_capacity(restored_state.pop("capacity")))
    if "_capacity" in restored_state:
        capacity_values.append(_positive_capacity(restored_state.pop("_capacity")))
    if "_capacity" in restored_slots:
        capacity_values.append(_positive_capacity(restored_slots["_capacity"]))
    if not capacity_values:
        raise ValueError("working memory state must include capacity")
    normalized_capacity = capacity_values[0]
    if any(value != normalized_capacity for value in capacity_values[1:]):
        raise ValueError("conflicting capacity values in working memory state")
    if "_capacity" in valid_slots:
        restored_slots["_capacity"] = normalized_capacity
    else:
        restored_state["_capacity"] = normalized_capacity
    instance.__dict__.update(restored_state)
    for name, value in restored_slots.items():
        setattr(instance, name, value)


class PhonologicalLoop:
    """
    Loop Fonológico (Phonological Loop).
    
    Mantém informação verbal/auditiva via rehearsal.
    Capacidade ~2s sem rehearsal, ~7±2 chunks.
    """

    @property
    def capacity(self) -> int:
        """Return the current normalized buffer capacity."""
        return self._capacity

    @capacity.setter
    def capacity(self, capacity: SupportsIndex) -> None:
        self._capacity = _positive_capacity(capacity)

    def __setstate__(self, state: Any) -> None:
        _restore_working_memory_state(self, state)

    def __init__(self, capacity: SupportsIndex = 7):
        self.capacity = capacity
        self.buffer: List[WorkingMemoryItem] = []
        self.decay_rate = 0.5  # Por segundo

    def store(self, content: str) -> WorkingMemoryItem:
        """Armazenar informação verbal"""
        # Se buffer cheio, remover item mais antigo
        if len(self.buffer) >= self.capacity:
            oldest = min(self.buffer, key=lambda x: x.last_rehearsed)
            self.buffer.remove(oldest)

        # Criar item
        item = WorkingMemoryItem(
            item_id=str(uuid.uuid4()),
            content=content,
            modality="verbal",
            activation=1.0,
            entered_at=time.time(),
            last_rehearsed=time.time()
        )

        self.buffer.append(item)
        return item

    def rehearse(self, item_id: str) -> bool:
        """Rehearse item para manter ativo"""
        item = next((i for i in self.buffer if i.item_id == item_id), None)

        if not item:
            return False

        # Boost activation
        item.activation = min(1.0, item.activation + 0.3)
        item.last_rehearsed = time.time()

        return True

    def decay_items(self):
        """Aplicar decay temporal"""
        current_time = time.time()

        for item in self.buffer:
            # Tempo desde último rehearsal
            time_since_rehearsal = current_time - item.last_rehearsed

            # Decay exponencial
            decay = self.decay_rate * time_since_rehearsal
            item.activation = max(0.0, item.activation - decay)

        # Remover itens com activation muito baixa
        self.buffer = [i for i in self.buffer if i.activation > 0.1]

    def retrieve(self, query: str) -> Optional[WorkingMemoryItem]:
        """Recuperar item por conteúdo"""
        # Busca por similaridade simples
        query_lower = str(query).lower()

        for item in self.buffer:
            if query_lower in str(item.content).lower():
                return item

        return None


class VisuospatialSketchpad:
    """
    Esboço Visuoespacial (Visuospatial Sketchpad).
    
    Mantém informação visual e espacial.
    Capacidade ~3-4 objetos.
    """

    @property
    def capacity(self) -> int:
        """Return the current normalized buffer capacity."""
        return self._capacity

    @capacity.setter
    def capacity(self, capacity: SupportsIndex) -> None:
        self._capacity = _positive_capacity(capacity)

    def __setstate__(self, state: Any) -> None:
        _restore_working_memory_state(self, state)

    def __init__(self, capacity: SupportsIndex = 4):
        self.capacity = capacity
        self.buffer: List[WorkingMemoryItem] = []

    def store_visual(self, content: Any) -> WorkingMemoryItem:
        """Armazenar informação visual"""
        if len(self.buffer) >= self.capacity:
            # Remover menos ativado
            least_active = min(self.buffer, key=lambda x: x.activation)
            self.buffer.remove(least_active)

        item = WorkingMemoryItem(
            item_id=str(uuid.uuid4()),
            content=content,
            modality="visual",
            activation=1.0,
            entered_at=time.time(),
            last_rehearsed=time.time()
        )

        self.buffer.append(item)
        return item

    def store_spatial(self, content: Any) -> WorkingMemoryItem:
        """Armazenar informação espacial"""
        if len(self.buffer) >= self.capacity:
            least_active = min(self.buffer, key=lambda x: x.activation)
            self.buffer.remove(least_active)

        item = WorkingMemoryItem(
            item_id=str(uuid.uuid4()),
            content=content,
            modality="spatial",
            activation=1.0,
            entered_at=time.time(),
            last_rehearsed=time.time()
        )

        self.buffer.append(item)
        return item

    def refresh(self, item_id: str) -> bool:
        """Refresh visual item"""
        item = next((i for i in self.buffer if i.item_id == item_id), None)

        if not item:
            return False

        item.activation = min(1.0, item.activation + 0.2)
        item.last_rehearsed = time.time()

        return True


class CentralExecutive:
    """
    Executivo Central (Central Executive).
    
    Controla e coordena working memory.
    Gerencia atenção, switching, e atualização.
    """

    def __init__(self, attention_system=None):
        self.attention_system = attention_system

        # Controle
        self.current_task: Optional[str] = None
        self.task_switching_cost = 0.2
        self.attention_allocations: List[Dict[str, Any]] = []

    def allocate_attention(self, target: str, strength: float = 0.8):
        """Alocar atenção para target"""
        normalized_strength = max(0.0, min(1.0, float(strength)))
        allocation = {
            "target": target,
            "strength": normalized_strength,
            "delegated": self.attention_system is not None,
        }
        self.attention_allocations.append(allocation)
        self.current_task = target
        return allocation

    def switch_task(self, new_task: str) -> float:
        """Mudar tarefa (com custo)"""
        if self.current_task == new_task:
            return 0.0  # Sem custo se mesma tarefa

        self.current_task = new_task
        return self.task_switching_cost

    def prioritize_item(self, item: WorkingMemoryItem, priority: float):
        """Priorizar item"""
        # Boost activation baseado em prioridade
        boost = priority * 0.3
        item.activation = min(1.0, item.activation + boost)


class RealWorkingMemorySystem:
    """
    POST6: Working Memory System — REAL Limited Capacity Buffer
    
    Working Memory = buffer temporário para processamento ativo.
    Capacidade limitada (~7 chunks verbal, ~4 visual).
    
    Componentes:
    - PhonologicalLoop: Info verbal/auditiva
    - VisuospatialSketchpad: Info visual/espacial
    - CentralExecutive: Controle e coordenação
    - Episodic Buffer: Integração multimodal
    
    Multiplica:
    • × POST1 (Memory): WM ↔ LTM transfer
    • × POST25 (Attention): Atenção em WM
    • × POST17 (Consciousness): Consciência do buffer
    • × POST12 (Multimodal): Buffers multimodais
    
    Baseado em:
    - Baddeley & Hitch's Working Memory Model (1974)
    - Baddeley's Episodic Buffer (2000)
    - Cowan's Embedded Processes Model (1999)
    - Oberauer's Concentric Model (2002)
    """

    def __init__(self, long_term_memory=None, attention=None, consciousness=None):
        self.ltm = long_term_memory
        self.attention = attention
        self.consciousness = consciousness

        # Componentes
        self.phonological_loop = PhonologicalLoop(capacity=7)
        self.visuospatial_sketchpad = VisuospatialSketchpad(capacity=4)
        self.central_executive = CentralExecutive(attention_system=attention)

        # Episodic buffer (integração)
        self.episodic_buffer: List[WorkingMemoryItem] = []
        self.episodic_capacity = 4

        # Estatísticas
        self.total_items_stored = 0
        self.rehearsal_count = 0
        self.items_transferred_to_ltm = 0

        print("🧠💾 POST 6 - RealWorkingMemorySystem initialized (Limited capacity buffer active)")

    def store_verbal(self, content: str) -> WorkingMemoryItem:
        """
        Armazenar informação verbal em WM.
        
        Args:
            content: Conteúdo verbal
        
        Returns:
            WorkingMemoryItem
        """
        item = self.phonological_loop.store(content)
        self.total_items_stored += 1

        # Consciência do buffer
        if self.consciousness:
            self.consciousness.attend_to(
                "working_memory",
                f"Stored in WM: {content[:50]}...",
                "cognitive",
                salience=0.5
            )

        return item

    def store_visual(self, content: Any) -> WorkingMemoryItem:
        """Armazenar informação visual em WM"""
        item = self.visuospatial_sketchpad.store_visual(content)
        self.total_items_stored += 1

        return item

    def store_spatial(self, content: Any) -> WorkingMemoryItem:
        """Armazenar informação espacial em WM"""
        item = self.visuospatial_sketchpad.store_spatial(content)
        self.total_items_stored += 1

        return item

    def rehearse_verbal(self, item_id: str):
        """Rehearse item verbal"""
        success = self.phonological_loop.rehearse(item_id)

        if success:
            self.rehearsal_count += 1

    def maintain_items(self):
        """Manter itens ativos (decay + rehearsal)"""
        # Apply decay
        self.phonological_loop.decay_items()

        # Auto-rehearsal dos itens mais importantes
        # (simplificado: rehearse itens com alta activation)
        for item in self.phonological_loop.buffer[:3]:  # Top 3
            self.phonological_loop.rehearse(item.item_id)
            self.rehearsal_count += 1

    def retrieve_from_wm(self, query: str) -> Optional[WorkingMemoryItem]:
        """
        Recuperar item de WM.
        
        Args:
            query: Query de busca
        
        Returns:
            WorkingMemoryItem ou None
        """
        # Buscar em phonological loop
        item = self.phonological_loop.retrieve(query)

        if item:
            # Boost activation ao recuperar
            item.activation = min(1.0, item.activation + 0.2)
            return item

        return None

    def transfer_to_ltm(self, item: WorkingMemoryItem):
        """
        Transferir item de WM para LTM.
        
        Consolidação: WM → LTM
        """
        if not self.ltm:
            return

        # Store in long-term memory
        self.ltm.store(
            str(item.content),
            "long_term",
            importance=item.activation
        )

        self.items_transferred_to_ltm += 1

    def integrate_multimodal(self, verbal_content: str, visual_content: Any) -> WorkingMemoryItem:
        """
        Integrar informação multimodal em episodic buffer.
        
        Args:
            verbal_content: Info verbal
            visual_content: Info visual
        
        Returns:
            Integrated item
        """
        # Criar item integrado
        integrated_content = {
            "verbal": verbal_content,
            "visual": visual_content
        }

        # Se buffer cheio, remover mais antigo
        if len(self.episodic_buffer) >= self.episodic_capacity:
            oldest = min(self.episodic_buffer, key=lambda x: x.entered_at)
            self.episodic_buffer.remove(oldest)

        # Criar item multimodal
        item = WorkingMemoryItem(
            item_id=str(uuid.uuid4()),
            content=integrated_content,
            modality="multimodal",
            activation=1.0,
            entered_at=time.time(),
            last_rehearsed=time.time()
        )

        self.episodic_buffer.append(item)
        self.total_items_stored += 1

        return item

    def get_capacity_usage(self) -> Dict[str, float]:
        """
        Obter uso de capacidade.
        
        Returns:
            Dict com % de uso de cada buffer
        """
        return {
            "phonological_loop": len(self.phonological_loop.buffer) / self.phonological_loop.capacity,
            "visuospatial_sketchpad": len(self.visuospatial_sketchpad.buffer) / self.visuospatial_sketchpad.capacity,
            "episodic_buffer": len(self.episodic_buffer) / self.episodic_capacity
        }

    def get_active_items(self) -> List[WorkingMemoryItem]:
        """Obter todos os itens ativos em WM"""
        all_items = (
            self.phonological_loop.buffer +
            self.visuospatial_sketchpad.buffer +
            self.episodic_buffer
        )

        # Ordenar por activation
        all_items.sort(key=lambda x: x.activation, reverse=True)

        return all_items

    def get_statistics(self) -> Dict:
        """Estatísticas do Working Memory System"""
        capacity_usage = self.get_capacity_usage()

        return {
            "total_items_stored": self.total_items_stored,
            "rehearsal_count": self.rehearsal_count,
            "items_transferred_to_ltm": self.items_transferred_to_ltm,
            "current_phonological_items": len(self.phonological_loop.buffer),
            "current_visuospatial_items": len(self.visuospatial_sketchpad.buffer),
            "current_episodic_items": len(self.episodic_buffer),
            "phonological_capacity_usage": capacity_usage["phonological_loop"],
            "visuospatial_capacity_usage": capacity_usage["visuospatial_sketchpad"],
            "episodic_capacity_usage": capacity_usage["episodic_buffer"]
        }
