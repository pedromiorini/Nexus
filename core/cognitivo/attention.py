"""Attention and focus (POST25) extraída de core/constitutional_brain.py.

Extração strangler: o código abaixo foi movido sem alteração de comportamento;
core.constitutional_brain reexporta as classes no ponto original.
Não há relógio injetável: clear, duração de foco e timestamps usam time.time().
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = [
    "AttentionShift",
    "AttentionalFocus",
    "AttentionalSpotlight",
    "BottomUpAttention",
    "RealAttentionSystem",
    "StimulusItem",
    "TopDownAttention",
]


# ═══════════════════════════════════════════════════════════════════════════
# MÓDULO 11A3: ATTENTION & FOCUS SYSTEM (POST 25)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StimulusItem:
    """Item de estímulo que compete por atenção"""
    stimulus_id: str
    stimulus_type: str  # "visual", "auditory", "internal"
    content: Any
    salience: float  # Proeminência do estímulo (0-1)
    priority: float  # Prioridade baseada em objetivos (0-1)
    timestamp: float
    contrast: Optional[float] = None  # Contraste observado, quando disponível
    motion: Optional[float] = None  # Movimento observado, quando disponível


@dataclass
class AttentionalFocus:
    """Foco atencional atual"""
    focus_id: str
    focused_item: StimulusItem
    focus_strength: float  # Intensidade do foco (0-1)
    start_time: float
    duration: Optional[float]


@dataclass
class AttentionShift:
    """Mudança de atenção"""
    shift_id: str
    from_focus: Optional[str]
    to_focus: str
    shift_reason: str  # "salience", "goal", "voluntary"
    cost: float  # Custo cognitivo da mudança (0-1)
    timestamp: float


class BottomUpAttention:
    """
    Atenção Bottom-Up (guiada por estímulos).
    
    Detecta estímulos salientes no ambiente.
    "Algo chamou minha atenção!"
    """

    def __init__(self):
        # Fatores de saliência
        self.salience_factors = {
            "novelty": 0.3,      # Novidade
            "intensity": 0.25,    # Intensidade
            "contrast": 0.25,     # Contraste com background
            "motion": 0.2         # Movimento/mudança
        }

    def compute_salience(self, stimulus: StimulusItem, context: Dict[str, Any] | None = None) -> float:
        """
        Computar saliência de estímulo.
        
        Args:
            stimulus: Item de estímulo
            context: Contexto atual
        
        Returns:
            Salience score (0-1)
        """
        context = context or {}

        # Fatores de saliência
        novelty = self._assess_novelty(stimulus, context)
        intensity = self._assess_intensity(stimulus)
        contrast = self._assess_contrast(stimulus, context)
        motion = self._assess_motion(stimulus)

        # Weighted sum
        salience = (
            self.salience_factors["novelty"] * novelty +
            self.salience_factors["intensity"] * intensity +
            self.salience_factors["contrast"] * contrast +
            self.salience_factors["motion"] * motion
        )

        return min(1.0, salience)

    def _assess_novelty(self, stimulus: StimulusItem, context: Dict) -> float:
        """Avaliar novidade do estímulo"""
        # Novo se não visto antes
        seen_stimuli = context.get("seen_stimuli", [])

        if stimulus.stimulus_id not in seen_stimuli:
            return 0.9  # Alta novidade

        return 0.2  # Baixa novidade (familiar)

    def _assess_intensity(self, stimulus: StimulusItem) -> float:
        """Avaliar intensidade do estímulo"""
        # Baseado no tipo
        if stimulus.stimulus_type == "visual":
            # Estímulos visuais fortes
            return 0.7
        elif stimulus.stimulus_type == "auditory":
            # Estímulos auditivos moderados
            return 0.6
        else:
            # Estímulos internos mais fracos
            return 0.4

    def _assess_contrast(self, stimulus: StimulusItem, context: Dict) -> float:
        """Avaliar contraste com background"""
        if stimulus.contrast is not None:
            return min(1.0, max(0.0, stimulus.contrast))

        # Compatibilidade com produtores que fornecem o sinal no contexto.
        context_contrast = context.get("contrast")
        if context_contrast is not None:
            return min(1.0, max(0.0, float(context_contrast)))

        # Fallback heurístico explícito para estímulos sem observação.
        return 0.5

    def _assess_motion(self, stimulus: StimulusItem) -> float:
        """Avaliar movimento/mudança"""
        if stimulus.motion is not None:
            return min(1.0, max(0.0, stimulus.motion))

        # Fallback heurístico explícito para estímulos sem observação.
        return 0.6


class TopDownAttention:
    """
    Atenção Top-Down (guiada por objetivos).
    
    Direciona atenção baseado em objetivos e expectativas.
    "Eu preciso focar nisso!"
    """

    def __init__(self):
        # Objetivos ativos
        self.active_goals: List[str] = []

        # Pesos de relevância por objetivo
        self.goal_relevance: Dict[str, float] = {}

    def set_goals(self, goals: List[str]):
        """Definir objetivos ativos"""
        self.active_goals = goals

        # Inicializar relevâncias
        for goal in goals:
            if goal not in self.goal_relevance:
                self.goal_relevance[goal] = 0.5

    def compute_goal_relevance(self, stimulus: StimulusItem) -> float:
        """
        Computar relevância do estímulo para objetivos.
        
        Args:
            stimulus: Item de estímulo
        
        Returns:
            Relevance score (0-1)
        """
        if not self.active_goals:
            return 0.3  # Sem objetivos = relevância baixa

        # Checar relevância para cada objetivo
        relevances = []

        for goal in self.active_goals:
            relevance = self._assess_relevance_to_goal(stimulus, goal)
            weight = self.goal_relevance.get(goal, 0.5)
            relevances.append(relevance * weight)

        # Máxima relevância
        return max(relevances) if relevances else 0.3

    def _assess_relevance_to_goal(self, stimulus: StimulusItem, goal: str) -> float:
        """Avaliar relevância de estímulo para objetivo"""
        # Simplificado: match por palavras-chave
        stimulus_str = str(stimulus.content).lower()
        goal_words = goal.lower().split()

        # Conta quantas palavras do objetivo aparecem no estímulo
        matches = sum(1 for word in goal_words if word in stimulus_str)

        if matches > 0:
            return min(1.0, matches / len(goal_words))

        return 0.1  # Baixa relevância default

    def update_goal_priority(self, goal: str, priority: float):
        """Atualizar prioridade de objetivo"""
        self.goal_relevance[goal] = min(1.0, max(0.0, priority))


class AttentionalSpotlight:
    """
    Spotlight Atencional (Foco limitado).
    
    Mantém foco limitado em um subconjunto de estímulos.
    Baseado em Posner's Spotlight Model.
    """

    def __init__(self, capacity: int = 3):
        self.capacity = capacity  # Quantos itens podem estar em foco
        self.spotlight: List[StimulusItem] = []

        # Histórico de focos
        self.focus_history: List[AttentionalFocus] = []
        self.current_focus: Optional[AttentionalFocus] = None

    def focus_on(self, stimulus: StimulusItem, strength: float = 0.8):
        """
        Focar em estímulo.
        
        Args:
            stimulus: Item a focar
            strength: Força do foco (0-1)
        """
        # Se já no spotlight, aumentar força
        if stimulus in self.spotlight:
            # Já focado
            return

        # Se spotlight cheio, remover item menos saliente
        if len(self.spotlight) >= self.capacity:
            least_salient = min(self.spotlight, key=lambda x: x.salience)
            self.spotlight.remove(least_salient)

        # Adicionar ao spotlight
        self.spotlight.append(stimulus)

        # Criar focus
        focus = AttentionalFocus(
            focus_id=str(uuid.uuid4()),
            focused_item=stimulus,
            focus_strength=strength,
            start_time=time.time(),
            duration=None
        )

        self.current_focus = focus
        self.focus_history.append(focus)

    def release_focus(self, stimulus_id: str):
        """Liberar foco de estímulo"""
        self.spotlight = [s for s in self.spotlight if s.stimulus_id != stimulus_id]

        if self.current_focus and self.current_focus.focused_item.stimulus_id == stimulus_id:
            self.current_focus.duration = time.time() - self.current_focus.start_time
            self.current_focus = None

    def get_focused_items(self) -> List[StimulusItem]:
        """Obter itens em foco"""
        return self.spotlight.copy()

    def is_focused(self, stimulus_id: str) -> bool:
        """Verificar se estímulo está em foco"""
        return any(s.stimulus_id == stimulus_id for s in self.spotlight)


class RealAttentionSystem:
    """
    POST25: Attention & Focus System — REAL Selective Attention
    
    Sistema de atenção = selecionar subconjunto relevante de informação.
    
    Componentes:
    - BottomUpAttention: Atenção guiada por estímulos
    - TopDownAttention: Atenção guiada por objetivos
    - AttentionalSpotlight: Foco limitado
    - Attention Shifting: Mudança de foco
    
    Multiplica:
    • × POST17 (Consciousness): Consciência atencional
    • × POST12 (Multimodal): Atenção multimodal
    • × POST24 (Metacognition): Controle atencional
    • × POST28 (Budget): Recursos atencionais limitados
    
    Baseado em:
    - Posner's Spotlight Model (1980)
    - Feature Integration Theory (Treisman & Gelade, 1980)
    - Biased Competition Model (Desimone & Duncan, 1995)
    - Load Theory (Lavie, 1995)
    """

    def __init__(self, consciousness=None, multimodal=None, metacognition=None):
        self.consciousness = consciousness
        self.multimodal = multimodal
        self.metacognition = metacognition

        # Componentes
        self.bottom_up = BottomUpAttention()
        self.top_down = TopDownAttention()
        self.spotlight = AttentionalSpotlight(capacity=3)

        # Estímulos disponíveis
        self.available_stimuli: List[StimulusItem] = []

        # Histórico de shifts
        self.attention_shifts: List[AttentionShift] = []

        # Estatísticas
        self.total_attentional_episodes = 0
        self.total_shifts = 0
        self.voluntary_shifts = 0
        self.involuntary_shifts = 0

        print("👁️🎯 POST 25 - RealAttentionSystem initialized (Selective attention active)")

    def register_stimulus(self, 
                         stimulus_type: str,
                         content: Any,
                         context: Dict[str, Any] | None = None) -> StimulusItem:
        """
        Registrar novo estímulo.
        
        Args:
            stimulus_type: Tipo ("visual", "auditory", "internal")
            content: Conteúdo do estímulo
            context: Contexto
        
        Returns:
            StimulusItem criado
        """
        context = context or {}

        # Criar estímulo
        stimulus = StimulusItem(
            stimulus_id=str(uuid.uuid4()),
            stimulus_type=stimulus_type,
            content=content,
            salience=0.0,  # Será computado
            priority=0.0,  # Será computado
            timestamp=time.time()
        )

        # Computar saliência (bottom-up)
        stimulus.salience = self.bottom_up.compute_salience(stimulus, context)

        # Computar prioridade (top-down)
        stimulus.priority = self.top_down.compute_goal_relevance(stimulus)

        # Adicionar à lista
        self.available_stimuli.append(stimulus)

        return stimulus

    def attend(self, goals: List[str] | None = None) -> List[StimulusItem]:
        """
        Processo de atenção completo.
        
        Args:
            goals: Objetivos ativos para top-down attention
        
        Returns:
            Itens selecionados para atenção
        """
        # Definir objetivos
        if goals:
            self.top_down.set_goals(goals)

        # Se não há estímulos, retornar vazio
        if not self.available_stimuli:
            return []

        # Selecionar estímulos baseado em saliência + prioridade
        selected = self._select_for_attention()

        # Focar nos selecionados
        for stimulus in selected:
            if not self.spotlight.is_focused(stimulus.stimulus_id):
                self.spotlight.focus_on(stimulus)

        self.total_attentional_episodes += 1

        # Consciência atencional
        if self.consciousness:
            self.consciousness.attend_to(
                "attention",
                f"Attending to {len(selected)} stimuli",
                "attentional",
                salience=0.6
            )

        return selected

    def _select_for_attention(self) -> List[StimulusItem]:
        """Selecionar estímulos para atenção"""
        # Computar attention score = salience + priority
        for stimulus in self.available_stimuli:
            stimulus.attention_score = (stimulus.salience + stimulus.priority) / 2.0

        # Ordenar por attention_score
        sorted_stimuli = sorted(
            self.available_stimuli,
            key=lambda x: getattr(x, 'attention_score', 0),
            reverse=True
        )

        # Selecionar top N (limitado por capacity)
        selected = sorted_stimuli[:self.spotlight.capacity]

        return selected

    def shift_attention(self, to_stimulus_id: str, reason: str = "voluntary") -> bool:
        """
        Mudar atenção para estímulo.
        
        Args:
            to_stimulus_id: ID do estímulo alvo
            reason: Razão da mudança ("voluntary", "salience", "goal")
        
        Returns:
            True se mudança bem-sucedida
        """
        # Encontrar estímulo alvo
        target = next((s for s in self.available_stimuli if s.stimulus_id == to_stimulus_id), None)

        if not target:
            return False

        # Custo de mudança (switching cost)
        cost = self._compute_shift_cost(target)

        # Criar shift
        shift = AttentionShift(
            shift_id=str(uuid.uuid4()),
            from_focus=self.spotlight.current_focus.focus_id if self.spotlight.current_focus else None,
            to_focus=to_stimulus_id,
            shift_reason=reason,
            cost=cost,
            timestamp=time.time()
        )

        self.attention_shifts.append(shift)
        self.total_shifts += 1

        if reason == "voluntary":
            self.voluntary_shifts += 1
        else:
            self.involuntary_shifts += 1

        # Executar mudança
        self.spotlight.focus_on(target)

        return True

    def _compute_shift_cost(self, target: StimulusItem) -> float:
        """Computar custo de mudança de atenção"""
        # Custo depende de:
        # 1. Distância do foco atual
        # 2. Tipo de estímulo

        base_cost = 0.2

        # Custo maior se mudando entre modalidades
        if self.spotlight.current_focus:
            current_type = self.spotlight.current_focus.focused_item.stimulus_type
            if current_type != target.stimulus_type:
                base_cost += 0.3  # Cross-modal shift cost

        return min(1.0, base_cost)

    def clear_attended_stimuli(self):
        """Limpar estímulos já atendidos"""
        # Remove estímulos antigos
        current_time = time.time()
        self.available_stimuli = [
            s for s in self.available_stimuli
            if current_time - s.timestamp < 60.0  # Keep last 60s
        ]

    def get_statistics(self) -> Dict:
        """Estatísticas do Attention System"""
        return {
            "total_attentional_episodes": self.total_attentional_episodes,
            "total_shifts": self.total_shifts,
            "voluntary_shifts": self.voluntary_shifts,
            "involuntary_shifts": self.involuntary_shifts,
            "available_stimuli": len(self.available_stimuli),
            "focused_items": len(self.spotlight.spotlight),
            "focus_capacity": self.spotlight.capacity
        }
