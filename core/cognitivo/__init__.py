"""Capacidades cognitivas extraídas do monólito core/constitutional_brain.py."""

from core.cognitivo.analogical_reasoning import (
    AnalogyMapping,
    AnalogySituation,
    RealAnalogicalReasoningSystem,
)
from core.cognitivo.attention import (
    AttentionShift,
    AttentionalFocus,
    AttentionalSpotlight,
    BottomUpAttention,
    RealAttentionSystem,
    StimulusItem,
    TopDownAttention,
)
from core.cognitivo.contextual_memory import (
    ContextEncoder,
    ContextMatch,
    ContextualMemory,
    RealContextualMemorySystem,
)
from core.cognitivo.emotional_memory_weighting import (
    EmotionalMemoryWeight,
    RealEmotionalMemoryWeighting,
)
from core.cognitivo.episodic_future_thinking import (
    FutureEvent,
    ProspectiveMemory,
    RealEpisodicFutureThinkingSystem,
)
from core.cognitivo.incremental_learner import (
    ConsolidationReport,
    LearningEvent,
    RealIncrementalLearner,
)
from core.cognitivo.mental_simulation import (
    MentalScenario,
    MentalSimulationResult,
    RealMentalSimulationSystem,
)
from core.cognitivo.working_memory import (
    CentralExecutive,
    PhonologicalLoop,
    RealWorkingMemorySystem,
    RehearsalEvent,
    VisuospatialSketchpad,
    WorkingMemoryItem,
)

__all__ = [
    "AnalogyMapping",
    "AnalogySituation",
    "AttentionShift",
    "AttentionalFocus",
    "AttentionalSpotlight",
    "BottomUpAttention",
    "CentralExecutive",
    "ConsolidationReport",
    "ContextEncoder",
    "ContextMatch",
    "ContextualMemory",
    "EmotionalMemoryWeight",
    "FutureEvent",
    "LearningEvent",
    "MentalScenario",
    "MentalSimulationResult",
    "PhonologicalLoop",
    "ProspectiveMemory",
    "RealAnalogicalReasoningSystem",
    "RealAttentionSystem",
    "RealContextualMemorySystem",
    "RealEmotionalMemoryWeighting",
    "RealEpisodicFutureThinkingSystem",
    "RealIncrementalLearner",
    "RealMentalSimulationSystem",
    "RealWorkingMemorySystem",
    "RehearsalEvent",
    "StimulusItem",
    "TopDownAttention",
    "VisuospatialSketchpad",
    "WorkingMemoryItem",
]
