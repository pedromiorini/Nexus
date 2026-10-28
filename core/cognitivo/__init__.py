"""Capacidades cognitivas extraídas do monólito core/constitutional_brain.py."""

from core.cognitivo.attention import (
    AttentionShift,
    AttentionalFocus,
    AttentionalSpotlight,
    BottomUpAttention,
    RealAttentionSystem,
    StimulusItem,
    TopDownAttention,
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
    "AttentionShift",
    "AttentionalFocus",
    "AttentionalSpotlight",
    "BottomUpAttention",
    "CentralExecutive",
    "ConsolidationReport",
    "FutureEvent",
    "LearningEvent",
    "MentalScenario",
    "MentalSimulationResult",
    "PhonologicalLoop",
    "ProspectiveMemory",
    "RealAttentionSystem",
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
