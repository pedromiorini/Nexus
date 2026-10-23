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
    "PhonologicalLoop",
    "RealAttentionSystem",
    "RealWorkingMemorySystem",
    "RehearsalEvent",
    "StimulusItem",
    "TopDownAttention",
    "VisuospatialSketchpad",
    "WorkingMemoryItem",
]
