"""C1–C3: o CentralRouter ainda não chama o ciclo de decisão.

Critérios congelados em
docs/cognitivo/PONTE_CENTRALROUTER_CARACTERIZACAO_2026-11-10.md.
C4 e C5 são os testes já existentes e os gates, não este arquivo.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.constitutional_brain import (
    CentralRouter,
    CompleteNexusBrain,
    EthicsGuard,
    ModuleType,
)

MODULE_NAMES = frozenset({
    "BUDGET",
    "CAUSAL",
    "EPISODIC",
    "INCREMENTAL_LEARNER",
    "JAILBREAK",
    "KNOWLEDGE_GRAPH",
    "LOG",
    "MCTS_PLANNER",
    "MEMORY",
    "REASONING",
    "SWARM",
    "WORLD_MODEL",
})

ZERO_STATISTICS = {
    "total_events_tracked": 0,
    "total_relations_detected": 0,
    "total_sequences_built": 0,
    "avg_sequence_length": 0.0,
    "active_events": 0,
    "temporal_relations": 0,
}


def _record(calls):
    def record(*args, **kwargs):
        calls.append((args, kwargs))

    return record


class RouterDoesNotCallTheDecisionCycle(unittest.TestCase):
    def test_route_ping_with_empty_modules_does_not_call_the_cycle(self):
        calls = []
        log = SimpleNamespace(log_event=lambda *args, **kwargs: "hash")
        jailbreak = SimpleNamespace(detect=lambda prompt: SimpleNamespace(detected=False))
        guard = EthicsGuard(log, jailbreak)
        router = CentralRouter({}, guard)
        with patch("core.cognitivo.decision_cycle.run_decision_cycle", _record(calls)), patch(
            "core.cognitivo.decision_cycle.select_operator", _record(calls)
        ):
            result = router.route("ping", {})
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertEqual(calls, [])


class TemporalReasoningStaysUntrackedAfterConstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = CompleteNexusBrain()

    def test_statistics_start_at_zero_and_no_event_is_tracked(self):
        engine = self.brain.temporal_reasoning
        self.assertEqual(engine.get_statistics(), ZERO_STATISTICS)
        self.assertEqual(engine.events, [])

    def test_integration_map_keeps_temporal_none_and_module_type_is_unchanged(self):
        temporal = self.brain.integration_orchestration.all_modules["temporal"]
        self.assertIsNone(temporal)
        self.assertEqual(frozenset(member.name for member in ModuleType), MODULE_NAMES)
