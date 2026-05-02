# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Importar a bridge que acabamos de criar
from nexus_constitutional_bridge_v2 import NexusConstitutionalBridge

# Importar o CompleteNexusBrain para o teste de integração
# Em um cenário real, NEXUS_CORE_v3_94 seria um módulo importável
# Para este teste, vamos simular a estrutura necessária

# Mock para as classes do Nexus Vita
class MockPredictor:
    def __init__(self, accuracy=0.8):
        self.recent_accuracy = accuracy

class MockRules:
    def __init__(self, n_rules=10):
        self._n_rules = n_rules
    def n_rules(self):
        return self._n_rules

class MockLTM:
    def __init__(self, size=100, collective_episodes=5):
        self.size = size
        self._collective_episodes = [{} for _ in range(collective_episodes)]
    def collective_episodes(self):
        return self._collective_episodes

class MockCreativeEngine:
    def __init__(self, creativity_score=0.7, promoted_count=3):
        self.creativity_score = creativity_score
        self.promoted_count = promoted_count

class MockAnomalyDetector:
    def __init__(self, anomaly_count=2):
        self.anomaly_count = anomaly_count

class MockRD:
    def __init__(self, var_val=0.1, mean_val=0.5):
        self._var_val = var_val
        self._mean_val = mean_val
    def var(self):
        return self._var_val
    def mean(self):
        return self._mean_val

class MockPlanner:
    def __init__(self, plan_count=1):
        self.plan_count = plan_count

class MockNovelty:
    def __init__(self, boost_count=1):
        self.boost_count = boost_count

class MockArchMut:
    def __init__(self, mutation_history=1):
        self.mutation_history = [{} for _ in range(mutation_history)]

class MockResonance:
    def __init__(self, resonating=True):
        self.resonating = resonating

class MockRuleEmergence:
    def __init__(self, emerged=True):
        self.emerged = emerged

class MockLangMonitor:
    def __init__(self, alignment_history=[0.9]):
        self.alignment_history = alignment_history

class MockGW:
    def __init__(self, gw_colony=0):
        self.gw_colony = gw_colony

class MockSTM:
    def __init__(self, size=50):
        self.size = size

class MockVitaInstance:
    def __init__(self, fitness=0.8, meta_cognition=0.6):
        self.fitness = fitness
        self.meta_cognition = meta_cognition
        self.predictor = MockPredictor()
        self.rules = MockRules()
        self.ltm = MockLTM()
        self.creative_engine = MockCreativeEngine()
        self.anomaly_detector = MockAnomalyDetector()
        self.rd_u = MockRD()
        self.rd_params = {"F": 0.03, "K": 0.06}
        self.planner = MockPlanner()
        self.novelty = MockNovelty()
        self.arch_mut = MockArchMut()
        self.sleep_count = 5
        self.stm = MockSTM()

class MockNexusFederation:
    def __init__(self, n_colonies=2, n_instances_per_colony=3):
        self.n = n_colonies
        self._round = 100
        self.converged = True
        self.resonance = MockResonance()
        self.rule_emergence = MockRuleEmergence()
        self.lang_monitor = MockLangMonitor()
        self.gw = MockGW()
        self._instances = []
        for _ in range(n_colonies * n_instances_per_colony):
            self._instances.append(MockVitaInstance())

    def _all_instances(self):
        return self._instances

# Mock para o CompleteNexusBrain (apenas o necessário para o teste de integração)
class MockCompleteNexusBrain:
    def __init__(self):
        self.vita_federation = None
        self.vita_bridge = None
        self.modules_activated = []
        self.reasoning_result = {}

    def integrate_vita(self, federation: MockNexusFederation):
        self.vita_federation = federation
        self.vita_bridge = NexusConstitutionalBridge()

    def think(self, question: str, uci_global: float):
        # Simular a parte relevante do método think para a integração do Vita
        if self.vita_federation and self.vita_bridge:
            self.modules_activated.append("vita_integration")
            vita_brain_state = self.vita_bridge.get_brain_state(self.vita_federation, uci_global)
            if isinstance(self.reasoning_result, dict):
                self.reasoning_result["vita_state"] = vita_brain_state
            else:
                self.reasoning_result = {"constitutional_reasoning": self.reasoning_result, "vita_state": vita_brain_state}
        return self.reasoning_result


class TestNexusVitaIntegration(unittest.TestCase):

    def setUp(self):
        self.bridge = NexusConstitutionalBridge()
        self.federation = MockNexusFederation()
        self.uci_global = 0.75

    def test_get_brain_state_format(self):
        """Verifica se o formato do estado do cérebro retornado pela bridge está correto."""
        brain_state = self.bridge.get_brain_state(self.federation, self.uci_global)

        self.assertIsInstance(brain_state, dict)
        self.assertIn("module", brain_state)
        self.assertEqual(brain_state["module"], "NexusVita")
        self.assertIn("version", brain_state)
        self.assertEqual(brain_state["version"], "5.5")
        self.assertIn("phase", brain_state)
        self.assertEqual(brain_state["phase"], 36)
        self.assertIn("emotions", brain_state)
        self.assertIn("cognition", brain_state)
        self.assertIn("embodiment", brain_state)
        self.assertIn("social", brain_state)
        self.assertIn("memory", brain_state)
        self.assertIn("meta", brain_state)

    def test_emotions_data(self):
        """Verifica se os dados de emoções são calculados corretamente."""
        brain_state = self.bridge.get_brain_state(self.federation, self.uci_global)
        emotions = brain_state["emotions"]

        self.assertIsInstance(emotions["valence"], float)
        self.assertGreaterEqual(emotions["valence"], -1.0)
        self.assertLessEqual(emotions["valence"], 1.0)
        self.assertIsInstance(emotions["curiosity"], float)

    def test_cognition_data(self):
        """Verifica se os dados de cognição são calculados corretamente."""
        brain_state = self.bridge.get_brain_state(self.federation, self.uci_global)
        cognition = brain_state["cognition"]

        self.assertEqual(cognition["uci"], self.uci_global)
        self.assertIsInstance(cognition["predictive_acc"], float)
        self.assertIsInstance(cognition["n_rules"], float)

    def test_integration_with_constitutional_brain(self):
        """Verifica se o CompleteNexusBrain integra o estado do Vita corretamente."""
        constitutional_brain = MockCompleteNexusBrain()
        constitutional_brain.integrate_vita(self.federation)

        question = "What is the meaning of life?"
        uci_global_for_think = 0.8
        result = constitutional_brain.think(question, uci_global_for_think)

        self.assertIn("vita_state", result)
        self.assertEqual(result["vita_state"]["module"], "NexusVita")
        self.assertIn("vita_integration", constitutional_brain.modules_activated)
        self.assertEqual(result["vita_state"]["cognition"]["uci"], uci_global_for_think)

    def test_empty_federation(self):
        """Testa o comportamento da bridge com uma federação vazia."""
        empty_federation = MockNexusFederation(n_colonies=0, n_instances_per_colony=0)
        brain_state = self.bridge.get_brain_state(empty_federation, self.uci_global)

        self.assertIsInstance(brain_state, dict)
        self.assertEqual(brain_state["emotions"]["valence"], -1.0) # Default for empty
        self.assertEqual(brain_state["cognition"]["n_rules"], 0.0)
        self.assertEqual(brain_state["social"]["n_instances"], 0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
