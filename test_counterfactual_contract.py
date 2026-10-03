import unittest

from core.constitutional_brain import RealCausalReasoning, RealCounterfactualReasoningEngine


class PredictiveWorldModel:
    def predict_outcome(self, situation, intervention):
        return f"model:{situation}:{intervention or 'none'}"


class CounterfactualContractTests(unittest.TestCase):
    def test_world_model_protocol_is_used_when_available(self):
        engine = RealCounterfactualReasoningEngine(world_model=PredictiveWorldModel())
        self.assertEqual(engine._predict_outcome("situation", "intervention"), "model:situation:intervention")

    def test_causal_protocol_builds_chain_when_available(self):
        causal = RealCausalReasoning()
        engine = RealCounterfactualReasoningEngine(causal_reasoning=causal)
        chain = engine._build_causal_chain("alarm", "patch")
        self.assertEqual(chain, ["alarm causes patch"])

    def test_heuristic_fallback_remains_explicit(self):
        engine = RealCounterfactualReasoningEngine()
        self.assertEqual(engine._predict_outcome("situation", None), "outcome_without_intervention")
        self.assertEqual(len(engine._build_causal_chain("situation", "change")), 5)


if __name__ == "__main__":
    unittest.main()
