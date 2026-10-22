"""Contrato R3: extração causal com banco de marcadores (Khoo et al., 2000), "cause" como
palavra inteira e escopo de negação estilo NegEx (Chapman et al., 2001).

(a) origem: marcadores além de "causes" (because [of], leads to, results in, due to) com a
    ordem causa/efeito correta;
(b) não regressão: "X causes Y" continua idêntico (confiança 0.9, tipo direct_cause);
(c) inválido/negativo: negação no escopo e substrings ("because", "causeway") não geram relação.
"""

import unittest

from core.cognitivo.causal_world_model import RealCausalReasoning


def pairs(text):
    return [(r.cause, r.effect, r.pattern_type) for r in RealCausalReasoning().extract_causal_relations(text)]


class CausalExtractionContractTests(unittest.TestCase):
    def test_direct_cause_unchanged(self):
        rels = RealCausalReasoning().extract_causal_relations("Smoking causes cancer")
        self.assertEqual([(r.cause, r.effect, r.confidence, r.pattern_type) for r in rels],
                         [("Smoking", "cancer", 0.9, "direct_cause")])

    def test_marker_bank_orders_cause_and_effect(self):
        self.assertEqual(pairs("Stress leads to insomnia")[0][:2], ("Stress", "insomnia"))
        self.assertEqual(pairs("Deforestation results in erosion")[0][:2], ("Deforestation", "erosion"))
        self.assertEqual(pairs("The service crashed because the disk was full")[0][:2],
                         ("the disk was full", "The service crashed"))
        self.assertEqual(pairs("Latency increased because of the cache eviction")[0][:2],
                         ("the cache eviction", "Latency increased"))
        self.assertEqual(pairs("Delays occurred due to the storm")[0][:2], ("the storm", "Delays occurred"))

    def test_negation_scope_blocks_relation(self):
        for text in ("Coffee does not cause cancer", "Vaccines never cause autism",
                     "Logging doesn't cause outages", "There is no evidence that caffeine causes anxiety"):
            self.assertEqual(pairs(text), [], text)

    def test_whole_word_and_correlation(self):
        for text in ("The causeway was closed", "We left early because", "Causal models are hard",
                     "The cause is unknown", "Sleep is associated with mood", ""):
            self.assertEqual(pairs(text), [], text)

    def test_one_relation_per_sentence_first_marker_wins(self):
        self.assertEqual(pairs("Stress leads to insomnia because of cortisol"),
                         [("Stress", "insomnia because of cortisol", "causal_verb")])

    def test_negated_marker_stops_search(self):
        self.assertEqual(pairs("Coffee does not cause cancer in most healthy adults because of genetics"), [])

    def test_graph_and_counter_follow_extractions(self):
        cr = RealCausalReasoning()
        cr.extract_causal_relations("Stress leads to insomnia")
        cr.extract_causal_relations("Coffee does not cause cancer")
        self.assertEqual(cr.get_statistics(), {"total_relations_extracted": 1, "graph_nodes": 1})


if __name__ == "__main__":
    unittest.main()
