from fractions import Fraction
import math
import unittest

from core.constitutional_brain import RealPredictiveCodingSystem


class POMDPBeliefFilterContractTests(unittest.TestCase):
    def test_one_step_update_uses_current_to_next_transition_direction(self):
        result = RealPredictiveCodingSystem.update_belief_state(
            states=("safe", "compromised"),
            prior=(1.0, 0.0),
            transition_model=((0.75, 0.25), (0.1, 0.9)),
            observation_likelihoods=(0.2, 0.8),
        )

        self.assertAlmostEqual(result["safe"], 3.0 / 7.0, places=14)
        self.assertAlmostEqual(result["compromised"], 4.0 / 7.0, places=14)
        self.assertAlmostEqual(math.fsum(result.values()), 1.0, places=14)

    def test_common_subnormal_likelihood_scale_does_not_create_false_zero_evidence(self):
        result = RealPredictiveCodingSystem.update_belief_state(
            states=("a", "b"),
            prior=(0.3, 0.7),
            transition_model=((1.0, 0.0), (0.0, 1.0)),
            observation_likelihoods=(5e-324, 5e-324),
        )

        self.assertAlmostEqual(result["a"], 0.3, places=12)
        self.assertAlmostEqual(result["b"], 0.7, places=12)

    def test_jointly_tiny_prior_and_transition_keep_nonzero_evidence(self):
        result = RealPredictiveCodingSystem.update_belief_state(
            states=("rare", "common"),
            prior=(1e-200, 1.0),
            transition_model=((1e-200, 1.0), (0.0, 1.0)),
            observation_likelihoods=(1.0, 0.0),
        )

        self.assertEqual(result, {"rare": 1.0, "common": 0.0})

    def test_true_zero_evidence_is_rejected_instead_of_fabricating_a_posterior(self):
        with self.assertRaisesRegex(ValueError, "zero probability"):
            RealPredictiveCodingSystem.update_belief_state(
                states=("a", "b"),
                prior=(1.0, 0.0),
                transition_model=((1.0, 0.0), (0.0, 1.0)),
                observation_likelihoods=(0.0, 1.0),
            )

    def test_state_count_is_bounded_before_model_materialization(self):
        states = tuple(f"s{index}" for index in range(RealPredictiveCodingSystem.MAX_BELIEF_STATES + 1))
        with self.assertRaisesRegex(ValueError, "state count exceeds"):
            RealPredictiveCodingSystem.update_belief_state(states, (), (), ())

    def test_probability_values_must_be_real_finite_and_in_range(self):
        identity = ((1.0, 0.0), (0.0, 1.0))
        invalid_inputs = (
            ((True, False), identity, (0.5, 0.5)),
            ((math.nan, 0.0), identity, (0.5, 0.5)),
            ((math.inf, 0.0), identity, (0.5, 0.5)),
            ((1.1, 0.0), identity, (0.5, 0.5)),
            ((0.5, 0.5), ((1.1, 0.0), (0.0, 1.0)), (0.5, 0.5)),
            ((0.5, 0.5), identity, (0.5, -0.1)),
            ((0.5, 0.5), identity, (0.5, math.inf)),
            ((1e308, 1e308), identity, (1.0, 1.0)),
        )
        for prior, transition_model, likelihoods in invalid_inputs:
            with self.subTest(prior=prior, transition_model=transition_model, likelihoods=likelihoods):
                with self.assertRaises(ValueError):
                    RealPredictiveCodingSystem.update_belief_state(
                        ("a", "b"), prior, transition_model, likelihoods
                    )

    def test_nonzero_probability_below_float_range_is_rejected(self):
        tiny = Fraction(1, 10**1000)
        with self.assertRaisesRegex(ValueError, "below float precision"):
            RealPredictiveCodingSystem.update_belief_state(
                ("a", "b"),
                (0.5, 0.5),
                ((1.0, 0.0), (0.0, 1.0)),
                (tiny, 1.0),
            )

    def test_prior_and_transition_rows_must_be_normalized(self):
        with self.assertRaisesRegex(ValueError, "prior must be normalized"):
            RealPredictiveCodingSystem.update_belief_state(
                ("a", "b"), (0.4, 0.4), ((1.0, 0.0), (0.0, 1.0)), (0.5, 0.5)
            )
        with self.assertRaisesRegex(ValueError, "each transition_model row"):
            RealPredictiveCodingSystem.update_belief_state(
                ("a", "b"), (0.5, 0.5), ((0.6, 0.6), (0.0, 1.0)), (0.5, 0.5)
            )

    def test_state_labels_and_dimensions_are_validated(self):
        valid_matrix = ((1.0, 0.0), (0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "unique"):
            RealPredictiveCodingSystem.update_belief_state(
                ("a", "a"), (0.5, 0.5), valid_matrix, (0.5, 0.5)
            )
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            RealPredictiveCodingSystem.update_belief_state(
                ("a", " "), (0.5, 0.5), valid_matrix, (0.5, 0.5)
            )
        with self.assertRaisesRegex(ValueError, "prior length"):
            RealPredictiveCodingSystem.update_belief_state(
                ("a", "b"), (1.0,), valid_matrix, (0.5, 0.5)
            )
        with self.assertRaisesRegex(ValueError, "square matrix"):
            RealPredictiveCodingSystem.update_belief_state(
                ("a", "b"), (0.5, 0.5), ((1.0,), (0.0, 1.0)), (0.5, 0.5)
            )
        with self.assertRaisesRegex(ValueError, "likelihoods length"):
            RealPredictiveCodingSystem.update_belief_state(
                ("a", "b"), (0.5, 0.5), valid_matrix, (1.0,)
            )

    def test_input_sequences_are_not_mutated(self):
        states = ["a", "b"]
        prior = [0.5, 0.5]
        transition_model = [[0.9, 0.1], [0.2, 0.8]]
        likelihoods = [0.7, 0.3]
        before = (states[:], prior[:], [row[:] for row in transition_model], likelihoods[:])

        RealPredictiveCodingSystem.update_belief_state(
            states, prior, transition_model, likelihoods
        )

        self.assertEqual((states, prior, transition_model, likelihoods), before)


if __name__ == "__main__":
    unittest.main()
