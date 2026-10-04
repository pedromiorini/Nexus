import unittest

from core.constitutional_brain import BottomUpAttention, StimulusItem


class AttentionSignalContractTests(unittest.TestCase):
    def _stimulus(self, **kwargs):
        values = {
            "stimulus_id": "s1",
            "stimulus_type": "visual",
            "content": {"label": "object"},
            "salience": 0.5,
            "priority": 0.5,
            "timestamp": 1.0,
        }
        values.update(kwargs)
        return StimulusItem(**values)

    def test_explicit_contrast_and_motion_drive_salience(self):
        attention = BottomUpAttention()
        stimulus = self._stimulus(contrast=1.4, motion=-0.2)

        self.assertEqual(attention._assess_contrast(stimulus, {}), 1.0)
        self.assertEqual(attention._assess_motion(stimulus), 0.0)
        score = attention.compute_salience(stimulus, {"seen_stimuli": ["s1"]})
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_context_contrast_is_used_when_stimulus_signal_is_absent(self):
        attention = BottomUpAttention()
        stimulus = self._stimulus()
        self.assertAlmostEqual(attention._assess_contrast(stimulus, {"contrast": 0.25}), 0.25)

    def test_missing_signals_keep_documented_heuristic_fallbacks(self):
        attention = BottomUpAttention()
        stimulus = self._stimulus()
        self.assertEqual(attention._assess_contrast(stimulus, {}), 0.5)
        self.assertEqual(attention._assess_motion(stimulus), 0.6)


if __name__ == "__main__":
    unittest.main()
