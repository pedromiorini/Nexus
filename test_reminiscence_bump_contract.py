import unittest

from core.constitutional_brain import RealReminiscenceBumpClustering


class ReminiscenceBumpContractTests(unittest.TestCase):
    def test_peak_in_formative_range_is_marked_typical(self):
        clustering = RealReminiscenceBumpClustering()
        bump = clustering.identify_reminiscence_bump([(1, 15), (2, 16), (3, 40)])
        self.assertTrue(bump.is_typical)
        self.assertEqual(bump.age_range, (15, 20))

    def test_peak_outside_formative_range_is_marked_atypical(self):
        clustering = RealReminiscenceBumpClustering()
        bump = clustering.identify_reminiscence_bump([(1, 40), (2, 41), (3, 15)])
        self.assertFalse(bump.is_typical)
        self.assertEqual(bump.age_range, (40, 45))


if __name__ == "__main__":
    unittest.main()
