import unittest

from core.temporal_facts import TemporalFactStore


class Clock:
    def __call__(self):
        return 10.0


class FiniteStoredFloatWithInfiniteConversion(float):
    def __float__(self):
        return float("inf")


class FiniteFloatSubclass(float):
    pass


class TemporalNumberFinitenessContract(unittest.TestCase):
    def test_rejects_non_finite_result_of_float_subclass_conversion(self):
        store = TemporalFactStore(functional_predicates=("mora_em",), clock=Clock())
        store.assert_fact("ana", "mora_em", "Curitiba", valid_from=0.0)
        before = store.export_snapshot()

        with self.assertRaisesRegex(ValueError, "número finito"):
            store.facts_as_of(FiniteStoredFloatWithInfiniteConversion(1.0))

        self.assertEqual(store.export_snapshot(), before)

    def test_accepts_finite_numeric_values_and_float_subclasses(self):
        store = TemporalFactStore(functional_predicates=("mora_em",), clock=Clock())
        store.assert_fact("ana", "mora_em", "Curitiba", valid_from=0.0)

        for instant in (1, 1.0, FiniteFloatSubclass(1.0)):
            with self.subTest(instant=instant):
                self.assertEqual(
                    [fact.object for fact in store.facts_as_of(instant)], ["Curitiba"]
                )

    def test_rejects_non_finite_query_instants_without_state_change(self):
        store = TemporalFactStore(functional_predicates=("mora_em",), clock=Clock())
        store.assert_fact("ana", "mora_em", "Curitiba", valid_from=0.0)
        before = store.export_snapshot()

        for instant in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(instant=instant):
                with self.assertRaisesRegex(ValueError, "número finito"):
                    store.facts_as_of(instant)
                self.assertEqual(store.export_snapshot(), before)


if __name__ == "__main__":
    unittest.main()
