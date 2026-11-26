import unittest

from core.tool_evidence_gate import evaluate_evidence_omission


class EvidenceGateSubclassContractTests(unittest.TestCase):
    def test_dict_subclass_cannot_forge_missing_evidence_or_run_overrides(self):
        class Payload(dict):
            def __init__(self):
                super().__init__()
                self.calls = []

            def __contains__(self, key):
                self.calls.append("__contains__")
                return True

            def __getitem__(self, key):
                self.calls.append("__getitem__")
                return True

            def items(self):
                self.calls.append("items")
                return ()

            def __iter__(self):
                self.calls.append("__iter__")
                return iter(())

        payload = Payload()
        allowed = evaluate_evidence_omission(["system_isolated"], payload)
        self.assertIs(allowed, False)
        self.assertEqual(payload.calls, [])

    def test_required_str_subclass_is_normalized_without_overrides(self):
        calls = []

        class Key(str):
            def __str__(self):
                calls.append("__str__")
                raise AssertionError("str override executed")

            def __hash__(self):
                calls.append("__hash__")
                raise AssertionError("hash override executed")

            def __eq__(self, other):
                calls.append("__eq__")
                raise AssertionError("eq override executed")

        allowed = evaluate_evidence_omission(
            [Key("system_isolated")], {"system_isolated": True}
        )
        self.assertIs(allowed, True)
        self.assertEqual(calls, [])

    def test_payload_str_subclass_key_is_read_without_key_overrides(self):
        calls = []

        class Key(str):
            __hash__ = str.__hash__

            def __str__(self):
                calls.append("__str__")
                raise AssertionError("str override executed")

            def __eq__(self, other):
                calls.append("__eq__")
                raise AssertionError("eq override executed")

        payload = {Key("system_isolated"): True}
        allowed = evaluate_evidence_omission(["system_isolated"], payload)
        self.assertIs(allowed, True)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
