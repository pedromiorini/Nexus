import unittest

from core.tool_evidence_gate import EvidenceGateRejection, evaluate_evidence_omission


class EvidenceOmissionContractTests(unittest.TestCase):
    def test_ts2a_missing_required_key_is_denied(self):
        allowed = evaluate_evidence_omission(["system_isolated"], {"status": "ok"})
        self.assertIs(allowed, False)

    def test_ts2b_exact_true_is_allowed(self):
        allowed = evaluate_evidence_omission(
            ["system_isolated"],
            {"status": "ok", "system_isolated": True},
        )
        self.assertIs(allowed, True)

    def test_ts2c_empty_required_allows_any_dict(self):
        self.assertIs(evaluate_evidence_omission([], {"status": "ok"}), True)
        self.assertIs(evaluate_evidence_omission([], {}), True)

    def test_ts2d_non_true_values_are_denied_without_exception(self):
        for value in (False, 1, "true", None):
            with self.subTest(value=value):
                allowed = evaluate_evidence_omission(
                    ["system_isolated"],
                    {"system_isolated": value},
                )
                self.assertIs(allowed, False)

    def test_ts2e_every_required_key_must_be_true(self):
        self.assertIs(
            evaluate_evidence_omission(
                ["system_isolated", "sandbox"],
                {"system_isolated": True},
            ),
            False,
        )
        self.assertIs(
            evaluate_evidence_omission(
                ["system_isolated", "sandbox"],
                {"system_isolated": True, "sandbox": True},
            ),
            True,
        )

    def test_required_none_bool_string_and_non_str_item_are_rejected(self):
        samples = (
            None,
            True,
            "system_isolated",
            ["system_isolated", 1],
            ["system_isolated", None],
        )
        for required in samples:
            with self.subTest(required=required):
                with self.assertRaises(EvidenceGateRejection) as ctx:
                    evaluate_evidence_omission(required, {"system_isolated": True})
                self.assertEqual(ctx.exception.code, "required_evidence_invalid")

    def test_non_dict_result_is_rejected(self):
        for result in (None, ["system_isolated"], "ok", True, 1):
            with self.subTest(result_type=type(result).__name__):
                with self.assertRaises(EvidenceGateRejection) as ctx:
                    evaluate_evidence_omission(["system_isolated"], result)
                self.assertEqual(ctx.exception.code, "tool_result_not_dict")

    def test_empty_required_with_non_dict_is_rejected(self):
        with self.assertRaises(EvidenceGateRejection) as ctx:
            evaluate_evidence_omission([], None)
        self.assertEqual(ctx.exception.code, "tool_result_not_dict")

    def test_invalid_required_wins_when_result_is_also_invalid(self):
        with self.assertRaises(EvidenceGateRejection) as ctx:
            evaluate_evidence_omission(None, None)
        self.assertEqual(ctx.exception.code, "required_evidence_invalid")

    def test_dict_subclass_with_true_value_is_allowed(self):
        class Payload(dict):
            pass

        allowed = evaluate_evidence_omission(
            ["system_isolated"],
            Payload(system_isolated=True),
        )
        self.assertIs(allowed, True)


if __name__ == "__main__":
    unittest.main()
