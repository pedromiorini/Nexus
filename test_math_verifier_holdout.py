"""Synthetic-only tests for the offline math-verifier holdout harness."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core.math_result_verifier as math_result_verifier
from core.constitutional_brain import ToolExecutionResult
from tools import math_verifier_holdout as harness


class _FixedTool:
    def execute_tool(self, tool_name: str, params: dict) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=tool_name,
            success=True,
            result=0,
            error=None,
            execution_time=0.001,
            cached=False,
            timestamp=1.0,
        )


class MathVerifierHoldoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _small_freeze(self, name: str = "synthetic") -> Path:
        destination = self.tmp_path / name
        harness.freeze(destination, count=3)
        return destination

    @staticmethod
    def _prediction(case: dict[str, str], **overrides: object) -> dict:
        record = {
            "id": case["id"],
            "expression": case["expression"],
            "tool_name": "math_calculate",
            "success": True,
            "result": 0,
            "error": None,
            "execution_time": 0.01,
            "cached": False,
            "timestamp": 1.0,
        }
        record.update(overrides)
        return record

    def test_freeze_is_random_label_free_and_integrity_checkable(self) -> None:
        directory = self._small_freeze()
        manifest, cases, manifest_hash = harness.verify_frozen(directory)

        self.assertEqual(manifest["count"], 3)
        self.assertEqual(len(manifest["seed_hex"]), 64)
        self.assertEqual(len(manifest_hash), 64)
        self.assertEqual(manifest["oracle_version"], harness.ORACLE_VERSION)
        self.assertEqual(manifest["harness_version"], harness.SCHEMA_VERSION)
        self.assertEqual(len(manifest["source_bundle_sha256"]), 64)
        self.assertIn("requirements-ci.lock", manifest["sources"])
        self.assertEqual(len({case["expression"] for case in cases}), 3)
        for case in cases:
            left, right = case["expression"].split(" + ")
            self.assertTrue(10 <= len(left) <= 12 and 10 <= len(right) <= 12)
            self.assertFalse({"label", "answer", "expected", "result"} & set(case))
        self.assertEqual(
            {path.name for path in directory.iterdir()},
            {harness.INPUT_NAME, harness.MANIFEST_NAME, harness.MANIFEST_HASH_NAME},
        )

    def test_input_or_manifest_tampering_is_detected(self) -> None:
        directory = self._small_freeze()
        path = directory / harness.INPUT_NAME
        payload = json.loads(path.read_text("utf-8"))
        payload["cases"][0]["expression"] = "1234567890 + 9876543210"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(harness.HarnessError, "input SHA256"):
            harness.verify_frozen(directory)

        manifest_path = directory / harness.MANIFEST_NAME
        manifest_path.write_text(manifest_path.read_text("utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(harness.HarnessError, "manifest digest"):
            harness.verify_frozen(directory)

    def test_predict_and_score_without_freeze_create_no_artifacts(self) -> None:
        directory = self.tmp_path / "not-frozen"
        directory.mkdir()
        with self.assertRaises(harness.HarnessError):
            harness._predict_with_tool(directory, _FixedTool())
        with self.assertRaises(harness.HarnessError):
            harness.score(directory)
        self.assertEqual(list(directory.iterdir()), [])

    def test_pairing_is_one_to_one_and_type_strict(self) -> None:
        _, cases, _ = harness.verify_frozen(self._small_freeze())
        valid = [self._prediction(case) for case in cases]
        self.assertEqual(harness._validate_pairing(cases, valid), valid)

        duplicate_id = [valid[0], valid[0], valid[2]]
        with self.assertRaisesRegex(harness.HarnessError, "ids"):
            harness._validate_pairing(cases, duplicate_id)

        wrong_expression = [dict(row) for row in valid]
        wrong_expression[0]["expression"] = cases[1]["expression"]
        with self.assertRaisesRegex(harness.HarnessError, "expression"):
            harness._validate_pairing(cases, wrong_expression)

        wrong_success_type = [dict(row) for row in valid]
        wrong_success_type[0]["success"] = 1
        with self.assertRaisesRegex(harness.HarnessError, "types"):
            harness._validate_pairing(cases, wrong_success_type)

    def test_predict_seals_and_never_overwrites_and_score_report_is_exclusive(self) -> None:
        directory = self._small_freeze()
        harness._predict_with_tool(directory, _FixedTool())
        self.assertTrue((directory / harness.PREDICTIONS_NAME).is_file())
        self.assertTrue((directory / harness.SEAL_NAME).is_file())
        before = (directory / harness.PREDICTIONS_NAME).read_bytes()

        with self.assertRaises(FileExistsError):
            harness._predict_with_tool(directory, _FixedTool())
        self.assertEqual((directory / harness.PREDICTIONS_NAME).read_bytes(), before)

        report = harness.score(directory)
        self.assertEqual(report["n_matched"], 3)
        self.assertEqual(report["n_correct"], 0)
        self.assertEqual(report["strategy_updates"], 0)
        self.assertIs(report["production_enabled"], False)
        self.assertIs(report["generalization_claim"], False)
        self.assertIs(report["labels_exposed"], True)
        with self.assertRaises(FileExistsError):
            harness.score(directory)

    def test_prediction_tamper_fails_seal_before_scoring(self) -> None:
        directory = self._small_freeze()
        harness._predict_with_tool(directory, _FixedTool())
        path = directory / harness.PREDICTIONS_NAME
        payload = json.loads(path.read_text("utf-8"))
        payload["predictions"][0]["result"] = 123
        path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(harness.HarnessError, "seal hash"):
            harness.score(directory)
        self.assertFalse((directory / harness.REPORT_NAME).exists())

    def test_freeze_cannot_overwrite_existing_directory(self) -> None:
        directory = self._small_freeze()
        original = (directory / harness.INPUT_NAME).read_bytes()
        with self.assertRaises(FileExistsError):
            harness.freeze(directory, count=3)
        self.assertEqual((directory / harness.INPUT_NAME).read_bytes(), original)

    def test_changed_source_hash_is_rejected(self) -> None:
        directory = self._small_freeze()
        with patch.object(harness, "_source_hashes", return_value={"changed.py": "0" * 64}):
            with self.assertRaisesRegex(harness.HarnessError, "source hashes"):
                harness.verify_frozen(directory)

    def test_freeze_and_predict_do_not_invoke_oracle(self) -> None:
        evaluator_error = AssertionError("oracle must not be called")
        with patch.object(
            math_result_verifier, "make_result_evaluator", side_effect=evaluator_error
        ) as make_evaluator:
            with patch(
                "core.constitutional_brain.RealToolUse", return_value=_FixedTool()
            ) as real_tool_use:
                directory = self._small_freeze()
                harness.predict(directory)
        real_tool_use.assert_called_once()
        make_evaluator.assert_not_called()

    def test_missing_seal_is_rejected_before_oracle_callback(self) -> None:
        directory = self._small_freeze()
        harness._predict_with_tool(directory, _FixedTool())
        (directory / harness.SEAL_NAME).unlink()

        with patch.object(
            math_result_verifier,
            "make_result_evaluator",
            side_effect=AssertionError("oracle callback must not be created"),
        ) as make_evaluator:
            with self.assertRaisesRegex(harness.HarnessError, "required artifact missing"):
                harness.score(directory)
        make_evaluator.assert_not_called()
        self.assertFalse((directory / harness.REPORT_NAME).exists())

    def test_pairing_rejects_wrong_cardinality_and_extra_id(self) -> None:
        _, cases, _ = harness.verify_frozen(self._small_freeze())
        valid = [self._prediction(case) for case in cases]

        with self.assertRaisesRegex(harness.HarnessError, "one-to-one"):
            harness._validate_pairing(cases, valid + [dict(valid[0])])

        extra_id = [dict(row) for row in valid]
        extra_id[0]["id"] = "not-an-input"
        with self.assertRaisesRegex(harness.HarnessError, "ids"):
            harness._validate_pairing(cases, extra_id)

    def test_frozen_cases_require_canonical_ordered_ids(self) -> None:
        _, cases, _ = harness.verify_frozen(self._small_freeze())
        malformed = [dict(case) for case in cases]
        malformed[1]["id"] = "case-00003"

        with self.assertRaisesRegex(harness.HarnessError, "canonical sequence"):
            harness._validate_cases(
                {"schema": harness.SCHEMA_VERSION, "cases": malformed}, len(cases)
            )

    def test_freeze_rejects_invalid_case_counts(self) -> None:
        for count in (0, -1, 10_001, True, 1.5):
            with self.subTest(count=count):
                destination = self.tmp_path / f"invalid-{count!r}"
                with self.assertRaisesRegex(ValueError, "count must be an integer"):
                    harness.freeze(destination, count=count)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
