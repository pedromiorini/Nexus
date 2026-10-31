"""Allowlist contract for prompt-free constitutional audit event payloads."""
import contextlib
import io
import json
import unittest
from types import SimpleNamespace as NS

from core.constitutional_brain import (
    EthicsGuard,
    ExecutionPlan,
    ModuleResult,
    ModuleType,
    RealImmutableLog,
    RequestType,
)
from test_router_fail_closed_contract import _router


PROMPTS = (
    "CANARY_PAYLOAD_SHORT: alpha query",
    "CANARY_PAYLOAD_LONG: alpha query " + "neutral context " * 250,
)


class _SpoofedAllowlistString(str):
    def __new__(cls, text, accepted_code):
        instance = super().__new__(cls, text)
        instance.accepted_code = accepted_code
        return instance

    def __hash__(self):
        return hash(self.accepted_code)

    def __eq__(self, other):
        if type(other) is str:
            return str.__eq__(self.accepted_code, other)
        return NotImplemented


class AuditEventPayloadPrivacyContract(unittest.TestCase):
    def _new_log(self):
        return RealImmutableLog(":memory:")

    def _guard(self, log, *, detected=False):
        detector = NS(detect=lambda _prompt: NS(detected=detected))
        with contextlib.redirect_stdout(io.StringIO()):
            return EthicsGuard(log, detector)

    def _payloads(self, log, event_type):
        rows = log.conn.execute(
            "SELECT event_data FROM log_entries WHERE event_type = ? ORDER BY sequence",
            (event_type,),
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def _assert_single_payload(self, log, event_type, prompt, expected):
        payloads = self._payloads(log, event_type)
        self.assertEqual(len(payloads), 1, (event_type, payloads))
        payload = payloads[0]
        self.assertEqual(payload, expected)
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("CANARY_PAYLOAD_", encoded)
        self.assertNotIn(prompt, encoded)
        self.assertNotIn("prompt", payload)
        self.assertNotIn("question", payload)
        self.assertNotIn("answer", payload)
        self.assertEqual(payload["schema_version"], 1)

    def test_ethics_pre_check_passed_keeps_only_versioned_metadata(self):
        for prompt in PROMPTS:
            with self.subTest(length=len(prompt)), self._new_log() as log:
                guard = self._guard(log)
                plan = ExecutionPlan(RequestType.QUERY, [], [], 0)
                result = guard.pre_execution_check(prompt, plan)
                self.assertTrue(result["cleared"])
                self.assertTrue(plan.ethical_clearance)
                self._assert_single_payload(
                    log,
                    "ETHICS_PRE_CHECK_PASSED",
                    prompt,
                    {
                        "schema_version": 1,
                        "modules": ["log"],
                        "module_count": 1,
                        "policy_codes": ["P11"],
                    },
                )

    def test_ethics_pre_check_maps_non_enum_module_to_unknown(self):
        marker = "CANARY_FREEFORM_MODULE_VALUE"
        with self._new_log() as log:
            guard = self._guard(log)
            plan = ExecutionPlan(RequestType.QUERY, [NS(value=marker)], [], 0)
            result = guard.pre_execution_check(PROMPTS[0], plan)
            self.assertTrue(result["cleared"])
            payloads = self._payloads(log, "ETHICS_PRE_CHECK_PASSED")
            self.assertEqual(len(payloads), 1)
            self.assertEqual(
                payloads[0],
                {
                    "schema_version": 1,
                    "modules": ["unknown", "log"],
                    "module_count": 2,
                    "policy_codes": ["P11"],
                },
            )
            self.assertNotIn(marker, json.dumps(payloads[0], sort_keys=True))

    def test_ethics_post_rejection_keeps_policy_and_result_codes(self):
        swarm = NS(
            final_decision=NS(value="reject"),
            votes=[NS(justification="ethical rejection")],
        )
        result = ModuleResult(ModuleType.SWARM, True, swarm, 0.0)
        for prompt in PROMPTS:
            with self.subTest(length=len(prompt)), self._new_log() as log:
                validation = self._guard(log).post_execution_validation(prompt, [result])
                self.assertFalse(validation["validated"])
                self.assertIn("P4", validation["reason"])
                self._assert_single_payload(
                    log,
                    "ETHICS_POST_REJECTION",
                    prompt,
                    {
                        "schema_version": 1,
                        "policy_code": "P4",
                        "reason_code": "SWARM_ETHICAL_REJECTION",
                        "result": "rejected",
                    },
                )

    def test_ethics_post_violation_keeps_policy_modules_and_count(self):
        result = ModuleResult(
            ModuleType.REASONING,
            True,
            {"note": "Please modify system settings."},
            0.0,
        )
        for prompt in PROMPTS:
            with self.subTest(length=len(prompt)), self._new_log() as log:
                validation = self._guard(log).post_execution_validation(prompt, [result])
                self.assertFalse(validation["validated"])
                self.assertEqual(validation["sanitized_results"], [])
                self._assert_single_payload(
                    log,
                    "ETHICS_POST_VIOLATION",
                    prompt,
                    {
                        "schema_version": 1,
                        "policy_codes": ["P6"],
                        "modules": ["reasoning"],
                        "violation_count": 1,
                    },
                )

    def test_ethics_post_violation_maps_non_enum_module_only_in_log(self):
        marker = "CANARY_FREEFORM_MODULE_VALUE"
        result = ModuleResult(
            NS(value=marker),
            True,
            {"note": "Please modify system settings."},
            0.0,
        )
        with self._new_log() as log:
            validation = self._guard(log).post_execution_validation(PROMPTS[0], [result])
            self.assertEqual(
                validation["reason"], f"P6: {marker} attempted self-modification"
            )
            payloads = self._payloads(log, "ETHICS_POST_VIOLATION")
            self.assertEqual(len(payloads), 1)
            self.assertEqual(payloads[0]["modules"], ["unknown"])
            self.assertNotIn(marker, json.dumps(payloads[0], sort_keys=True))

    def test_constitutional_violation_sanitizes_log_only(self):
        for prompt in PROMPTS:
            with self.subTest(length=len(prompt)), self._new_log() as log:
                guard = self._guard(log, detected=True)
                check = guard.pre_execution_check(
                    prompt, ExecutionPlan(RequestType.QUERY, [], [], 0)
                )
                self.assertFalse(check["cleared"])
                self.assertEqual(len(guard.violation_history), 1)
                self.assertEqual(guard.violation_history[0]["prompt"], prompt[:100])
                self.assertEqual(guard.violation_history[0]["principle"], "P13")
                payloads = self._payloads(log, "CONSTITUTIONAL_VIOLATION")
                self.assertEqual(len(payloads), 1)
                payload = payloads[0]
                self.assertEqual(set(payload), {"schema_version", "timestamp", "principle", "result"})
                self.assertEqual(payload["schema_version"], 1)
                self.assertEqual(payload["principle"], "P13")
                self.assertEqual(payload["result"], "blocked")
                self.assertIsInstance(payload["timestamp"], (int, float))
                encoded = json.dumps(payload, sort_keys=True)
                self.assertNotIn("CANARY_PAYLOAD_", encoded)
                self.assertNotIn("prompt", payload)
                self.assertNotIn("reason", payload)

    def test_router_blocked_keeps_metadata_and_is_logged_once(self):
        for prompt in PROMPTS:
            with self.subTest(length=len(prompt)), self._new_log() as log:
                router = _router(
                    [],
                    LOG=log,
                    BUDGET=NS(request_budget=lambda *_args, **_kwargs: False),
                )
                result = router.route(prompt)
                self.assertIs(result["success"], False)
                self.assertEqual(result["error"], "BUDGET_DENIED")
                self._assert_single_payload(
                    log,
                    "ROUTER_BLOCKED",
                    prompt,
                    {
                        "schema_version": 1,
                        "error": "BUDGET_DENIED",
                        "module": "budget",
                        "request_type": "query",
                    },
                )

    def test_router_blocked_maps_unrecognized_metadata_in_log_only(self):
        marker = "CANARY_FREEFORM_BLOCK_METADATA"
        with self._new_log() as log:
            router = _router([], LOG=log)
            plan = ExecutionPlan(RequestType.QUERY, [], [], 0)
            router.stats["total_requests"] = 1
            response = router._finish_blocked(
                PROMPTS[0],
                NS(value=marker),
                {"error": marker, "module": NS(value=marker), "reason": marker},
                plan,
                [],
                0.0,
            )
            self.assertEqual(response["error"], marker)
            self.assertEqual(response["request_type"], marker)
            self.assertEqual(response["reason"], marker)
            payloads = self._payloads(log, "ROUTER_BLOCKED")
            self.assertEqual(len(payloads), 1)
            self.assertEqual(
                payloads[0],
                {
                    "schema_version": 1,
                    "error": "UNKNOWN_BLOCK",
                    "module": "unknown",
                    "request_type": "unknown",
                },
            )
            self.assertNotIn(marker, json.dumps(payloads[0], sort_keys=True))

    def test_router_blocked_rejects_str_subclass_that_mimics_allowlisted_error(self):
        marker = "CANARY_FREEFORM_ERROR_CODE"
        spoofed_error = _SpoofedAllowlistString(marker, "BUDGET_DENIED")
        with self._new_log() as log:
            router = _router([], LOG=log)
            plan = ExecutionPlan(RequestType.QUERY, [], [], 0)
            router.stats["total_requests"] = 1
            response = router._finish_blocked(
                PROMPTS[0],
                RequestType.QUERY,
                {
                    "error": spoofed_error,
                    "module": ModuleType.BUDGET,
                    "reason": "resource budget denied",
                },
                plan,
                [],
                0.0,
            )
            self.assertIs(response["error"], spoofed_error)
            self.assertEqual(response["reason"], "resource budget denied")
            payloads = self._payloads(log, "ROUTER_BLOCKED")
            self.assertEqual(len(payloads), 1)
            self.assertEqual(payloads[0]["error"], "UNKNOWN_BLOCK")
            self.assertNotIn(marker, json.dumps(payloads[0], sort_keys=True))

    def test_routed_request_contains_no_prompt_or_response(self):
        for prompt in PROMPTS:
            with self.subTest(length=len(prompt)), self._new_log() as log:
                router = _router([], LOG=log)
                result = router.route(prompt)
                self.assertIs(result["success"], True)
                self._assert_single_payload(
                    log,
                    "ROUTED_REQUEST",
                    prompt,
                    {"schema_version": 1},
                )

    def test_router_block_log_uses_stable_code_not_exception_message(self):
        secret_exception = "CANARY_EXCEPTION_MESSAGE: backend detail"

        def fail(_prompt):
            raise RuntimeError(secret_exception)

        with self._new_log() as log:
            router = _router([], LOG=log, JAILBREAK=NS(detect=fail))
            result = router.route("CANARY_PAYLOAD_SHORT: alpha query")
            self.assertEqual(result["error"], "JAILBREAK_CHECK_FAILED")
            payloads = self._payloads(log, "ROUTER_BLOCKED")
            self.assertEqual(len(payloads), 1)
            self.assertEqual(
                payloads[0],
                {
                    "schema_version": 1,
                    "error": "JAILBREAK_CHECK_FAILED",
                    "module": "jailbreak",
                    "request_type": "query",
                },
            )
            self.assertNotIn(secret_exception, json.dumps(payloads[0], sort_keys=True))

    def test_constitutional_violation_log_drops_non_allowlisted_metadata(self):
        prompt = "CANARY_PAYLOAD_SHORT: request body"
        freeform_reason = "CANARY_EXCEPTION_MESSAGE: do not persist this detail"
        freeform_principle = "CANARY_POLICY_MARKER"
        with self._new_log() as log:
            guard = self._guard(log)
            guard._log_violation(freeform_principle, freeform_reason, prompt)
            self.assertEqual(guard.violation_history[0]["principle"], freeform_principle)
            self.assertEqual(guard.violation_history[0]["reason"], freeform_reason)
            self.assertEqual(guard.violation_history[0]["prompt"], prompt)
            payloads = self._payloads(log, "CONSTITUTIONAL_VIOLATION")
            self.assertEqual(len(payloads), 1)
            payload = payloads[0]
            self.assertEqual(set(payload), {"schema_version", "timestamp", "principle", "result"})
            self.assertEqual(payload["principle"], "UNKNOWN")
            self.assertNotIn("CANARY_", json.dumps(payload, sort_keys=True))

    def test_constitutional_violation_rejects_str_subclass_that_mimics_principle(self):
        marker = "CANARY_FREEFORM_PRINCIPLE"
        spoofed_principle = _SpoofedAllowlistString(marker, "P13")
        prompt = PROMPTS[0]
        reason = "freeform reason retained in history"
        with self._new_log() as log:
            guard = self._guard(log)
            guard._log_violation(spoofed_principle, reason, prompt)
            self.assertEqual(len(guard.violation_history), 1)
            self.assertIs(guard.violation_history[0]["principle"], spoofed_principle)
            self.assertEqual(guard.violation_history[0]["reason"], reason)
            self.assertEqual(guard.violation_history[0]["prompt"], prompt[:100])
            payloads = self._payloads(log, "CONSTITUTIONAL_VIOLATION")
            self.assertEqual(len(payloads), 1)
            self.assertEqual(payloads[0]["principle"], "UNKNOWN")
            self.assertNotIn(marker, json.dumps(payloads[0], sort_keys=True))


if __name__ == "__main__":
    unittest.main()
