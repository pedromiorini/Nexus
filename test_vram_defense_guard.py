import unittest
from copy import deepcopy
from unittest.mock import patch

from core.vram_defense_guard import VramDefenseGuard, VramSnapshot
from core.constitutional_brain import CentralRouter


class VramDefenseGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = VramDefenseGuard(soft_limit=0.80, hard_limit=0.92)

    def test_normal_pressure(self):
        decision = self.guard.evaluate(VramSnapshot(True, reserved_bytes=40, total_bytes=100))
        self.assertEqual(decision.action, "continue")
        self.assertEqual(decision.severity, "normal")

    def test_soft_pressure(self):
        decision = self.guard.evaluate(VramSnapshot(True, reserved_bytes=85, total_bytes=100))
        self.assertEqual(decision.action, "defer_and_release")
        self.assertIn("release_cached_tensors", self.guard.mitigation_plan(decision)["steps"])

    def test_hard_pressure(self):
        decision = self.guard.evaluate(VramSnapshot(True, reserved_bytes=95, total_bytes=100))
        self.assertEqual(decision.action, "emergency_release")
        self.assertEqual(decision.severity, "critical")

    def test_cpu_fallback(self):
        decision = self.guard.evaluate(VramSnapshot(False))
        self.assertEqual(decision.action, "observe")
        self.assertEqual(self.guard.mitigation_plan(decision)["steps"], ["use_cpu_fallback"])

    def test_statistics(self):
        self.guard.evaluate(VramSnapshot(True, reserved_bytes=10, total_bytes=100))
        self.guard.evaluate(VramSnapshot(True, reserved_bytes=95, total_bytes=100))
        self.assertEqual(self.guard.statistics()["samples"], 2)
        self.assertEqual(self.guard.statistics()["actions"]["emergency_release"], 1)

    def test_router_owns_vram_guard(self):
        router = CentralRouter({}, object())
        self.assertIsInstance(router.vram_guard, VramDefenseGuard)
        decision = router.vram_guard.evaluate(VramSnapshot(True, reserved_bytes=95, total_bytes=100))
        self.assertEqual(decision.action, "emergency_release")

    def test_mitigation_executes_safe_steps_and_audits(self):
        decision = self.guard.evaluate(VramSnapshot(True, reserved_bytes=95, total_bytes=100))
        stopped = []
        result = self.guard.execute_mitigation(decision, stop_callback=lambda: stopped.append(True))
        self.assertIn("stop_noncritical_tasks", result.completed)
        self.assertIn("gc_collect", result.completed)
        self.assertIn("require_recheck", result.skipped)
        self.assertEqual(stopped, [True])
        self.assertEqual(len(self.guard.audit_log), 1)

    def test_mitigation_callback_failure_is_contained(self):
        decision = self.guard.evaluate(VramSnapshot(True, reserved_bytes=85, total_bytes=100))
        result = self.guard.execute_mitigation(decision, defer_callback=lambda: (_ for _ in ()).throw(RuntimeError("blocked")))
        self.assertFalse(result.successful)
        self.assertTrue(any("defer_next_task" in item for item in result.errors))


class VramDefenseGuardBoundaryTests(unittest.TestCase):
    """Fronteiras e validações que os mutantes do gate de mutação exigem."""

    def test_limits_are_inclusive_at_the_boundary(self):
        guard = VramDefenseGuard(soft_limit=0.80, hard_limit=0.92)
        expected = ((79, "continue"), (80, "defer_and_release"), (91, "defer_and_release"), (92, "emergency_release"))
        for reserved, action in expected:
            with self.subTest(reserved=reserved):
                self.assertEqual(guard.evaluate(VramSnapshot(True, reserved_bytes=reserved, total_bytes=100)).action, action)

    def test_constructor_validates_limits(self):
        for soft, hard in ((0.0, 0.5), (-0.1, 0.5), (0.9, 0.8), (0.8, 0.8), (0.5, 1.01)):
            with self.subTest(soft=soft, hard=hard):
                with self.assertRaises(ValueError):
                    VramDefenseGuard(soft_limit=soft, hard_limit=hard)
        self.assertEqual(VramDefenseGuard(soft_limit=0.5, hard_limit=1.0).hard_limit, 1.0)

    def test_constructor_rejects_boolean_limits(self):
        for soft, hard in ((False, 0.92), (True, 0.92), (0.5, False), (0.5, True),
                           (False, False), (False, True), (True, False), (True, True)):
            with self.subTest(soft=soft, hard=hard):
                with self.assertRaises(TypeError):
                    VramDefenseGuard(soft_limit=soft, hard_limit=hard)

    def test_constructor_preserves_numeric_limits_and_inclusive_actions(self):
        for hard in (1, 1.0):
            with self.subTest(hard_type=type(hard).__name__):
                guard = VramDefenseGuard(soft_limit=0.5, hard_limit=hard)
                self.assertIs(guard.hard_limit, hard)
                self.assertEqual(guard.soft_limit, 0.5)
                self.assertEqual(guard.evaluate(VramSnapshot(True, reserved_bytes=50, total_bytes=100)).action,
                                 "defer_and_release")
                self.assertEqual(guard.evaluate(VramSnapshot(True, reserved_bytes=100, total_bytes=100)).action,
                                 "emergency_release")

    def test_utilization_is_capped_at_one_and_safe_without_total(self):
        self.assertEqual(VramSnapshot(True, reserved_bytes=150, total_bytes=100).utilization, 1.0)
        self.assertEqual(VramSnapshot(True, reserved_bytes=10, total_bytes=0).utilization, 0.0)

    def test_mitigation_without_callbacks_skips_instead_of_failing(self):
        guard = VramDefenseGuard()
        decision = guard.evaluate(VramSnapshot(True, reserved_bytes=85, total_bytes=100))
        result = guard.execute_mitigation(decision)
        self.assertTrue(result.successful)
        self.assertEqual(result.skipped, ["defer_next_task"])
        self.assertEqual(result.completed, ["release_cached_tensors", "gc_collect"])


class VramSnapshotCounterTests(unittest.TestCase):
    FIELDS = ("allocated_bytes", "reserved_bytes", "total_bytes")

    class IntSubclass(int):
        pass

    INVALID_TYPES = (
        True, False, 0.0, 1.5, -1.0, float("nan"), float("inf"),
        float("-inf"), None, "1", b"1", [], {}, object(), IntSubclass(1),
    )

    def test_each_counter_rejects_invalid_types(self):
        for field in self.FIELDS:
            for value in self.INVALID_TYPES:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(TypeError, field):
                        VramSnapshot(True, **{field: value})

    def test_each_counter_rejects_negative_integers(self):
        for field in self.FIELDS:
            for value in (-1, -(2**64)):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        VramSnapshot(True, **{field: value})

    def test_invalid_construction_has_no_guard_side_effects_or_decision(self):
        for field in self.FIELDS:
            for value in (*self.INVALID_TYPES, -1):
                for available in (True, False):
                    with self.subTest(field=field, value=value, available=available):
                        guard = VramDefenseGuard()
                        guard.execute_mitigation(guard.evaluate(VramSnapshot(False)))
                        before = deepcopy((guard.history, guard.mitigation_history,
                                           guard.audit_log, guard.statistics()))
                        error = ValueError if type(value) is int else TypeError
                        with patch("core.vram_defense_guard.DefenseDecision") as decision:
                            # Argument construction must fail before evaluate is entered.
                            with self.assertRaises(error):
                                guard.evaluate(VramSnapshot(available, **{field: value}))
                            # The same guarantee holds for a failing measurement provider.
                            with patch.object(guard, "snapshot", side_effect=lambda: VramSnapshot(
                                available, **{field: value}
                            )):
                                with self.assertRaises(error):
                                    guard.evaluate()
                            decision.assert_not_called()
                        self.assertEqual((guard.history, guard.mitigation_history,
                                          guard.audit_log, guard.statistics()), before)

    def test_valid_counter_boundaries_and_timestamp_are_preserved(self):
        for field in self.FIELDS:
            for value in (0, 1, 2**64):
                with self.subTest(field=field, value=value):
                    snapshot = VramSnapshot(True, timestamp=123.5, **{field: value})
                    self.assertIs(getattr(snapshot, field), value)
                    self.assertEqual(snapshot.timestamp, 123.5)
        snapshot = VramSnapshot(True, 2**64, 2**64, 2**64)
        self.assertEqual(snapshot.utilization, 1.0)
        self.assertEqual(VramDefenseGuard().evaluate(snapshot).action, "emergency_release")

    def test_zero_totals_and_unavailable_defaults_remain_valid(self):
        guard = VramDefenseGuard()
        unavailable = VramSnapshot(False)
        self.assertEqual((unavailable.allocated_bytes, unavailable.reserved_bytes,
                          unavailable.total_bytes), (0, 0, 0))
        self.assertIsInstance(unavailable.timestamp, float)
        self.assertEqual(unavailable.utilization, 0.0)
        self.assertEqual(guard.evaluate(unavailable).action, "observe")
        self.assertEqual(guard.evaluate(VramSnapshot(True)).action, "continue")
        self.assertEqual(VramSnapshot(True, 10, 20, 0).utilization, 0.0)


if __name__ == "__main__":
    unittest.main()
