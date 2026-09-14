import unittest

from core.vram_defense_guard import DefenseDecision, VramDefenseGuard, VramSnapshot
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


if __name__ == "__main__":
    unittest.main()
