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


if __name__ == "__main__":
    unittest.main()
