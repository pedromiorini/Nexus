import unittest

from core.constitutional_brain import CentralExecutive


class WorkingMemoryContractTests(unittest.TestCase):
    def test_attention_allocation_is_recorded_and_clamped(self):
        executive = CentralExecutive()
        allocation = executive.allocate_attention("planning", strength=2.0)
        self.assertEqual(allocation, {"target": "planning", "strength": 1.0, "delegated": False})
        self.assertEqual(executive.current_task, "planning")
        self.assertEqual(executive.attention_allocations, [allocation])

    def test_external_attention_presence_is_reported_without_unknown_call(self):
        executive = CentralExecutive(attention_system=object())
        allocation = executive.allocate_attention("monitoring", strength=-1.0)
        self.assertEqual(allocation["strength"], 0.0)
        self.assertTrue(allocation["delegated"])
        self.assertEqual(executive.current_task, "monitoring")


if __name__ == "__main__":
    unittest.main()
