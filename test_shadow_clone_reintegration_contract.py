import time
import unittest

from core.constitutional_brain import RealHardwareAwareSelfOptimization, ShadowClone


class BootstrapSink:
    def __init__(self):
        self.received = []

    def ingest_knowledge(self, knowledge):
        self.received.append(dict(knowledge))


class ShadowCloneReintegrationContractTests(unittest.TestCase):
    def test_bootstrap_protocol_receives_clone_knowledge(self):
        bootstrap = BootstrapSink()
        optimizer = RealHardwareAwareSelfOptimization(bootstrap_loop=bootstrap)
        clone = ShadowClone("clone-1", time.time(), "exploration", {"patterns": 3}, {"quality": 0.8}, True)
        optimizer.shadow_clones[clone.clone_id] = clone
        result = optimizer.reintegrate_shadow_clone(clone.clone_id)
        self.assertEqual(result, {"patterns": 3})
        self.assertEqual(bootstrap.received, [{"patterns": 3}])
        self.assertTrue(optimizer.bootstrap_reintegration_events[-1]["integrated"])
        self.assertNotIn(clone.clone_id, optimizer.shadow_clones)

    def test_missing_bootstrap_protocol_is_recorded_without_claiming_integration(self):
        optimizer = RealHardwareAwareSelfOptimization(bootstrap_loop=object())
        clone = ShadowClone("clone-2", time.time(), "training", {"examples": 2}, {}, True)
        optimizer.shadow_clones[clone.clone_id] = clone
        optimizer.reintegrate_shadow_clone(clone.clone_id)
        event = optimizer.bootstrap_reintegration_events[-1]
        self.assertTrue(event["bootstrap_available"])
        self.assertFalse(event["integrated"])


if __name__ == "__main__":
    unittest.main()
