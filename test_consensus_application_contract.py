import unittest

from core.constitutional_brain import ConsensusProposal, RealDistributedConsensusProtocol


class ConsensusApplicationContractTests(unittest.TestCase):
    def make_protocol(self):
        return RealDistributedConsensusProtocol(node_id="node-a")

    def test_parameter_change_is_applied_with_bounds(self):
        protocol = self.make_protocol()
        proposal = ConsensusProposal("p1", "node-a", "parameter_change", {"parameter": "heartbeat_interval", "value": 2.5}, 1, 0, "accepted")
        self.assertTrue(protocol._apply_accepted_proposal(proposal))
        self.assertEqual(protocol.heartbeat_interval, 2.5)
        self.assertEqual(protocol.applied_parameter_changes[-1]["parameter"], "heartbeat_interval")

    def test_invalid_parameter_change_is_rejected_without_mutation(self):
        protocol = self.make_protocol()
        before = protocol.quorum_size
        proposal = ConsensusProposal("p2", "node-a", "parameter_change", {"parameter": "unknown", "value": 0.4}, 1, 0, "accepted")
        self.assertFalse(protocol._apply_accepted_proposal(proposal))
        self.assertEqual(protocol.quorum_size, before)
        self.assertEqual(protocol.applied_parameter_changes, [])

    def test_decision_is_recorded_as_local_audit_event(self):
        protocol = self.make_protocol()
        proposal = ConsensusProposal("p3", "node-a", "decision", {"decision": "defer-next-task"}, 1, 0, "accepted")
        self.assertTrue(protocol._apply_accepted_proposal(proposal))
        self.assertEqual(protocol.applied_decisions, [{"proposal_id": "p3", "decision": "defer-next-task"}])

    def test_blank_decision_is_rejected(self):
        protocol = self.make_protocol()
        proposal = ConsensusProposal("p4", "node-a", "decision", {"decision": "  "}, 1, 0, "accepted")
        self.assertFalse(protocol._apply_accepted_proposal(proposal))
        self.assertEqual(protocol.applied_decisions, [])


if __name__ == "__main__":
    unittest.main()
