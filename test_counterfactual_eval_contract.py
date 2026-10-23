"""Contrato do avaliador CF pré-registrado em b3e7c04."""
import unittest

import tools.cognitivo_eval as ev


class CounterfactualEvalContractTests(unittest.TestCase):
    def test_ids_and_capability(self):
        self.assertIn("counterfactual", ev.CAPABILITIES)
        ids = [c["id"] for c in ev.eval_counterfactual()]
        self.assertEqual(ids, ["CF1", "CF2", "CF3", "CF4", "CF5", "CF6", "CF7"])

    def test_cli(self):
        self.assertIn(ev.main(["--capability", "counterfactual", "--criteria", "v1"]),
                      (ev.EXIT_OK, ev.EXIT_FAIL))


if __name__ == "__main__":
    unittest.main()
