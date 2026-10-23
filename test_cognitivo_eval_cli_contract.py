"""Contrato do harness tools/cognitivo_eval.py (entrada ausente/vazia/desconhecida -> exit 2)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tools import cognitivo_eval


class CognitivoEvalCliContractTests(unittest.TestCase):
    def _run(self, argv):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            code = cognitivo_eval.main(argv)
        return code, err.getvalue()

    def test_missing_capability_exits_2(self):
        code, err = self._run([])
        self.assertEqual(code, cognitivo_eval.EXIT_USAGE)
        self.assertIn("ausente", err)

    def test_empty_capability_exits_2(self):
        for value in ("", "   "):
            with self.subTest(value=value):
                self.assertEqual(self._run(["--capability", value])[0], 2)

    def test_unknown_capability_exits_2(self):
        code, err = self._run(["--capability", "telepathy"])
        self.assertEqual(code, 2)
        self.assertIn("desconhecida", err)

    def test_exit_codes_are_distinct(self):
        self.assertEqual(len({cognitivo_eval.EXIT_OK, cognitivo_eval.EXIT_FAIL, cognitivo_eval.EXIT_USAGE}), 3)

    def test_json_report_lists_checks_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            code, _ = self._run(["--capability", "predictive", "--json", str(path)])
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn(code, (0, 1))
        self.assertEqual({c["id"] for c in report["predictive"]}, {"PC1", "PC2", "PC3", "PC4"})
        self.assertEqual(code == 1, bool(report["failed"]))
        pc4 = next(c for c in report["predictive"] if c["id"] == "PC4")
        self.assertTrue(pc4["passed"])

    def test_unknown_criteria_version_is_usage_error(self):
        code, err = self._run(["--capability", "predictive", "--criteria", "v9"])
        self.assertEqual(code, cognitivo_eval.EXIT_USAGE)
        self.assertIn("critérios", err)

    def test_v1_and_v2_predictive_reports_are_distinct(self):
        from tools.cognitivo_eval import eval_predictive
        v1 = {c["id"]: c for c in eval_predictive("v1")}
        v2 = {c["id"]: c for c in eval_predictive("v2")}
        self.assertNotIn("criteria", v1["PC1"])
        self.assertEqual(v2["PC1"]["seeds"], list(range(100, 120)))
        self.assertIn("per_seed_pass_rate", v2["PC3"])

    def test_v3_rejects_missing_or_empty_input(self):
        with self.assertRaises(ValueError):
            cognitivo_eval._mcts_m3_v3(instances=())
        with self.assertRaises(ValueError):
            cognitivo_eval._mcts_m3_v3(seeds=())
        with self.assertRaises(ValueError):
            cognitivo_eval._m3_precheck([])
        for bad in (None, -1, True, "0"):
            with self.assertRaises(ValueError):
                cognitivo_eval.SyntheticTree(bad)

    def test_invalid_scenario_has_own_exit_code(self):
        fake = [{"id": "M3P", "value": 0.0, "limit": "", "passed": False, "scenario_invalid": True}]
        original = cognitivo_eval.EVALUATORS["mcts"]
        cognitivo_eval.EVALUATORS["mcts"] = lambda criteria="v3": fake
        try:
            code, _ = self._run(["--capability", "mcts", "--criteria", "v3"])
        finally:
            cognitivo_eval.EVALUATORS["mcts"] = original
        self.assertEqual(code, cognitivo_eval.EXIT_INVALID_SCENARIO)
        self.assertNotIn(cognitivo_eval.EXIT_INVALID_SCENARIO,
                         (cognitivo_eval.EXIT_OK, cognitivo_eval.EXIT_FAIL, cognitivo_eval.EXIT_USAGE))

    def test_synthetic_tree_has_unique_exact_optimum_in_deceptive_branch(self):
        tree = cognitivo_eval.SyntheticTree(0)
        self.assertEqual(len(tree.leaves), 6 ** 6)
        self.assertEqual(tree.optimum, 1.0)
        self.assertEqual(tree.needle[0], tree.deceptive)
        self.assertEqual(sum(v == tree.optimum for v in tree.leaves.values()), 1)
        self.assertEqual(tree.value(tree.needle[:3]), 0.0)
        check = cognitivo_eval._m3_precheck([tree])
        self.assertTrue(check["passed"], check)

if __name__ == "__main__":
    unittest.main()
