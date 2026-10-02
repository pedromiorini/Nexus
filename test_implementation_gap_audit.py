import json
import tempfile
import unittest
from pathlib import Path

from tools.implementation_gap_audit import audit, main


class ImplementationGapAuditTests(unittest.TestCase):
    def test_detects_pass_only_and_not_implemented(self):
        source = """
def empty():
    \"\"\"documented\"\"\"
    pass

def concrete():
    return 1

def missing():
    raise NotImplementedError('later')
"""
        result = audit(source, "sample.py")
        self.assertEqual(result["pass_only"], [{"name": "empty", "line": 2}])
        self.assertEqual(result["not_implemented"], [{"line": 10}])
        self.assertTrue(result["ok"])

    def test_cli_writes_json_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.py"
            output = Path(directory) / "gaps.json"
            source.write_text("def empty():\n    pass\n", encoding="utf-8")
            self.assertEqual(main([str(source), "--output", str(output)]), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["pass_only_count"], 1)


if __name__ == "__main__":
    unittest.main()
