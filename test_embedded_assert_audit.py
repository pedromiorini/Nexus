import json
import unittest

from tools.embedded_assert_audit import audit


class EmbeddedAssertAuditTests(unittest.TestCase):
    def test_b101_findings_inside_main_block_are_accepted(self):
        source = "if __name__ == '__main__':\n    assert True\n"
        payload = {"results": [{"test_id": "B101", "filename": "demo.py", "line_number": 2}]}
        result = audit(payload, source, "demo.py")
        self.assertTrue(result["ok"])
        self.assertEqual(result["b101_count"], 1)
        self.assertEqual(result["outside_main_block"], [])
        self.assertEqual(result["assert_lines_outside_main"], [])

    def test_b101_outside_main_block_is_rejected(self):
        source = "assert True\nif __name__ == '__main__':\n    print('demo')\n"
        payload = {"results": [{"test_id": "B101", "filename": "demo.py", "line_number": 1}]}
        result = audit(payload, source, "demo.py")
        self.assertFalse(result["ok"])
        self.assertEqual(result["outside_main_block"][0]["line_number"], 1)
        self.assertEqual(result["assert_lines_outside_main"], [1])

    def test_unreported_assert_outside_main_is_rejected(self):
        source = "assert True\nif __name__ == '__main__':\n    print('demo')\n"
        result = audit({"results": []}, source, "demo.py")
        self.assertFalse(result["ok"])
        self.assertEqual(result["assert_lines_outside_main"], [1])

    def test_non_b101_findings_do_not_affect_audit(self):
        source = "if __name__ == '__main__':\n    print('demo')\n"
        payload = {"results": [{"test_id": "B311", "filename": "demo.py", "line_number": 1}]}
        result = audit(payload, source, "demo.py")
        self.assertTrue(result["ok"])
        self.assertEqual(result["b101_count"], 0)
        self.assertEqual(result["assert_lines_outside_main"], [])


if __name__ == "__main__":
    unittest.main()
