import unittest

from tools.embedded_assert_audit import audit


NESTED_GUARD_CONTEXTS = (
    ("function", "def helper():\n", ""),
    ("async_function", "async def helper():\n", ""),
    ("class", "class Helper:\n", ""),
    ("if", "if enabled:\n", ""),
    ("for", "for item in items:\n", ""),
    ("while", "while enabled:\n", ""),
    ("with", "with context():\n", ""),
    ("try", "try:\n", "finally:\n    pass\n"),
    ("else", "if enabled:\n    pass\nelse:\n", ""),
)


class EmbeddedAssertAuditTests(unittest.TestCase):
    def test_b101_findings_inside_main_block_are_accepted(self):
        source = "if __name__ == '__main__':\n    assert True\n"
        payload = {"results": [{"test_id": "B101", "filename": "demo.py", "line_number": 2}]}
        result = audit(payload, source, "demo.py")
        self.assertTrue(result["ok"])
        self.assertEqual(result["b101_count"], 1)
        self.assertEqual(result["main_block_ranges"], [(1, 2)])
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

    def test_module_guard_without_finding_is_accepted(self):
        source = 'if __name__ == "__main__":\n    assert False\n'
        result = audit({}, source, "demo.py")
        self.assertTrue(result["ok"])
        self.assertEqual(result["main_block_ranges"], [(1, 2)])
        self.assertEqual(result["assert_lines_outside_main"], [])

    def test_nested_guards_with_b101_are_rejected(self):
        for context, prefix, suffix in NESTED_GUARD_CONTEXTS:
            nested = prefix + "    if __name__ == '__main__':\n        assert False\n" + suffix
            line = prefix.count("\n") + 2
            for module_demo in ("", "if __name__ == '__main__':\n    print('demo')\n"):
                with self.subTest(context=context, module_demo=bool(module_demo)):
                    source = nested + module_demo
                    finding = {"test_id": "B101", "filename": "demo.py", "line_number": line, "code": "assert False"}
                    result = audit({"results": [finding]}, source, "demo.py")
                    expected_ranges = [(nested.count("\n") + 1, nested.count("\n") + 2)] if module_demo else []
                    self.assertEqual(result["main_block_ranges"], expected_ranges)
                    self.assertEqual(result["outside_main_block"], [{"line_number": line, "code": "assert False"}])
                    self.assertEqual(result["assert_lines_outside_main"], [line])
                    self.assertEqual(result["b101_count"], 1)
                    self.assertFalse(result["ok"])

    def test_nested_guards_without_findings_are_rejected(self):
        for context, prefix, suffix in NESTED_GUARD_CONTEXTS:
            nested = prefix + "    if __name__ == '__main__':\n        assert False\n" + suffix
            line = prefix.count("\n") + 2
            for module_demo in ("", "if __name__ == '__main__':\n    print('demo')\n"):
                with self.subTest(context=context, module_demo=bool(module_demo)):
                    result = audit({}, nested + module_demo, "demo.py")
                    expected_ranges = [(nested.count("\n") + 1, nested.count("\n") + 2)] if module_demo else []
                    self.assertEqual(result["main_block_ranges"], expected_ranges)
                    self.assertEqual(result["outside_main_block"], [])
                    self.assertEqual(result["assert_lines_outside_main"], [line])
                    self.assertEqual(result["b101_count"], 0)
                    self.assertFalse(result["ok"])

    def test_nested_b101_without_assert_is_rejected(self):
        source = "def helper():\n    if __name__ == '__main__':\n        print('demo')\nif __name__ == '__main__':\n    pass\n"
        payload = {"results": [{"test_id": "B101", "filename": "demo.py", "line_number": 3}]}
        result = audit(payload, source, "demo.py")
        self.assertEqual(result["main_block_ranges"], [(4, 5)])
        self.assertEqual(result["outside_main_block"], [{"line_number": 3, "code": ""}])
        self.assertEqual(result["assert_lines_outside_main"], [])
        self.assertFalse(result["ok"])

    def test_module_guard_else_is_not_exempt(self):
        source = "if __name__ == '__main__':\n    assert True\nelse:\n    assert False\n"
        for findings in ([], [{"test_id": "B101", "filename": "demo.py", "line_number": 4}]):
            with self.subTest(findings=findings):
                result = audit({"results": findings}, source, "demo.py")
                self.assertEqual(result["main_block_ranges"], [(1, 2)])
                self.assertEqual(result["assert_lines_outside_main"], [4])
                self.assertEqual(result["outside_main_block"], [{"line_number": 4, "code": ""}] if findings else [])
                self.assertFalse(result["ok"])

    def test_nonconventional_guards_are_not_exempt(self):
        for condition in ("'__main__' == __name__", "__name__ != '__main__'", "__name__ == '__main__' == label", "__name__ == '__main__' and enabled"):
            with self.subTest(condition=condition):
                result = audit({}, f"if {condition}:\n    assert False\n", "demo.py")
                self.assertEqual(result["main_block_ranges"], [])
                self.assertEqual(result["assert_lines_outside_main"], [2])
                self.assertFalse(result["ok"])

    def test_nested_assert_in_module_demo_body_remains_exempt(self):
        source = "if __name__ == '__main__':\n    if enabled:\n        assert False\n"
        payload = {"results": [{"test_id": "B101", "filename": "demo.py", "line_number": 3}]}
        result = audit(payload, source, "demo.py")
        self.assertEqual(result["main_block_ranges"], [(1, 3)])
        self.assertEqual(result["outside_main_block"], [])
        self.assertEqual(result["assert_lines_outside_main"], [])
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
