"""Contratos sintáticos de NotImplementedError, sem executar o fonte auditado."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.implementation_gap_audit import audit, main


class RaiseClassInventoryContractTests(unittest.TestCase):
    def test_explicit_class_and_instance_targets_are_equivalent(self):
        targets = (
            "NotImplementedError", "NotImplementedError()",
            "NotImplementedError('later')", "builtins.NotImplementedError",
            "builtins.NotImplementedError()", "package.errors.NotImplementedError",
            "package.errors.NotImplementedError('later')",
        )
        for target in targets:
            with self.subTest(target=target):
                source = f"def missing():\n    raise {target}\n"
                result = audit(source, "sample.py")
                self.assertEqual(result["not_implemented"], [{"line": 2}])
                self.assertEqual(result["not_implemented_count"], 1)
                self.assertEqual(result["pass_only"], [])
                self.assertTrue(result["ok"])

    def test_chained_causes_are_not_additional_targets(self):
        for target in ("NotImplementedError", "builtins.NotImplementedError()"):
            for cause in ("None", "ValueError()", "NotImplementedError()"):
                with self.subTest(target=target, cause=cause):
                    result = audit(f"raise {target} from {cause}\n", "sample.py")
                    self.assertEqual(result["not_implemented"], [{"line": 1}])
                    self.assertEqual(result["not_implemented_count"], 1)
        self.assertEqual(
            audit("raise ValueError() from NotImplementedError\n", "sample.py")["not_implemented"],
            [],
        )

    def test_nested_sync_and_async_raises_preserve_lines_and_counts(self):
        source = (
            "class Pending:\n"
            "    def missing(self):\n"
            "        raise NotImplementedError\n"
            "    async def later(self):\n"
            "        raise builtins.NotImplementedError\n"
        )
        result = audit(source, "nested.py")
        self.assertEqual(result["not_implemented"], [{"line": 3}, {"line": 5}])
        self.assertEqual(result["not_implemented_count"], 2)
        self.assertEqual(result["filename"], "nested.py")

    def test_non_targets_are_ignored_and_source_is_never_executed(self):
        source = (
            "raise RuntimeError('audit must not execute this')\n"
            "# raise NotImplementedError\n"
            "text = 'raise builtins.NotImplementedError'\n"
            "pending = NotImplementedError\n"
            "def reference():\n"
            "    return NotImplementedError\n"
            "def empty():\n"
            "    pass\n"
            "try:\n"
            "    raise ValueError() from NotImplementedError()\n"
            "except NotImplementedError:\n"
            "    raise\n"
            "def unrelated():\n"
            "    raise NotImplemented\n"
        )
        result = audit(source, "negative.py")
        self.assertEqual(result["not_implemented"], [])
        self.assertEqual(result["not_implemented_count"], 0)
        self.assertEqual(result["pass_only"], [{"name": "empty", "line": 7}])
        self.assertEqual(result["pass_only_count"], 1)

    def test_cli_keeps_schema_and_reports_bare_class(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pending.py"
            output = Path(directory) / "inventory.json"
            source.write_text("raise NotImplementedError\n", encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(source), "--output", str(output)])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue()), payload)
            self.assertEqual(set(payload), {
                "filename", "pass_only", "not_implemented", "pass_only_count",
                "not_implemented_count", "ok",
            })
            self.assertEqual(payload["not_implemented"], [{"line": 1}])
            self.assertEqual(payload["not_implemented_count"], 1)


if __name__ == "__main__":
    unittest.main()
