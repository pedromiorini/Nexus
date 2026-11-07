"""Contract for the POST 62 extraction and its legacy public path."""
import ast
import hashlib
import inspect
import pickle
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import unittest

import core.constitutional_brain as legacy
import core.system_health_diagnostics as extracted


CLASS_NAMES = (
    "HealthIndicator",
    "DiagnosticReport",
    "SystemAlert",
    "RealSystemHealthDiagnosticsEngine",
)
# Base fingerprint after removing trailing whitespace only; computed from the four definitions at 632ce845.
BASE_CLASS_SOURCE_RSTRIP_SHA256 = "f1329d0769bbf5d5b46ce0712b2576487b2f34ec8cf3f137de06b02be0f15cbb"


def class_source_fingerprint():
    source = Path(extracted.__file__).read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in CLASS_NAMES
    ]
    if tuple(node.name for node in nodes) != CLASS_NAMES:
        return None

    definitions = []
    for node in nodes:
        first_line = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
        definitions.append("\n".join(line.rstrip() for line in lines[first_line - 1 : node.end_lineno]))
    exact_block = "\n".join(definitions) + "\n"
    return hashlib.sha256(exact_block.encode("utf-8")).hexdigest()


class SystemHealthDiagnosticsExtractionContractTests(unittest.TestCase):
    def test_extracted_class_definitions_match_base_after_trailing_space_normalization(self):
        self.assertEqual(class_source_fingerprint(), BASE_CLASS_SOURCE_RSTRIP_SHA256)

    def test_legacy_api_reexports_the_extracted_objects_and_reflection_path(self):
        for name in CLASS_NAMES:
            with self.subTest(name=name):
                extracted_class = getattr(extracted, name)
                self.assertIs(getattr(legacy, name), extracted_class)
                self.assertEqual(extracted_class.__module__, "core.constitutional_brain")
                self.assertIs(inspect.getmodule(extracted_class), legacy)

    def test_legacy_pickle_path_round_trips_all_public_types(self):
        indicator = extracted.HealthIndicator("i", "api", "healthy", 1.0, "ok")
        values = [
            indicator,
            extracted.DiagnosticReport("r", 1.0, 0.8, [indicator], ["check"]),
            extracted.SystemAlert("a", "warning", "api", "slow", 1.0),
        ]
        with redirect_stdout(StringIO()):
            values.append(extracted.RealSystemHealthDiagnosticsEngine())

        for value in values:
            with self.subTest(class_name=type(value).__name__):
                payload = pickle.dumps(value)
                self.assertIn(b"core.constitutional_brain", payload)
                restored = pickle.loads(payload)
                self.assertIs(type(restored), type(value))
                self.assertEqual(restored.__dict__, value.__dict__)


if __name__ == "__main__":
    unittest.main()
