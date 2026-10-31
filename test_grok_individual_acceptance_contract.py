"""Aceite individual explícito do Grok; não implica aceite do Axis ou publicação."""
import hashlib
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tools.agente.validar_registro import ROOT, validate


class GrokIndividualAcceptanceContract(unittest.TestCase):
    def test_only_grok_accepts_literal_sha_from_6367039207(self):
        reg_path = Path("docs/agentes/REGISTRO_AGENTES.yaml")
        reg = yaml.safe_load((ROOT / reg_path).read_text(encoding="utf-8"))
        agents = {a["id"]: a for a in reg["agentes"]}
        sha = "8f03762d2a2d128eea955597f1b55210e387c08385bdc5544514ea2f39701b5d"
        path = f"docs/agentes/versoes/3.4.2/{sha}.md"
        self.assertEqual(agents["grok-bot"]["instrucoes"],
                         {"arquivo": path, "sha256": sha, "versao_aceita": "3.4.2"})
        self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), sha)
        old = "6778552adef5433c85bf9209b494456f4f497ffdf6f8e190b19eab45d54c20a6"
        self.assertEqual(agents["axis"]["instrucoes"], {
            "arquivo": f"docs/agentes/versoes/3.4.1/{old}.md",
            "sha256": old, "versao_aceita": "3.4.1"})
        self.assertEqual(agents["cursor-grok-4-7"]["instrucoes"], {
            "arquivo": f"docs/agentes/versoes/3.4.1/{old}.md",
            "sha256": old, "versao_aceita": "3.4.1"})
        errors, warnings = validate(ROOT, reg_path, Path("docs/agentes/CLAIMS.yaml"),
                                    datetime(2026, 10, 23, tzinfo=timezone.utc))
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 2)
        warning_text = "\n".join(warnings)
        self.assertIn("axis", warning_text)
        self.assertIn("cursor-grok-4-7", warning_text)
        self.assertNotIn("grok-bot", warning_text)


if __name__ == "__main__":
    unittest.main()
