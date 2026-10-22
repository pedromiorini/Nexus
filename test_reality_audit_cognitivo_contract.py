"""Contrato: o auditor de realidade cobre core/cognitivo e falha fechado sem entrada.

Teste dos três casos (critérios escritos antes da mudança):
(a) caso de origem: alegações no código extraído para core/cognitivo voltam a ser contadas
    (o inventário não pode cair só porque o texto mudou de arquivo);
(b) não regressão: os números do núcleo (`core_*`) continuam medindo só o monólito e os
    testes existentes de test_reality_audit.py passam sem alteração;
(c) caso incompatível: núcleo ausente ou vazio sai com código próprio (2/3), sem gravar relatório.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools import reality_audit

ROOT = Path(__file__).resolve().parent


class RealityAuditCognitivoContractTests(unittest.TestCase):
    def test_extracted_modules_are_inventoried(self):
        evidence = reality_audit.collect_static_evidence()
        paths = [item["path"] for item in evidence["extracted_modules"]]
        self.assertIn("core/cognitivo/episodic_memory.py", paths)
        self.assertIn("core/cognitivo/predictive_coding.py", paths)
        core_only = len(reality_audit.CORE.read_text(encoding="utf-8").splitlines())
        self.assertEqual(evidence["core_lines"], core_only)

    def test_claims_in_extracted_code_are_counted(self):
        extracted = (ROOT / "core" / "cognitivo" / "predictive_coding.py").read_text(encoding="utf-8")
        self.assertIn("Bayesian prediction machine", extracted)
        inventory = reality_audit.audit_code_claim_inventory()
        labels = {item["label"] for item in inventory["items"]}
        self.assertIn("máquina bayesiana afirmada", labels)
        only_core = reality_audit.audit_code_claim_inventory(reality_audit.CORE.read_text(encoding="utf-8"))
        self.assertGreater(inventory["by_level"]["strong_capability_claim"],
                           only_core["by_level"]["strong_capability_claim"])

    def _run_in(self, core_text):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        (tmp / "tools").mkdir()
        shutil.copy(ROOT / "tools" / "reality_audit.py", tmp / "tools" / "reality_audit.py")
        (tmp / "README.md").write_text("x\n", encoding="utf-8")
        if core_text is not None:
            (tmp / "core").mkdir()
            (tmp / "core" / "constitutional_brain.py").write_text(core_text, encoding="utf-8")
        proc = subprocess.run([sys.executable, str(tmp / "tools" / "reality_audit.py")],
                              capture_output=True, text=True, timeout=60)
        return proc, tmp

    def test_missing_core_exits_2_without_report(self):
        proc, tmp = self._run_in(None)
        self.assertEqual(proc.returncode, reality_audit.EXIT_MISSING_INPUT, proc.stderr)
        self.assertEqual(reality_audit.EXIT_MISSING_INPUT, 2)
        self.assertFalse((tmp / "audit_reality.json").exists())

    def test_empty_core_exits_3_without_report(self):
        proc, tmp = self._run_in("  \n")
        self.assertEqual(proc.returncode, reality_audit.EXIT_EMPTY_INPUT, proc.stderr)
        self.assertEqual(reality_audit.EXIT_EMPTY_INPUT, 3)
        self.assertFalse((tmp / "audit_reality.json").exists())


if __name__ == "__main__":
    unittest.main()
