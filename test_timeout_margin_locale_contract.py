"""Contrato: dígito não ASCII não pode alargar [0-9] e sair como sucesso.

Observação em main af28cc1, arquivo com as linhas 500 e ٣:
- LC_ALL=en_US.UTF-8: a regex casa, (( )) emite "syntax error" e o processo sai 0
  com "OK: max 500s <= 612s".
- LC_ALL=C: a mesma guarda sai 3 e escreve "entrada invalida".

Hipóteses:
- H1: [0-9] no bash é só ASCII. Falsifica se en_US.UTF-8 casa ٣.
- H2: erro de aritmética aborta a guarda. Falsifica se o status final é 0.
- H3: o arquivo ausente e o vazio já são o código 3. Continua valendo nos dois locales.
- H4: duração ASCII dentro da margem deixa de passar. Falsifica se 500 sai diferente de 0.

Códigos já da guarda: 0 dentro da margem, 6 acima de 85%, 3 entrada inválida.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GUARD = ROOT / "tools/agente/timeout_margin.sh"


def run(locale, timeout, content):
    with tempfile.TemporaryDirectory(prefix="tm-locale-") as tmp:
        path = Path(tmp) / "d.txt"
        if content is not None:
            path.write_bytes(content.encode())
            arg = str(path)
        else:
            arg = str(Path(tmp) / "ausente.txt")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "LC_ALL": locale,
        }
        return subprocess.run(
            ["bash", str(GUARD), timeout, arg],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=20,
        )


class TimeoutMarginLocaleContractTests(unittest.TestCase):
    def test_non_ascii_digit_is_invalid_in_utf8_and_c(self):
        for locale in ("en_US.UTF-8", "C"):
            with self.subTest(locale=locale):
                result = run(locale, "12", "500\n٣\n")
                self.assertEqual(result.returncode, 3, (result.stdout, result.stderr))
                self.assertIn("entrada invalida", result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertNotIn("syntax error", result.stderr)

    def test_missing_and_empty_keep_their_exit_code(self):
        for locale in ("en_US.UTF-8", "C"):
            for content in (None, ""):
                with self.subTest(locale=locale, content=content):
                    result = run(locale, "12", content)
                    self.assertEqual(result.returncode, 3, (result.stdout, result.stderr))
                    self.assertIn("entrada invalida", result.stderr)

    def test_ascii_duration_within_margin_stays_success(self):
        for locale in ("en_US.UTF-8", "C"):
            with self.subTest(locale=locale):
                result = run(locale, "12", "500\n")
                self.assertEqual(result.returncode, 0, (result.stdout, result.stderr))
                self.assertIn("OK: max 500s", result.stdout)


if __name__ == "__main__":
    unittest.main()
