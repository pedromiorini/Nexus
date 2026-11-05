"""Auditoria dos mutantes de ``veto_key`` retirados do catálogo ``cognitivo_mutation`` (errata E-VK6).

``docs/cognitivo/MUTANTES_VK_RETIRADOS.md`` traz o texto exato (arquivo, trecho, substituição) de cada um.
Este teste confirma (a) que o trecho ainda aplica exatamente uma vez no arquivo de produção, (b) que o doc cita o
nome e o trecho, (c) que os que são equivalentes não divergem de ``veto_key`` num fuzz determinístico (e numa
varredura de todos os code points, para os preenchedores), e (d) que o não equivalente (M3a) segue no catálogo.
Se um trecho deixar de aplicar uma vez, o mutante virou ``not_applied``: reavaliar a equivalência.
"""
import random
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOC = (ROOT / "docs/cognitivo/MUTANTES_VK_RETIRADOS.md").read_text(encoding="utf-8")
EFFECTS = "core/cognitivo/action_effects.py"

# (nome, arquivo, trecho, substituição)
EQUIVALENTES = [
    ("vk_sep_not_greedy", EFFECTS, r'\u00b7]+")', r'\u00b7]+?")'),
    ("vk_ascii_allows_dash", EFFECTS, '_ASCII_KEY = re.compile(r"[A-Za-z0-9_]+")',
     '_ASCII_KEY = re.compile(r"[A-Za-z0-9_-]+")'),
    ("vk_text_not_stripped_before_separators", EFFECTS, '_SEPARATORS.sub("_", text.strip())',
     '_SEPARATORS.sub("_", text)'),
    ("vk_filler_hangul_compat_kept", EFFECTS, r'\u3164', ''),
    ("vk_filler_halfwidth_hangul_kept", EFFECTS, r'\uffa0', ''),
]
NAO_EQUIVALENTE = ("vk_mcts_non_str_candidate_vetoed", "core/cognitivo/mcts_planner.py",
                   'if not isinstance(action, str) or any(v["action"] == action for v in vetoed):',
                   'if any(v["action"] == action for v in vetoed):')

ALFABETO = (list("SHUTDOWN_ ab01") + ["\t", "-", ".", ":", "/", "\\", ",", "\u2010", "\u2015", "\u2212", "\u00b7",
                                      "\u00a0", "(", ")", "()", ";", "!", "?", "_", "\u200b", "\u3164", "\uffa0",
                                      "\u115f", "\u1160", "\u2800", "\u0301", "\u20dd", "\u0405", "\ufe63",
                                      "\uff0d", "\u00e9", "\x00"])


def mutante(file, needle, replacement):
    source = (ROOT / file).read_text(encoding="utf-8")
    assert source.count(needle) == 1, (file, needle, source.count(needle))
    namespace = {"__name__": "mutante_veto_key"}
    exec(compile(source.replace(needle, replacement), file, "exec"), namespace)  # nosec B102 - mutante local
    return namespace["veto_key"]


class MutantesRetiradosSaoAuditaveis(unittest.TestCase):
    def test_trechos_aplicam_uma_vez_e_estao_no_doc(self):
        for name, file, needle, replacement in EQUIVALENTES + [NAO_EQUIVALENTE]:
            with self.subTest(name=name):
                source = (ROOT / file).read_text(encoding="utf-8")
                self.assertEqual(source.count(needle), 1)
                self.assertIn(name, DOC)
                self.assertIn(needle, DOC)
                if replacement:
                    self.assertIn(replacement, DOC)

    def test_equivalentes_nao_divergem_no_fuzz(self):
        from core.cognitivo.action_effects import veto_key
        rnd = random.Random(11)
        entradas = ["".join(rnd.choice(ALFABETO) for _ in range(rnd.randint(0, 10))) for _ in range(150000)]
        for name, file, needle, replacement in EQUIVALENTES:
            fn = mutante(file, needle, replacement)
            with self.subTest(name=name):
                divergentes = [t for t in entradas if fn(t) != veto_key(t)]
                self.assertEqual(divergentes[:3], [])

    def test_preenchedores_redundantes_por_nfkc_em_todo_code_point(self):
        import unicodedata
        from core.cognitivo.action_effects import veto_key
        self.assertEqual({unicodedata.normalize("NFKC", c) for c in "\u3164\uffa0"}, {"\u1160"})
        for name, file, needle, replacement in EQUIVALENTES[-2:]:
            fn = mutante(file, needle, replacement)
            with self.subTest(name=name):
                divergentes = [hex(i) for i in range(0x110000) if fn("A" + chr(i) + "B") != veto_key("A" + chr(i) + "B")]
                self.assertEqual(divergentes[:3], [])

    def test_o_nao_equivalente_segue_no_catalogo(self):
        from tools import cognitivo_mutation as cm
        achados = [m for m in cm.MUTATIONS if m.name == NAO_EQUIVALENTE[0]]
        self.assertEqual(len(achados), 1)
        self.assertEqual((achados[0].needle, achados[0].replacement), NAO_EQUIVALENTE[2:])
        retirados = {name for name, *_ in EQUIVALENTES}
        self.assertFalse(retirados & {m.name for m in cm.MUTATIONS})


if __name__ == "__main__":
    unittest.main()
