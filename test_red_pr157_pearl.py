"""RED da revisão da Pearl sobre a #157 (``dca56c4``): S1 e S3 (S2 fica de fora, decisão do Pedro).

Portado de ``/workspace/pearl157_probes/test_red_pr157_pearl.py`` (sem ``PEARL_REPO``, sem a classe S2).
Pré-registro: C9–C12 em ``docs/cognitivo/AUDITORIA_2026-10-22.md`` (seção «`veto_key` rodada 2»).

- S1 (C9): variantes do nome vetado com sintaxe de chamada espaçada ou invisível que não é Cf/Cc têm de canonicalizar.
- C10: guardas (passam antes e depois): S1 não muda chave que já existia, não alarga o que continua ``None``.
- S3 (C11): guarda contra candidato não-``str`` com ``__eq__``/``__hash__`` hostil; só fica RED sob os mutantes M3a/M3b.
"""
import contextlib
import io
import random
import re
import unicodedata
import unittest

from core.cognitivo import decision_cycle as DC
from core.cognitivo import mcts_planner as MP
from core.cognitivo.action_effects import veto_key, veto_lookup
from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator
from test_red_pr156_veto_names import INOCENTES

B = "SHUTDOWN_SERVICE"


class S1ConsultaNaoPodeSerAbertaPorNomeSemChave(unittest.TestCase):
    """Se o nome só difere do vetado por sintaxe de chamada ou invisível não-Cf/Cc, a chave tem de ser a vetada."""
    VARIANTES = [
        B + " ( ) ;", B + " ; ;", B + "() ;", B + "();\t!", B + " (  ) ?",
        B + "\u3164", "SHUTDOWN\uffa0_SERVICE", B + "\u2800", B + "\u115f\u1160", "SHUT\u034fDOWN_SERVICE",
        B + "\ufe0f", "SHUTDOWN_\u0301SERVICE", B + "\u180b", B + "\U000e0100",
    ]

    def test_chave_canonica_das_variantes(self):
        for v in self.VARIANTES:
            with self.subTest(v=ascii(v)):
                self.assertEqual(veto_key(v), B)

    def test_decision_cycle_padrao_nao_seleciona_a_variante(self):
        for v in self.VARIANTES:
            with self.subTest(v=ascii(v)):
                self.assertIsNone(DC.select_operator([v]).selected)


def _veto_key_dca56c4(name):
    """Referência congelada: ``veto_key`` exatamente como em ``dca56c4`` (antes de S1)."""
    if not isinstance(name, str):
        return None
    sep = re.compile(r"[\s\-.:/\\,\u2010-\u2015\u2212\u00b7]+")
    call = re.compile(r"(?:\(_*\))?[;!?]*\Z")
    text = unicodedata.normalize("NFKC", str.__str__(name))
    text = "".join(c for c in text if c.isspace() or unicodedata.category(c) not in ("Cf", "Cc"))
    text = call.sub("", sep.sub("_", text.strip()))
    text = re.sub(r"_+", "_", text).strip("_")
    if not text or not re.fullmatch(r"[A-Za-z0-9_]+", text):
        return None
    return text.upper()


class C10S1NaoMudaChaveExistenteNemAlarga(unittest.TestCase):
    ALPHA = (list("SHUTDOWN_ ab01") + [" ", "-", ".", "(", ")", "()", ";", "!", "?", "_", "\u200b", "\u3164", "\u2800",
                                      "\u0301", "\ufe0f", "\u0405", "\u00e9", "\u2122", "\x00"])

    def test_diferencial_com_dca56c4(self):
        rnd = random.Random(5)
        changed, bad_shape, not_idem, widened = [], [], [], 0
        for _ in range(400000):
            text = "".join(rnd.choice(self.ALPHA) for _ in range(rnd.randint(0, 10)))
            old, new = _veto_key_dca56c4(text), veto_key(text)
            if old is not None and new != old:
                changed.append(text)
            if new is not None:
                if re.fullmatch(r"[A-Z0-9_]+", new) is None:
                    bad_shape.append(text)
                elif veto_key(new) != new:
                    not_idem.append(text)
            if old is None and new is not None:
                widened += 1
        self.assertEqual(changed[:5], [])
        self.assertEqual(bad_shape[:5], [])
        self.assertEqual(not_idem[:5], [])
        self.assertGreaterEqual(widened, 0)  # estatística apenas: S1 pode (e deve) canonicalizar novas variantes

    def test_inocentes_continuam_sem_veto(self):
        for name in INOCENTES:
            with self.subTest(name=name):
                self.assertEqual(DC.select_operator([name]).selected, name)
                self.assertEqual(veto_lookup(name, {B: "r"}), (False, None))

    def test_o_que_continua_none(self):
        for text in ["shutdown_service (dry run)", "(dry run)", "\u0405HUTDOWN_SERVICE", "SHUTDOWN_SERVI\u00dfE", "()", ";",
                     " ; ", "\u3164", "\u0301", "_"]:
            with self.subTest(text=ascii(text)):
                self.assertIsNone(veto_key(text))


class C12GuardaMarcasMnMe(unittest.TestCase):
    """Guarda acrescentada na etapa de mutantes (errata E-VK6): as 14 variantes da Pearl só têm Mn; sem esta guarda o
    mutante ``vk_keeps_me`` (Me fora do filtro) sobrevivia. Passa no GREEN; morre sob ``vk_keeps_me``/``vk_keeps_mn``."""

    def test_marcas_envolventes_me(self):
        for c in ("\u20dd", "\u0488"):
            for v in (B + c, "SHUTDOWN_" + c + "SERVICE", "SHUT" + c + "DOWN_SERVICE"):
                with self.subTest(v=ascii(v)):
                    self.assertEqual(veto_key(v), B)
                    self.assertIsNone(DC.select_operator([v]).selected)

    def test_toda_marca_mn_e_me_do_unicode_e_removida(self):
        marcas = [chr(i) for i in range(0x110000) if unicodedata.category(chr(i)) in ("Mn", "Me")]
        self.assertGreater(len(marcas), 1000)
        ruins = [hex(ord(c)) for c in marcas if veto_key("SHUTDOWN_" + c + "SERVICE") != B]
        self.assertEqual(ruins[:5], [])


class S3EntradaNaoStrNaoChamaMetodoDoChamador(unittest.TestCase):
    """Guarda (passa hoje; falha sob M3a = ``plan`` sem ``not isinstance(action, str)``, e sob M3b = ``veto_lookup``
    sem ``isinstance``)."""

    class Eq:
        def __eq__(self, other):
            raise RuntimeError("__eq__ chamado em candidato não-str")

        __hash__ = object.__hash__

    class Hash:
        def __hash__(self):
            raise RuntimeError("__hash__ chamado em candidato não-str")

    def test_plan_nao_chama_eq_de_candidato_nao_str(self):
        causal = RealCausalReasoning()
        world = RealWorldModelSimulator(causal)
        planner = MP.RealMCTSPlanner(world_model=world, causal_reasoning=causal)
        random.seed(1)
        with contextlib.redirect_stdout(io.StringIO()):
            result = planner.plan(goal="g", available_actions=[B, "MONITOR", self.Eq()], max_iterations=6)
        self.assertEqual([v["action"] for v in result.vetoed_actions], [B])

    def test_veto_lookup_nao_chama_hash_de_nao_str(self):
        self.assertEqual(veto_lookup(self.Hash(), {B: "r"}), (False, None))


if __name__ == "__main__":
    unittest.main()
