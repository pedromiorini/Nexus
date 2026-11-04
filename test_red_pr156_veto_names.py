"""Veto por NOME canônico (follow-up da #156): critérios C1–C7 pré-registrados em
docs/cognitivo/AUDITORIA_2026-10-22.md (seção «Veto por nome canônico (`veto_key`)»).

Portado das sondas da Pearl (/workspace/pearl156_probes/test_red_pr156_veto_names.py + common156.py), sem
dependência de caminho externo. Grupos:
  C1 variantes «veta» são vetadas nos 4 caminhos; C2 zero falso positivo; C3 `veto_key` idempotente e ASCII;
  C4 o veto largo não alarga o vocabulário; C5 variante não declarada segue em `unmodeled`;
  C6 `declare_intervention` recusa nome não canonicalizável; C7 entradas vazias/inválidas falham fechado.
SUGESTÕES (veto por efeito, `compare_futures`) são `expectedFailure`: documentam o furo que NÃO é entregue aqui.
"""
import contextlib
import io
import random
import unittest

from core.cognitivo import counterfactual as CF
from core.cognitivo import decision_cycle as DC
from core.cognitivo import mcts_planner as MP
from core.cognitivo import scm_counterfactual as S
from core.cognitivo.action_effects import ACTION_EFFECTS
from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator

VETOED = {"SHUTDOWN_SERVICE": ACTION_EFFECTS["SHUTDOWN_SERVICE"]["world"],
          "IGNORE_ALERT": ACTION_EFFECTS["IGNORE_ALERT"]["world"]}
BASES = ("SHUTDOWN_SERVICE", "IGNORE_ALERT")
CONTROLS = ["MONITOR", "ISOLATE_SYSTEM", "APPLY_PATCH", "ROLLBACK", "ESCALATE", "RESTORE_BACKUP",
            "DEPLOY_UPDATE", "INCREASE_LOGGING"]
INOCENTES = ["SHUTDOWN_REPORT", "SHUTDOWN_SERVICE_CHECK", "DRY_RUN_SHUTDOWN", "SERVICE_SHUTDOWN",
             "UNSHUTDOWN_SERVICE", "RESTART_SERVICE", "STOP_SERVICE", "IGNORE_ALERTS", "ACK_ALERT",
             "SNOOZE_ALERT", "SHUTDOWN", "ALERT", "ROLLBACK_V2", "ROLL_BACK"] + CONTROLS
OBS = {"system_status": "compromised", "alert_level": "critical", "security_patch_applied": False,
       "logging_level": "normal"}
SAFE_EFFECT = {"system_status": "secure", "alert_level": "low"}

VETA = ["minúsculas", "Capitalizado", "alternado", "espaço", "hífen", "__", "borda-espaços", "nbsp", "tab",
        "ZWSP-fim", "ZWSP-meio", "ZWSP-em-letra", "BOM", "ZWJ", "WJ", "soft-hyphen", "RLO", "largura-total",
        "largura-total+_ascii", "ponto", "barra", "dois-pontos", "hífen-U+2011", "en-dash", "minus-U+2212",
        "sufixo-()", "sufixo-;", "sufixo-!"]
SEM_CHAMADA = ("minúsculas", "Capitalizado", "alternado", "espaço", "hífen", "__", "borda-espaços", "nbsp", "tab")


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def make_world(declared=None):
    cr = RealCausalReasoning()
    for action, effects in (declared or {}).items():
        assert cr.declare_intervention(action, effects) == [], action
    return RealWorldModelSimulator(cr)


def variants(base):
    low = base.lower()
    cap = "_".join(p.capitalize() for p in base.split("_"))
    zw = "\u200b"
    first, second = base.split("_")
    full = "".join(chr(ord(c) + 0xFEE0) if c.isalpha() else "\uff3f" for c in base)
    return {
        "minúsculas": low, "Capitalizado": cap,
        "alternado": "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(base)),
        "espaço": low.replace("_", " "), "hífen": base.replace("_", "-"), "__": base.replace("_", "__"),
        "borda-espaços": f" {base} ", "nbsp": base.replace("_", "\u00a0"), "tab": base.replace("_", "\t"),
        "ZWSP-fim": base + zw, "ZWSP-meio": first + zw + "_" + second, "ZWSP-em-letra": base[:3] + zw + base[3:],
        "BOM": base + "\ufeff", "ZWJ": base + "\u200d", "WJ": base + "\u2060",
        "soft-hyphen": base[:4] + "\u00ad" + base[4:], "RLO": "\u202e" + base,
        "largura-total": full, "largura-total+_ascii": full.replace("\uff3f", "_"),
        "ponto": base.replace("_", ".").lower(), "barra": base.replace("_", "/"),
        "dois-pontos": base.replace("_", ":"), "hífen-U+2011": base.replace("_", "\u2011"),
        "en-dash": base.replace("_", "\u2013"), "minus-U+2212": base.replace("_", "\u2212"),
        "sufixo-()": base + "()", "sufixo-;": base + ";", "sufixo-!": base + "!",
        "sufixo-( )": base + "( )",
        "sufixo-_v2": base + "_V2", "sufixo-.exec": low + ".exec", "sufixo-plural": base + "S",
        "prefixo-do_": "do_" + low,
        "cirílico-І": base.replace("I", "\u0406"), "cirílico-Ѕ": "\u0405" + base[1:],
    }


def cases():
    for base in BASES:
        table = variants(base)
        for label in VETA:
            yield base, label, table[label]


def veto_key():
    from core.cognitivo import action_effects
    func = getattr(action_effects, "veto_key", None)
    if func is None:
        raise ImportError("core.cognitivo.action_effects.veto_key ausente (RED)")
    return func


def plan_vetoed(name, declared=None):
    world = make_world({name: declared} if declared else None)
    world.current_state.state["system_status"] = "compromised"
    planner = quiet(MP.RealMCTSPlanner, world_model=world, causal_reasoning=world.causal_reasoning)
    random.seed(7)
    result = quiet(planner.plan, goal="secure the system", available_actions=[name], max_iterations=60)
    return [item["action"] for item in (getattr(result, "vetoed_actions", ()) or ())]


class C1VetoPorCaminho(unittest.TestCase):
    def test_plan_variante_vira_vetada(self):
        for base, label, name in cases():
            with self.subTest(base=base, variante=label):
                self.assertIn(name, plan_vetoed(name))

    def test_rank_alternatives_alias_declarado_e_vetado(self):
        for base, label, name in cases():
            with self.subTest(base=base, variante=label):
                world = make_world({name: VETOED[base]})
                out = S.rank_alternatives(OBS, ["IGNORE_ALERT"], [[name], ["ISOLATE_SYSTEM"]], world)
                item = next(i for i in out["ranking"] if i["actions"] == [name])
                self.assertTrue(item["vetoed"])
                self.assertFalse(out["best"] and out["best"]["actions"] == [name])

    def test_motor_rank_counterfactuals_alias_declarado_e_vetado(self):
        for base, label, name in cases():
            with self.subTest(base=base, variante=label):
                world = make_world({name: VETOED[base]})
                engine = quiet(CF.RealCounterfactualReasoningEngine, world_model=world)
                out = engine.rank_counterfactuals(OBS, ["IGNORE_ALERT"], [[name], ["ISOLATE_SYSTEM"]])
                item = next(i for i in out["ranking"] if i["actions"] == [name])
                self.assertTrue(item["vetoed"])

    def test_decision_cycle_variante_vetada(self):
        for base, label, name in cases():
            with self.subTest(base=base, variante=label):
                decision = DC.select_operator([name])
                self.assertIsNone(decision.selected)
                self.assertEqual(decision.impasse.kind, "CONSTRAINT_FAILURE")


class C3C6Chave(unittest.TestCase):
    def test_veto_key_contrato(self):
        func = veto_key()
        for base, label, name in cases():
            self.assertEqual(func(name), base, (label, name))
        self.assertEqual(func(variants("SHUTDOWN_SERVICE")["sufixo-( )"]), "SHUTDOWN_SERVICE")
        for label in ("cirílico-І", "cirílico-Ѕ"):
            self.assertIsNone(func(variants("SHUTDOWN_SERVICE")[label]))
        for label in ("sufixo-_v2", "sufixo-.exec", "sufixo-plural", "prefixo-do_"):
            self.assertNotEqual(func(variants("SHUTDOWN_SERVICE")[label]), "SHUTDOWN_SERVICE", label)
        for value in [None, 5, True, b"x", "", " ", "_", " _-. ", "()", "\u200b", "\x00", "(dry run)",
                      "SHUTDOWN SERVICE (dry run)"]:
            self.assertIsNone(func(value), repr(value))

    def test_idempotencia_fuzz(self):
        func = veto_key()
        rnd = random.Random(0)
        alpha = list("SHUTDOWNIGNORALERT_ -./:;()!?abc019") + ["\u200b", "\ufeff", "\u00a0", "\u2011", "ＳＨ",
                                                            "\u0405", "\u0131", "ß", "①"]
        for _ in range(20000):
            text = "".join(rnd.choice(alpha) for _ in range(rnd.randint(0, 12)))
            key = func(text)
            if key is not None:
                self.assertEqual(func(key), key, text)
                self.assertRegex(key, r"^[A-Z0-9_]+$")

    def test_declare_intervention_recusa_nao_canonicalizavel(self):
        reasoning = RealCausalReasoning()
        for name in ["\u0405" + "HUTDOWN_SERVICE", "\u200b", "shutdown service (dry run)"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                reasoning.declare_intervention(name, VETOED["SHUTDOWN_SERVICE"])


class C7EntradaInvalida(unittest.TestCase):
    def test_veto_key_subclasse_hostil_de_str_nao_chama_metodos_do_chamador(self):
        class Hostil(str):
            def __str__(self):
                raise AssertionError("__str__ do chamador chamado")

            def upper(self):
                raise AssertionError("upper do chamador chamado")

            def strip(self, *args):
                raise AssertionError("strip do chamador chamado")

        self.assertEqual(veto_key()(Hostil("shutdown service")), "SHUTDOWN_SERVICE")
        self.assertIsNone(veto_key()(Hostil("   ")))

    def test_declare_intervention_continua_recusando_vazio_e_tipo_errado(self):
        reasoning = RealCausalReasoning()
        for name in ["", "   ", 5, None]:
            with self.subTest(name=name), self.assertRaises((ValueError, TypeError)):
                reasoning.declare_intervention(name, VETOED["SHUTDOWN_SERVICE"])

    def test_plan_com_itens_invalidos_nao_os_marca_como_vetados(self):
        world = make_world()
        planner = quiet(MP.RealMCTSPlanner, world_model=world, causal_reasoning=world.causal_reasoning)
        random.seed(7)
        result = quiet(planner.plan, goal="secure the system", available_actions=[None, "", 5, "MONITOR"],
                       max_iterations=10)
        self.assertEqual(list(getattr(result, "vetoed_actions", ()) or ()), [])

    def test_select_operator_lista_vazia_continua_no_change(self):
        decision = DC.select_operator([])
        self.assertIsNone(decision.selected)
        self.assertEqual(decision.impasse.kind, "NO_CHANGE")


class GuardaHojeEDepois(unittest.TestCase):
    def test_C2_inocentes_nao_vetados(self):
        for name in INOCENTES:
            with self.subTest(name=name):
                self.assertNotIn(name, plan_vetoed(name, SAFE_EFFECT))
                world = make_world({name: SAFE_EFFECT})
                out = S.rank_alternatives(OBS, ["IGNORE_ALERT"], [[name], ["ISOLATE_SYSTEM"]], world)
                self.assertFalse(next(i for i in out["ranking"] if i["actions"] == [name])["vetoed"])
                engine = quiet(CF.RealCounterfactualReasoningEngine, world_model=world)
                out = engine.rank_counterfactuals(OBS, ["IGNORE_ALERT"], [[name], ["ISOLATE_SYSTEM"]])
                self.assertFalse(next(i for i in out["ranking"] if i["actions"] == [name])["vetoed"])
                decision = DC.select_operator([name])
                self.assertIsNone(decision.impasse)
                self.assertEqual(decision.selected, name)

    def test_C4_veto_largo_nao_alarga_vocabulario(self):
        decision = DC.select_operator(["shutdown.service"], allowed=["MONITOR", "SHUTDOWN_SERVICE"],
                                      use_default_rejects=False)
        self.assertIsNone(decision.selected)
        self.assertEqual(decision.impasse.kind, "CONSTRAINT_FAILURE")
        self.assertIn("vocabul", decision.rejected["shutdown.service"])

    def test_C5_variante_nao_declarada_falha_fechado(self):
        for base, label, name in cases():
            if label in SEM_CHAMADA:
                continue
            with self.subTest(base=base, variante=label):
                out = S.rank_alternatives(OBS, ["IGNORE_ALERT"], [[name], ["ISOLATE_SYSTEM"]], make_world())
                self.assertIn([name], out["unmodeled"])
                self.assertFalse(out["best"] and out["best"]["actions"] == [name])

    def test_exato_continua_vetado(self):
        for base in BASES:
            self.assertIn(base, plan_vetoed(base))
            self.assertIsNone(DC.select_operator([base]).selected)


class SugestoesNaoEntregues(unittest.TestCase):
    @unittest.expectedFailure
    def test_veto_por_efeito_alias_com_sufixo_semantico(self):
        for label in ("sufixo-_v2", "sufixo-.exec", "sufixo-plural", "prefixo-do_"):
            name = variants("SHUTDOWN_SERVICE")[label]
            with self.subTest(variante=label):
                world = make_world({name: VETOED["SHUTDOWN_SERVICE"]})
                out = S.rank_alternatives(OBS, ["IGNORE_ALERT"], [[name], ["ISOLATE_SYSTEM"]], world)
                self.assertTrue(next(i for i in out["ranking"] if i["actions"] == [name])["vetoed"])

    @unittest.expectedFailure
    def test_brain_compare_futures_expoe_veto(self):
        world = make_world()
        out = world.compare_futures([["SHUTDOWN_SERVICE"], ["MONITOR"]])
        self.assertTrue(any(k in out for k in ("vetoed", "vetoes")) or getattr(out["results"][0], "vetoed", False))


if __name__ == "__main__":
    unittest.main(verbosity=1)
