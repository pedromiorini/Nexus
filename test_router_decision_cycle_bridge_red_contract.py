"""RED B1–B5 da ponte CentralRouter, congelado em
docs/cognitivo/PONTE_CENTRALROUTER_SECAO_B_2026-11-10.md.

O commit GREEN (B2) tira os expectedFailure. Um sucesso inesperado
reprova a suíte. A guarda B4a fica sem o marcador.
"""
import contextlib
import hashlib
import io
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from core.cognitivo import decision_cycle as decision_cycle
from core.constitutional_brain import CentralRouter, EthicsGuard, ModuleType
from test_router_fail_closed_contract import _Guard, _boom, _modules

GOLDEN = {
    "should I plan my week": "planning",
    "What should I do about the pharmacist?": "planning",
    "pharmacist": "ethical",
    "copyright policy": "ethical",
    "what did I say about the right choice": "ethical",
    "design a fair strategy": "planning",
    "why is the sky blue": "analysis",
    "remember the last time": "recall",
    "write a poem": "creative",
    "Tell me about item 9": "query",
    "learn this fact": "query",
    "ping": "query",
}
AMBIGUOUS = {
    "should I plan my week": ["planning", "ethical"],
    "design a fair strategy": ["planning", "ethical", "creative"],
}
INPUT_CODES = {"DC_BAD_INPUT", "DC_NO_INPUT"}


def _router(calls, **overrides):
    modules = _modules(calls)
    for key, value in overrides.items():
        module = ModuleType[key]
        if value is None:
            modules.pop(module, None)
        else:
            modules[module] = value
    log = NS(log_event=lambda *args, **kwargs: "h")
    jailbreak = NS(detect=lambda prompt: NS(detected=False))
    with contextlib.redirect_stdout(io.StringIO()):
        router = CentralRouter(modules, EthicsGuard(log, jailbreak))
    router.vram_guard = _Guard()
    return router


def _blocks(test, result, error, calls):
    test.assertIs(result.get("success"), False, result.get("error"))
    test.assertEqual(result.get("error"), error)
    test.assertIs(result.get("degraded"), True)
    test.assertEqual(calls, [], "nenhum stage (nem jailbreak) roda depois de impasse do ciclo")
    test.assertIn("impasse", result)


class B1Wiring(unittest.TestCase):
    """O ciclo é chamado 1x por requisição não cacheada, pelo atributo do módulo."""

    @unittest.expectedFailure
    def test_cycle_called_once_and_type_matches_golden(self):
        for prompt, expected in GOLDEN.items():
            with self.subTest(prompt=prompt):
                seen = []
                real = decision_cycle.run_decision_cycle

                def spy(*args, **kwargs):
                    seen.append(1)
                    return real(*args, **kwargs)

                with patch("core.cognitivo.decision_cycle.run_decision_cycle", spy):
                    result = _router([]).route(prompt, {"k": 1})
                self.assertEqual(len(seen), 1)
                self.assertEqual(result["request_type"], expected)


class B2ImpasseBlocks(unittest.TestCase):
    """Impasses alcançáveis sem apply bloqueiam antes de qualquer stage."""

    cases = {
        "PROPOSER_FAILURE": (_boom("segredo-xyz"),),
        "NO_CHANGE": (lambda state: (),),
        "CONSTRAINT_FAILURE": (lambda state: ("ethical\u200b",),),
        "TIE": (lambda state: ("planning",), lambda state: ("ethical",)),
    }

    @unittest.expectedFailure
    def test_each_impasse_kind_blocks(self):
        for kind, proposers in self.cases.items():
            with self.subTest(kind=kind):
                calls = []
                router = _router(calls)
                router.request_type_proposers = proposers
                if kind == "TIE":
                    router.request_type_precedence = ()
                result = router.route("Tell me about item 1", {"k": 1})
                _blocks(self, result, "ROUTER_IMPASSE", calls)
                self.assertEqual(result["impasse"]["kind"], kind)
                self.assertNotIn("segredo-xyz", repr(result))
                again = router.route("Tell me about item 1", {"k": 1})
                self.assertIs(again.get("cached", False), False)


class B3InputErrorBlocks(unittest.TestCase):
    """DecisionInputError vira bloqueio, nunca exceção solta."""

    @unittest.expectedFailure
    def test_bad_input_blocks_with_code(self):
        for bad in ("abc", [1], None):
            with self.subTest(bad=bad):
                calls = []
                router = _router(calls)
                router.request_type_proposers = bad
                result = router.route("Tell me about item 1", {"k": 1})
                _blocks(self, result, "ROUTER_DECISION_INPUT_INVALID", calls)
                self.assertIn(result["impasse"]["code"], INPUT_CODES)

    @unittest.expectedFailure
    def test_everything_failing_blocks_on_the_cycle_first(self):
        calls = []
        absent = {module.name: None for module in ModuleType}
        router = _router(calls, **absent)
        router.request_type_proposers = (_boom("a"), _boom("b"))
        result = router.route("Is it ethical to lie?", {"k": 1})
        _blocks(self, result, "ROUTER_IMPASSE", calls)


class B4NoSteering(unittest.TestCase):
    """O chamador não escolhe o tipo. B4a é a guarda que passa na main."""

    def test_context_cannot_steer_away_from_ethical(self):
        calls = []
        context = {
            "request_type": "query",
            "proposals": ["query"],
            "allowed": ["query"],
            "operators": ["query"],
        }
        result = _router(calls).route("Is it ethical to lie?", context)
        self.assertEqual(result["request_type"], "ethical")
        self.assertIn("swarm", calls)

    @unittest.expectedFailure
    def test_unicode_variant_from_a_proposer_is_rejected(self):
        for variant in ("ethical\u200b", "\uff45thical", "ethic al", "query;"):
            with self.subTest(v=variant):
                calls = []
                router = _router(calls)

                def propose(state, variant=variant):
                    return (variant,)

                router.request_type_proposers = (propose,)
                result = router.route("Tell me about item 1", {"k": 1})
                _blocks(self, result, "ROUTER_IMPASSE", calls)
                self.assertEqual(result["impasse"]["kind"], "CONSTRAINT_FAILURE")


class B5Observability(unittest.TestCase):
    """Ambiguidade no sucesso; impasse com origem; contador por tipo."""

    @unittest.expectedFailure
    def test_success_records_all_proposals(self):
        for prompt, proposed in AMBIGUOUS.items():
            with self.subTest(prompt=prompt):
                result = _router([]).route(prompt, {"k": 1})
                self.assertEqual(result["decision"]["selected"], GOLDEN[prompt])
                self.assertEqual(result["decision"]["proposed"], proposed + ["query"])

    @unittest.expectedFailure
    def test_impasse_has_origin_and_is_counted(self):
        router = _router([])
        router.request_type_proposers = (_boom(),)
        result = router.route("Tell me about item 1", {"k": 1})
        origin = hashlib.sha256(b"Tell me about item 1").hexdigest()[:16]
        self.assertEqual(result["impasse"]["origin"], origin)
        counted = router.get_statistics()["impasses_by_kind"]
        self.assertEqual(counted, {"PROPOSER_FAILURE": 1})
