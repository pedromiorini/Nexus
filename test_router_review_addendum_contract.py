"""RT9–RT13: adendo da revisão do Laird ao CentralRouter fail-closed.

Pré-registrado em docs/cognitivo/AUDITORIA_2026-10-22.md (seção «adendo da revisão do Laird»,
protocolo RT-v2). Seeds fixas random.Random(20261031 + k), k = 0..19; módulos falsos escritos aqui.
"""
import contextlib
import io
import itertools
import random
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from core.constitutional_brain import CompleteNexusBrain, ModuleType, RequestType
from test_router_fail_closed_contract import _boom, _ethical, _query, _router

SEEDS = [20261031 + k for k in range(20)]


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class _Recorder:
    """Módulo LOG falso que guarda (event_type, data)."""

    def __init__(self):
        self.events = []

    def log_event(self, event_type, data):
        self.events.append((event_type, data))
        return "hash"

    def blocked(self):
        return [event for event in self.events if event[0] == "ROUTER_BLOCKED"]


class _BrainCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = _quiet(CompleteNexusBrain, enable_router=True)

    def setUp(self):
        modules = self.brain.router.modules
        self._saved = dict(modules)
        self.addCleanup(lambda: (modules.clear(), modules.update(self._saved)))
        self.reset()

    def reset(self):
        """Volta os módulos do roteador ao estado original (com LOG gravador) entre subtestes."""
        modules = self.brain.router.modules
        modules.clear()
        modules.update(self._saved)
        self.log = _Recorder()
        modules[ModuleType.LOG] = self.log

    def think(self, question):
        return _quiet(self.brain.think, question)


BLOCKERS = (
    ("BUDGET_DENIED", "budget", ModuleType.BUDGET, lambda: NS(request_budget=lambda *a, **k: False)),
    ("JAILBREAK_DETECTED", "jailbreak", ModuleType.JAILBREAK, lambda: NS(detect=lambda p: NS(detected=True))),
)


class RT9ThinkPropagatesBlock(_BrainCase):
    def test_blocked_route_surfaces_error_and_degraded(self):
        for k, seed in enumerate(SEEDS):
            code, name, module_type, factory = BLOCKERS[k % 2]
            with self.subTest(seed=seed, code=code):
                self.reset()
                self.brain.router.modules[module_type] = factory()
                result = self.think(_query(random.Random(seed)))
                self.assertEqual(result.get("error"), code)
                self.assertIs(result.get("degraded"), True)
                self.assertEqual(result.get("failed_modules"), [name])
                self.assertTrue(result.get("reason"))
                self.assertIs(result.get("success"), False)

    def test_clean_route_has_no_error_and_is_not_degraded(self):
        for seed in SEEDS[:5]:
            with self.subTest(seed=seed):
                result = self.think(_query(random.Random(seed)))
                self.assertFalse(result.get("error"))
                self.assertIs(result.get("degraded"), False)
                self.assertEqual(result.get("failed_modules"), [])


OPTIONAL = (
    (ModuleType.MEMORY, "retrieve"),
    (ModuleType.KNOWLEDGE_GRAPH, "rag_query"),
    (ModuleType.SWARM, "deliberate"),
)


class RT10ThinkPropagatesDegraded(_BrainCase):
    def test_optional_failure_surfaces_degraded(self):
        for k, seed in enumerate(SEEDS):
            module_type, fn_name = OPTIONAL[k % 3]
            with self.subTest(seed=seed, module=module_type.value):
                self.reset()
                self.brain.router.modules[module_type] = NS(**{fn_name: _boom("down")})
                result = self.think(_query(random.Random(seed)))
                self.assertFalse(result.get("error"))
                self.assertIs(result.get("degraded"), True)
                self.assertEqual(result.get("failed_modules"), [module_type.value])


CAUSES = (
    ("budget", "BUDGET_DENIED", {"BUDGET": NS(request_budget=lambda *a, **k: False)}, _query),
    ("jailbreak", "JAILBREAK_DETECTED", {"JAILBREAK": NS(detect=lambda p: NS(detected=True))}, _query),
    ("swarm", "MANDATORY_MODULE_FAILED", {"SWARM": NS(deliberate=_boom())}, _ethical),
)


class RT11BlockIsLoggedOnce(unittest.TestCase):
    def _route(self, overrides, prompt, log):
        router = _router([], LOG=log, **overrides)
        return router, router.route(prompt)

    def test_each_cause_logs_exactly_one_router_blocked(self):
        for k, seed in enumerate(SEEDS):
            name, code, overrides, make_prompt = CAUSES[k % 3]
            with self.subTest(seed=seed, cause=name):
                prompt = make_prompt(random.Random(seed))
                log = _Recorder()
                _, result = self._route(overrides, prompt, log)
                self.assertEqual(result["error"], code)
                blocked = log.blocked()
                self.assertEqual(len(blocked), 1, log.events)
                data = blocked[0][1]
                self.assertEqual(data["error"], code)
                self.assertEqual(data["module"], name)
                self.assertEqual(data["request_type"], result["request_type"])
                self.assertEqual(data["schema_version"], 1)
                self.assertNotIn("prompt", data)

    def test_long_prompt_is_not_persisted_in_router_block_event(self):
        log = _Recorder()
        prompt = "ROUTER_BLOCKED_CANARY:" + "x" * 300
        self._route(CAUSES[0][2], prompt, log)
        blocked = log.blocked()
        self.assertEqual(len(blocked), 1, log.events)
        data = blocked[0][1]
        self.assertEqual(data["schema_version"], 1)
        self.assertNotIn("prompt", data)
        self.assertNotIn("ROUTER_BLOCKED_CANARY", repr(data))

    def test_success_logs_no_router_blocked(self):
        for seed in SEEDS[:5]:
            with self.subTest(seed=seed):
                log = _Recorder()
                _, result = self._route({}, _query(random.Random(seed)), log)
                self.assertIs(result["success"], True)
                self.assertEqual(log.blocked(), [])

    def test_failing_or_missing_log_does_not_change_the_block(self):
        for name, code, overrides, make_prompt in CAUSES:
            for log in (NS(log_event=_boom("log down")), None):
                with self.subTest(cause=name, log=log is not None):
                    prompt = make_prompt(random.Random(SEEDS[0]))
                    _, result = self._route(overrides, prompt, log)
                    self.assertIs(result["success"], False)
                    self.assertEqual(result["error"], code)

    def test_blocked_think_logs_exactly_one(self):
        class Case(_BrainCase):
            def runTest(self):
                pass

        Case.setUpClass()
        case = Case()
        case.setUp()
        try:
            case.brain.router.modules[ModuleType.BUDGET] = NS(request_budget=lambda *a, **k: False)
            case.think("Tell me about item 7")
            self.assertEqual(len(case.log.blocked()), 1, case.log.events)
        finally:
            case.doCleanups()


class RT12BlockedRequestsAreCounted(unittest.TestCase):
    def test_blocks_enter_the_statistics(self):
        for k, seed in enumerate(SEEDS):
            name, code, overrides, make_prompt = CAUSES[k % 3]
            n = 1 + k % 4
            with self.subTest(seed=seed, cause=name, n=n):
                calls = []
                router = _router(calls, **overrides)
                per_request = []
                clock = itertools.count(1000.0, 0.01)
                with mock.patch("time.time", side_effect=lambda: next(clock)):
                    for i in range(n):
                        before = len(calls)
                        result = router.route(make_prompt(random.Random(seed + i)))
                        self.assertEqual(result["error"], code)
                        # o módulo substituído não grava em `calls`; budget/jailbreak respondem com sucesso, swarm levanta
                        per_request.append(len(calls) - before + (name in ("budget", "jailbreak")))
                stats = router.get_statistics()
                self.assertEqual(stats["total_requests"], n)
                self.assertEqual(stats["blocked_requests"], n)
                self.assertEqual(stats["blocked_by_error"], {code: n})
                self.assertEqual(stats["cache_hits"], 0)
                self.assertGreater(router.stats["total_latency_ms"], 0)
                self.assertAlmostEqual(router.stats["avg_modules_per_request"], sum(per_request) / n)

    def test_success_does_not_count_as_blocked(self):
        router = _router([])
        router.route("Tell me about item 1")
        stats = router.get_statistics()
        self.assertEqual(stats["total_requests"], 1)
        self.assertEqual(stats["blocked_requests"], 0)
        self.assertEqual(stats["blocked_by_error"], {})


class RT13FailureCauseIsReported(_BrainCase):
    def test_optional_failure_carries_the_message(self):
        for k, seed in enumerate(SEEDS):
            rng = random.Random(seed)
            message = f"boom-{rng.randrange(10**9)}"
            module_type, fn_name = OPTIONAL[k % 3]
            with self.subTest(seed=seed, module=module_type.value):
                router = _router([], **{module_type.name: NS(**{fn_name: _boom(message)})})
                result = router.route(_query(rng))
                self.assertIs(result["success"], True)
                self.assertEqual(result["failed_modules"], [module_type.value])
                detail = result["failed_modules_detail"]
                self.assertEqual([item["module"] for item in detail], [module_type.value])
                self.assertIn(message, detail[0]["error"])

    def test_blocks_carry_the_cause(self):
        for seed in SEEDS[:5]:
            message = f"detector-{seed}"
            with self.subTest(seed=seed):
                result = _router([], JAILBREAK=NS(detect=_boom(message))).route(_query(random.Random(seed)))
                self.assertEqual(result["error"], "JAILBREAK_CHECK_FAILED")
                self.assertEqual(result["failed_modules"], ["jailbreak"])
                detail = result["failed_modules_detail"]
                self.assertEqual([item["module"] for item in detail], ["jailbreak"])
                self.assertIn(message, detail[0]["error"])
        denied = _router([], BUDGET=NS(request_budget=lambda *a, **k: False)).route("Tell me about item 2")
        self.assertEqual(denied["failed_modules_detail"], [{"module": "budget", "error": "resource budget denied"}])

    def test_clean_route_has_empty_detail(self):
        result = _router([]).route("Tell me about item 3")
        self.assertEqual(result["failed_modules"], [])
        self.assertEqual(result["failed_modules_detail"], [])

    def test_failed_modules_stays_a_list_of_names(self):
        result = _router([], KNOWLEDGE_GRAPH=NS(rag_query=_boom())).route("Tell me about item 4")
        self.assertEqual(result["failed_modules"], ["kg"])

    def test_detail_keeps_the_first_failure_of_each_module(self):
        # Acrescentado depois do 1º mutation run: `rv_detail_duplicates` sobreviveu porque nenhum plano
        # real executa o mesmo módulo duas vezes; o teste chama a regra diretamente.
        from core.constitutional_brain import CentralRouter, ModuleResult

        results = [
            ModuleResult(ModuleType.MEMORY, False, None, 0.0, "first"),
            ModuleResult(ModuleType.KNOWLEDGE_GRAPH, True, {}, 0.0),
            ModuleResult(ModuleType.MEMORY, False, None, 0.0, "second"),
        ]
        self.assertEqual(
            CentralRouter._failed_module_details(results), [{"module": "memory", "error": "first"}]
        )

    def test_think_propagates_the_detail(self):
        self.brain.router.modules[ModuleType.KNOWLEDGE_GRAPH] = NS(rag_query=_boom("kg offline"))
        result = self.think("Tell me about item 5")
        self.assertEqual([item["module"] for item in result["failed_modules_detail"]], ["kg"])
        self.assertIn("kg offline", result["failed_modules_detail"][0]["error"])
        self.brain.router.modules[ModuleType.BUDGET] = NS(request_budget=lambda *a, **k: False)
        blocked = self.think("Tell me about item 6")
        self.assertEqual(blocked["failed_modules_detail"][0]["module"], "budget")


class RT14ContractsStillHold(unittest.TestCase):
    def test_existing_decisions_are_unchanged(self):
        # Errata E2 (RT-v3): o orçamento que levanta passou a bloquear (decisão do Pedro, 30/10/2026).
        # Swarm: prompt sem red flag do P4 e sem gatilho ético continua só degradando.
        raising = _router([], BUDGET=NS(request_budget=_boom())).route("Tell me about item 8")
        self.assertIs(raising["success"], False)
        self.assertEqual(raising["error"], "BUDGET_CHECK_FAILED")
        self.assertEqual(raising["failed_modules"], ["budget"])
        swarm = _router([], SWARM=NS(deliberate=_boom())).route("Tell me about item 9")
        self.assertIs(swarm["success"], True)
        self.assertEqual(swarm["failed_modules"], ["swarm"])
        self.assertEqual(_router([]).analyze_request("Tell me about item 9"), RequestType.QUERY)


if __name__ == "__main__":
    unittest.main()
