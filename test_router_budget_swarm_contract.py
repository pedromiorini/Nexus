"""BS1–BS9: orçamento fail-closed e Swarm obrigatório além de ETHICAL_DECISION no CentralRouter.

Contrato pré-registrado em docs/cognitivo/AUDITORIA_2026-10-22.md (seção «orçamento fail-closed e
Swarm obrigatório», protocolo RT-v3, errata E2 e E3). Seeds fixas random.Random(20261101 + k),
k = 0..19; módulos falsos escritos aqui.
"""
import inspect
import random
import unittest
from types import SimpleNamespace as NS

from core.constitutional_brain import (
    CentralRouter,
    EthicsGuard,
    ExecutionPlan,
    ModuleType,
    RealCognitiveBudgetEnforcer,
    RequestType,
    ResourceType,
)
from test_router_fail_closed_contract import _boom, _Guard, _modules, _query
from test_router_review_addendum_contract import _BrainCase, _Recorder

SEEDS = [20261101 + k for k in range(20)]
RED_FLAGS = ["harm", "manipulate", "deceive", "exploit", "discriminate", "violate", "bypass", "ignore previous"]
NON_ETHICAL_FLAGS = [flag for flag in RED_FLAGS if flag != "harm"]


class _Ethics:
    """Guard falso sem requires_swarm (como os dublês dos contratos anteriores)."""

    def pre_execution_check(self, prompt, plan):
        return {"cleared": True, "modified_plan": plan}

    def post_execution_validation(self, prompt, results):
        return {"validated": True}


class _EthicsWith(_Ethics):
    def __init__(self, requires_swarm):
        self.requires_swarm = requires_swarm


class _PlanEthics(_Ethics):
    """Guard falso que devolve um plano modificado (estágios escolhidos pelo teste)."""

    def __init__(self, stages):
        self.stages = stages

    def pre_execution_check(self, prompt, plan):
        modified = ExecutionPlan(
            request_type=plan.request_type,
            modules_required=list(plan.modules_required),
            execution_stages=[list(stage) for stage in self.stages],
            estimated_cost=plan.estimated_cost,
            cache_key=plan.cache_key,
        )
        return {"cleared": True, "modified_plan": modified}


def _build(calls, ethics=None, **overrides):
    modules = _modules(calls)
    for key, value in overrides.items():
        module = ModuleType[key]
        if value is None:
            modules.pop(module, None)
        else:
            modules[module] = value
    router = CentralRouter(modules, ethics if ethics is not None else _Ethics())
    router.vram_guard = _Guard()
    return router


def _real_guard():
    log = NS(log_event=lambda *args, **kwargs: "hash")
    jailbreak = NS(detect=lambda prompt: NS(detected=False))
    return EthicsGuard(log, jailbreak)


def _flagged(rng):
    return f"Please {rng.choice(NON_ETHICAL_FLAGS)} item {rng.randrange(10**6)}"


def _blocked(test, result, error, failed, calls=None, allowed_calls=None):
    test.assertIs(result["success"], False)
    test.assertEqual(result["error"], error)
    test.assertIs(result["degraded"], True)
    test.assertEqual(result["failed_modules"], failed)
    for key in ("reason", "prompt", "failed_modules_detail", "request_type"):
        test.assertIn(key, result)
    if calls is not None:
        test.assertLessEqual(set(calls), set(allowed_calls), calls)


class BS1BudgetRaising(unittest.TestCase):
    def test_raising_budget_blocks_logs_once_and_is_not_cached(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                text = f"budget-down-{rng.randrange(10**6)}"
                runs, calls = [], []

                def failing(*args, **kwargs):
                    runs.append(1)
                    raise RuntimeError(text)

                log = _Recorder()
                router = _build(calls, BUDGET=NS(request_budget=failing), LOG=log)
                prompt = _query(rng)
                first = router.route(prompt)
                _blocked(self, first, "BUDGET_CHECK_FAILED", ["budget"], calls, {"jailbreak"})
                self.assertIn(text, first["reason"])
                self.assertEqual(len(log.blocked()), 1)
                self.assertEqual(log.blocked()[0][1]["error"], "BUDGET_CHECK_FAILED")
                second = router.route(prompt)
                self.assertIs(second["success"], False)
                self.assertNotIn("cached", second)
                self.assertEqual(len(runs), 2)
                self.assertEqual(router.stats["blocked_by_error"]["BUDGET_CHECK_FAILED"], 2)


class BS2MissingOrNoVerdict(unittest.TestCase):
    def test_missing_module_blocks(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                calls = []
                result = _build(calls, BUDGET=None).route(_query(random.Random(seed)))
                _blocked(self, result, "BUDGET_CHECK_FAILED", ["budget"], calls, {"jailbreak"})

    def test_non_boolean_verdicts_block(self):
        verdicts = [None, 0, 1, "yes", []]
        for k, seed in enumerate(SEEDS):
            verdict = verdicts[k % len(verdicts)]
            with self.subTest(seed=seed, verdict=verdict):
                calls = []
                router = _build(calls, BUDGET=NS(request_budget=lambda *a, v=verdict, **kw: v))
                result = router.route(_query(random.Random(seed)))
                _blocked(self, result, "BUDGET_CHECK_FAILED", ["budget"], calls, {"jailbreak"})

    def test_false_is_still_denied_and_true_passes(self):
        denied = _build([], BUDGET=NS(request_budget=lambda *a, **kw: False)).route("Tell me about item 4")
        _blocked(self, denied, "BUDGET_DENIED", ["budget"])
        clean = _build([], BUDGET=NS(request_budget=lambda *a, **kw: True)).route("Tell me about item 4")
        self.assertIs(clean["success"], True)
        self.assertIs(clean["degraded"], False)
        self.assertEqual(clean["failed_modules"], [])


class BS3NoExemption(unittest.TestCase):
    PROMPTS = ["", "   "]
    CONTEXTS = [None, {}, {"budget_enabled": False, "disable_budget": True, "budget": "disabled"}]

    def test_absent_or_raising_budget_blocks_for_empty_input_and_any_context(self):
        for seed in SEEDS:
            rng = random.Random(seed)
            prompt = self.PROMPTS[rng.randrange(2)] if seed % 2 == 0 else _query(rng)
            context = self.CONTEXTS[seed % 3]
            absent = seed % 4 < 2
            with self.subTest(seed=seed, prompt=prompt, context=context, absent=absent):
                calls = []
                budget = None if absent else NS(request_budget=_boom())
                result = _build(calls, BUDGET=budget).route(prompt, context)
                _blocked(self, result, "BUDGET_CHECK_FAILED", ["budget"], calls, {"jailbreak"})

    def test_empty_prompt_with_working_budget_behaves_as_before(self):
        for prompt in self.PROMPTS:
            with self.subTest(prompt=prompt):
                result = _build([]).route(prompt)
                self.assertIs(result["success"], True)
                self.assertIs(result["degraded"], False)


class BS4SwarmOnRedFlags(unittest.TestCase):
    def test_swarm_failure_blocks_with_p4_red_flag(self):
        for k, seed in enumerate(SEEDS):
            with self.subTest(seed=seed):
                prompt = _flagged(random.Random(seed))
                router = _build([], ethics=_real_guard(), SWARM=None if k % 2 else NS(deliberate=_boom()))
                self.assertNotEqual(router.analyze_request(prompt), RequestType.ETHICAL_DECISION)
                result = router.route(prompt)
                _blocked(self, result, "MANDATORY_MODULE_FAILED", ["swarm"])

    def test_healthy_swarm_runs_and_succeeds(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                calls = []
                result = _build(calls, ethics=_real_guard()).route(_flagged(random.Random(seed)))
                self.assertIs(result["success"], True)
                self.assertIs(result["degraded"], False)
                self.assertIn("swarm", calls)

    def test_prompt_without_flag_only_degrades(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                prompt = _query(random.Random(seed))
                result = _build([], ethics=_real_guard(), SWARM=NS(deliberate=_boom())).route(prompt)
                self.assertIs(result["success"], True)
                self.assertIs(result["degraded"], True)
                self.assertEqual(result["failed_modules"], ["swarm"])


class BS5GuardCompatibility(unittest.TestCase):
    def _route(self, ethics, prompt):
        return _build([], ethics=ethics, SWARM=NS(deliberate=_boom())).route(prompt)

    def test_guard_without_hook_keeps_ethical_only(self):
        result = self._route(_Ethics(), "Please deceive item 7")
        self.assertIs(result["success"], True)
        self.assertEqual(result["failed_modules"], ["swarm"])

    def test_raising_hook_is_treated_as_mandatory(self):
        result = self._route(_EthicsWith(_boom()), "Tell me about item 7")
        _blocked(self, result, "MANDATORY_MODULE_FAILED", ["swarm"])

    def test_ethical_decision_is_a_floor(self):
        result = self._route(_EthicsWith(lambda prompt: False), "Is this moral case 7?")
        _blocked(self, result, "MANDATORY_MODULE_FAILED", ["swarm"])

    def test_hook_true_makes_a_query_mandatory(self):
        result = self._route(_EthicsWith(lambda prompt: True), "Tell me about item 7")
        _blocked(self, result, "MANDATORY_MODULE_FAILED", ["swarm"])

    def test_real_requires_swarm(self):
        guard = _real_guard()
        self.assertEqual(
            guard.ethical_red_flags,
            ["harm", "manipulate", "deceive", "exploit", "discriminate", "violate", "bypass", "ignore previous"],
        )
        for flag in RED_FLAGS:
            with self.subTest(flag=flag):
                self.assertIs(guard.requires_swarm(f"x {flag} y"), True)
                self.assertIs(guard.requires_swarm(f"x {flag.upper()} y"), True)
        for text in ("", "   ", "Tell me about item 7"):
            with self.subTest(text=text):
                self.assertIs(guard.requires_swarm(text), False)


class BS6ModifiedPlan(unittest.TestCase):
    def test_plan_without_budget_or_without_mandatory_swarm_blocks(self):
        for k, seed in enumerate(SEEDS):
            rng = random.Random(seed)
            with self.subTest(seed=seed):
                if k % 2 == 0:
                    ethics = _PlanEthics([[ModuleType.JAILBREAK], [ModuleType.LOG]])
                    result = _build([], ethics=ethics).route(_query(rng))
                    _blocked(self, result, "BUDGET_CHECK_FAILED", ["budget"])
                else:
                    ethics = _PlanEthics([[ModuleType.JAILBREAK, ModuleType.BUDGET], [ModuleType.LOG]])
                    result = _build([], ethics=ethics).route(f"Is this moral case {rng.randrange(10**6)}?")
                    _blocked(self, result, "MANDATORY_MODULE_FAILED", ["swarm"])

    def test_unmodified_plan_still_passes(self):
        for seed in SEEDS[:5]:
            result = _build([]).route(_query(random.Random(seed)))
            self.assertIs(result["success"], True)


class BS7Integrated(_BrainCase):
    def test_real_enforcer_at_the_limit_is_denied_and_reset_passes(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                enforcer = RealCognitiveBudgetEnforcer()
                router = _build([], BUDGET=enforcer)
                prompt = _query(random.Random(seed))
                self.assertIs(router.route(prompt)["success"], True)
                enforcer.usage[ResourceType.REASONING_CYCLES] = enforcer.limits[ResourceType.REASONING_CYCLES]
                _blocked(self, router.route(prompt + " again"), "BUDGET_DENIED", ["budget"])
                enforcer.start_session()
                self.assertIs(router.route(prompt + " third")["success"], True)

    def test_think_surfaces_a_failing_router_budget(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                self.reset()
                self.brain.router.modules[ModuleType.BUDGET] = NS(request_budget=_boom("router-budget-down"))
                before = self.brain.router.get_statistics()["blocked_by_error"].get("BUDGET_CHECK_FAILED", 0)
                result = self.think(_query(random.Random(seed)))
                self.assertEqual(result.get("error"), "BUDGET_CHECK_FAILED")
                self.assertIs(result.get("degraded"), True)
                self.assertEqual(result.get("failed_modules"), ["budget"])
                after = self.brain.router.get_statistics()["blocked_by_error"].get("BUDGET_CHECK_FAILED", 0)
                self.assertEqual(after, before + 1)

    def test_think_with_intact_budget_is_not_blocked(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                self.reset()
                result = self.think(_query(random.Random(seed)))
                self.assertFalse(result.get("error"))
                self.assertNotEqual(result.get("success"), False)


class BS8NoNewSurface(unittest.TestCase):
    def test_no_budget_opt_out_exists(self):
        for name in ("enabled", "disabled", "budget_enabled", "budget_disabled"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(RealCognitiveBudgetEnforcer(), name))
                self.assertFalse(hasattr(_build([]), name))

    def test_requires_swarm_is_the_only_new_public_guard_method(self):
        public = {name for name, _ in inspect.getmembers(EthicsGuard, inspect.isfunction) if not name.startswith("_")}
        self.assertEqual(
            public,
            {"review_action_plan", "pre_execution_check", "post_execution_validation", "get_statistics", "requires_swarm"},
        )
        self.assertIn("requires_swarm", public)


class BS9Contract(unittest.TestCase):
    def test_signature_and_block_keys(self):
        params = list(inspect.signature(CentralRouter._aggregate_results).parameters)
        self.assertEqual(params, ["self", "results", "request_type", "context"])
        result = _build([], BUDGET=NS(request_budget=_boom())).route("x" * 300)
        _blocked(self, result, "BUDGET_CHECK_FAILED", ["budget"])
        self.assertEqual(len(result["prompt"]), 100)
        self.assertEqual(result["request_type"], RequestType.QUERY.value)


if __name__ == "__main__":
    unittest.main()
