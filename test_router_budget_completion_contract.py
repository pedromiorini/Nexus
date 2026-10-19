"""Contratos operacionais A04/A05; doubles não avaliam capacidade cognitiva."""
import concurrent.futures
import threading
import unittest
from types import SimpleNamespace

from core.constitutional_brain import (
    CentralRouter, EthicsGuard, ModuleType, RealCognitiveBudgetEnforcer,
    RealJailbreakDetector, RequestType, ResourceType,
)


class Reasoner:
    def __init__(self, result=None, fail=False):
        self.calls = 0
        self.result = {"answer": "ok", "confidence": 0.7} if result is None else result
        self.fail = fail

    def reason(self, _prompt):
        self.calls += 1
        if self.fail:
            raise RuntimeError("private failure detail")
        return self.result


def make_router(reasoner=None, budget=None):
    detector = RealJailbreakDetector()
    log = SimpleNamespace(log_event=lambda *args: "test-hash")
    budget = RealCognitiveBudgetEnforcer() if budget is None else budget
    reasoner = Reasoner() if reasoner is None else reasoner
    router = CentralRouter({
        ModuleType.LOG: log, ModuleType.JAILBREAK: detector,
        ModuleType.BUDGET: budget, ModuleType.REASONING: reasoner,
    }, EthicsGuard(log, detector))
    router.vram_guard = SimpleNamespace(
        evaluate=lambda: SimpleNamespace(action="normal"),
        execute_mitigation=lambda _decision: SimpleNamespace(applied=False),
        mitigation_plan=lambda _decision: {},
    )
    router.routing_rules[RequestType.QUERY] = [ModuleType.REASONING]
    return router, budget, reasoner


class RouterBudgetCompletionTests(unittest.TestCase):
    def test_denied_budget_stops_before_work_and_never_caches(self):
        router, budget, reasoner = make_router()
        budget.consume(ResourceType.REASONING_CYCLES, 100)
        work = []
        router.modules[ModuleType.MEMORY] = SimpleNamespace(
            retrieve=lambda *args, **kwargs: work.append(True))
        router.routing_rules[RequestType.QUERY].insert(0, ModuleType.MEMORY)
        for _ in range(2):
            result = router.route("bounded computation")
            self.assertIs(result["success"], False)
        self.assertEqual(reasoner.calls, 0)
        self.assertEqual(work, [])
        self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 100)
        self.assertEqual(router.execution_cache, {})

    def test_budget_missing_error_and_non_boolean_permission_fail_closed(self):
        for bad in (None, SimpleNamespace(request_budget=lambda *args: 1),
                    SimpleNamespace(request_budget=lambda *args: {"allowed": False}),
                    SimpleNamespace(request_budget=lambda *args: (_ for _ in ()).throw(RuntimeError()))):
            with self.subTest(budget=bad):
                router, _, reasoner = make_router()
                if bad is None:
                    router.modules.pop(ModuleType.BUDGET)
                else:
                    router.modules[ModuleType.BUDGET] = bad
                result = router.route("bounded computation")
                self.assertIs(result["success"], False)
                self.assertEqual(reasoner.calls, 0)
                self.assertEqual(router.execution_cache, {})

    def test_success_consumes_one_and_cache_hit_consumes_nothing(self):
        router, budget, reasoner = make_router()
        first = router.route("bounded computation")
        self.assertIs(first["success"], True)
        self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 1)
        budget.consume(ResourceType.REASONING_CYCLES, 99)
        second = router.route("bounded computation")
        self.assertIs(second["cached"], True)
        self.assertEqual(reasoner.calls, 1)
        self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 100)

    def test_failed_attempt_is_charged_retried_and_not_cached(self):
        router, budget, reasoner = make_router(Reasoner(fail=True))
        for _ in range(2):
            result = router.route("bounded computation")
            self.assertIs(result["success"], False)
        self.assertEqual(reasoner.calls, 2)
        self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 2)
        self.assertEqual(router.execution_cache, {})
        self.assertEqual(router.stats["module_errors"][ModuleType.REASONING], 2)

    def test_missing_reasoner_never_consumes_or_succeeds(self):
        router, budget, _ = make_router()
        router.modules.pop(ModuleType.REASONING)
        self.assertIs(router.route("bounded computation")["success"], False)
        self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 0)
        self.assertEqual(router.execution_cache, {})

    def test_failed_reasoner_is_retained_then_completes_after_recovery(self):
        router, budget, reasoner = make_router(Reasoner(fail=True))
        task = router.deferred_queue.enqueue("bounded computation", {})
        first = router.reprocess_deferred_tasks()
        self.assertEqual(first["completed"], 0)
        self.assertEqual(first["retried"], 1)
        self.assertEqual(router.deferred_queue.statistics()["depth"], 1)
        self.assertNotIn("private", task.last_reason)
        self.assertEqual(router.stats["deferred_reprocessing"]["failed"], 1)
        reasoner.fail = False
        second = router.reprocess_deferred_tasks()
        self.assertEqual(second["completed"], 1)
        self.assertEqual(router.deferred_queue.statistics()["depth"], 0)
        self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 2)

    def test_structured_failure_is_not_operational_success(self):
        for output in ({"success": False, "answer": "not completed"},
                       {"success": 1, "answer": "not verified"},
                       {}, {"error": "unavailable"}, {"answer": None},
                       {"answer": ""}, {"answer": "   "}, []):
            with self.subTest(output=output):
                router, budget, reasoner = make_router(Reasoner(output))
                self.assertIs(router.route("bounded computation")["success"], False)
                self.assertEqual(reasoner.calls, 1)
                self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 1)
                self.assertEqual(router.execution_cache, {})

    def test_reservation_happens_after_context_copy_and_at_reasoner_entry(self):
        router, budget, reasoner = make_router()
        usages = []
        original = reasoner.reason
        reasoner.reason = lambda prompt: (usages.append(budget.get_status()["usage"]["reasoning_cycles"]), original(prompt))[1]
        self.assertIs(router.route("bounded computation")["success"], True)
        self.assertEqual(usages, [1])

    def test_last_unit_is_atomic_across_independent_routers(self):
        budget = RealCognitiveBudgetEnforcer()
        budget.consume(ResourceType.REASONING_CYCLES, 99)
        routers = [make_router(budget=budget) for _ in range(12)]
        barrier = threading.Barrier(len(routers))
        def route(item):
            barrier.wait(timeout=10)
            return item[0].route("bounded computation")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(routers)) as pool:
            results = list(pool.map(route, routers))
        self.assertEqual(sum(result["success"] for result in results), 1)
        self.assertEqual(sum(item[2].calls for item in routers), 1)
        self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 100)

    def test_permission_can_be_revoked_between_preflight_and_reasoning(self):
        router, budget, reasoner = make_router()
        def retrieve(*args, **kwargs):
            budget.consume(ResourceType.REASONING_CYCLES, 100)
            return []
        router.modules[ModuleType.MEMORY] = SimpleNamespace(retrieve=retrieve)
        router.routing_rules[RequestType.QUERY].insert(0, ModuleType.MEMORY)
        self.assertIs(router.route("bounded computation")["success"], False)
        self.assertEqual(reasoner.calls, 0)
        self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 100)

    def test_non_reasoning_success_is_not_charged_and_empty_recall_is_valid(self):
        router, budget, _ = make_router()
        router.analyze_request = lambda *args: RequestType.MEMORY_RECALL
        router.routing_rules[RequestType.MEMORY_RECALL] = [ModuleType.MEMORY]
        router.modules[ModuleType.MEMORY] = SimpleNamespace(retrieve=lambda *args, **kwargs: [])
        self.assertIs(router.route("retrieve stored item")["success"], True)
        self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 0)

    def test_required_modules_by_type_cannot_be_removed_by_routing_rules(self):
        requirements = {
            RequestType.QUERY: [ModuleType.REASONING],
            RequestType.PLANNING: [ModuleType.MCTS_PLANNER],
            RequestType.MEMORY_RECALL: [ModuleType.MEMORY],
            RequestType.LEARNING: [ModuleType.INCREMENTAL_LEARNER],
            RequestType.CREATIVE: [ModuleType.WORLD_MODEL],
            RequestType.ANALYSIS: [ModuleType.CAUSAL, ModuleType.REASONING],
            RequestType.ETHICAL_DECISION: [ModuleType.REASONING, ModuleType.SWARM],
        }
        for request_type, required in requirements.items():
            for missing in required:
                with self.subTest(request_type=request_type, missing=missing):
                    router, _, _ = make_router()
                    router.analyze_request = lambda *args: request_type
                    router.routing_rules[request_type] = [m for m in required if m != missing]
                    router._invoke_module = lambda mod, *args: (
                        {"allowed": True} if mod == ModuleType.BUDGET else {"answer": "ok"})
                    result = router.route("bounded computation")
                    self.assertIs(result["success"], False)
                    self.assertIn(missing.value, result["failed_modules"])
                    self.assertEqual(router.execution_cache, {})

    def test_non_answer_request_types_complete_with_their_own_product(self):
        fixtures = {
            RequestType.PLANNING: (ModuleType.MCTS_PLANNER, "plan", {"plan": []}),
            RequestType.MEMORY_RECALL: (ModuleType.MEMORY, "retrieve", []),
            RequestType.CREATIVE: (ModuleType.WORLD_MODEL, "compare_futures", {"results": []}),
            RequestType.LEARNING: (ModuleType.INCREMENTAL_LEARNER, None, None),
        }
        for request_type, (module, method, output) in fixtures.items():
            with self.subTest(request_type=request_type):
                router, budget, _ = make_router()
                router.analyze_request = lambda *args: request_type
                router.routing_rules[request_type] = [module]
                router.modules[module] = (SimpleNamespace(**{method: lambda *args, **kwargs: output})
                                          if method else object())
                result = router.route("bounded computation")
                self.assertIs(result["success"], True)
                self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 0)
                self.assertIs(router.route("bounded computation")["cached"], True)

    def test_optional_failure_does_not_override_required_success(self):
        router, _, _ = make_router()
        router.routing_rules[RequestType.QUERY].append(ModuleType.KNOWLEDGE_GRAPH)
        self.assertIs(router.route("bounded computation")["success"], True)
        self.assertEqual(router.stats["module_errors"][ModuleType.KNOWLEDGE_GRAPH], 1)

    def test_missing_atomic_api_never_falls_back_to_check_then_consume(self):
        consumed = []
        budget = SimpleNamespace(request_budget=lambda *args: True,
                                 consume=lambda *args: consumed.append(True))
        router, _, reasoner = make_router(budget=budget)
        self.assertIs(router.route("bounded computation")["success"], False)
        self.assertEqual(reasoner.calls, 0)
        self.assertEqual(consumed, [])

    def test_unusable_reasoner_interface_does_not_consume(self):
        for reasoner in (object(), SimpleNamespace(reason=None)):
            with self.subTest(reasoner=reasoner):
                router, budget, _ = make_router(reasoner)
                self.assertIs(router.route("bounded computation")["success"], False)
                self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 0)

    def test_context_copy_failure_at_reasoning_does_not_consume(self):
        class CannotCopy:
            def __deepcopy__(self, memo):
                raise ValueError("not copyable")
        router, budget, reasoner = make_router()
        router.modules[ModuleType.MEMORY] = SimpleNamespace(retrieve=lambda *args, **kwargs: CannotCopy())
        router.routing_rules[RequestType.QUERY].insert(0, ModuleType.MEMORY)
        self.assertIs(router.route("bounded computation")["success"], False)
        self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 0)
        self.assertEqual(reasoner.calls, 0)

    def test_ethics_added_swarm_is_required_even_for_query(self):
        router, budget, reasoner = make_router()
        result = router.route("bounded exploit description")
        self.assertIs(result["success"], False)
        self.assertIn("swarm", result["failed_modules"])
        self.assertEqual(reasoner.calls, 1)
        self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 1)
        self.assertEqual(router.execution_cache, {})

    def test_none_from_required_module_is_not_completion(self):
        router, _, reasoner = make_router()
        reasoner.reason = lambda prompt: None
        self.assertIs(router.route("bounded computation")["success"], False)
        self.assertEqual(router.execution_cache, {})

    def test_absent_or_failed_recall_result_is_not_completion(self):
        for output in (None, {"success": False}, {"success": 0}):
            with self.subTest(output=output):
                router, budget, _ = make_router()
                router.analyze_request = lambda *args: RequestType.MEMORY_RECALL
                router.routing_rules[RequestType.MEMORY_RECALL] = [ModuleType.MEMORY]
                router.modules[ModuleType.MEMORY] = SimpleNamespace(
                    retrieve=lambda *args, **kwargs: output)
                self.assertIs(router.route("bounded computation")["success"], False)
                self.assertEqual(router.execution_cache, {})
                self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 0)

    def test_required_non_reasoner_error_payload_is_not_completion(self):
        fixtures = {
            RequestType.PLANNING: (ModuleType.MCTS_PLANNER, "plan"),
            RequestType.MEMORY_RECALL: (ModuleType.MEMORY, "retrieve"),
            RequestType.CREATIVE: (ModuleType.WORLD_MODEL, "compare_futures"),
            RequestType.ANALYSIS: (ModuleType.CAUSAL, "enrich_context"),
            RequestType.ETHICAL_DECISION: (ModuleType.SWARM, "deliberate"),
        }
        for request_type, (module, method) in fixtures.items():
            for output in ({"error": "unavailable"},
                           {"success": True, "error": "unavailable"},
                           {"error": ""}):
                with self.subTest(request_type=request_type, output=output):
                    router, _, _ = make_router()
                    router.analyze_request = lambda *args: request_type
                    router.routing_rules[request_type] = [module, ModuleType.REASONING]
                    router.modules[module] = SimpleNamespace(**{
                        method: lambda *args, **kwargs: output})
                    for _ in range(2):
                        self.assertIs(router.route("bounded computation")["success"], False)
                    self.assertEqual(router.execution_cache, {})

    def test_ethics_denial_does_not_consume(self):
        router, budget, reasoner = make_router()
        router.ethics.pre_execution_check = lambda *args: {"cleared": False, "reason": "denied"}
        self.assertIs(router.route("bounded computation")["success"], False)
        self.assertEqual(budget.get_status()["usage"]["reasoning_cycles"], 0)
        self.assertEqual(reasoner.calls, 0)


class AtomicBudgetTests(unittest.TestCase):
    def test_reserve_boundaries_and_non_mutating_denial(self):
        budget = RealCognitiveBudgetEnforcer()
        resource = ResourceType.REASONING_CYCLES
        self.assertIs(budget.request_budget(resource, 100), True)
        self.assertEqual(budget.get_status()["usage"][resource.value], 0)
        self.assertIs(budget.try_consume(resource, 100), True)
        self.assertIs(budget.try_consume(resource, 1), False)
        self.assertEqual(budget.get_status()["usage"][resource.value], 100)
        budget.start_session()
        self.assertEqual(budget.get_status()["usage"][resource.value], 0)

    def test_invalid_amounts_fail_before_mutation(self):
        resource = ResourceType.REASONING_CYCLES
        budget = RealCognitiveBudgetEnforcer()
        for method in (budget.request_budget, budget.consume, budget.try_consume):
            for amount in (True, False, "1", None, float("nan"), float("inf"), -1):
                with self.subTest(method=method.__name__, amount=amount):
                    with self.assertRaises((TypeError, ValueError)):
                        method(resource, amount)
                    self.assertEqual(budget.get_status()["usage"][resource.value], 0)


if __name__ == "__main__":
    unittest.main()
