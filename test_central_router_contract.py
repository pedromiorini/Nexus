import io
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from core.constitutional_brain import CentralRouter, ModuleType, RequestType
from core.deferred_task_queue import DeferredTaskQueue


class CentralRouterContractTests(unittest.TestCase):
    def test_route_forwards_caller_context_and_bypasses_prompt_only_cache(self):
        class Ethics:
            def pre_execution_check(self, prompt, plan):
                return {"cleared": True, "modified_plan": plan}

            def post_execution_validation(self, prompt, results):
                return {"validated": True}

        class Guard:
            def evaluate(self):
                return SimpleNamespace(action="normal")

            def execute_mitigation(self, decision):
                return SimpleNamespace()

            def mitigation_plan(self, decision):
                return {}

        class WorldModel:
            def __init__(self):
                self.actions = []

            def compare_futures(self, actions):
                self.actions.append(actions)
                return {"actions": actions}

        world_model = WorldModel()
        modules = {
            ModuleType.JAILBREAK: SimpleNamespace(
                detect=lambda prompt: SimpleNamespace(detected=False)
            ),
            ModuleType.BUDGET: SimpleNamespace(request_budget=lambda *args: True),
            ModuleType.LOG: SimpleNamespace(log_event=lambda *args: "test-hash"),
            ModuleType.WORLD_MODEL: world_model,
        }
        router = CentralRouter(modules, Ethics())
        router.vram_guard = Guard()
        router.routing_rules[RequestType.PLANNING] = [ModuleType.WORLD_MODEL]
        router._aggregate_results = lambda results, request_type, context: {
            "observed_context": dict(context),
            "nested_output": {"values": [1]},
        }

        first_actions = [["left"], ["right"]]
        second_actions = [["north"], ["south"]]
        first = router.route(
            "plan a route",
            {"possible_actions": first_actions, "original_prompt": "forged"},
        )
        second = router.route("plan a route", {"possible_actions": second_actions})

        self.assertEqual(world_model.actions, [first_actions, second_actions])
        self.assertFalse(second["cached"])
        self.assertEqual(first["observed_context"]["possible_actions"], first_actions)
        self.assertEqual(first["observed_context"]["original_prompt"], "plan a route")

        execute_stage = router._execute_stage

        def mutate_nested_context(modules, prompt, context):
            context["nested"]["value"] = 2
            return execute_stage(modules, prompt, context)

        router._execute_stage = mutate_nested_context
        caller_context = {"possible_actions": [["up"]], "nested": {"value": 1}}
        router.route("plan with nested context", caller_context)
        self.assertEqual(caller_context["nested"]["value"], 1)
        router._execute_stage = execute_stage

        uncached = router.route("plan without explicit context")
        uncached["nested_output"]["values"].append(2)
        cached = router.route("plan without explicit context")
        self.assertTrue(cached["cached"])
        self.assertEqual(cached["nested_output"]["values"], [1])
        cached["nested_output"]["values"].append(3)
        cached_again = router.route("plan without explicit context")
        self.assertEqual(cached_again["nested_output"]["values"], [1])

    def test_debug_logging_never_includes_user_prompt(self):
        prompt = "USER_PROMPT_SECRET: " + "x" * (50 - len("USER_PROMPT_SECRET: "))
        router = CentralRouter.__new__(CentralRouter)
        router.modules = {}
        router.stats = {"total_requests": 0}
        router.deferred_queue = DeferredTaskQueue()

        class Guard:
            def __init__(self):
                self.actions = iter(("emergency_release", "normal"))

            def evaluate(self):
                return SimpleNamespace(action=next(self.actions), reason="critical_vram")

            def execute_mitigation(self, _decision):
                return SimpleNamespace(applied=True)

            def mitigation_plan(self, _decision):
                return {}

        router.vram_guard = Guard()
        output = io.StringIO()

        with patch("core.constitutional_brain._NEXUS_DEBUG", True), redirect_stdout(output):
            router._route(prompt, _deferred_reprocess=True)

        logs = output.getvalue()
        self.assertIn("Router.route() CALLED", logs)
        self.assertNotIn(prompt, logs)

    def test_caller_context_cannot_suppress_emergency_deferral_enqueue(self):
        router = CentralRouter.__new__(CentralRouter)
        router.modules = {}
        router.stats = {"total_requests": 0}
        router.deferred_queue = DeferredTaskQueue()

        class Guard:
            def __init__(self):
                self.actions = iter(("emergency_release", "normal"))

            def evaluate(self):
                return SimpleNamespace(action=next(self.actions), reason="critical_vram")

            def execute_mitigation(self, _decision):
                return SimpleNamespace(applied=True)

            def mitigation_plan(self, _decision):
                return {}

        router.vram_guard = Guard()
        caller_context = {"_deferred_reprocess": True}

        result = router.route("work", caller_context)

        self.assertTrue(result["deferred"])
        self.assertTrue(result["deferred_task_id"])
        self.assertEqual(router.deferred_queue.statistics()["depth"], 1)
        self.assertEqual(caller_context, {"_deferred_reprocess": True})
        with self.assertRaises(TypeError):
            router.route("work", {}, _deferred_reprocess=True)

    def test_execute_stage_isolates_mutable_context_between_modules(self):
        router = CentralRouter.__new__(CentralRouter)
        router.modules = {
            ModuleType.JAILBREAK: object(),
            ModuleType.BUDGET: object(),
        }
        router.stats = {
            "module_errors": defaultdict(int),
            "module_call_count": defaultdict(int),
        }
        observed = []

        def mutate_context(_module_name, _module, _prompt, context):
            observed.append(context["nested"]["value"])
            context["nested"]["value"] += 1
            return {"ok": True}

        router._invoke_module = mutate_context
        original = {"nested": {"value": 1}}

        results = router._execute_stage(
            [ModuleType.JAILBREAK, ModuleType.BUDGET], "prompt", original
        )

        self.assertEqual(observed, [1, 1])
        self.assertEqual(original, {"nested": {"value": 1}})
        self.assertTrue(all(result.success for result in results))

    def test_statistics_expose_request_count_and_rates(self):
        router = CentralRouter.__new__(CentralRouter)
        router.stats = {
            "total_requests": 3,
            "cache_hits": 1,
            "total_latency_ms": 120.0,
            "requests_by_type": defaultdict(int),
            "module_call_count": {module: 0 for module in ModuleType},
            "module_errors": {module: 0 for module in ModuleType},
            "avg_modules_per_request": 2.0,
            "deferred_reprocessing": {
                "attempts": 0,
                "completed": 0,
                "failed": 0,
                "discarded": 0,
                "total_latency_ms": 0.0,
            },
        }
        router.deferred_queue = DeferredTaskQueue()

        statistics = router.get_statistics()

        self.assertEqual(statistics["total_requests"], 3)
        self.assertEqual(statistics["cache_hits"], 1)
        self.assertAlmostEqual(statistics["cache_hit_rate"], 1 / 3)
        self.assertAlmostEqual(statistics["avg_latency_ms"], 40.0)
        self.assertEqual(statistics["avg_modules_per_request"], 2.0)


if __name__ == "__main__":
    unittest.main()
