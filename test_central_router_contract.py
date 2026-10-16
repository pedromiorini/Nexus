import unittest
from collections import defaultdict
from types import SimpleNamespace

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
            "observed_context": dict(context)
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

        router.route("plan without explicit context")
        cached = router.route("plan without explicit context")
        self.assertTrue(cached["cached"])

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
