import unittest
from collections import defaultdict

from core.constitutional_brain import CentralRouter, ModuleType
from core.deferred_task_queue import DeferredTaskQueue


class CentralRouterContractTests(unittest.TestCase):
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
