"""Contrato: pressure_critical só é bool exato.

Na base medida, recheck_and_pop e consume tratam o flag por verdade
Python. 1, NaN, infinito e a string "false" bloqueiam a fila; 0, None
e recipientes vazios deixam a tarefa sair. Bool True/False permanece.
"""
import math
import unittest

from core.deferred_task_queue import DeferredTaskQueue


class _Truthy:
    def __bool__(self):
        return True


class _Falsy:
    def __bool__(self):
        return False


class _Int(int):
    pass


def _snapshot(queue):
    payload = queue.export_snapshot()
    return (
        queue.statistics(),
        payload["tasks"],
        payload["statistics"],
    )


class DeferredPressureFlagContractTests(unittest.TestCase):
    def test_exact_bool_keeps_block_and_release(self):
        blocked = DeferredTaskQueue()
        blocked.enqueue("stay", priority=3)
        before = _snapshot(blocked)
        self.assertIsNone(blocked.recheck_and_pop(True))
        result = blocked.consume(lambda task: True, pressure_critical=True)
        self.assertEqual(result["blocked"], True)
        self.assertEqual(result["processed"], 0)
        self.assertEqual(_snapshot(blocked), before)

        released = DeferredTaskQueue()
        released.enqueue("go", priority=3)
        seen = []
        popped = released.recheck_and_pop(False)
        self.assertEqual(popped.prompt, "go")
        self.assertEqual(popped.attempts, 1)
        released.enqueue("next", priority=1)
        result = released.consume(
            lambda task: seen.append(task.prompt) or True,
            pressure_critical=False,
        )
        self.assertEqual(seen, ["next"])
        self.assertEqual(result["blocked"], False)
        self.assertEqual(result["completed"], 1)

    def test_non_bool_flag_does_not_pop_or_block_or_run(self):
        invalid = (
            1,
            0,
            1.5,
            0.0,
            "false",
            "true",
            "",
            None,
            [],
            {},
            (),
            b"",
            float("nan"),
            float("inf"),
            float("-inf"),
            math.nan,
            _Truthy(),
            _Falsy(),
            _Int(1),
            _Int(0),
        )
        for flag in invalid:
            with self.subTest(flag=flag):
                queue = DeferredTaskQueue()
                queue.enqueue("pending", priority=4, reason="keep")
                before = _snapshot(queue)
                calls = []

                with self.assertRaisesRegex(TypeError, "pressure_critical deve ser bool"):
                    queue.recheck_and_pop(flag)
                self.assertEqual(_snapshot(queue), before)

                with self.assertRaisesRegex(TypeError, "pressure_critical deve ser bool"):
                    queue.consume(
                        lambda task: calls.append(task.prompt) or True,
                        pressure_critical=flag,
                    )
                self.assertEqual(calls, [])
                self.assertEqual(_snapshot(queue), before)
                self.assertEqual(queue.export_snapshot()["tasks"][0]["prompt"], "pending")

    def test_invalid_flag_is_rejected_before_a_bad_batch_is_not_required(self):
        queue = DeferredTaskQueue()
        queue.enqueue("pending")
        before = _snapshot(queue)
        with self.assertRaisesRegex(ValueError, "max_batch"):
            queue.consume(lambda task: True, max_batch=True, pressure_critical=1)
        self.assertEqual(_snapshot(queue), before)

    def test_blocking_bool_does_not_call_the_processor(self):
        queue = DeferredTaskQueue()
        queue.enqueue("pending")
        calls = []
        result = queue.consume(
            lambda task: calls.append(task.prompt) or True,
            max_batch=3,
            pressure_critical=True,
        )
        self.assertEqual(calls, [])
        self.assertEqual(result["blocked"], True)
        self.assertEqual(queue.statistics()["depth"], 1)


if __name__ == "__main__":
    unittest.main()
