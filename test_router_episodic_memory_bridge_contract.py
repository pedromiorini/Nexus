"""C1–C15 da ligação da memória episódica.

Critérios aprovados em docs/cognitivo/PONTE_EPISODIC_MEMORY_CARACTERIZACAO_2026-11-12.md.
Este arquivo só observa o comportamento atual. Não altera produção.
"""
import contextlib
import inspect
import io
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import core.cognitivo as cognitivo_pkg
import core.cognitivo.episodic_memory as episodic_module
import core.constitutional_brain as brain
from core.constitutional_brain import (
    CentralRouter,
    CompleteNexusBrain,
    ConsolidationReport,
    EthicsGuard,
    ModuleType,
    RealEpisodicMemory,
)

ROOT = Path(__file__).resolve().parent
ZERO_EPISODIC = {
    "episodes_created": 0,
    "episodes_closed": 0,
    "total_links": 0,
    "avg_importance": 0.0,
}
EPISODIC_METHODS = (
    "search_episodes",
    "start_episode",
    "close_episode",
    "link_memory_to_episode",
    "get_episode_memories",
    "get_statistics",
)
MODULE_NAMES = frozenset({
    "BUDGET",
    "CAUSAL",
    "EPISODIC",
    "INCREMENTAL_LEARNER",
    "JAILBREAK",
    "KNOWLEDGE_GRAPH",
    "LOG",
    "MCTS_PLANNER",
    "MEMORY",
    "REASONING",
    "SWARM",
    "WORLD_MODEL",
})
BLOCKING_ERRORS = {
    "JAILBREAK_DETECTED",
    "JAILBREAK_CHECK_FAILED",
    "BUDGET_CHECK_FAILED",
    "BUDGET_DENIED",
    "MANDATORY_MODULE_FAILED",
}
RECALL = "remember the last time"


@contextlib.contextmanager
def silence():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def raised_frames(exc_type, fn):
    """Captura o traceback na hora. assertRaises zera __traceback__."""
    try:
        with silence():
            fn()
    except exc_type as exc:
        frames = [
            (Path(frame.filename).name, frame.lineno, frame.name)
            for frame in traceback.extract_tb(exc.__traceback__)
            if Path(frame.filename).name == "constitutional_brain.py"
        ]
        return exc, frames
    except Exception as exc:
        raise AssertionError(
            f"esperava {exc_type.__name__}, veio {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{exc_type.__name__} não foi levantado")


class EpisodicRecorder:
    """Registrador do harness H. search_episodes devolve [] salvo substituição."""

    def __init__(self, search_raises=None):
        self.calls = []
        self.search_raises = search_raises

    def search_episodes(self, *args, **kwargs):
        self.calls.append(("search_episodes", args, dict(kwargs)))
        if self.search_raises is not None:
            raise self.search_raises
        return []

    def start_episode(self, *args, **kwargs):
        self.calls.append(("start_episode", args, dict(kwargs)))
        return "ep-stub"

    def close_episode(self, *args, **kwargs):
        self.calls.append(("close_episode", args, dict(kwargs)))

    def link_memory_to_episode(self, *args, **kwargs):
        self.calls.append(("link_memory_to_episode", args, dict(kwargs)))

    def get_episode_memories(self, *args, **kwargs):
        self.calls.append(("get_episode_memories", args, dict(kwargs)))
        return []

    def get_statistics(self, *args, **kwargs):
        self.calls.append(("get_statistics", args, dict(kwargs)))
        return {}


class _ObserveGuard:
    def evaluate(self):
        return SimpleNamespace(action="observe")

    def execute_mitigation(self, decision):
        return SimpleNamespace()

    def mitigation_plan(self, decision):
        return {}


def ethics_guard(detected=False):
    log = SimpleNamespace(log_event=lambda *args, **kwargs: "hash")
    jailbreak = SimpleNamespace(
        detect=lambda prompt: SimpleNamespace(
            detected=detected, severity=0.0, attack_type=SimpleNamespace(value="none"),
        ),
    )
    with silence():
        return EthicsGuard(log, jailbreak)


def _record(name, value):
    def fn(*args, **kwargs):
        return value
    fn.__name__ = name
    return fn


def harness_modules(recorder, jailbreak_detected=False):
    detected = SimpleNamespace(detected=jailbreak_detected)
    return {
        ModuleType.JAILBREAK: SimpleNamespace(detect=_record("jailbreak", detected)),
        ModuleType.BUDGET: SimpleNamespace(request_budget=_record("budget", True)),
        ModuleType.LOG: SimpleNamespace(log_event=_record("log", "hash")),
        ModuleType.MEMORY: SimpleNamespace(retrieve=_record("memory", [])),
        ModuleType.KNOWLEDGE_GRAPH: SimpleNamespace(
            rag_query=_record("kg", {"graph_entities": [], "confidence": 0.0}),
        ),
        ModuleType.REASONING: SimpleNamespace(
            reason=_record("reasoning", {"answer": "ok", "confidence": 0.7}),
        ),
        ModuleType.SWARM: SimpleNamespace(
            deliberate=_record("swarm", SimpleNamespace(
                final_decision=SimpleNamespace(value="approve"), votes=[],
            )),
        ),
        ModuleType.CAUSAL: SimpleNamespace(enrich_context=_record("causal", {})),
        ModuleType.WORLD_MODEL: SimpleNamespace(compare_futures=_record("world", {})),
        ModuleType.MCTS_PLANNER: SimpleNamespace(plan=_record("mcts", {})),
        ModuleType.INCREMENTAL_LEARNER: SimpleNamespace(),
        ModuleType.EPISODIC: recorder,
    }


def harness_h(recorder=None, *, include_episodic=True, jailbreak_detected=False):
    """Harness H: EthicsGuard real, módulos dublês, guarda com ação observe."""
    if recorder is None:
        recorder = EpisodicRecorder()
    modules = harness_modules(recorder, jailbreak_detected=jailbreak_detected)
    if not include_episodic:
        modules.pop(ModuleType.EPISODIC, None)
    with silence():
        router = CentralRouter(modules, ethics_guard())
    router.vram_guard = _ObserveGuard()
    return router, recorder


def empty_router():
    """Harness Vazio: mapa sem módulos e o VramDefenseGuard real."""
    with silence():
        return CentralRouter({}, ethics_guard())


def brain_kwargs():
    signature = inspect.signature(CompleteNexusBrain.__init__)
    kwargs = {}
    for name in signature.parameters:
        if name == "self":
            continue
        if name == "db_prefix":
            kwargs[name] = ":memory:"
        elif name in ("enable_router", "enable_integration_orchestration"):
            kwargs[name] = True
        elif name.startswith("enable_"):
            kwargs[name] = False
    return kwargs


def make_brain():
    with silence():
        return CompleteNexusBrain(**brain_kwargs())


def wrap_episodic(memory):
    """Registra e delega aos métodos reais da instância."""
    calls = []
    for name in EPISODIC_METHODS:
        original = getattr(memory, name)

        def wrapped(*args, original=original, name=name, **kwargs):
            result = original(*args, **kwargs)
            calls.append((name, args, dict(kwargs), result))
            return result

        setattr(memory, name, wrapped)
    return calls


def method_names(calls):
    return [call[0] for call in calls]


class C1PackageReexport(unittest.TestCase):
    def test_monolith_reexports_the_same_classes_and_the_package_does_not(self):
        self.assertIs(brain.RealEpisodicMemory, episodic_module.RealEpisodicMemory)
        self.assertIs(brain.Episode, episodic_module.Episode)
        self.assertEqual(brain.RealEpisodicMemory.__module__, "core.cognitivo.episodic_memory")
        self.assertFalse(hasattr(cognitivo_pkg, "RealEpisodicMemory"))
        self.assertNotIn("RealEpisodicMemory", cognitivo_pkg.__all__)
        with self.assertRaises(ImportError):
            from core.cognitivo import RealEpisodicMemory as _reexported  # noqa: F401
        source = (ROOT / "core/constitutional_brain.py").read_text(encoding="utf-8").splitlines()
        self.assertIn(
            "from core.cognitivo.episodic_memory import Episode, RealEpisodicMemory",
            source[2033],
        )
        package = (ROOT / "core/cognitivo/__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("episodic_memory", package)


class C2BrainConstruction(unittest.TestCase):
    def test_harness_b_wires_one_empty_episodic_memory(self):
        signature = inspect.signature(CompleteNexusBrain.__init__)
        self.assertNotIn("enable_episodic", signature.parameters)
        calls = []
        real_start = RealEpisodicMemory.start_episode

        def spy(self, *args, **kwargs):
            calls.append((args, kwargs))
            return real_start(self, *args, **kwargs)

        with patch.object(RealEpisodicMemory, "start_episode", spy), silence():
            created = CompleteNexusBrain(**brain_kwargs())
        self.addCleanup(created.episodic_memory.close)
        self.assertEqual(calls, [])
        self.assertIs(type(created.episodic_memory), episodic_module.RealEpisodicMemory)
        self.assertEqual(created.episodic_memory.get_statistics(), ZERO_EPISODIC)
        self.assertIs(created.router.modules[ModuleType.EPISODIC], created.episodic_memory)
        integration = created.integration_orchestration.all_modules
        self.assertNotIn("episodic", integration)
        self.assertIsNone(integration["episodic_future"])
        statistics = created.get_statistics()
        self.assertEqual(statistics["episodic"], ZERO_EPISODIC)
        self.assertEqual(statistics["interactions"], 0)


class C3EmptyModulesDoNotSearch(unittest.TestCase):
    def test_ping_and_recall_fail_at_jailbreak_before_any_episodic_call(self):
        router = empty_router()
        self.assertEqual(router.vram_guard.evaluate().action, "observe")
        calls = []

        def spy(name):
            def inner(self, *args, **kwargs):
                calls.append(name)
                raise AssertionError(name)
            return inner

        patches = [
            patch.object(RealEpisodicMemory, name, spy(name))
            for name in EPISODIC_METHODS
        ]
        for item in patches:
            item.start()
        self.addCleanup(lambda: [item.stop() for item in patches])
        for prompt in ("ping", RECALL):
            with self.subTest(prompt=prompt):
                with silence():
                    result = router.route(prompt, {})
                self.assertIsInstance(result, dict)
                self.assertIs(result["success"], False)
                self.assertEqual(result["error"], "JAILBREAK_CHECK_FAILED")
                self.assertEqual(result["failed_modules"], ["jailbreak"])
                self.assertNotEqual(result["error"], "VRAM_PRESSURE")
        self.assertEqual(calls, [])


class C4WhoCallsSearch(unittest.TestCase):
    TABLE = (
        ("ping", "query", 0),
        ("", "query", 0),
        ("learn this fact", "query", 0),
        ("Tell me about item 9", "query", 0),
        ("should I plan my week", "planning", 0),
        ("what did I say about the right choice", "ethical", 0),
        (RECALL, "recall", 1),
    )

    def test_only_recall_calls_search_episodes(self):
        for prompt, request_type, searches in self.TABLE:
            with self.subTest(prompt=prompt):
                router, recorder = harness_h()
                with silence():
                    result = router.route(prompt, {})
                names = method_names(recorder.calls)
                self.assertEqual(names.count("search_episodes"), searches)
                self.assertEqual(request_type, result["request_type"])
                for other in EPISODIC_METHODS:
                    if other != "search_episodes":
                        self.assertEqual(names.count(other), 0, names)
                if searches == 0:
                    self.assertIs(result["success"], True)
                    self.assertIs(result["degraded"], False)
                    self.assertNotIn("episodic", result["modules_executed"])
                if prompt == "what did I say about the right choice":
                    self.assertIn("ethical", result["decision"]["proposed"])
                    self.assertIn("recall", result["decision"]["proposed"])
                    self.assertEqual(result["decision"]["selected"], "ethical")
                if searches == 1:
                    self.assertEqual(names, ["search_episodes"])


class C5RecallCallAndPublicResult(unittest.TestCase):
    def test_search_kwargs_are_dropped_from_the_public_dict_and_cached(self):
        router, recorder = harness_h()
        with silence():
            first = router.route(RECALL, {})
            second = router.route(RECALL, {})
        self.assertEqual(
            recorder.calls,
            [("search_episodes", (), {"query": RECALL, "limit": 3})],
        )
        self.assertIs(first["success"], True)
        self.assertIs(first["degraded"], False)
        self.assertEqual(first["request_type"], "recall")
        self.assertEqual(first["answer"], "No answer generated")
        self.assertEqual(
            first["modules_executed"],
            ["jailbreak", "budget", "memory", "episodic", "log"],
        )
        self.assertNotIn("episodic", first)
        self.assertNotIn("episode_id", first)
        self.assertEqual(first["episodes"], [])
        self.assertEqual(second["episodes"], [])
        self.assertIs(second["cached"], True)
        self.assertEqual(method_names(recorder.calls).count("search_episodes"), 1)


class C6ContextDoesNotRetarget(unittest.TestCase):
    def test_request_type_in_context_still_searches_and_skips_cache(self):
        router, recorder = harness_h()
        context = {"request_type": "query"}
        with silence():
            first = router.route(RECALL, context)
            second = router.route(RECALL, context)
        expected = ("search_episodes", (), {"query": RECALL, "limit": 3})
        self.assertEqual(recorder.calls, [expected, expected])
        self.assertEqual(first["request_type"], "recall")
        self.assertIs(second["cached"], False)
        self.assertEqual(method_names(recorder.calls).count("search_episodes"), 2)


class C7EpisodicFailureDegrades(unittest.TestCase):
    def test_missing_module_degrades_without_a_call(self):
        router, recorder = harness_h(include_episodic=False)
        with silence():
            result = router.route(RECALL, {})
        self.assertIs(result["success"], True)
        self.assertIs(result["degraded"], True)
        self.assertEqual(result["failed_modules"], ["episodic"])
        self.assertEqual(
            result["failed_modules_detail"],
            [{"module": "episodic", "error": "Module not found in brain"}],
        )
        self.assertEqual(result["request_type"], "recall")
        self.assertEqual(recorder.calls, [])
        self.assertNotIn(result.get("error"), BLOCKING_ERRORS)

    def test_search_exception_degrades_and_is_not_cached(self):
        recorder = EpisodicRecorder(search_raises=ValueError("query must be non-empty"))
        router, recorder = harness_h(recorder)
        with silence():
            first = router.route(RECALL, {})
            second = router.route(RECALL, {})
        self.assertIs(first["success"], True)
        self.assertIs(first["degraded"], True)
        self.assertEqual(first["failed_modules"], ["episodic"])
        self.assertEqual(
            first["failed_modules_detail"],
            [{"module": "episodic", "error": "query must be non-empty"}],
        )
        self.assertIs(first["cached"], False)
        self.assertIs(second["cached"], False)
        self.assertEqual(method_names(recorder.calls).count("search_episodes"), 2)
        self.assertNotIn(first.get("error"), BLOCKING_ERRORS)
        self.assertNotIn(second.get("error"), BLOCKING_ERRORS)


class C8JailbreakCutsTheSearch(unittest.TestCase):
    def test_detected_jailbreak_does_not_call_episodic_methods(self):
        router, recorder = harness_h(jailbreak_detected=True)
        with silence():
            result = router.route(RECALL, {})
        self.assertIs(result["success"], False)
        self.assertEqual(result["error"], "JAILBREAK_DETECTED")
        self.assertEqual(result["failed_modules"], ["jailbreak"])
        self.assertEqual(recorder.calls, [])


class C9InvalidInputPropagates(unittest.TestCase):
    def test_non_strings_raise_at_the_prompt_slice(self):
        expectations = (
            (True, "bool"),
            (False, "bool"),
            (None, "NoneType"),
            (1, "int"),
        )
        for prompt, type_name in expectations:
            with self.subTest(prompt=prompt):
                router, recorder = harness_h()
                caught, frames = raised_frames(TypeError, lambda: router.route(prompt, {}))
                self.assertIn(type_name, str(caught))
                self.assertIn("not subscriptable", str(caught))
                self.assertIn(("constitutional_brain.py", 9905, "_route"), frames)
                self.assertEqual(recorder.calls, [])

    def test_bytes_and_list_raise_inside_the_ethics_check(self):
        cases = (
            (b"remember", TypeError, "a bytes-like object is required, not 'str'"),
            (["remember the last time"], AttributeError, "'list' object has no attribute 'lower'"),
        )
        for prompt, exc_type, message in cases:
            with self.subTest(prompt=prompt):
                router, recorder = harness_h()
                caught, frames = raised_frames(exc_type, lambda: router.route(prompt, {}))
                self.assertEqual(str(caught), message)
                self.assertIn(("constitutional_brain.py", 9370, "pre_execution_check"), frames)
                self.assertEqual(recorder.calls, [])

    def test_non_dict_context_raises_before_the_search(self):
        router, recorder = harness_h()
        caught, frames = raised_frames(ValueError, lambda: router.route("ping", ["nope"]))
        self.assertEqual(str(caught), "context must be a dictionary")
        self.assertIn(("constitutional_brain.py", 9897, "_route"), frames)
        self.assertEqual(recorder.calls, [])


class C10RealMemoryInTheMap(unittest.TestCase):
    def _router(self, memory):
        modules = harness_modules(memory)
        with silence():
            router = CentralRouter(modules, ethics_guard())
        router.vram_guard = _ObserveGuard()
        return router

    def test_closed_episode_list_is_not_copied_into_the_public_result(self):
        memory = RealEpisodicMemory(":memory:")
        self.addCleanup(memory.close)
        episode_id = memory.start_episode(RECALL)
        memory.close_episode(episode_id)
        seen = []
        original = memory.search_episodes

        def wrapped(*args, **kwargs):
            found = original(*args, **kwargs)
            seen.append((args, dict(kwargs), [episode.episode_id for episode in found]))
            return found

        memory.search_episodes = wrapped
        with silence():
            result = self._router(memory).route(RECALL, {})
        self.assertEqual(seen, [((), {"query": RECALL, "limit": 3}, [episode_id])])
        self.assertNotIn("episode_id", result)
        self.assertNotIn("episodic", result)
        self.assertIn("episodic", result["modules_executed"])

    def test_open_episode_stays_open_and_the_public_result_omits_the_list(self):
        memory = RealEpisodicMemory(":memory:")
        self.addCleanup(memory.close)
        closed_id = memory.start_episode(RECALL)
        memory.close_episode(closed_id)
        open_id = memory.start_episode("remember the last time open")
        seen = []
        original = memory.search_episodes

        def wrapped(*args, **kwargs):
            found = original(*args, **kwargs)
            seen.append((args, dict(kwargs)))
            return found

        memory.search_episodes = wrapped
        with silence():
            result = self._router(memory).route(RECALL, {})
        self.assertEqual(seen, [((), {"query": RECALL, "limit": 3})])
        self.assertNotIn("episode_id", result)
        self.assertNotIn("episodic", result)
        self.assertEqual(memory.current_episode_id, open_id)

    def test_forced_recall_of_empty_query_degrades_without_creating_episodes(self):
        memory = RealEpisodicMemory(":memory:")
        self.addCleanup(memory.close)
        router = self._router(memory)
        router.request_type_proposers = (lambda state: ("recall",),)
        with silence():
            result = router.route("", {})
        self.assertIs(result["success"], True)
        self.assertIs(result["degraded"], True)
        self.assertEqual(
            result["failed_modules_detail"],
            [{"module": "episodic", "error": "query must be non-empty"}],
        )
        self.assertEqual(memory.get_statistics(), ZERO_EPISODIC)


class C11LearningIsNotADefaultProposer(unittest.TestCase):
    def test_forced_learning_searches_and_does_not_call_the_learner(self):
        router, recorder = harness_h()
        router.request_type_proposers = (lambda state: ("learning",),)
        with silence():
            result = router.route("ping", {})
        self.assertEqual(result["request_type"], "learning")
        self.assertEqual(
            recorder.calls,
            [("search_episodes", (), {"query": "ping", "limit": 3})],
        )
        self.assertIn("episodic", result["modules_executed"])
        self.assertIn("learner", result["modules_executed"])
        self.assertIs(result["success"], True)
        self.assertEqual(result["answer"], "No answer generated")


class C12ThinkWithRouter(unittest.TestCase):
    def _wrapped(self):
        created = make_brain()
        self.addCleanup(created.episodic_memory.close)
        return created, wrap_episodic(created.episodic_memory)

    def test_ping_starts_and_closes_without_search(self):
        created, calls = self._wrapped()
        with silence():
            result = created.think("ping", use_router=True)
        self.assertNotIn("error", result)
        self.assertEqual(method_names(calls), ["start_episode", "close_episode"])
        self.assertEqual(calls[0][1], ())
        self.assertEqual(calls[0][2], {
            "title": "Q: ping", "trigger_query": "ping", "importance": 0.7,
        })
        self.assertEqual(result["episode_id"], calls[0][3])
        self.assertEqual(calls[1][1], (result["episode_id"],))
        self.assertEqual(calls[1][2], {})

    def test_recall_searches_between_start_and_close(self):
        created, calls = self._wrapped()
        with silence():
            result = created.think(RECALL, use_router=True)
        self.assertNotIn("error", result)
        self.assertEqual(
            method_names(calls),
            ["start_episode", "search_episodes", "close_episode"],
        )
        self.assertEqual(calls[0][2], {
            "title": "Q: remember the last time",
            "trigger_query": RECALL,
            "importance": 0.7,
        })
        self.assertEqual(calls[1][1], ())
        self.assertEqual(calls[1][2], {"query": RECALL, "limit": 3})
        self.assertEqual(calls[2][1], (result["episode_id"],))
        self.assertEqual(calls[2][2], {})
        self.assertNotIn("get_episode_memories", method_names(calls))

    def test_long_question_truncates_only_the_title(self):
        created, calls = self._wrapped()
        question = "remember " + ("x" * 80)
        with silence():
            result = created.think(question, use_router=True)
        self.assertNotIn("error", result)
        self.assertEqual(calls[0][2]["title"], "Q: " + question[:50])
        self.assertEqual(calls[0][2]["trigger_query"], question)

    def test_jailbreak_returns_before_any_episodic_call(self):
        created, calls = self._wrapped()
        created.jailbreak_detector.detect = lambda prompt: SimpleNamespace(
            detected=True, severity=1.0, attack_type=SimpleNamespace(value="probe"),
        )
        with silence():
            result = created.think("ping", use_router=True)
        self.assertEqual(result["error"], "JAILBREAK_DETECTED")
        self.assertNotIn("episode_id", result)
        self.assertEqual(calls, [])

    def test_denied_budget_returns_before_any_episodic_call(self):
        created, calls = self._wrapped()
        created.budget.request_budget = lambda *args, **kwargs: False
        with silence():
            result = created.think("ping", use_router=True)
        self.assertEqual(result["error"], "BUDGET_EXCEEDED")
        self.assertNotIn("episode_id", result)
        self.assertEqual(calls, [])

    def test_self_modification_is_an_ethical_violation_before_the_episode(self):
        created, calls = self._wrapped()
        with silence():
            result = created.think("please rewrite yourself", use_router=True)
        self.assertEqual(result["error"], "ETHICAL_VIOLATION")
        self.assertEqual(
            result["ethical_concern"],
            "P6 VIOLATION: Self-modification attempt blocked",
        )
        self.assertEqual(calls, [])

    def test_route_exception_closes_the_episode(self):
        created, calls = self._wrapped()

        def boom(*args, **kwargs):
            raise RuntimeError("probe-boom")

        created.router.route = boom
        with silence():
            result = created.think("ping", use_router=True)
        self.assertEqual(result["error"], "EXCEPTION_DURING_THINK")
        self.assertEqual(result["exception"], "probe-boom")
        self.assertIn("episode_id", result)
        self.assertEqual(method_names(calls), ["start_episode", "close_episode"])
        self.assertEqual(calls[1][1], (result["episode_id"],))
        self.assertIsNone(created.episodic_memory.current_episode_id)


class C13ThinkWithoutRouter(unittest.TestCase):
    def test_legacy_think_learns_from_the_open_episode_and_does_not_search(self):
        created = make_brain()
        self.addCleanup(created.episodic_memory.close)
        calls = wrap_episodic(created.episodic_memory)
        with silence():
            result = created.think("ping", use_router=False)
        self.assertNotIn("error", result)
        self.assertEqual(
            method_names(calls),
            ["start_episode", "get_episode_memories", "close_episode"],
        )
        self.assertEqual(calls[0][2], {
            "title": "Q: ping", "trigger_query": "ping", "importance": 0.7,
        })
        self.assertEqual(calls[1][1], (result["episode_id"],))
        self.assertEqual(calls[2][1], (result["episode_id"],))
        self.assertNotIn("search_episodes", method_names(calls))


class C14HistoryAndConsolidation(unittest.TestCase):
    def test_history_and_consolidate_read_the_two_closed_episodes(self):
        created = make_brain()
        self.addCleanup(created.episodic_memory.close)
        calls = wrap_episodic(created.episodic_memory)
        question = "remember " + ("x" * 80)
        with silence():
            legacy = created.think("ping", use_router=False)
            routed = created.think(question, use_router=True)
        self.assertNotIn("error", legacy)
        self.assertNotIn("error", routed)
        calls.clear()
        with silence():
            history = created.get_conversation_history(limit=5)
        self.assertEqual(len(history), 2)
        self.assertEqual(set(history[0]), {
            "episode_id", "title", "trigger_query", "importance", "duration", "memory_count",
        })
        self.assertEqual(history[0]["title"], "Q: " + question[:50])
        self.assertEqual(history[0]["trigger_query"], question)
        self.assertIsInstance(history[0]["duration"], float)
        self.assertEqual(history[0]["memory_count"], 0)
        self.assertEqual(set(history[1]), set(history[0]))
        self.assertEqual(calls[0][0], "search_episodes")
        self.assertEqual(calls[0][1], ())
        self.assertEqual(calls[0][2], {"status": "closed", "limit": 5})
        memory_calls = [call for call in calls if call[0] == "get_episode_memories"]
        self.assertEqual(len(memory_calls), 2)
        self.assertEqual(
            {call[1] for call in memory_calls},
            {(item["episode_id"],) for item in history},
        )
        calls.clear()
        with silence():
            report = created.consolidate_learning(last_n=10)
        self.assertIsInstance(report, ConsolidationReport)
        self.assertEqual(calls[0][0], "search_episodes")
        self.assertEqual(calls[0][1], ())
        self.assertEqual(calls[0][2], {"status": "closed", "limit": 10})
        consolidated = [call for call in calls if call[0] == "get_episode_memories"]
        self.assertEqual(len(consolidated), 2)
        self.assertEqual(
            {call[1] for call in consolidated},
            {(item["episode_id"],) for item in history},
        )


class C15ExistingContractsStayUntouched(unittest.TestCase):
    UNTOUCHED = (
        "test_episodic_memory_extraction_contract.py",
        "test_episodic_retrieval_contract.py",
        "test_episodic_like_literal_contract.py",
        "test_episodic_memory_search_limits_contract.py",
        "test_episodic_memory_statistics_contract.py",
        "test_episodic_memory_concurrency_contract.py",
        "test_router_fail_closed_contract.py",
        "test_router_decision_cycle_bridge_contract.py",
        "tools/cognitivo_mutation.py",
    )

    def test_claims_stay_empty_and_module_type_does_not_grow(self):
        claims = (ROOT / "docs/agentes/CLAIMS.yaml").read_text(encoding="utf-8")
        self.assertIn("claims: []", claims)
        self.assertEqual(frozenset(member.name for member in ModuleType), MODULE_NAMES)
        for relative in self.UNTOUCHED:
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
